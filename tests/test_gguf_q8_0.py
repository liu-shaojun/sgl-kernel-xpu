import pytest
import torch

from sgl_kernel import gguf_q8_0_grouped_mm, gguf_q8_0_prepare


def _make_raw_q8_0(rows: int, blocks: int):
    torch.manual_seed(7)
    scales = (torch.rand(rows, blocks, dtype=torch.float16) * 0.02 + 0.001).contiguous()
    weights = torch.randint(-128, 128, (rows, blocks, 32), dtype=torch.int8)
    raw = torch.cat(
        (
            scales.view(torch.uint8).reshape(rows, blocks, 2),
            weights.view(torch.uint8),
        ),
        dim=-1,
    ).reshape(rows, blocks * 34)
    return raw, weights.reshape(rows, blocks * 32), scales


def _reference(activations, weights, scales, rows_per_expert):
    experts, n, k = weights.shape
    dequant = (
        weights.float().reshape(experts, n, k // 32, 32) * scales.float().unsqueeze(-1)
    ).reshape(experts, n, k)
    outputs = []
    row = 0
    for expert, count in enumerate(rows_per_expert.cpu().tolist()):
        outputs.append(activations[row : row + count].float() @ dequant[expert].T)
        row += count
    return torch.cat(outputs).to(activations.dtype)


def test_q8_0_prepare_is_byte_exact():
    raw, expected_weights, expected_scales = _make_raw_q8_0(3, 5)
    weights, scales = gguf_q8_0_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), expected_scales, rtol=0, atol=0)
    assert (
        weights.numel() * weights.element_size()
        + scales.numel() * scales.element_size()
        == raw.numel()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 9, 17, 33, 65, 129, 257])
def test_q8_0_grouped_mm_matches_reference(dtype, m):
    n, k = 64, 256
    raw, _, _ = _make_raw_q8_0(n, k // 32)
    weights, scales = gguf_q8_0_prepare(raw.to("xpu"))
    activations = torch.randn(m, k, dtype=dtype, device="xpu") * 0.1
    rows = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q8_0_grouped_mm(
        activations, weights.reshape(1, n, k), scales.reshape(1, n, k // 32), rows
    )
    expected = _reference(
        activations.cpu(),
        weights.reshape(1, n, k).cpu(),
        scales.reshape(1, n, k // 32).cpu(),
        rows,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


def test_q8_0_grouped_mm_multi_expert_and_n_tail():
    experts, n, k = 3, 72, 256
    raw, _, _ = _make_raw_q8_0(experts * n, k // 32)
    weights, scales = gguf_q8_0_prepare(raw.to("xpu"))
    rows = torch.tensor([1, 5, 2], dtype=torch.int32, device="xpu")
    activations = (
        torch.randn(int(rows.sum()), k, dtype=torch.bfloat16, device="xpu") * 0.1
    )
    weights = weights.reshape(experts, n, k)
    scales = scales.reshape(experts, n, k // 32)
    actual = gguf_q8_0_grouped_mm(activations, weights, scales, rows)
    expected = _reference(activations.cpu(), weights.cpu(), scales.cpu(), rows)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


@pytest.mark.parametrize(
    "raw",
    [
        torch.empty((1, 34), dtype=torch.int8),
        torch.empty((34,), dtype=torch.uint8),
        torch.empty((1, 35), dtype=torch.uint8),
    ],
)
def test_q8_0_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_q8_0_prepare(raw)
