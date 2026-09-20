import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

from sgl_kernel import gguf_q6_k_grouped_mm, gguf_q6_k_prepare


def _make_raw_q6_k(rows: int, blocks: int):
    torch.manual_seed(11)
    signed = torch.randint(-32, 32, (rows, blocks, 256), dtype=torch.int8)
    values = (signed.to(torch.int16) + 32).to(torch.uint8)
    scale_codes = torch.randint(-8, 9, (rows, blocks, 16), dtype=torch.int8)
    base_scales = (
        torch.rand(rows, blocks, dtype=torch.float16) * 0.02 + 0.001
    ).contiguous()

    ql = torch.empty((rows, blocks, 128), dtype=torch.uint8)
    qh = torch.empty((rows, blocks, 64), dtype=torch.uint8)
    for half in range(2):
        base = half * 128
        q0 = values[:, :, base : base + 32]
        q32 = values[:, :, base + 32 : base + 64]
        q64 = values[:, :, base + 64 : base + 96]
        q96 = values[:, :, base + 96 : base + 128]
        ql[:, :, half * 64 : half * 64 + 32] = (q0 & 0x0F) | ((q64 & 0x0F) << 4)
        ql[:, :, half * 64 + 32 : half * 64 + 64] = (q32 & 0x0F) | ((q96 & 0x0F) << 4)
        qh[:, :, half * 32 : half * 32 + 32] = (
            ((q0 >> 4) & 0x03)
            | (((q32 >> 4) & 0x03) << 2)
            | (((q64 >> 4) & 0x03) << 4)
            | (((q96 >> 4) & 0x03) << 6)
        )

    raw = torch.cat(
        (
            ql,
            qh,
            scale_codes.view(torch.uint8),
            base_scales.view(torch.uint8).reshape(rows, blocks, 2),
        ),
        dim=-1,
    ).reshape(rows, blocks * 210)
    scales = base_scales.unsqueeze(-1) * scale_codes.to(torch.float16)
    return raw, signed.reshape(rows, blocks * 256), scales.reshape(rows, blocks * 16)


def _reference(activations, weights, scales, rows_per_expert):
    experts, n, k = weights.shape
    dequant = (
        weights.float().reshape(experts, n, k // 16, 16) * scales.float().unsqueeze(-1)
    ).reshape(experts, n, k)
    outputs = []
    row = 0
    for expert, count in enumerate(rows_per_expert.cpu().tolist()):
        outputs.append(activations[row : row + count].float() @ dequant[expert].T)
        row += count
    return torch.cat(outputs).to(activations.dtype)


def test_q6_k_prepare_matches_reference():
    raw, expected_weights, expected_scales = _make_raw_q6_k(3, 2)
    weights, scales = gguf_q6_k_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), expected_scales, rtol=0, atol=0)
    assert weights.numel() + scales.numel() * scales.element_size() == 3 * 2 * 288
    assert weights.numel() + scales.numel() * scales.element_size() < 3 * 2 * 512


def test_q6_k_prepared_values_match_gguf_reference():
    raw, weights, scales = _make_raw_q6_k(3, 2)
    actual = (
        weights.float().reshape(3, -1, 16) * scales.float().unsqueeze(-1)
    ).reshape(3, -1)
    expected = torch.from_numpy(
        dequantize(raw.numpy(), GGMLQuantizationType.Q6_K).copy()
    )
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 9, 17, 33, 65, 129, 257])
@pytest.mark.parametrize("k", [256, 512])
def test_q6_k_grouped_mm_matches_reference(dtype, m, k):
    n = 64
    raw, _, _ = _make_raw_q6_k(n, k // 256)
    weights, scales = gguf_q6_k_prepare(raw.to("xpu"))
    activations = torch.randn(m, k, dtype=dtype, device="xpu") * 0.1
    rows = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q6_k_grouped_mm(
        activations, weights.reshape(1, n, k), scales.reshape(1, n, k // 16), rows
    )
    expected = _reference(
        activations.cpu(),
        weights.reshape(1, n, k).cpu(),
        scales.reshape(1, n, k // 16).cpu(),
        rows,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


def test_q6_k_grouped_mm_multi_expert_and_n_tail():
    experts, n, k = 3, 72, 256
    raw, _, _ = _make_raw_q6_k(experts * n, k // 256)
    weights, scales = gguf_q6_k_prepare(raw.to("xpu"))
    rows = torch.tensor([1, 5, 2], dtype=torch.int32, device="xpu")
    activations = (
        torch.randn(int(rows.sum()), k, dtype=torch.bfloat16, device="xpu") * 0.1
    )
    weights = weights.reshape(experts, n, k)
    scales = scales.reshape(experts, n, k // 16)
    actual = gguf_q6_k_grouped_mm(activations, weights, scales, rows)
    expected = _reference(activations.cpu(), weights.cpu(), scales.cpu(), rows)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


def test_q6_k_grouped_mm_rejects_partial_block_k():
    activations = torch.empty((1, 32), dtype=torch.bfloat16, device="xpu")
    weights = torch.empty((1, 8, 32), dtype=torch.int8, device="xpu")
    scales = torch.empty((1, 8, 2), dtype=torch.float16, device="xpu")
    rows = torch.tensor([1], dtype=torch.int32, device="xpu")

    with pytest.raises(RuntimeError, match="256-value Q6_K block"):
        gguf_q6_k_grouped_mm(activations, weights, scales, rows)


@pytest.mark.parametrize(
    "raw",
    [
        torch.empty((1, 210), dtype=torch.int8),
        torch.empty((210,), dtype=torch.uint8),
        torch.empty((1, 211), dtype=torch.uint8),
    ],
)
def test_q6_k_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_q6_k_prepare(raw)
