import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

from sgl_kernel import gguf_q5_k_grouped_mm, gguf_q5_k_prepare

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

Q5_K_BLOCK_BYTES = 176


def _set_fp16(raw: torch.Tensor, byte_offset: int, value: float) -> None:
    bits = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
    raw[..., byte_offset : byte_offset + 2] = bits


def _make_raw_q5_k(rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260920 + rows + blocks)
    raw = torch.randint(
        0,
        256,
        (rows, blocks, Q5_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.0078125)
    _set_fp16(raw, 2, 0.00390625)
    return raw.reshape(rows, blocks * Q5_K_BLOCK_BYTES)


def _prepared_reference(raw: torch.Tensor):
    rows = raw.shape[0]
    blocks = raw.shape[1] // Q5_K_BLOCK_BYTES
    block = raw.reshape(rows, blocks, Q5_K_BLOCK_BYTES)
    bases = block[..., :4].contiguous().view(torch.float16).reshape(rows, blocks, 2)
    meta = block[..., 4:16].to(torch.int32)
    scale6 = torch.cat(
        (
            meta[..., :4] & 0x3F,
            (meta[..., 8:12] & 0x0F) | ((meta[..., :4] >> 2) & 0x30),
        ),
        dim=-1,
    )
    min6 = torch.cat(
        (
            meta[..., 4:8] & 0x3F,
            (meta[..., 8:12] >> 4) | ((meta[..., 4:8] >> 2) & 0x30),
        ),
        dim=-1,
    )
    scales = (bases[..., :1] * scale6.to(torch.float16)).reshape(rows, -1)
    minimums = (bases[..., 1:] * min6.to(torch.float16)).reshape(rows, -1)

    qh = block[..., 16:48]
    qs = block[..., 48:].reshape(rows, blocks, 4, 32)
    values = []
    for group in range(8):
        low = (qs[..., group // 2, :] >> ((group & 1) * 4)) & 0x0F
        high = ((qh >> group) & 0x01) << 4
        values.append(low | high)
    weights = torch.stack(values, dim=2).reshape(rows, blocks * 256).to(torch.int8)
    return weights, scales, minimums


def _reference(activations, weights, scales, minimums, rows_per_expert):
    experts, n, k = weights.shape
    dequantized = (
        weights.float().reshape(experts, n, k // 32, 32)
        * scales.float().unsqueeze(-1)
        - minimums.float().unsqueeze(-1)
    ).reshape(experts, n, k)
    outputs = []
    row = 0
    for expert, count in enumerate(rows_per_expert.cpu().tolist()):
        outputs.append(activations[row : row + count].float() @ dequantized[expert].T)
        row += count
    return torch.cat(outputs).to(activations.dtype)


def test_q5_k_prepare_matches_reference():
    raw = _make_raw_q5_k(3, 2)
    expected = _prepared_reference(raw)
    actual = gguf_q5_k_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    assert sum(t.numel() * t.element_size() for t in actual) == 3 * 2 * 288
    assert sum(t.numel() * t.element_size() for t in actual) < 3 * 2 * 512


def test_q5_k_prepared_values_match_gguf_reference():
    raw = _make_raw_q5_k(3, 2)
    weights, scales, minimums = _prepared_reference(raw)
    actual = (
        weights.float().reshape(3, -1, 32) * scales.float().unsqueeze(-1)
        - minimums.float().unsqueeze(-1)
    ).reshape(3, -1)
    expected = torch.from_numpy(dequantize(raw.numpy(), GGMLQuantizationType.Q5_K).copy())
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 9, 17, 33, 65, 129, 257])
@pytest.mark.parametrize("k", [256, 512])
def test_q5_k_grouped_mm_matches_reference(dtype, m, k):
    n = 64
    raw = _make_raw_q5_k(n, k // 256)
    weights, scales, minimums = gguf_q5_k_prepare(raw.to("xpu"))
    activations = torch.randn(m, k, dtype=dtype, device="xpu") * 0.1
    rows = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q5_k_grouped_mm(
        activations,
        weights.reshape(1, n, k),
        scales.reshape(1, n, k // 32),
        minimums.reshape(1, n, k // 32),
        rows,
    )
    expected = _reference(
        activations.cpu(),
        weights.reshape(1, n, k).cpu(),
        scales.reshape(1, n, k // 32).cpu(),
        minimums.reshape(1, n, k // 32).cpu(),
        rows,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.2)


def test_q5_k_grouped_mm_multi_expert_and_n_tail():
    experts, n, k = 3, 72, 256
    raw = _make_raw_q5_k(experts * n, k // 256)
    weights, scales, minimums = gguf_q5_k_prepare(raw.to("xpu"))
    rows = torch.tensor([1, 5, 2], dtype=torch.int32, device="xpu")
    activations = torch.randn(int(rows.sum()), k, dtype=torch.bfloat16, device="xpu") * 0.1
    weights = weights.reshape(experts, n, k)
    scales = scales.reshape(experts, n, k // 32)
    minimums = minimums.reshape(experts, n, k // 32)
    actual = gguf_q5_k_grouped_mm(activations, weights, scales, minimums, rows)
    expected = _reference(activations.cpu(), weights.cpu(), scales.cpu(), minimums.cpu(), rows)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.2)


def test_q5_k_grouped_mm_zero_scale_keeps_minimum():
    n, k = 64, 256
    raw = torch.zeros((n, 1, Q5_K_BLOCK_BYTES), dtype=torch.uint8)
    _set_fp16(raw, 0, 0.0)
    _set_fp16(raw, 2, 0.03125)
    raw[..., 8:12] = 0x3F
    weights, scales, minimums = gguf_q5_k_prepare(raw.reshape(n, -1).to("xpu"))
    activations = torch.ones((1, k), dtype=torch.bfloat16, device="xpu")
    rows = torch.tensor([1], dtype=torch.int32, device="xpu")
    actual = gguf_q5_k_grouped_mm(
        activations,
        weights.reshape(1, n, k),
        scales.reshape(1, n, k // 32),
        minimums.reshape(1, n, k // 32),
        rows,
    )
    expected = _reference(
        activations.cpu(),
        weights.reshape(1, n, k).cpu(),
        scales.reshape(1, n, k // 32).cpu(),
        minimums.reshape(1, n, k // 32).cpu(),
        rows,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.2)


def test_q5_k_grouped_mm_rejects_partial_block_k():
    activations = torch.empty((1, 32), dtype=torch.bfloat16, device="xpu")
    weights = torch.empty((1, 8, 32), dtype=torch.int8, device="xpu")
    scales = torch.empty((1, 8, 1), dtype=torch.float16, device="xpu")
    rows = torch.tensor([1], dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="256-value Q5_K block"):
        gguf_q5_k_grouped_mm(activations, weights, scales, scales, rows)


@pytest.mark.parametrize(
    "raw",
    [
        torch.empty((1, 176), dtype=torch.int8),
        torch.empty((176,), dtype=torch.uint8),
        torch.empty((1, 177), dtype=torch.uint8),
    ],
)
def test_q5_k_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_q5_k_prepare(raw)
