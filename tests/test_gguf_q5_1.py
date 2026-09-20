import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

from sgl_kernel import gguf_q5_1_grouped_mm, gguf_q5_1_prepare

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

Q5_1_BLOCK_BYTES = 24


def _set_fp16(raw: torch.Tensor, byte_offset: int, value: float) -> None:
    bits = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
    raw[..., byte_offset : byte_offset + 2] = bits


def _make_raw_q5_1(rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260924 + rows + blocks)
    raw = torch.randint(
        0,
        256,
        (rows, blocks, Q5_1_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.0078125)
    _set_fp16(raw, 2, -0.125)
    return raw.reshape(rows, blocks * Q5_1_BLOCK_BYTES)


def _prepared_reference(raw: torch.Tensor):
    rows = raw.shape[0]
    blocks = raw.shape[1] // Q5_1_BLOCK_BYTES
    block = raw.reshape(rows, blocks, Q5_1_BLOCK_BYTES)
    scales = block[..., :2].contiguous().view(torch.float16).reshape(rows, blocks)
    offsets = block[..., 2:4].contiguous().view(torch.float16).reshape(rows, blocks)
    qh = block[..., 4:8].to(torch.int32)
    qs = block[..., 8:24].to(torch.int32)
    values = []
    for i in range(32):
        low4 = (qs[..., i & 15] >> ((i // 16) * 4)) & 0x0F
        high1 = (qh[..., i // 8] >> (i & 7)) & 1
        values.append((low4 | (high1 << 4)).to(torch.int8))
    return (
        torch.stack(values, dim=2).reshape(rows, blocks * 32),
        scales,
        -offsets,
    )


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


def test_q5_1_prepare_matches_reference():
    raw = _make_raw_q5_1(3, 8)
    expected = _prepared_reference(raw)
    actual = gguf_q5_1_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    assert sum(t.numel() * t.element_size() for t in actual) == 3 * 8 * 36
    assert sum(t.numel() * t.element_size() for t in actual) < 3 * 8 * 64


def test_q5_1_prepared_values_match_gguf_reference():
    raw = _make_raw_q5_1(3, 8)
    weights, scales, minimums = _prepared_reference(raw)
    actual = (
        weights.float().reshape(3, -1, 32) * scales.float().unsqueeze(-1)
        - minimums.float().unsqueeze(-1)
    ).reshape(3, -1)
    expected = torch.from_numpy(
        dequantize(raw.numpy(), GGMLQuantizationType.Q5_1).copy()
    )
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 17, 65, 257])
@pytest.mark.parametrize("k", [256, 704])
def test_q5_1_grouped_mm_matches_reference(dtype, m, k):
    n = 64
    raw = _make_raw_q5_1(n, k // 32)
    weights, scales, minimums = gguf_q5_1_prepare(raw.to("xpu"))
    activations = torch.randn(m, k, dtype=dtype, device="xpu") * 0.1
    rows = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q5_1_grouped_mm(
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
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


@pytest.mark.parametrize(
    "raw",
    [
        torch.empty((1, 24), dtype=torch.int8),
        torch.empty((24,), dtype=torch.uint8),
        torch.empty((1, 25), dtype=torch.uint8),
    ],
)
def test_q5_1_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_q5_1_prepare(raw)
