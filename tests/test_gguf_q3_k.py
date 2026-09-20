import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

from sgl_kernel import gguf_q3_k_prepare, gguf_q6_k_grouped_mm

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

Q3_K_BLOCK_BYTES = 110


def _set_fp16(raw: torch.Tensor, byte_offset: int, value: float) -> None:
    bits = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
    raw[..., byte_offset : byte_offset + 2] = bits


def _make_raw_q3_k(rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260922 + rows + blocks)
    raw = torch.randint(
        0,
        256,
        (rows, blocks, Q3_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 108, 0.0078125)
    return raw.reshape(rows, blocks * Q3_K_BLOCK_BYTES)


def _prepared_reference(raw: torch.Tensor):
    rows = raw.shape[0]
    blocks = raw.shape[1] // Q3_K_BLOCK_BYTES
    block = raw.reshape(rows, blocks, Q3_K_BLOCK_BYTES)
    hmask = block[..., :32]
    qs = block[..., 32:96]
    scale_meta = block[..., 96:108].to(torch.int32)
    d = block[..., 108:].contiguous().view(torch.float16).reshape(rows, blocks, 1)

    scales = []
    weights = []
    for group in range(16):
        low4 = (scale_meta[..., group & 7] >> ((group // 8) * 4)) & 0x0F
        high2 = (
            scale_meta[..., 8 + (group & 3)] >> ((group // 4) * 2)
        ) & 0x03
        scales.append(d[..., 0] * (low4 | (high2 << 4)).sub(32).to(torch.float16))

        q_set = group // 8
        q_shift = ((group & 7) // 2) * 2
        q_half = group & 1
        low2 = (
            qs[..., q_set * 32 + q_half * 16 : q_set * 32 + (q_half + 1) * 16]
            >> q_shift
        ) & 0x03
        h_bit = group // 2
        h_half = group & 1
        high = (
            (hmask[..., h_half * 16 : (h_half + 1) * 16] >> h_bit) & 0x01
        ) ^ 0x01
        weights.append(low2.to(torch.int8) - (high << 2).to(torch.int8))

    return (
        torch.stack(weights, dim=2).reshape(rows, blocks * 256),
        torch.stack(scales, dim=2).reshape(rows, blocks * 16),
    )


def _reference(activations, weights, scales, rows_per_expert):
    experts, n, k = weights.shape
    dequantized = (
        weights.float().reshape(experts, n, k // 16, 16)
        * scales.float().unsqueeze(-1)
    ).reshape(experts, n, k)
    outputs = []
    row = 0
    for expert, count in enumerate(rows_per_expert.cpu().tolist()):
        outputs.append(activations[row : row + count].float() @ dequantized[expert].T)
        row += count
    return torch.cat(outputs).to(activations.dtype)


def test_q3_k_prepare_matches_reference():
    raw = _make_raw_q3_k(3, 2)
    expected = _prepared_reference(raw)
    actual = gguf_q3_k_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    assert sum(t.numel() * t.element_size() for t in actual) == 3 * 2 * 288
    assert sum(t.numel() * t.element_size() for t in actual) < 3 * 2 * 512


def test_q3_k_prepared_values_match_gguf_reference():
    raw = _make_raw_q3_k(3, 2)
    weights, scales = _prepared_reference(raw)
    actual = (
        weights.float().reshape(3, -1, 16) * scales.float().unsqueeze(-1)
    ).reshape(3, -1)
    expected = torch.from_numpy(dequantize(raw.numpy(), GGMLQuantizationType.Q3_K).copy())
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 17, 65, 257])
def test_q3_k_grouped_mm_matches_reference(dtype, m):
    n, k = 64, 256
    raw = _make_raw_q3_k(n, k // 256)
    weights, scales = gguf_q3_k_prepare(raw.to("xpu"))
    activations = torch.randn(m, k, dtype=dtype, device="xpu") * 0.1
    rows = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q6_k_grouped_mm(
        activations,
        weights.reshape(1, n, k),
        scales.reshape(1, n, k // 16),
        rows,
    )
    expected = _reference(
        activations.cpu(),
        weights.reshape(1, n, k).cpu(),
        scales.reshape(1, n, k // 16).cpu(),
        rows,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


@pytest.mark.parametrize(
    "raw",
    [
        torch.empty((1, 110), dtype=torch.int8),
        torch.empty((110,), dtype=torch.uint8),
        torch.empty((1, 111), dtype=torch.uint8),
    ],
)
def test_q3_k_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_q3_k_prepare(raw)
