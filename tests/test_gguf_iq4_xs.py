import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

from sgl_kernel import gguf_iq4_xs_prepare, gguf_q8_0_grouped_mm

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

IQ4_XS_BLOCK_BYTES = 136
IQ4_XS_VALUES = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)


def _set_fp16(raw: torch.Tensor, byte_offset: int, value: float) -> None:
    bits = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
    raw[..., byte_offset : byte_offset + 2] = bits


def _make_raw_iq4_xs(rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260920 + rows + blocks)
    raw = torch.randint(
        0,
        256,
        (rows, blocks, IQ4_XS_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.0009765625)
    return raw.reshape(rows, blocks * IQ4_XS_BLOCK_BYTES)


def _prepared_reference(raw: torch.Tensor):
    rows = raw.shape[0]
    blocks = raw.shape[1] // IQ4_XS_BLOCK_BYTES
    block = raw.reshape(rows, blocks, IQ4_XS_BLOCK_BYTES)
    d = block[..., :2].contiguous().view(torch.float16).reshape(rows, blocks, 1)
    scales_h = (
        block[..., 2:4].contiguous().view(torch.int16).reshape(rows, blocks).to(torch.int32)
    )
    scales_l = block[..., 4:8].to(torch.int32)

    scales = []
    weights = []
    qs = block[..., 8:]
    for group in range(8):
        low4 = (scales_l[..., group // 2] >> ((group & 1) * 4)) & 0x0F
        high2 = (scales_h >> (2 * group)) & 0x03
        scales.append(d[..., 0] * (low4 | (high2 << 4)).sub(32).to(torch.float16))

        packed = qs[..., group * 16 : (group + 1) * 16].to(torch.int64)
        indices = torch.cat((packed & 0x0F, packed >> 4), dim=-1)
        weights.append(IQ4_XS_VALUES[indices])

    return (
        torch.stack(weights, dim=2).reshape(rows, blocks * 256),
        torch.stack(scales, dim=2).reshape(rows, blocks * 8),
    )


def _reference(activations, weights, scales, rows_per_expert):
    experts, n, k = weights.shape
    dequantized = (
        weights.float().reshape(experts, n, k // 32, 32)
        * scales.float().unsqueeze(-1)
    ).reshape(experts, n, k)
    outputs = []
    row = 0
    for expert, count in enumerate(rows_per_expert.cpu().tolist()):
        outputs.append(activations[row : row + count].float() @ dequantized[expert].T)
        row += count
    return torch.cat(outputs).to(activations.dtype)


def test_iq4_xs_prepare_matches_reference():
    raw = _make_raw_iq4_xs(3, 2)
    expected = _prepared_reference(raw)
    actual = gguf_iq4_xs_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    assert sum(t.numel() * t.element_size() for t in actual) == 3 * 2 * 272
    assert sum(t.numel() * t.element_size() for t in actual) < 3 * 2 * 512


def test_iq4_xs_prepared_values_match_gguf_reference():
    raw = _make_raw_iq4_xs(3, 2)
    weights, scales = _prepared_reference(raw)
    actual = (
        weights.float().reshape(3, -1, 32) * scales.float().unsqueeze(-1)
    ).reshape(3, -1)
    expected = torch.from_numpy(dequantize(raw.numpy(), GGMLQuantizationType.IQ4_XS).copy())
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 17, 65, 257])
def test_iq4_xs_grouped_mm_matches_reference(dtype, m):
    n, k = 64, 256
    raw = _make_raw_iq4_xs(n, k // 256)
    weights, scales = gguf_iq4_xs_prepare(raw.to("xpu"))
    activations = torch.randn(m, k, dtype=dtype, device="xpu") * 0.1
    rows = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q8_0_grouped_mm(
        activations,
        weights.reshape(1, n, k),
        scales.reshape(1, n, k // 32),
        rows,
    )
    expected = _reference(
        activations.cpu(),
        weights.reshape(1, n, k).cpu(),
        scales.reshape(1, n, k // 32).cpu(),
        rows,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.12)


@pytest.mark.parametrize(
    "raw",
    [
        torch.empty((1, 136), dtype=torch.int8),
        torch.empty((136,), dtype=torch.uint8),
        torch.empty((1, 137), dtype=torch.uint8),
    ],
)
def test_iq4_xs_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_iq4_xs_prepare(raw)
