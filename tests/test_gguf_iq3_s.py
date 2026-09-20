import pytest
import torch
from gguf import GGMLQuantizationType, dequantize
from gguf.quants import IQ3_S

from sgl_kernel import gguf_iq3_s_prepare, gguf_q8_0_grouped_mm

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

IQ3_S_BLOCK_BYTES = 110


def _set_fp16(raw: torch.Tensor, byte_offset: int, value: float) -> None:
    bits = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
    raw[..., byte_offset : byte_offset + 2] = bits


def _make_raw_iq3_s(rows: int, blocks: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260923 + rows + blocks)
    raw = torch.randint(
        0,
        256,
        (rows, blocks, IQ3_S_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.00390625)
    return raw.reshape(rows, blocks * IQ3_S_BLOCK_BYTES)


def _prepared_reference(raw: torch.Tensor):
    rows = raw.shape[0]
    blocks = raw.shape[1] // IQ3_S_BLOCK_BYTES
    block = raw.reshape(rows, blocks, IQ3_S_BLOCK_BYTES)
    d = block[..., :2].contiguous().view(torch.float16).reshape(rows, blocks, 1)
    qs = block[..., 2:66].to(torch.int64)
    qh = block[..., 66:74].to(torch.int64)
    signs = block[..., 74:106].to(torch.int64)
    scale_meta = block[..., 106:110].to(torch.int64)

    IQ3_S.init_grid()
    grid = torch.from_numpy(IQ3_S.grid.reshape(512, 4).copy()).to(torch.int8)
    weights = torch.empty((rows, blocks, 256), dtype=torch.int8)
    scales = torch.empty((rows, blocks, 8), dtype=torch.float16)
    for group in range(8):
        scale_nibble = (scale_meta[..., group // 2] >> ((group & 1) * 4)) & 0x0F
        scales[..., group] = d[..., 0] * (1 + 2 * scale_nibble).to(torch.float16)
        for grid_in_group in range(8):
            grid_idx = group * 8 + grid_in_group
            lut_idx = qs[..., grid_idx] | (
                ((qh[..., grid_idx // 8] >> (grid_idx & 7)) & 1) << 8
            )
            values = grid[lut_idx]
            for component in range(4):
                value_idx = grid_idx * 4 + component
                negative = (
                    (signs[..., value_idx // 8] >> (value_idx & 7)) & 1
                ).bool()
                weights[..., value_idx] = torch.where(
                    negative, -values[..., component], values[..., component]
                )
    return weights.reshape(rows, blocks * 256), scales.reshape(rows, blocks * 8)


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


def test_iq3_s_prepare_matches_reference():
    raw = _make_raw_iq3_s(3, 2)
    expected = _prepared_reference(raw)
    actual = gguf_iq3_s_prepare(raw.to("xpu"))
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    assert sum(t.numel() * t.element_size() for t in actual) == 3 * 2 * 272
    assert sum(t.numel() * t.element_size() for t in actual) < 3 * 2 * 512


def test_iq3_s_prepared_values_match_gguf_reference():
    raw = _make_raw_iq3_s(3, 2)
    weights, scales = _prepared_reference(raw)
    actual = (
        weights.float().reshape(3, -1, 32) * scales.float().unsqueeze(-1)
    ).reshape(3, -1)
    expected = torch.from_numpy(
        dequantize(raw.numpy(), GGMLQuantizationType.IQ3_S).copy()
    )
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 17, 65, 257])
def test_iq3_s_grouped_mm_matches_reference(dtype, m):
    n, k = 64, 256
    raw = _make_raw_iq3_s(n, k // 256)
    weights, scales = gguf_iq3_s_prepare(raw.to("xpu"))
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
        torch.empty((1, 110), dtype=torch.int8),
        torch.empty((110,), dtype=torch.uint8),
        torch.empty((1, 111), dtype=torch.uint8),
    ],
)
def test_iq3_s_prepare_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_iq3_s_prepare(raw)
