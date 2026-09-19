import pytest
import torch

from sgl_kernel.gguf import Q4_K_BLOCK_BYTES, gguf_q4_k_repack

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


def _set_fp16(raw: torch.Tensor, byte_offset: int, value: float) -> None:
    bits = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
    raw[..., byte_offset : byte_offset + 2] = bits


def _reference(raw: torch.Tensor, dtype: torch.dtype):
    rows = raw.shape[0]
    blocks = raw.shape[1] // Q4_K_BLOCK_BYTES
    block = raw.reshape(rows, blocks, Q4_K_BLOCK_BYTES)
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
    scales = bases[..., :1].float() * scale6.float()
    minimums = bases[..., 1:].float() * min6.float()
    scales = scales.reshape(rows, -1)
    minimums = minimums.reshape(rows, -1)
    zero_scale = scales == 0

    qs = block[..., 16:].reshape(rows, blocks, 4, 32)
    values = torch.cat((qs & 0x0F, qs >> 4), dim=-1).reshape(rows, -1, 32)
    values = torch.where(zero_scale.unsqueeze(-1), 0, values).reshape(rows, -1)
    packed = (values[:, 0::2] | (values[:, 1::2] << 4)).contiguous()
    zeros = torch.where(zero_scale, minimums, minimums / scales)
    scales = torch.where(zero_scale, torch.ones_like(scales), scales)
    return packed, scales.to(dtype), zeros.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gguf_q4_k_repack_matches_reference(dtype):
    generator = torch.Generator().manual_seed(20260919)
    raw = torch.randint(
        0,
        256,
        (3, 2, Q4_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.125)
    _set_fp16(raw, 2, 0.0625)
    expected = _reference(raw.reshape(3, -1), dtype)

    actual = gguf_q4_k_repack(raw.reshape(3, -1).to("xpu"), dtype)
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


def test_gguf_q4_k_repack_zero_scale_boundary():
    raw = torch.zeros((1, 1, Q4_K_BLOCK_BYTES), dtype=torch.uint8)
    _set_fp16(raw, 0, 0.0)
    _set_fp16(raw, 2, 1.0)
    raw[..., 8:12] = 1
    expected = _reference(raw.reshape(1, -1), torch.bfloat16)

    actual = gguf_q4_k_repack(raw.reshape(1, -1).to("xpu"), torch.bfloat16)
    torch.xpu.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "raw",
    [
        torch.zeros((1, Q4_K_BLOCK_BYTES - 1), dtype=torch.uint8),
        torch.zeros((0, Q4_K_BLOCK_BYTES), dtype=torch.uint8),
        torch.zeros((Q4_K_BLOCK_BYTES,), dtype=torch.uint8),
        torch.zeros((1, Q4_K_BLOCK_BYTES), dtype=torch.int8),
    ],
)
def test_gguf_q4_k_repack_rejects_bad_input(raw):
    with pytest.raises((TypeError, ValueError)):
        gguf_q4_k_repack(raw, torch.bfloat16)
