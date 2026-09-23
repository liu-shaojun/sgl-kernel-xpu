import pytest
import torch

from sgl_kernel.gguf import (
    Q4_K_BLOCK_BYTES,
    Q4_K_GROUP_SIZE,
    Q4_K_METADATA_BYTES,
    gguf_q4_k_grouped_mm,
    gguf_q4_k_prepare,
    gguf_q4_k_repack,
)

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


def _dequantize_reference(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    packed, scales, zeros = _reference(raw, torch.float32)
    values = torch.empty(
        (packed.shape[0], packed.shape[1] * 2), dtype=torch.float32
    )
    values[:, 0::2] = packed & 0x0F
    values[:, 1::2] = packed >> 4
    scales = scales.repeat_interleave(Q4_K_GROUP_SIZE, dim=-1)
    zeros = zeros.repeat_interleave(Q4_K_GROUP_SIZE, dim=-1)
    return ((values - zeros) * scales).to(dtype)


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


def test_gguf_q4_k_prepare_preserves_compact_metadata():
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
    raw[..., 4:16] = 0x55
    expected_packed, _, _ = _reference(raw.reshape(3, -1), torch.float32)
    expected_metadata = raw[..., :Q4_K_METADATA_BYTES].reshape(3, -1)

    packed, metadata = gguf_q4_k_prepare(raw.reshape(3, -1).to("xpu"))
    torch.xpu.synchronize()
    torch.testing.assert_close(packed.cpu(), expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(metadata.cpu(), expected_metadata, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 9, 17])
def test_gguf_q4_k_grouped_mm_matches_reference(dtype, m):
    generator = torch.Generator().manual_seed(20260919 + m)
    n, blocks = 64, 2
    raw = torch.randint(
        0,
        256,
        (n, blocks, Q4_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.015625)
    _set_fp16(raw, 2, 0.0078125)
    raw_2d = raw.reshape(n, -1)
    weight = _dequantize_reference(raw_2d, dtype)
    activations = torch.randn((m, weight.shape[1]), dtype=dtype, generator=generator)

    packed, metadata = gguf_q4_k_prepare(raw_2d.to("xpu"))
    rows_per_expert = torch.tensor([m], dtype=torch.int32, device="xpu")
    actual = gguf_q4_k_grouped_mm(
        activations.to("xpu"),
        packed.unsqueeze(0),
        metadata.unsqueeze(0),
        rows_per_expert,
    )
    torch.xpu.synchronize()

    expected = torch.matmul(activations.float(), weight.float().T).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.25)


def test_gguf_q4_k_grouped_mm_multi_expert_and_n_tail():
    generator = torch.Generator().manual_seed(20260919)
    experts, n, blocks = 2, 72, 2
    rows_per_expert_cpu = torch.tensor([3, 30], dtype=torch.int32)
    raw = torch.randint(
        0,
        256,
        (experts, n, blocks, Q4_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.015625)
    _set_fp16(raw, 2, 0.0078125)
    raw_2d = raw.reshape(experts * n, -1)
    weight = _dequantize_reference(raw_2d, torch.bfloat16).reshape(
        experts, n, -1
    )
    activations = torch.randn(
        (int(rows_per_expert_cpu.sum()), weight.shape[-1]),
        dtype=torch.bfloat16,
        generator=generator,
    )

    packed, metadata = gguf_q4_k_prepare(raw_2d.to("xpu"))
    actual = gguf_q4_k_grouped_mm(
        activations.to("xpu"),
        packed.reshape(experts, n, -1),
        metadata.reshape(experts, n, -1),
        rows_per_expert_cpu.to("xpu"),
    )
    torch.xpu.synchronize()

    expected_parts = []
    row_start = 0
    for expert, rows in enumerate(rows_per_expert_cpu.tolist()):
        expected_parts.append(
            torch.matmul(
                activations[row_start : row_start + rows].float(),
                weight[expert].float().T,
            )
        )
        row_start += rows
    expected = torch.cat(expected_parts).to(torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.25)


@pytest.mark.parametrize(
    ("n", "blocks"),
    [
        (128, 2),  # m64_n128
        (64, 2),  # m64_n128_skip
        (256, 5),  # m64_n256
        (1288, 5),  # m64_n256_skip
    ],
)
def test_gguf_q4_k_grouped_mm_prefill_policies(n, blocks):
    generator = torch.Generator().manual_seed(20260919 + n)
    m = 65
    raw = torch.randint(
        0,
        256,
        (n, blocks, Q4_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_fp16(raw, 0, 0.015625)
    _set_fp16(raw, 2, 0.0078125)
    raw_2d = raw.reshape(n, -1)
    weight = _dequantize_reference(raw_2d, torch.bfloat16)
    activations = torch.randn(
        (m, weight.shape[1]), dtype=torch.bfloat16, generator=generator
    )

    packed, metadata = gguf_q4_k_prepare(raw_2d.to("xpu"))
    actual = gguf_q4_k_grouped_mm(
        activations.to("xpu"),
        packed.unsqueeze(0),
        metadata.unsqueeze(0),
        torch.tensor([m], dtype=torch.int32, device="xpu"),
    )
    torch.xpu.synchronize()

    expected = torch.matmul(activations.float(), weight.float().T).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.3)


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
