"""Intel XPU helpers for GGUF quantized weights."""

from __future__ import annotations

import torch

Q4_K_BLOCK_BYTES = 144
Q4_K_VALUES_PER_BLOCK = 256
Q4_K_GROUP_SIZE = 32


def gguf_q4_k_repack(
    raw_qweight: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack raw GGUF Q4_K blocks on XPU for the W4A16 grouped GEMM.

    Args:
        raw_qweight: Contiguous uint8 tensor shaped ``[N, blocks * 144]``.
        dtype: Scale/zero dtype, normally matching the activation dtype.

    Returns:
        ``(packed_qweight, scales, zeros)`` with shapes ``[N, K/2]`` and
        ``[N, K/32]`` where ``K = blocks * 256``.
    """
    if raw_qweight.dtype != torch.uint8:
        raise TypeError(f"raw_qweight must be uint8, got {raw_qweight.dtype}")
    if raw_qweight.ndim != 2:
        raise ValueError(f"raw_qweight must be rank 2, got {raw_qweight.ndim}")
    if raw_qweight.shape[0] == 0:
        raise ValueError("raw_qweight must contain at least one row")
    if raw_qweight.shape[1] == 0 or raw_qweight.shape[1] % Q4_K_BLOCK_BYTES:
        raise ValueError(
            "raw_qweight width must be a non-zero multiple of "
            f"{Q4_K_BLOCK_BYTES}, got {raw_qweight.shape[1]}"
        )
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"dtype must be bfloat16, float16, or float32, got {dtype}")

    rows = raw_qweight.shape[0]
    blocks = raw_qweight.shape[1] // Q4_K_BLOCK_BYTES
    packed = torch.empty(
        (rows, blocks * (Q4_K_VALUES_PER_BLOCK // 2)),
        dtype=torch.uint8,
        device=raw_qweight.device,
    )
    scales = torch.empty(
        (rows, blocks * (Q4_K_VALUES_PER_BLOCK // Q4_K_GROUP_SIZE)),
        dtype=dtype,
        device=raw_qweight.device,
    )
    zeros = torch.empty_like(scales)
    torch.ops.sgl_kernel.gguf_q4_k_repack(raw_qweight, packed, scales, zeros)
    return packed, scales, zeros


__all__ = [
    "Q4_K_BLOCK_BYTES",
    "Q4_K_GROUP_SIZE",
    "Q4_K_VALUES_PER_BLOCK",
    "gguf_q4_k_repack",
]
