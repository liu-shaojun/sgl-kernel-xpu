"""Intel XPU helpers for GGUF quantized weights."""

from __future__ import annotations

import torch

Q4_K_BLOCK_BYTES = 144
Q4_K_VALUES_PER_BLOCK = 256
Q4_K_GROUP_SIZE = 32
Q4_K_METADATA_BYTES = 16
Q8_0_BLOCK_BYTES = 34
Q8_0_VALUES_PER_BLOCK = 32
Q6_K_BLOCK_BYTES = 210
Q6_K_VALUES_PER_BLOCK = 256
Q6_K_GROUP_SIZE = 16
Q5_K_BLOCK_BYTES = 176
Q5_K_VALUES_PER_BLOCK = 256
Q5_K_GROUP_SIZE = 32
IQ4_XS_BLOCK_BYTES = 136
IQ4_XS_VALUES_PER_BLOCK = 256
IQ4_XS_GROUP_SIZE = 32


def gguf_q4_k_grouped_mm(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    metadata: torch.Tensor,
    rows_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Multiply activations by compact-metadata Q4_K expert weights."""
    if activations.ndim != 2:
        raise ValueError("activations must be rank 2 [total_m, K]")
    if packed_weights.ndim != 3:
        raise ValueError("packed_weights must be rank 3 [E, N, K/2]")
    output = torch.empty(
        (activations.shape[0], packed_weights.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    torch.ops.sgl_kernel.gguf_q4_k_grouped_mm(
        output,
        activations,
        packed_weights,
        metadata,
        rows_per_expert,
        packed_weights.shape[0],
    )
    return output


def gguf_q4_k_prepare(
    raw_qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare raw Q4_K blocks for direct compact-metadata GEMM.

    The packed values use adjacent-element nibbles. Metadata retains the exact
    16-byte GGUF scale/min header per 256-value block, so the two outputs use
    the same number of bytes as the input.
    """
    _validate_q4_k(raw_qweight)
    rows = raw_qweight.shape[0]
    blocks = raw_qweight.shape[1] // Q4_K_BLOCK_BYTES
    packed = torch.empty(
        (rows, blocks * (Q4_K_VALUES_PER_BLOCK // 2)),
        dtype=torch.uint8,
        device=raw_qweight.device,
    )
    metadata = torch.empty(
        (rows, blocks * Q4_K_METADATA_BYTES),
        dtype=torch.uint8,
        device=raw_qweight.device,
    )
    torch.ops.sgl_kernel.gguf_q4_k_prepare(raw_qweight, packed, metadata)
    return packed, metadata


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
    _validate_q4_k(raw_qweight)

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


def gguf_q8_0_prepare(
    raw_qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split raw Q8_0 blocks into signed values and their original FP16 scales.

    The outputs consume exactly the same number of bytes as the GGUF input.
    """
    _validate_q8_0(raw_qweight)
    rows = raw_qweight.shape[0]
    blocks = raw_qweight.shape[1] // Q8_0_BLOCK_BYTES
    weights = torch.empty(
        (rows, blocks * Q8_0_VALUES_PER_BLOCK),
        dtype=torch.int8,
        device=raw_qweight.device,
    )
    scales = torch.empty((rows, blocks), dtype=torch.float16, device=raw_qweight.device)
    torch.ops.sgl_kernel.gguf_q8_0_prepare(raw_qweight, weights, scales)
    return weights, scales


def gguf_q8_0_grouped_mm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    rows_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Multiply activations by compact Q8_0 expert weights."""
    if activations.ndim != 2:
        raise ValueError("activations must be rank 2 [total_m, K]")
    if weights.ndim != 3:
        raise ValueError("weights must be rank 3 [E, N, K]")
    output = torch.empty(
        (activations.shape[0], weights.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    torch.ops.sgl_kernel.gguf_q8_0_grouped_mm(
        output,
        activations,
        weights,
        scales,
        rows_per_expert,
        weights.shape[0],
    )
    return output


def gguf_q6_k_prepare(
    raw_qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare Q6_K as signed INT8 values with one FP16 scale per 16 values."""
    _validate_q6_k(raw_qweight)
    rows = raw_qweight.shape[0]
    blocks = raw_qweight.shape[1] // Q6_K_BLOCK_BYTES
    weights = torch.empty(
        (rows, blocks * Q6_K_VALUES_PER_BLOCK),
        dtype=torch.int8,
        device=raw_qweight.device,
    )
    scales = torch.empty(
        (rows, blocks * (Q6_K_VALUES_PER_BLOCK // Q6_K_GROUP_SIZE)),
        dtype=torch.float16,
        device=raw_qweight.device,
    )
    torch.ops.sgl_kernel.gguf_q6_k_prepare(raw_qweight, weights, scales)
    return weights, scales


def gguf_q6_k_grouped_mm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    rows_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Multiply activations by prepared Q6_K expert weights."""
    if activations.ndim != 2:
        raise ValueError("activations must be rank 2 [total_m, K]")
    if weights.ndim != 3:
        raise ValueError("weights must be rank 3 [E, N, K]")
    output = torch.empty(
        (activations.shape[0], weights.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    torch.ops.sgl_kernel.gguf_q6_k_grouped_mm(
        output,
        activations,
        weights,
        scales,
        rows_per_expert,
        weights.shape[0],
    )
    return output


def gguf_q5_k_prepare(
    raw_qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q5_K code values, FP16 scales, and FP16 minimums."""
    _validate_q5_k(raw_qweight)
    rows = raw_qweight.shape[0]
    blocks = raw_qweight.shape[1] // Q5_K_BLOCK_BYTES
    weights = torch.empty(
        (rows, blocks * Q5_K_VALUES_PER_BLOCK),
        dtype=torch.int8,
        device=raw_qweight.device,
    )
    metadata_shape = (
        rows,
        blocks * (Q5_K_VALUES_PER_BLOCK // Q5_K_GROUP_SIZE),
    )
    scales = torch.empty(metadata_shape, dtype=torch.float16, device=raw_qweight.device)
    minimums = torch.empty_like(scales)
    torch.ops.sgl_kernel.gguf_q5_k_prepare(raw_qweight, weights, scales, minimums)
    return weights, scales, minimums


def gguf_q5_k_grouped_mm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    minimums: torch.Tensor,
    rows_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Multiply activations by prepared affine Q5_K expert weights."""
    if activations.ndim != 2:
        raise ValueError("activations must be rank 2 [total_m, K]")
    if weights.ndim != 3:
        raise ValueError("weights must be rank 3 [E, N, K]")
    output = torch.empty(
        (activations.shape[0], weights.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    torch.ops.sgl_kernel.gguf_q5_k_grouped_mm(
        output,
        activations,
        weights,
        scales,
        minimums,
        rows_per_expert,
        weights.shape[0],
    )
    return output


def gguf_iq4_xs_prepare(
    raw_qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare IQ4_XS codebook values and one FP16 scale per 32 values."""
    _validate_iq4_xs(raw_qweight)
    rows = raw_qweight.shape[0]
    blocks = raw_qweight.shape[1] // IQ4_XS_BLOCK_BYTES
    weights = torch.empty(
        (rows, blocks * IQ4_XS_VALUES_PER_BLOCK),
        dtype=torch.int8,
        device=raw_qweight.device,
    )
    scales = torch.empty(
        (rows, blocks * (IQ4_XS_VALUES_PER_BLOCK // IQ4_XS_GROUP_SIZE)),
        dtype=torch.float16,
        device=raw_qweight.device,
    )
    torch.ops.sgl_kernel.gguf_iq4_xs_prepare(raw_qweight, weights, scales)
    return weights, scales


def _validate_q4_k(raw_qweight: torch.Tensor) -> None:
    """Validate the common raw Q4_K matrix contract."""
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


def _validate_q8_0(raw_qweight: torch.Tensor) -> None:
    """Validate the common raw Q8_0 matrix contract."""
    if raw_qweight.dtype != torch.uint8:
        raise TypeError(f"raw_qweight must be uint8, got {raw_qweight.dtype}")
    if raw_qweight.ndim != 2:
        raise ValueError(f"raw_qweight must be rank 2, got {raw_qweight.ndim}")
    if raw_qweight.shape[0] == 0:
        raise ValueError("raw_qweight must contain at least one row")
    if raw_qweight.shape[1] == 0 or raw_qweight.shape[1] % Q8_0_BLOCK_BYTES:
        raise ValueError(
            "raw_qweight width must be a non-zero multiple of "
            f"{Q8_0_BLOCK_BYTES}, got {raw_qweight.shape[1]}"
        )


def _validate_q6_k(raw_qweight: torch.Tensor) -> None:
    """Validate the common raw Q6_K matrix contract."""
    if raw_qweight.dtype != torch.uint8:
        raise TypeError(f"raw_qweight must be uint8, got {raw_qweight.dtype}")
    if raw_qweight.ndim != 2:
        raise ValueError(f"raw_qweight must be rank 2, got {raw_qweight.ndim}")
    if raw_qweight.shape[0] == 0:
        raise ValueError("raw_qweight must contain at least one row")
    if raw_qweight.shape[1] == 0 or raw_qweight.shape[1] % Q6_K_BLOCK_BYTES:
        raise ValueError(
            "raw_qweight width must be a non-zero multiple of "
            f"{Q6_K_BLOCK_BYTES}, got {raw_qweight.shape[1]}"
        )


def _validate_q5_k(raw_qweight: torch.Tensor) -> None:
    """Validate the common raw Q5_K matrix contract."""
    if raw_qweight.dtype != torch.uint8:
        raise TypeError(f"raw_qweight must be uint8, got {raw_qweight.dtype}")
    if raw_qweight.ndim != 2:
        raise ValueError(f"raw_qweight must be rank 2, got {raw_qweight.ndim}")
    if raw_qweight.shape[0] == 0:
        raise ValueError("raw_qweight must contain at least one row")
    if raw_qweight.shape[1] == 0 or raw_qweight.shape[1] % Q5_K_BLOCK_BYTES:
        raise ValueError(
            "raw_qweight width must be a non-zero multiple of "
            f"{Q5_K_BLOCK_BYTES}, got {raw_qweight.shape[1]}"
        )


def _validate_iq4_xs(raw_qweight: torch.Tensor) -> None:
    """Validate the common raw IQ4_XS matrix contract."""
    if raw_qweight.dtype != torch.uint8:
        raise TypeError(f"raw_qweight must be uint8, got {raw_qweight.dtype}")
    if raw_qweight.ndim != 2:
        raise ValueError(f"raw_qweight must be rank 2, got {raw_qweight.ndim}")
    if raw_qweight.shape[0] == 0:
        raise ValueError("raw_qweight must contain at least one row")
    if raw_qweight.shape[1] == 0 or raw_qweight.shape[1] % IQ4_XS_BLOCK_BYTES:
        raise ValueError(
            "raw_qweight width must be a non-zero multiple of "
            f"{IQ4_XS_BLOCK_BYTES}, got {raw_qweight.shape[1]}"
        )


__all__ = [
    "Q4_K_BLOCK_BYTES",
    "Q4_K_GROUP_SIZE",
    "Q4_K_METADATA_BYTES",
    "Q4_K_VALUES_PER_BLOCK",
    "Q8_0_BLOCK_BYTES",
    "Q8_0_VALUES_PER_BLOCK",
    "Q6_K_BLOCK_BYTES",
    "Q6_K_GROUP_SIZE",
    "Q6_K_VALUES_PER_BLOCK",
    "Q5_K_BLOCK_BYTES",
    "Q5_K_GROUP_SIZE",
    "Q5_K_VALUES_PER_BLOCK",
    "IQ4_XS_BLOCK_BYTES",
    "IQ4_XS_GROUP_SIZE",
    "IQ4_XS_VALUES_PER_BLOCK",
    "gguf_q4_k_grouped_mm",
    "gguf_q4_k_prepare",
    "gguf_q4_k_repack",
    "gguf_q8_0_grouped_mm",
    "gguf_q8_0_prepare",
    "gguf_q6_k_grouped_mm",
    "gguf_q6_k_prepare",
    "gguf_q5_k_grouped_mm",
    "gguf_q5_k_prepare",
    "gguf_iq4_xs_prepare",
]
