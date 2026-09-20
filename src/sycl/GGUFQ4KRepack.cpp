/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"
#include "sycl/Utils.h"
#include "sycl/gguf_iq3_s_grid.h"

namespace {

constexpr int kQ4KValuesPerBlock = 256;
constexpr int kQ4KBytesPerBlock = 144;
constexpr int kQ4KGroupsPerBlock = 8;
constexpr int kQ4KValuesPerGroup = 32;
constexpr int kW4A16BytesPerBlock = kQ4KValuesPerBlock / 2;
constexpr int kQ80ValuesPerBlock = 32;
constexpr int kQ80BytesPerBlock = 34;
constexpr int kQ6KValuesPerBlock = 256;
constexpr int kQ6KBytesPerBlock = 210;
constexpr int kQ6KScaleGroupsPerBlock = 16;
constexpr int kQ5KValuesPerBlock = 256;
constexpr int kQ5KBytesPerBlock = 176;
constexpr int kQ5KGroupsPerBlock = 8;
constexpr int kIQ4XSValuesPerBlock = 256;
constexpr int kIQ4XSBytesPerBlock = 136;
constexpr int kIQ4XSGroupsPerBlock = 8;
constexpr int kIQ4NLValuesPerBlock = 32;
constexpr int kIQ4NLBytesPerBlock = 18;
constexpr int kQ3KValuesPerBlock = 256;
constexpr int kQ3KBytesPerBlock = 110;
constexpr int kQ3KGroupsPerBlock = 16;
constexpr int kIQ3SValuesPerBlock = 256;
constexpr int kIQ3SBytesPerBlock = 110;
constexpr int kIQ3SGroupsPerBlock = 8;
constexpr int8_t kIQ4Values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <typename ScaleT>
class GGUFQ4KRepackKernel;

class GGUFQ4KPrepareKernel;
class GGUFQ80PrepareKernel;
class GGUFQ6KPrepareKernel;
class GGUFQ5KPrepareKernel;
class GGUFIQ4XSPrepareKernel;
class GGUFIQ4NLPrepareKernel;
class GGUFQ3KPrepareKernel;
class GGUFIQ3SPrepareKernel;

inline int64_t round_up(int64_t value, int64_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

template <typename ScaleT>
void launch_q4_k_repack(
    sycl::queue& queue,
    const uint8_t* raw,
    uint8_t* packed,
    ScaleT* scales,
    ScaleT* zeros,
    int64_t rows,
    int64_t blocks_per_row) {
  const int64_t group_count = rows * blocks_per_row * kQ4KGroupsPerBlock;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(group_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFQ4KRepackKernel<ScaleT>>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t linear_group = item.get_global_linear_id();
          if (linear_group >= group_count) {
            return;
          }

          const int group = linear_group % kQ4KGroupsPerBlock;
          const int64_t row_block = linear_group / kQ4KGroupsPerBlock;
          const uint8_t* block = raw + row_block * kQ4KBytesPerBlock;

          const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8);
          const uint16_t dmin_bits = static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8);
          const float d = static_cast<float>(sycl::bit_cast<sycl::half>(d_bits));
          const float dmin = static_cast<float>(sycl::bit_cast<sycl::half>(dmin_bits));

          const uint8_t* meta = block + 4;
          int scale6;
          int min6;
          if (group < 4) {
            scale6 = meta[group] & 0x3f;
            min6 = meta[group + 4] & 0x3f;
          } else {
            const int high_group = group - 4;
            scale6 = (meta[8 + high_group] & 0x0f) | ((meta[high_group] >> 2) & 0x30);
            min6 = (meta[8 + high_group] >> 4) | ((meta[4 + high_group] >> 2) & 0x30);
          }

          float scale = d * static_cast<float>(scale6);
          const float minimum = dmin * static_cast<float>(min6);
          const bool zero_scale = scale == 0.0f;
          const float zero = zero_scale ? minimum : minimum / scale;
          if (zero_scale) {
            scale = 1.0f;
          }

          scales[linear_group] = static_cast<ScaleT>(scale);
          zeros[linear_group] = static_cast<ScaleT>(zero);

          // GGUF Q4_K stores four 32-byte regions. Within each region the
          // low nibbles are one 32-value group and the high nibbles are the
          // following group. W4A16 instead packs adjacent logical values in
          // each byte, so rebuild the 16 output bytes for this group.
          const int region = group / 2;
          const int shift = (group & 1) * 4;
          const uint8_t* qsrc = block + 16 + region * kQ4KValuesPerGroup;
          uint8_t* qdst = packed + row_block * kW4A16BytesPerBlock + group * (kQ4KValuesPerGroup / 2);

#pragma unroll
          for (int i = 0; i < kQ4KValuesPerGroup / 2; ++i) {
            uint8_t q0 = (qsrc[2 * i] >> shift) & 0x0f;
            uint8_t q1 = (qsrc[2 * i + 1] >> shift) & 0x0f;
            if (zero_scale) {
              q0 = 0;
              q1 = 0;
            }
            qdst[i] = q0 | (q1 << 4);
          }
        });
  });
}

void launch_q4_k_prepare(
    sycl::queue& queue, const uint8_t* raw, uint8_t* packed, uint8_t* metadata, int64_t rows, int64_t blocks_per_row) {
  const int64_t group_count = rows * blocks_per_row * kQ4KGroupsPerBlock;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(group_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFQ4KPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t linear_group = item.get_global_linear_id();
          if (linear_group >= group_count) {
            return;
          }

          const int group = linear_group % kQ4KGroupsPerBlock;
          const int64_t row_block = linear_group / kQ4KGroupsPerBlock;
          const uint8_t* block = raw + row_block * kQ4KBytesPerBlock;
          uint8_t* mdst = metadata + row_block * 16 + group * 2;
          mdst[0] = block[group * 2];
          mdst[1] = block[group * 2 + 1];

          const int region = group / 2;
          const int shift = (group & 1) * 4;
          const uint8_t* qsrc = block + 16 + region * kQ4KValuesPerGroup;
          uint8_t* qdst = packed + row_block * kW4A16BytesPerBlock + group * (kQ4KValuesPerGroup / 2);
#pragma unroll
          for (int i = 0; i < kQ4KValuesPerGroup / 2; ++i) {
            const uint8_t q0 = (qsrc[2 * i] >> shift) & 0x0f;
            const uint8_t q1 = (qsrc[2 * i + 1] >> shift) & 0x0f;
            qdst[i] = q0 | (q1 << 4);
          }
        });
  });
}

void launch_q8_0_prepare(
    sycl::queue& queue, const uint8_t* raw, int8_t* weights, uint16_t* scales, int64_t rows, int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFQ80PrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }
          const uint8_t* src = raw + block_idx * kQ80BytesPerBlock;
          scales[block_idx] = static_cast<uint16_t>(src[0]) | (static_cast<uint16_t>(src[1]) << 8);
          int8_t* dst = weights + block_idx * kQ80ValuesPerBlock;
#pragma unroll
          for (int i = 0; i < kQ80ValuesPerBlock; ++i) {
            dst[i] = sycl::bit_cast<int8_t>(src[i + 2]);
          }
        });
  });
}

void launch_q6_k_prepare(
    sycl::queue& queue, const uint8_t* raw, int8_t* weights, uint16_t* scales, int64_t rows, int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFQ6KPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }

          const uint8_t* src = raw + block_idx * kQ6KBytesPerBlock;
          const uint8_t* ql = src;
          const uint8_t* qh = src + 128;
          const int8_t* scale_codes = reinterpret_cast<const int8_t*>(src + 192);
          const uint16_t d_bits = static_cast<uint16_t>(src[208]) | (static_cast<uint16_t>(src[209]) << 8);
          const sycl::half d = sycl::bit_cast<sycl::half>(d_bits);

          int8_t* dst = weights + block_idx * kQ6KValuesPerBlock;
#pragma unroll
          for (int part = 0; part < 8; ++part) {
            const int half = part / 4;
            const int within_half = part % 4;
            const int ql_base = half * 64 + (within_half & 1) * 32;
            const int qh_base = half * 32;
            const int nibble_shift = within_half >= 2 ? 4 : 0;
            const int high_shift = within_half * 2;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              const int low = (ql[ql_base + i] >> nibble_shift) & 0x0f;
              const int high = (qh[qh_base + i] >> high_shift) & 0x03;
              dst[part * 32 + i] = static_cast<int8_t>((low | (high << 4)) - 32);
            }
          }

          uint16_t* dst_scales = scales + block_idx * kQ6KScaleGroupsPerBlock;
#pragma unroll
          for (int group = 0; group < kQ6KScaleGroupsPerBlock; ++group) {
            const sycl::half scale = d * static_cast<sycl::half>(scale_codes[group]);
            dst_scales[group] = sycl::bit_cast<uint16_t>(scale);
          }
        });
  });
}

void launch_q5_k_prepare(
    sycl::queue& queue,
    const uint8_t* raw,
    int8_t* weights,
    uint16_t* scales,
    uint16_t* minimums,
    int64_t rows,
    int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFQ5KPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }

          const uint8_t* src = raw + block_idx * kQ5KBytesPerBlock;
          const uint16_t d_bits = static_cast<uint16_t>(src[0]) | (static_cast<uint16_t>(src[1]) << 8);
          const uint16_t dmin_bits = static_cast<uint16_t>(src[2]) | (static_cast<uint16_t>(src[3]) << 8);
          const sycl::half d = sycl::bit_cast<sycl::half>(d_bits);
          const sycl::half dmin = sycl::bit_cast<sycl::half>(dmin_bits);
          const uint8_t* meta = src + 4;
          const uint8_t* qh = src + 16;
          const uint8_t* qs = src + 48;

          int8_t* dst = weights + block_idx * kQ5KValuesPerBlock;
          uint16_t* dst_scales = scales + block_idx * kQ5KGroupsPerBlock;
          uint16_t* dst_minimums = minimums + block_idx * kQ5KGroupsPerBlock;
#pragma unroll
          for (int group = 0; group < kQ5KGroupsPerBlock; ++group) {
            int scale6;
            int min6;
            if (group < 4) {
              scale6 = meta[group] & 0x3f;
              min6 = meta[group + 4] & 0x3f;
            } else {
              const int high_group = group - 4;
              scale6 = (meta[8 + high_group] & 0x0f) | ((meta[high_group] >> 2) & 0x30);
              min6 = (meta[8 + high_group] >> 4) | ((meta[4 + high_group] >> 2) & 0x30);
            }
            dst_scales[group] = sycl::bit_cast<uint16_t>(d * static_cast<sycl::half>(scale6));
            dst_minimums[group] = sycl::bit_cast<uint16_t>(dmin * static_cast<sycl::half>(min6));

            const uint8_t* qsrc = qs + (group / 2) * 32;
            const int nibble_shift = (group & 1) * 4;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              const int low = (qsrc[i] >> nibble_shift) & 0x0f;
              const int high = (qh[i] >> group) & 0x01;
              dst[group * 32 + i] = static_cast<int8_t>(low | (high << 4));
            }
          }
        });
  });
}

void launch_iq4_xs_prepare(
    sycl::queue& queue, const uint8_t* raw, int8_t* weights, uint16_t* scales, int64_t rows, int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFIQ4XSPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }

          const uint8_t* src = raw + block_idx * kIQ4XSBytesPerBlock;
          const uint16_t d_bits = static_cast<uint16_t>(src[0]) | (static_cast<uint16_t>(src[1]) << 8);
          const sycl::half d = sycl::bit_cast<sycl::half>(d_bits);
          const uint16_t scales_h = static_cast<uint16_t>(src[2]) | (static_cast<uint16_t>(src[3]) << 8);
          const uint8_t* scales_l = src + 4;
          const uint8_t* qs = src + 8;
          int8_t* dst = weights + block_idx * kIQ4XSValuesPerBlock;
          uint16_t* dst_scales = scales + block_idx * kIQ4XSGroupsPerBlock;

#pragma unroll
          for (int group = 0; group < kIQ4XSGroupsPerBlock; ++group) {
            const int low4 = (scales_l[group / 2] >> ((group & 1) * 4)) & 0x0f;
            const int high2 = (scales_h >> (2 * group)) & 0x03;
            const int signed_scale = (low4 | (high2 << 4)) - 32;
            dst_scales[group] = sycl::bit_cast<uint16_t>(d * static_cast<sycl::half>(signed_scale));

            const uint8_t* qsrc = qs + group * 16;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              const int nibble = (qsrc[i & 15] >> ((i / 16) * 4)) & 0x0f;
              dst[group * 32 + i] = kIQ4Values[nibble];
            }
          }
        });
  });
}

void launch_iq4_nl_prepare(
    sycl::queue& queue, const uint8_t* raw, int8_t* weights, uint16_t* scales, int64_t rows, int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFIQ4NLPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }

          const uint8_t* src = raw + block_idx * kIQ4NLBytesPerBlock;
          scales[block_idx] = static_cast<uint16_t>(src[0]) | (static_cast<uint16_t>(src[1]) << 8);
          int8_t* dst = weights + block_idx * kIQ4NLValuesPerBlock;

#pragma unroll
          for (int i = 0; i < 32; ++i) {
            const int nibble = (src[2 + (i & 15)] >> ((i / 16) * 4)) & 0x0f;
            dst[i] = kIQ4Values[nibble];
          }
        });
  });
}

void launch_q3_k_prepare(
    sycl::queue& queue, const uint8_t* raw, int8_t* weights, uint16_t* scales, int64_t rows, int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFQ3KPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }

          const uint8_t* src = raw + block_idx * kQ3KBytesPerBlock;
          const uint8_t* hmask = src;
          const uint8_t* qs = src + 32;
          const uint8_t* scale_meta = src + 96;
          const uint16_t d_bits = static_cast<uint16_t>(src[108]) | (static_cast<uint16_t>(src[109]) << 8);
          const sycl::half d = sycl::bit_cast<sycl::half>(d_bits);
          int8_t* dst = weights + block_idx * kQ3KValuesPerBlock;
          uint16_t* dst_scales = scales + block_idx * kQ3KGroupsPerBlock;

#pragma unroll
          for (int group = 0; group < kQ3KGroupsPerBlock; ++group) {
            const int low4 = (scale_meta[group & 7] >> ((group / 8) * 4)) & 0x0f;
            const int high2 = (scale_meta[8 + (group & 3)] >> ((group / 4) * 2)) & 0x03;
            const int signed_scale = (low4 | (high2 << 4)) - 32;
            dst_scales[group] = sycl::bit_cast<uint16_t>(d * static_cast<sycl::half>(signed_scale));

            const int q_set = group / 8;
            const int q_shift = ((group & 7) / 2) * 2;
            const int q_half = group & 1;
            const int h_bit = group / 2;
            const int h_half = group & 1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              const int low2 = (qs[q_set * 32 + q_half * 16 + i] >> q_shift) & 0x03;
              const int high = ((hmask[h_half * 16 + i] >> h_bit) & 0x01) ^ 0x01;
              dst[group * 16 + i] = static_cast<int8_t>(low2 - (high << 2));
            }
          }
        });
  });
}

void launch_iq3_s_prepare(
    sycl::queue& queue, const uint8_t* raw, int8_t* weights, uint16_t* scales, int64_t rows, int64_t blocks_per_row) {
  const int64_t block_count = rows * blocks_per_row;
  constexpr int64_t kLocalSize = 256;
  const int64_t global_size = round_up(block_count, kLocalSize);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GGUFIQ3SPrepareKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), [=](sycl::nd_item<1> item) {
          const int64_t block_idx = item.get_global_linear_id();
          if (block_idx >= block_count) {
            return;
          }

          const uint8_t* src = raw + block_idx * kIQ3SBytesPerBlock;
          const uint16_t d_bits = static_cast<uint16_t>(src[0]) | (static_cast<uint16_t>(src[1]) << 8);
          const sycl::half d = sycl::bit_cast<sycl::half>(d_bits);
          const uint8_t* qs = src + 2;
          const uint8_t* qh = src + 66;
          const uint8_t* signs = src + 74;
          const uint8_t* scale_meta = src + 106;
          int8_t* dst = weights + block_idx * kIQ3SValuesPerBlock;
          uint16_t* dst_scales = scales + block_idx * kIQ3SGroupsPerBlock;

#pragma unroll
          for (int group = 0; group < kIQ3SGroupsPerBlock; ++group) {
            const int scale_nibble = (scale_meta[group / 2] >> ((group & 1) * 4)) & 0x0f;
            const sycl::half scale = d * static_cast<sycl::half>(1 + 2 * scale_nibble);
            dst_scales[group] = sycl::bit_cast<uint16_t>(scale);

#pragma unroll
            for (int grid_in_group = 0; grid_in_group < 8; ++grid_in_group) {
              const int grid_idx = group * 8 + grid_in_group;
              const int lut_idx = qs[grid_idx] | (((qh[grid_idx / 8] >> (grid_idx & 7)) & 0x01) << 8);
              const uint32_t packed_grid = kIQ3SGrid[lut_idx];
#pragma unroll
              for (int component = 0; component < 4; ++component) {
                const int value_idx = grid_idx * 4 + component;
                const int magnitude = static_cast<int>((packed_grid >> (component * 8)) & 0xff) >> 2;
                const bool negative = ((signs[value_idx / 8] >> (value_idx & 7)) & 0x01) != 0;
                dst[value_idx] = static_cast<int8_t>(negative ? -magnitude : magnitude);
              }
            }
          }
        });
  });
}

void check_same_device(const torch::Tensor& tensor, const torch::Tensor& raw, const char* name) {
  TORCH_CHECK(tensor.device() == raw.device(), name, " must be on the same device as raw_qweight");
}

}  // namespace

SGL_KERNEL_EXPORT void gguf_q4_k_repack(
    const torch::Tensor& raw_qweight, torch::Tensor& packed_qweight, torch::Tensor& scales, torch::Tensor& zeros) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(packed_qweight);
  CHECK_INPUT(scales);
  CHECK_INPUT(zeros);
  check_same_device(packed_qweight, raw_qweight, "packed_qweight");
  check_same_device(scales, raw_qweight, "scales");
  check_same_device(zeros, raw_qweight, "zeros");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(packed_qweight.scalar_type() == at::kByte, "packed_qweight must be uint8");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 144]");
  TORCH_CHECK(packed_qweight.dim() == 2, "packed_qweight must be rank 2 [N, blocks * 128]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks * 8]");
  TORCH_CHECK(zeros.dim() == 2, "zeros must be rank 2 [N, blocks * 8]");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kQ4KBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 144");

  const int64_t rows = raw_qweight.size(0);
  TORCH_CHECK(rows > 0, "raw_qweight must contain at least one row");
  const int64_t blocks_per_row = raw_qweight.size(1) / kQ4KBytesPerBlock;
  TORCH_CHECK(
      packed_qweight.size(0) == rows && packed_qweight.size(1) == blocks_per_row * kW4A16BytesPerBlock,
      "packed_qweight shape must be [N, blocks * 128]");
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == blocks_per_row * kQ4KGroupsPerBlock,
      "scales shape must be [N, blocks * 8]");
  TORCH_CHECK(zeros.sizes() == scales.sizes(), "zeros shape must match scales shape");
  TORCH_CHECK(zeros.scalar_type() == scales.scalar_type(), "zeros dtype must match scales dtype");
  TORCH_CHECK(
      scales.scalar_type() == at::kBFloat16 || scales.scalar_type() == at::kHalf || scales.scalar_type() == at::kFloat,
      "scales and zeros must be bfloat16, float16, or float32");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (scales.scalar_type() == at::kBFloat16) {
    launch_q4_k_repack(
        queue,
        raw_qweight.data_ptr<uint8_t>(),
        packed_qweight.data_ptr<uint8_t>(),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(scales.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(zeros.data_ptr()),
        rows,
        blocks_per_row);
  } else if (scales.scalar_type() == at::kHalf) {
    launch_q4_k_repack(
        queue,
        raw_qweight.data_ptr<uint8_t>(),
        packed_qweight.data_ptr<uint8_t>(),
        reinterpret_cast<sycl::half*>(scales.data_ptr()),
        reinterpret_cast<sycl::half*>(zeros.data_ptr()),
        rows,
        blocks_per_row);
  } else {
    launch_q4_k_repack(
        queue,
        raw_qweight.data_ptr<uint8_t>(),
        packed_qweight.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        rows,
        blocks_per_row);
  }
}

SGL_KERNEL_EXPORT void
gguf_q4_k_prepare(const torch::Tensor& raw_qweight, torch::Tensor& packed_qweight, torch::Tensor& metadata) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(packed_qweight);
  CHECK_INPUT(metadata);
  check_same_device(packed_qweight, raw_qweight, "packed_qweight");
  check_same_device(metadata, raw_qweight, "metadata");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(packed_qweight.scalar_type() == at::kByte, "packed_qweight must be uint8");
  TORCH_CHECK(metadata.scalar_type() == at::kByte, "metadata must be uint8");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 144]");
  TORCH_CHECK(packed_qweight.dim() == 2, "packed_qweight must be rank 2 [N, blocks * 128]");
  TORCH_CHECK(metadata.dim() == 2, "metadata must be rank 2 [N, blocks * 16]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kQ4KBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 144");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kQ4KBytesPerBlock;
  TORCH_CHECK(
      packed_qweight.size(0) == rows && packed_qweight.size(1) == blocks_per_row * kW4A16BytesPerBlock,
      "packed_qweight shape must be [N, blocks * 128]");
  TORCH_CHECK(
      metadata.size(0) == rows && metadata.size(1) == blocks_per_row * 16, "metadata shape must be [N, blocks * 16]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_q4_k_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      packed_qweight.data_ptr<uint8_t>(),
      metadata.data_ptr<uint8_t>(),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void
gguf_q8_0_prepare(const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 34]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 32]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kQ80BytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 34");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kQ80BytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kQ80ValuesPerBlock,
      "weights shape must be [N, blocks * 32]");
  TORCH_CHECK(scales.size(0) == rows && scales.size(1) == blocks_per_row, "scales shape must be [N, blocks]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_q8_0_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void
gguf_q6_k_prepare(const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 210]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 256]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks * 16]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kQ6KBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 210");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kQ6KBytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kQ6KValuesPerBlock,
      "weights shape must be [N, blocks * 256]");
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == blocks_per_row * kQ6KScaleGroupsPerBlock,
      "scales shape must be [N, blocks * 16]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_q6_k_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void gguf_q5_k_prepare(
    const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales, torch::Tensor& minimums) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  CHECK_INPUT(minimums);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");
  check_same_device(minimums, raw_qweight, "minimums");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(minimums.scalar_type() == at::kHalf, "minimums must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 176]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 256]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks * 8]");
  TORCH_CHECK(minimums.dim() == 2, "minimums must be rank 2 [N, blocks * 8]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kQ5KBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 176");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kQ5KBytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kQ5KValuesPerBlock,
      "weights shape must be [N, blocks * 256]");
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == blocks_per_row * kQ5KGroupsPerBlock,
      "scales shape must be [N, blocks * 8]");
  TORCH_CHECK(minimums.sizes() == scales.sizes(), "minimums shape must match scales shape");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_q5_k_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      reinterpret_cast<uint16_t*>(minimums.data_ptr()),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void
gguf_iq4_xs_prepare(const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 136]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 256]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks * 8]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kIQ4XSBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 136");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kIQ4XSBytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kIQ4XSValuesPerBlock,
      "weights shape must be [N, blocks * 256]");
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == blocks_per_row * kIQ4XSGroupsPerBlock,
      "scales shape must be [N, blocks * 8]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_iq4_xs_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void
gguf_iq4_nl_prepare(const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 18]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 32]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kIQ4NLBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 18");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kIQ4NLBytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kIQ4NLValuesPerBlock,
      "weights shape must be [N, blocks * 32]");
  TORCH_CHECK(scales.size(0) == rows && scales.size(1) == blocks_per_row, "scales shape must be [N, blocks]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_iq4_nl_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void
gguf_q3_k_prepare(const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 110]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 256]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks * 16]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kQ3KBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 110");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kQ3KBytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kQ3KValuesPerBlock,
      "weights shape must be [N, blocks * 256]");
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == blocks_per_row * kQ3KGroupsPerBlock,
      "scales shape must be [N, blocks * 16]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_q3_k_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      rows,
      blocks_per_row);
}

SGL_KERNEL_EXPORT void
gguf_iq3_s_prepare(const torch::Tensor& raw_qweight, torch::Tensor& weights, torch::Tensor& scales) {
  CHECK_INPUT(raw_qweight);
  CHECK_INPUT(weights);
  CHECK_INPUT(scales);
  check_same_device(weights, raw_qweight, "weights");
  check_same_device(scales, raw_qweight, "scales");

  TORCH_CHECK(raw_qweight.scalar_type() == at::kByte, "raw_qweight must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kChar, "weights must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be float16");
  TORCH_CHECK(raw_qweight.dim() == 2, "raw_qweight must be rank 2 [N, blocks * 110]");
  TORCH_CHECK(weights.dim() == 2, "weights must be rank 2 [N, blocks * 256]");
  TORCH_CHECK(scales.dim() == 2, "scales must be rank 2 [N, blocks * 8]");
  TORCH_CHECK(raw_qweight.size(0) > 0, "raw_qweight must contain at least one row");
  TORCH_CHECK(
      raw_qweight.size(1) > 0 && raw_qweight.size(1) % kIQ3SBytesPerBlock == 0,
      "raw_qweight width must be a non-zero multiple of 110");

  const int64_t rows = raw_qweight.size(0);
  const int64_t blocks_per_row = raw_qweight.size(1) / kIQ3SBytesPerBlock;
  TORCH_CHECK(
      weights.size(0) == rows && weights.size(1) == blocks_per_row * kIQ3SValuesPerBlock,
      "weights shape must be [N, blocks * 256]");
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == blocks_per_row * kIQ3SGroupsPerBlock,
      "scales shape must be [N, blocks * 8]");

  c10::DeviceGuard device_guard(raw_qweight.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  launch_iq3_s_prepare(
      queue,
      raw_qweight.data_ptr<uint8_t>(),
      weights.data_ptr<int8_t>(),
      reinterpret_cast<uint16_t*>(scales.data_ptr()),
      rows,
      blocks_per_row);
}
