// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"

#include <cstdint>

#if defined(__ANDROID__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size) {
  if (size <= 0) return 0;
  float32x4_t max_v4 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  int i = 0;
  for (; i <= size - 4; i += 4) {
    max_v4 = vmaxq_f32(max_v4, vld1q_f32(data + i));
  }
  float max_vals_arr[4];
  vst1q_f32(max_vals_arr, max_v4);
  float max_v = max_vals_arr[0];
  for (int j = 1; j < 4; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  // Second pass: find first index with max_v
  float32x4_t target = vdupq_n_f32(max_v);
  for (i = 0; i <= size - 4; i += 4) {
    uint32x4_t cmp = vceqq_f32(vld1q_f32(data + i), target);
    uint32_t mask[4];
    vst1q_u32(mask, cmp);
    if (mask[0] || mask[1] || mask[2] || mask[3]) {
      for (int j = 0; j < 4; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexInt16Neon(const int16_t* data, int size) {
  if (size <= 0) return 0;
  int16x8_t max_v8 = vdupq_n_s16(std::numeric_limits<int16_t>::lowest());
  int i = 0;
  for (; i <= size - 8; i += 8) {
    max_v8 = vmaxq_s16(max_v8, vld1q_s16(data + i));
  }
  int16_t max_vals_arr[8];
  vst1q_s16(max_vals_arr, max_v8);
  int16_t max_v = max_vals_arr[0];
  for (int j = 1; j < 8; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  int16x8_t target = vdupq_n_s16(max_v);
  for (i = 0; i <= size - 8; i += 8) {
    uint16x8_t cmp = vceqq_s16(vld1q_s16(data + i), target);
    uint16_t mask[8];
    vst1q_u16(mask, cmp);
    if (mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] ||
        mask[6] || mask[7]) {
      for (int j = 0; j < 8; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexInt8Neon(const int8_t* data, int size) {
  if (size <= 0) return 0;
  int8x16_t max_v16 = vdupq_n_s8(std::numeric_limits<int8_t>::lowest());
  int i = 0;
  for (; i <= size - 16; i += 16) {
    max_v16 = vmaxq_s8(max_v16, vld1q_s8(data + i));
  }
  int8_t max_vals_arr[16];
  vst1q_s8(max_vals_arr, max_v16);
  int8_t max_v = max_vals_arr[0];
  for (int j = 1; j < 16; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  int8x16_t target = vdupq_n_s8(max_v);
  for (i = 0; i <= size - 16; i += 16) {
    uint8x16_t cmp = vceqq_s8(vld1q_s8(data + i), target);
    uint8_t mask[16];
    vst1q_u8(mask, cmp);
    // Quick check if any lane matched
    uint64_t low = vget_lane_u64(vreinterpret_u64_u8(vget_low_u8(cmp)), 0);
    uint64_t high = vget_lane_u64(vreinterpret_u64_u8(vget_high_u8(cmp)), 0);
    if (low || high) {
      for (int j = 0; j < 16; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}
#endif

absl::StatusOr<int> ApplyGreedySampling(::litert::TensorBuffer& decoded_logits,
                                        bool use_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType logits_tensor_type,
                          decoded_logits.TensorType());
  if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
    return FindMaxIndex<float>(decoded_logits, use_neon_sampling);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int16) {
    return FindMaxIndex<int16_t>(decoded_logits, use_neon_sampling);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int8) {
    return FindMaxIndex<int8_t>(decoded_logits, use_neon_sampling);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for greedy sampling: ",
                     logits_tensor_type.ElementType()));
  }
}

}  // namespace litert::lm
