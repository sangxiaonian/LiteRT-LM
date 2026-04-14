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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

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

absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers) {
  static constexpr absl::string_view kInputPos = "input_pos";
  auto& input_pos_buffer = in_buffers.at(kInputPos);

  LITERT_ASSIGN_OR_RETURN(
      auto pos_lock,
      ::litert::TensorBufferScopedLock::Create(
          input_pos_buffer, ::litert::TensorBuffer::LockMode::kRead));
  int start_pos = static_cast<const int32_t*>(pos_lock.second)[0];

  auto perform_update =
      [&](::litert::TensorBuffer& cache,
          const ::litert::TensorBuffer& slice) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(auto cache_type, cache.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto slice_type, slice.TensorType());
    auto cache_dims = cache_type.Layout().Dimensions();
    auto slice_dims = slice_type.Layout().Dimensions();
    int rank = cache_type.Layout().Rank();

    LITERT_ASSIGN_OR_RETURN(size_t cache_bytes, cache.Size());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                            cache_type.Layout().NumElements());
    size_t element_size = cache_bytes / num_elements;
    bool last_dim_matches = (cache_dims[rank - 1] == slice_dims[rank - 1]);

    int64_t hidden_dim, cache_seq, slice_seq;
    int slice_rank = slice_type.Layout().Rank();
    if (last_dim_matches) {
      hidden_dim = cache_dims[rank - 1];
      cache_seq = cache_dims[rank - 2];
      slice_seq = slice_dims[slice_rank - 2];
    } else {
      cache_seq = cache_dims[rank - 1];
      hidden_dim = cache_dims[rank - 2];
      // In transposed layout, sequence dim is the LAST dimension of cache.
      // We look for hidden_dim in slice to find where the sequence dim is.
      if (slice_dims[slice_rank - 1] == hidden_dim) {
        slice_seq = slice_dims[slice_rank - 2];
      } else {
        slice_seq = slice_dims[slice_rank - 1];
      }
    }

    if (start_pos + slice_seq > cache_seq) {
      return absl::OutOfRangeError("KV-cache update out of range");
    }

    LITERT_ASSIGN_OR_RETURN(
        auto cache_lock, ::litert::TensorBufferScopedLock::Create(
                             cache, ::litert::TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(
        auto slice_lock, ::litert::TensorBufferScopedLock::Create(
                             slice, ::litert::TensorBuffer::LockMode::kRead));

    uint8_t* cache_ptr = static_cast<uint8_t*>(cache_lock.second);
    const uint8_t* slice_ptr = static_cast<const uint8_t*>(slice_lock.second);

    if (last_dim_matches) {
      std::memcpy(cache_ptr + (start_pos * hidden_dim * element_size),
                  slice_ptr, slice_seq * hidden_dim * element_size);
    } else {
#if defined(__ANDROID__) && defined(__ARM_NEON) && defined(__aarch64__)
      if (slice_seq == 1 && element_size == 1) {
        int64_t h = 0;
        for (; h <= hidden_dim - 16; h += 16) {
          uint8x16_t v = vld1q_u8(slice_ptr + h);
          cache_ptr[(h + 0) * cache_seq + start_pos] = vgetq_lane_u8(v, 0);
          cache_ptr[(h + 1) * cache_seq + start_pos] = vgetq_lane_u8(v, 1);
          cache_ptr[(h + 2) * cache_seq + start_pos] = vgetq_lane_u8(v, 2);
          cache_ptr[(h + 3) * cache_seq + start_pos] = vgetq_lane_u8(v, 3);
          cache_ptr[(h + 4) * cache_seq + start_pos] = vgetq_lane_u8(v, 4);
          cache_ptr[(h + 5) * cache_seq + start_pos] = vgetq_lane_u8(v, 5);
          cache_ptr[(h + 6) * cache_seq + start_pos] = vgetq_lane_u8(v, 6);
          cache_ptr[(h + 7) * cache_seq + start_pos] = vgetq_lane_u8(v, 7);
          cache_ptr[(h + 8) * cache_seq + start_pos] = vgetq_lane_u8(v, 8);
          cache_ptr[(h + 9) * cache_seq + start_pos] = vgetq_lane_u8(v, 9);
          cache_ptr[(h + 10) * cache_seq + start_pos] = vgetq_lane_u8(v, 10);
          cache_ptr[(h + 11) * cache_seq + start_pos] = vgetq_lane_u8(v, 11);
          cache_ptr[(h + 12) * cache_seq + start_pos] = vgetq_lane_u8(v, 12);
          cache_ptr[(h + 13) * cache_seq + start_pos] = vgetq_lane_u8(v, 13);
          cache_ptr[(h + 14) * cache_seq + start_pos] = vgetq_lane_u8(v, 14);
          cache_ptr[(h + 15) * cache_seq + start_pos] = vgetq_lane_u8(v, 15);
        }
        for (; h < hidden_dim; ++h) {
          cache_ptr[h * cache_seq + start_pos] = slice_ptr[h];
        }
        return absl::OkStatus();
      }
      if (slice_seq == 1 && element_size == 2) {
        int64_t h = 0;
        const uint16_t* s_ptr = reinterpret_cast<const uint16_t*>(slice_ptr);
        uint16_t* c_ptr = reinterpret_cast<uint16_t*>(cache_ptr);
        for (; h <= hidden_dim - 8; h += 8) {
          uint16x8_t v = vld1q_u16(s_ptr + h);
          c_ptr[(h + 0) * cache_seq + start_pos] = vgetq_lane_u16(v, 0);
          c_ptr[(h + 1) * cache_seq + start_pos] = vgetq_lane_u16(v, 1);
          c_ptr[(h + 2) * cache_seq + start_pos] = vgetq_lane_u16(v, 2);
          c_ptr[(h + 3) * cache_seq + start_pos] = vgetq_lane_u16(v, 3);
          c_ptr[(h + 4) * cache_seq + start_pos] = vgetq_lane_u16(v, 4);
          c_ptr[(h + 5) * cache_seq + start_pos] = vgetq_lane_u16(v, 5);
          c_ptr[(h + 6) * cache_seq + start_pos] = vgetq_lane_u16(v, 6);
          c_ptr[(h + 7) * cache_seq + start_pos] = vgetq_lane_u16(v, 7);
        }
        for (; h < hidden_dim; ++h) {
          c_ptr[h * cache_seq + start_pos] = s_ptr[h];
        }
        return absl::OkStatus();
      }
#endif
      for (int64_t h = 0; h < hidden_dim; ++h) {
        std::memcpy(cache_ptr + (h * cache_seq + start_pos) * element_size,
                    slice_ptr + (h * slice_seq) * element_size,
                    slice_seq * element_size);
      }
    }
    return absl::OkStatus();
  };

  for (int layer_id = 0;; ++layer_id) {
    char k_cache_name[32];
    snprintf(k_cache_name, sizeof(k_cache_name), "kv_cache_k_%d", layer_id);
    if (!in_buffers.contains(k_cache_name)) break;

    char v_cache_name[32];
    snprintf(v_cache_name, sizeof(v_cache_name), "kv_cache_v_%d", layer_id);
    char k_slice_name[32];
    snprintf(k_slice_name, sizeof(k_slice_name), "kv_slice_k_%d", layer_id);
    char v_slice_name[32];
    snprintf(v_slice_name, sizeof(v_slice_name), "kv_slice_v_%d", layer_id);

    auto& in_k_cache = in_buffers.at(k_cache_name);
    auto& in_v_cache = in_buffers.at(v_cache_name);
    const auto& k_slice = in_buffers.at(k_slice_name);
    const auto& v_slice = in_buffers.at(v_slice_name);

    LITERT_RETURN_IF_ERROR(perform_update(in_k_cache, k_slice));
    LITERT_RETURN_IF_ERROR(perform_update(in_v_cache, v_slice));

    if (out_buffers.contains(k_cache_name)) {
      auto& out_k_cache = out_buffers.at(k_cache_name);
      if (in_k_cache.Get() != out_k_cache.Get()) {
        LITERT_RETURN_IF_ERROR(perform_update(out_k_cache, k_slice));
      }
    }
    if (out_buffers.contains(v_cache_name)) {
      auto& out_v_cache = out_buffers.at(v_cache_name);
      if (in_v_cache.Get() != out_v_cache.Get()) {
        LITERT_RETURN_IF_ERROR(perform_update(out_v_cache, v_slice));
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace litert::lm
