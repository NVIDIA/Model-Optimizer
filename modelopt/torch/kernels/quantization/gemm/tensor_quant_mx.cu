/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda/std/bit>
#include <cuda/std/tuple>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>

#include "tensor_quant_mx.h"

namespace {

static __device__ __host__ __inline__ float quantize(const float x, const float scale,
                                                     const float unscale, const Types format) {
  //  Get the sign.
  int sign;
  if (x < 0.0)
    sign = -1;
  if (x > 0.0)
    sign = 1;
  // Absolute value of x.
  float xabs = fabsf(x);
  // Scale.
  xabs = xabs * scale;
  // Quantize.
  xabs = convert_to_types(xabs, format);
  // Unscale.
  xabs = xabs * unscale;

  // Apply sign.
  return sign * xabs;
}

/*
 * Get the maximum for a format for quantization purposes.
 *
 * Note: we split out e8mY formats as a) they don't need scaling for
 *       dynamic range, and if we _do_ scale, we can easily get numerical
 *       weirdnesses - 1/scale has a habit of flushing -> 0 and causing
 *       NaN in answers. For this reason, effectively disable scaling
 *       for these formats (scale=unscale=1)
 */

// Algorithms for E8M0 scaling factor computation: OCP and NV
// Both implementations require or assume the following:
//   * amax is the maximum magnitude input value,
//   * dmax is the maximum magnitude representable by config,
//   * the range of config is symmetric.

// OCP algorithm:
//   scale = 2 ** clamp(floor(log2(amax)) - floor(log2(dmax)), -127, 127),
__device__ __host__ cuda::std::tuple<float, float> compute_scale_e8m0_OCP(float amax,
                                                                          Types format) {
  // lg2_dmax = log2(maximum power-of-two representable by config)
  int32_t lg2_dmax = get_format_lg2_max(format);

  // lg2_amax = floor(log2(amax))
  // The following code assumes amax is a positive, finite FP32 value.
  union {
    float f32;
    uint32_t u32;
  } amax_bits;

  amax_bits.f32 = amax;

  uint32_t amax_exponent_field = (amax_bits.u32 >> 23) & 0xff;
  uint32_t amax_significand_field = amax_bits.u32 & 0x7fffff;
  int32_t lg2_amax = amax_exponent_field == 0
                         ? static_cast<int32_t>(cuda::std::bit_width(amax_significand_field)) - 150
                         : static_cast<int32_t>(amax_exponent_field) - 127;

  int unscale_exponent = std::clamp(lg2_amax - lg2_dmax, -127, 127);

  float scale = ldexpf(1.f, -unscale_exponent);
  float unscale = ldexpf(1.f, unscale_exponent);

  return cuda::std::make_tuple(scale, unscale);
}

// NVIDIA algorithm:
//   unscale = 2 ** clamp (ceil(log2(amax/dmax)), -127, 127)
__device__ __host__ cuda::std::tuple<float, float> compute_scale_e8m0_NV(float amax, Types format) {
  // TODO: currently this FP32 division uses RNE. When the exact result
  // is just above a binade boundary, it will be rounded down, in which
  // case this routine may return the floor(), not the ceil()!
  //
  // We can fix this by performing the fdiv using RU. On __device__,
  // this can be accomplished by __fdiv_ru intrinsic. On __host__,
  // we may have to use <fenv>.

  float ratio = amax / get_format_max(format);

  // scale_exponent = clamp(ceil(log2(ratio)), -127, 127)
  // The following code assumes ratio is a positive, finite FP32 value.
  union {
    float f32;
    uint32_t u32;
  } ratio_bits;

  ratio_bits.f32 = ratio;

  uint32_t ratio_exponent_field = (ratio_bits.u32 >> 23) & 0xff;
  uint32_t ratio_significand_field = ratio_bits.u32 & 0x7fffff;
  int32_t unscale_exponent =
      ratio_significand_field > 0 && ratio_exponent_field != 0xfe &&
              !(ratio_exponent_field == 0 && ratio_significand_field <= 0x400000)
          ? static_cast<int32_t>(ratio_exponent_field) - 126
          : static_cast<int32_t>(ratio_exponent_field) - 127;

  float scale = ldexpf(1.f, -unscale_exponent);
  float unscale = ldexpf(1.f, unscale_exponent);

  return cuda::std::make_tuple(scale, unscale);
}

// Compute scaling factor.
__device__ __host__ __inline__ cuda::std::tuple<float, float>
compute_scale(const float amax, const Types format, const Types scale_format) {
  if (amax == 0 || isinf(amax) || isnan(amax)) {
    return cuda::std::make_tuple(1.f, 1.f);
  } else {
    // intercept e8m0 specifically
    if (scale_format == Types::E8M0) {
      return compute_scale_e8m0_NV(amax, format);
    }
    // otherwise take the general path
    const float emax = get_format_max(format);
    double scale = emax / amax;
    double inv_scale = convert_to_types(1. / scale, scale_format);
    scale = 1. / inv_scale;
    return cuda::std::make_tuple(scale, inv_scale);
  }
}

__device__ __host__ __inline__ cuda::std::tuple<float, float>
compute_scale_with_global(const float global_amax, const float local_amax, const Types format,
                          const Types scale_format) {
  bool local_amax_invalid = local_amax == 0 || isinf(local_amax) || isnan(local_amax);
  bool global_amax_invalid = global_amax == 0 || isinf(global_amax) || isnan(global_amax);
  if (local_amax_invalid || global_amax_invalid) {
    return cuda::std::make_tuple(1.f, 1.f);
  } else {
    const float elem_format_max = get_format_max(format);
    const float scale_format_max = get_format_max(scale_format);

    const float local_unscale = local_amax / elem_format_max;
    // Note: if elem_format_max / global_amax > 1 and scale_format_max
    // is close to FLT_MAX, we can overflow here
    //                WAR is to just do these few calculations in double.
    const double two_level_scale =
        static_cast<double>(scale_format_max) * (elem_format_max / global_amax);

    const double local_unscale_q =
        convert_to_types(static_cast<float>(local_unscale * two_level_scale), scale_format) /
        two_level_scale;
    const float scale = 1. / local_unscale_q;

    return cuda::std::make_tuple(scale, static_cast<float>(local_unscale_q));
  }
}

}; // anonymous namespace

namespace {

template <int THREAD_X, int THREAD_Y = 4>
__device__ float compute_max_block(const float x, const int block_size,
                                   const float clamp = FLT_MAX) {
  float xabs = fabsf(x);
  xabs = (xabs > clamp) ? clamp : xabs;

  typedef cub::BlockReduce<float, THREAD_X> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage[THREAD_Y];

  float amax = BlockReduce(temp_storage[threadIdx.y]).Reduce(xabs, [](float a, float b) {
    return fmaxf(a, b);
  });

  __shared__ float amax_smem[THREAD_Y];
  if (threadIdx.x == 0)
    amax_smem[threadIdx.y] = amax;

  __syncthreads();

  return amax_smem[threadIdx.y];
}

template <int THREAD_Y>
__device__ float compute_max_warp(const float x, const int block_size,
                                  const float clamp = FLT_MAX) {
  float xabs = fabsf(x);
  xabs = (xabs > clamp) ? clamp : xabs;

  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[THREAD_Y];

  float amax = WarpReduce(temp_storage[threadIdx.y]).Reduce(xabs, [](float a, float b) {
    return fmaxf(a, b);
  });
  amax = __shfl_sync(0xffffffff, amax, 0);
  return amax;
}

template <int THREAD_X, int THREAD_Y>
__device__ float compute_max(const float x, const int block_size, const float clamp = FLT_MAX) {
  if (THREAD_X == 32) {
    return compute_max_warp<THREAD_Y>(x, block_size, clamp);
  } else {
    return compute_max_block<THREAD_X, THREAD_Y>(x, block_size, clamp);
  }
}

}; // namespace

template <typename T, typename T_y, int THREAD_X, int THREAD_Y>
__global__ void amax_and_convert_kernel(const T *x, T_y *y, const int64_t n, const int blocksize,
                                        const int actual_cols, const int padded_cols, Types format,
                                        Types scale_format, const float *global_amax = nullptr,
                                        const bool per_channel_scaling = false,
                                        const int ncols = -1) {
  // map thread -> element
  const uint64_t block_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int thread_id = threadIdx.x;
  const int warp_id = threadIdx.y;

  // Unique id in the padded matrix
  const int64_t idx = block_idx * blocksize + thread_id;

  const int blocks_per_row = padded_cols / blocksize;

  if (idx >= n)
    return;

  // bool output_scaled = (block_unscale_ptr != nullptr);

  float scale = 1.f, unscale = 1.f;

  // row and col are in terms of the "padded" size that we launched based off
  const int row = block_idx / blocks_per_row;
  const int block_in_row = block_idx % blocks_per_row;
  const int col = block_in_row * blocksize + thread_id;
  // Differentiate between real and padded elements
  const bool in_padding = col >= actual_cols;

  //  map threads to the actual memory locations
  //  Example: x = [2, 17], bs=16
  //  thread @ 0, 0 = 0
  //  thread @ 0, 15 = 15
  //  thread @ 1, 0 = 17
  const size_t real_idx = row * actual_cols + col;

  // read either real or padded (zero) value
  float thread_val = (in_padding) ? 0.f : static_cast<float>(x[real_idx]);
  float block_amax = compute_max<THREAD_X, THREAD_Y>(thread_val, blocksize);

  if (global_amax) {
    int global_idx = (per_channel_scaling) ? idx / ncols : 0;
    float global_amax_val = global_amax[global_idx];
    cuda::std::tie(scale, unscale) =
        compute_scale_with_global(global_amax_val, block_amax, format, scale_format);
  } else {
    cuda::std::tie(scale, unscale) = compute_scale(block_amax, format, scale_format);
  }

  // if thread isn't handling a real value, nothing to output.
  if (in_padding)
    return;

  y[real_idx] = quantize(thread_val, scale, unscale, format);
}

std::tuple<float *, bool> get_global_scaling(at::Tensor x, std::optional<at::Tensor> global_amax) {
  float *global_amax_ptr =
      global_amax.has_value() ? global_amax.value().data_ptr<float>() : nullptr;

  // if we need per-channel scaling, following must be true:
  // 1) 2d input tensor
  // 2) 1d blocksize
  // 3) # global_amax elements == # rows of collapsed matrix
  bool per_channel_scaling = false;
  int cols = x.size(-1);
  if (global_amax_ptr) {
    per_channel_scaling = (x.dim() == 2) && (global_amax.value().numel() == x.size(0));
  }

  return std::make_tuple(global_amax_ptr, per_channel_scaling);
}

using output_t = std::tuple<at::Tensor, std::optional<at::Tensor>>;

#define DISPATCH_FUSED_AMAX_KERNEL(T, TX, TY)                                                      \
  amax_and_convert_kernel<scalar_t, T, TX, TY>                                                     \
      <<<blocks, block, 0, c10::cuda::getCurrentCUDAStream()>>>(                                   \
          x.data_ptr<scalar_t>(), y.data_ptr<T>(), n, blocksize, actual_cols, padded_cols, format, \
          scale_format, global_amax_ptr, per_channel_scaling, x.size(-1))

#define DISPATCH_SINGLE_TYPE(T)                                                                    \
  if (blocksize == 8) {                                                                            \
    DISPATCH_FUSED_AMAX_KERNEL(T, 8, blocks_per_cta);                                              \
  } else if (blocksize == 16) {                                                                    \
    DISPATCH_FUSED_AMAX_KERNEL(T, 16, blocks_per_cta);                                             \
  } else if (blocksize == 32) {                                                                    \
    DISPATCH_FUSED_AMAX_KERNEL(T, 32, blocks_per_cta);                                             \
  } else {                                                                                         \
    throw std::invalid_argument("Blocksize for fused call must be one of {8, 16, 32}");            \
  }

#define DISPATCH_FLOAT_AND_HALF_AND_BFLOAT(TYPE, NAME, ...)                                        \
  switch (TYPE) {                                                                                  \
  case at::ScalarType::Float: {                                                                    \
    using scalar_t = float;                                                                        \
    __VA_ARGS__;                                                                                   \
    break;                                                                                         \
  }                                                                                                \
  case at::ScalarType::Half: {                                                                     \
    using scalar_t = at::Half;                                                                     \
    __VA_ARGS__;                                                                                   \
    break;                                                                                         \
  }                                                                                                \
  case at::ScalarType::BFloat16: {                                                                 \
    using scalar_t = at::BFloat16;                                                                 \
    __VA_ARGS__;                                                                                   \
    break;                                                                                         \
  }                                                                                                \
  default:                                                                                         \
    AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                \
  }

int pad_dim_to_blocksize(const int dim, const int bs) { return dim + (bs - (dim % bs)); }

at::Tensor fused_amax_convert(at::Tensor x, const int blocksize, Types format, Types scale_format,
                              std::optional<at::Tensor> global_amax) {
  std::optional<at::Tensor> global_amax_float = std::nullopt;

  if (global_amax.has_value()) {
    const at::Tensor &_global_amax = *global_amax;
    global_amax_float = at::_cast_Float(_global_amax);
  }
  // const bool native_mode = is_native(output_mode);
  auto y_options = x.options();

  at::Tensor y = at::empty_like(x, y_options);

  size_t n = x.numel();
  int ndim = x.dim();
  // Handle partial blocks by fake-padding to full blocks
  const int actual_cols = x.size(-1);
  const int padded_cols = pad_dim_to_blocksize(actual_cols, blocksize);
  // pad the total size, replace the final dimension with padded one
  n = n / actual_cols * padded_cols;

  const int blocks_per_cta = 4;

  const dim3 block(blocksize, blocks_per_cta);
  const int blocks = ceil_div(n, blocksize * blocks_per_cta);

  auto [global_amax_ptr, per_channel_scaling] = get_global_scaling(x, global_amax_float);

  DISPATCH_FLOAT_AND_HALF_AND_BFLOAT(x.scalar_type(), "fused_amax_convert_kernel",
                                     { DISPATCH_SINGLE_TYPE(scalar_t); });

  return y;
}

#undef DISPATCH_FUSED_AMAX_KERNEL
#undef DISPATCH_SINGLE_TYPE
#undef DISPATCH_FLOAT_AND_HALF_AND_BFLOAT

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_amax_convert", &fused_amax_convert,
        "Compute block amax and quantize the tensor to mx formats", py::arg("inputs"),
        py::arg("block_size"), py::arg("format"), py::arg("scale_format"),
        py::arg("global_amax") = py::none());
  m.def("convert_to_exmy", &convert_to_types, "Convert value to exmy format.", py::arg("x"),
        py::arg("format"));
  py::enum_<Types>(m, "Types")
      .value("E4M3", Types::E4M3)
      .value("E5M2", Types::E5M2)
      .value("INT8", Types::INT8)
      .value("E0M3", Types::E0M3)
      .value("E1M2", Types::E1M2)
      .value("E3M0", Types::E3M0)
      .value("E2M1", Types::E2M1)
      .value("E3M2", Types::E3M2)
      .value("E2M3", Types::E2M3)
      .value("E8M0", Types::E8M0);
}
