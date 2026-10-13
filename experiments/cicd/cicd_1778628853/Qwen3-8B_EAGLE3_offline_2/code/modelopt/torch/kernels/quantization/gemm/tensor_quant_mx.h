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

#pragma once

#ifndef CUDA_HOST_DEVICE
#ifdef __CUDA_ARCH__
#define CUDA_HOST_DEVICE __host__ __device__ __forceinline__
#define MODELOPT_FORMAT_FAIL(str) __trap();
#else
#define CUDA_HOST_DEVICE inline
#define MODELOPT_FORMAT_FAIL(str) throw std::runtime_error(str);
#endif
#endif

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT static
#endif

#include <cuda_fp8.h>
#include <float.h>

enum class Types { E4M3, E5M2, INT8, E0M3, E1M2, E3M0, E2M1, E3M2, E2M3, E8M0 };

// FP4
CONSTANT float e2m1_table_values[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
CONSTANT float e2m1_table_bounds[7] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5};

CONSTANT float e1m2_table_values[8] = {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5};
CONSTANT float e1m2_table_bounds[7] = {0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25};

CONSTANT float e0m3_table_values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
CONSTANT float e0m3_table_bounds[7] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5};

CONSTANT float e3m0_table_values[8] = {0, 0.25, 0.5, 1, 2, 4, 8, 16};
CONSTANT float e3m0_table_bounds[7] = {0.125, 0.375, 0.75, 1.5, 3, 6, 12};

// FP6
CONSTANT float e3m2_table_values[32] = {0,   0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
                                        0.5, 0.625,  0.75,  0.875,  1,    1.25,   1.5,   1.75,
                                        2,   2.5,    3,     3.5,    4,    5,      6,     7,
                                        8,   10,     12,    14,     16,   20,     24,    28};
CONSTANT float e3m2_table_bounds[31] = {
    0.03125, 0.09375, 0.15625, 0.21875, 0.28125, 0.34375, 0.40625, 0.46875, 0.5625, 0.6875, 0.8125,
    0.9375,  1.125,   1.375,   1.625,   1.875,   2.25,    2.75,    3.25,    3.75,   4.5,    5.5,
    6.5,     7.5,     9,       11,      13,      15,      18,      22,      26};

CONSTANT float e2m3_table_values[32] = {
    0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
    2, 2.25,  2.5,  2.75,  3,   3.25,  3.5,  3.75,  4, 4.5,   5,    5.5,   6,   6.5,   7,    7.5};
CONSTANT float e2m3_table_bounds[31] = {
    0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0625, 1.1875, 1.3125,
    1.4375, 1.5625, 1.6875, 1.8125, 1.9375, 2.125,  2.375,  2.625,  2.875,  3.125,  3.375,
    3.625,  3.875,  4.25,   4.75,   5.25,   5.75,   6.25,   6.75,   7.25};

CUDA_HOST_DEVICE float convert_to_fp8_e4m3(const float x) {
  return static_cast<float>(static_cast<__nv_fp8_e4m3>(static_cast<float>(x)));
}

CUDA_HOST_DEVICE float convert_to_fp8_e5m2(const float x) {
  return static_cast<float>(static_cast<__nv_fp8_e5m2>(static_cast<float>(x)));
}

CUDA_HOST_DEVICE float convert_int8_saturating(const float x) {
  // Regular conversion w/rounding
  int converted = std::rint(x);

  // saturate high & low
  converted = std::clamp(converted, (int)(-INT8_MAX), (int)(INT8_MAX));

  // do the actual conversion.
  return static_cast<int8_t>(converted);
}

// // Rounding function. Round to nearest, ties away from zero.
CUDA_HOST_DEVICE float round_away_from_zero(float x, const float *values, const float *bounds,
                                            int n) {
  // Within current bounds.
  for (int i = 0; i < n - 1; i++) {
    if (x < bounds[i])
      return values[i];
  }
  // Max bounds.
  return values[n - 1];
}

// Rounding function. Round to nearest, ties to even
CUDA_HOST_DEVICE float round_to_even(float x, const float *values, const float *bounds, int n) {
  // Within current bounds.
  for (int i = 0; i < n - 1; i++) {
    // If less than tie, round down
    if (x < bounds[i])
      return values[i];
    // If equals to tie, round to representable value that has 0 in lsb of the mantissa
    if (x == bounds[i]) {
      // Since lsb alternates between 0 and 1, we can round down
      // if the index is even, and round up if the index is odd
      if (i % 2 == 0)
        return values[i];
      else
        return values[i + 1];
    }
  }
  // Max bounds.
  return values[n - 1];
}

CUDA_HOST_DEVICE float convert_to_table_types(float x, const int n, const float *values,
                                              const float *bounds) {
  float sign;
  if (x < 0.f) {
    sign = -1.f;
    x = -x;
  } else {
    sign = 1.f;
  }
  // Saturate infs and nans.
  if (isinf(x) || isnan(x))
    return sign * values[n - 1];

  // Round.
  float value = round_to_even(x, values, bounds, n);

  return value * sign;
}

CUDA_HOST_DEVICE float convert_to_e3m0(float x, const int n, const float *values,
                                       const float *bounds) {
  float sign;
  if (x < 0.f) {
    sign = -1.f;
    x = -x;
  } else {
    sign = 1.f;
  }
  // Encoding table for 4-bit formats.
  // Saturate infs and nans.
  if (isinf(x) || isnan(x))
    return sign * values[n - 1];

  // Round.
  float value = round_away_from_zero(x, values, bounds, n);

  return value * sign;
}

CUDA_HOST_DEVICE float convert_to_types(const float x, const Types format) {
  switch (format) {
  case Types::E4M3:
    return convert_to_fp8_e4m3(x);
  case Types::E5M2:
    return convert_to_fp8_e5m2(x);
  case Types::INT8:
    return convert_int8_saturating(x);
  case Types::E2M1:
    return convert_to_table_types(x, 8, e2m1_table_values, e2m1_table_bounds);
  case Types::E0M3:
    return convert_to_table_types(x, 8, e0m3_table_values, e0m3_table_bounds);
  case Types::E1M2:
    return convert_to_table_types(x, 8, e1m2_table_values, e1m2_table_bounds);
  case Types::E3M0:
    return convert_to_e3m0(x, 8, e3m0_table_values, e3m0_table_bounds);
  case Types::E3M2:
    return convert_to_table_types(x, 32, e3m2_table_values, e3m2_table_bounds);
  case Types::E2M3:
    return convert_to_table_types(x, 32, e2m3_table_values, e2m3_table_bounds);
  default:
    return 0.f;
  }
}

CUDA_HOST_DEVICE
float get_format_max(const Types format) {
  switch (format) {
  case Types::E4M3:
    return 448.f;
  case Types::E5M2:
    return 57344.f;
  case Types::INT8:
    return 127.f;
  case Types::E2M1:
    return 6.f;
  case Types::E0M3:
    return 7.f;
  case Types::E3M0:
    return 16.f;
  case Types::E1M2:
    return 3.5f;
  case Types::E3M2:
    return 28.f;
  case Types::E2M3:
    return 7.5f;
  default:
    // error
    return 0.f;
  }
}

// Returns the exponent of the maximum power-of-two representable in the given format.
__host__ __device__ __inline__ int get_format_lg2_max(Types format) {
  switch (format) {
  case Types::E5M2:
    return 15;
  case Types::E4M3:
    return 8;
  case Types::E3M2:
  case Types::E3M0:
    return 4;
  case Types::E2M3:
  case Types::E2M1:
    return 2;
  case Types::E1M2:
    return 1;
  case Types::E0M3:
    return 2;
  case Types::INT8:
    // This follows the psx-encodings definition of "INT8",
    // which are the 8-bit two's complement integers, symmetrized
    // by removing -128.
    //
    // In the context of OCP MX-formats, "INT8" refers to a
    // symmetrized Q1.6 type. In that case, this function
    // should return 0.
    return 6;
  default:
    MODELOPT_FORMAT_FAIL("Unsupported format for E8M0 scaling");
  }
  return -1;
}

inline int ceil_div(const int a, const int b) { return (a + b - 1) / b; }
