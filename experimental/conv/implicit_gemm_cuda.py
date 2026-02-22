# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimized CUDA-based Conv3D Implicit GEMM with FP4 quantization using BF16 WMMA Tensor Cores.

This module provides an optimized CUDA kernel for Conv3D via implicit GEMM with
fused FP4 fake quantization. The kernel is compiled just-in-time using
PyTorch's cpp_extension.

Key optimizations:
1. BF16 WMMA tensor core operations (m16n16k16) with FP32 accumulators
2. On-the-fly spatial index computation (no global memory lookup tables)
3. Dual FP4_BLOCK_SIZE support (128 and 256) with optimized tile configs:
   - FP4_BLOCK_SIZE=128: BM=64, BN=64, BK=128, 8 warps (256 threads), ~35KB shared
   - FP4_BLOCK_SIZE=256: BM=64, BN=64, BK=256, 8 warps (256 threads), ~69KB shared
4. Register-fused FP4 quantization (quantize during A-tile load, eliminates sync)
5. Branchless FP4 quantization using predicated selects
6. BF16 shared memory (halves memory vs FP32)
7. L2-friendly block scheduling (swizzled grid)
8. FP8 E4M3 round-trip for scale quantization (matches Triton exactly)
"""

import torch
import torch.nn.functional as F

# C++ header for function declarations
CPP_SOURCE = r"""
torch::Tensor conv3d_implicit_gemm_cuda(
    torch::Tensor x_pad,
    torch::Tensor w_flat,
    torch::Tensor bias,
    torch::Tensor act_amax,
    int N_batch, int Cin, int Dp, int Hp, int Wp,
    int Cout, int OD, int OH, int OW,
    int kD, int kH, int kW,
    int sd, int sh, int sw,
    int dd, int dh, int dw,
    int M, int K,
    bool quant_act, bool has_bias,
    int fp4_block_size
);
"""

# Optimized CUDA kernel with BF16 WMMA tensor cores
CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// =============================================================================
// FP4 Quantization Helpers
// =============================================================================

__device__ __forceinline__ float fp4_quantize_value(float scaled) {
    float q;
    q = (scaled <= 5.0f) ? 4.0f : 6.0f;
    q = (scaled < 3.5f) ? 3.0f : q;
    q = (scaled <= 2.5f) ? 2.0f : q;
    q = (scaled < 1.75f) ? 1.5f : q;
    q = (scaled <= 1.25f) ? 1.0f : q;
    q = (scaled < 0.75f) ? 0.5f : q;
    q = (scaled <= 0.25f) ? 0.0f : q;
    return q;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float fp8_e4m3_round_trip(float x) {
    if (x == 0.0f) return 0.0f;

    unsigned int bits = __float_as_uint(x);
    unsigned int sign = bits >> 31;
    int exp = ((bits >> 23) & 0xff) - 127;
    unsigned int mantissa = bits & 0x7fffff;

    if (exp > 8) return sign ? -448.0f : 448.0f;
    if (exp < -9) return 0.0f;

    unsigned int mantissa_3bit = (mantissa + (1 << 19)) >> 20;
    if (mantissa_3bit > 7) {
        mantissa_3bit = 0;
        exp += 1;
        if (exp > 8) return sign ? -448.0f : 448.0f;
    }

    if (exp < -6) {
        int shift = -6 - exp;
        mantissa_3bit = (mantissa_3bit | 8) >> shift;
        exp = -6;
    }

    int fp32_exp = exp + 127;
    unsigned int fp32_mantissa = mantissa_3bit << 20;
    unsigned int fp32_bits = (sign << 31) | (fp32_exp << 23) | fp32_mantissa;

    return __uint_as_float(fp32_bits);
}

__device__ __forceinline__ float quantize_scale_fp8(float block_max, float global_scale) {
    float scaled = block_max / (6.0f * global_scale);
    scaled = fminf(scaled, 448.0f);
    float quantized = fp8_e4m3_round_trip(scaled);
    return quantized * global_scale;
}

// =============================================================================
// BF16 WMMA Conv3D Implicit GEMM Kernel
// =============================================================================
// Template parameters:
//   QUANT_ACT  - whether to apply FP4 quantization
//   HAS_BIAS   - whether bias is present
//   BLOCK_M    - M tile size (64)
//   BLOCK_N    - N tile size (32)
//   BLOCK_K    - K tile size (matches FP4_BLOCK_SIZE: 128 or 256)
//   WARPS_M    - warp tiling in M dimension (2)
//   WARPS_N    - warp tiling in N dimension (2)
//   L2_SWIZZLE_GROUP - group size for L2-friendly block scheduling
//
// Each warp computes a (WARP_M x WARP_N) output tile using 16x16x16 WMMA.
// WARP_M = BLOCK_M / WARPS_M, WARP_N = BLOCK_N / WARPS_N
// WARP_TILES_M = WARP_M / 16, WARP_TILES_N = WARP_N / 16
//
// Shared memory layout (BF16):
//   As[BLOCK_M][BK_STRIDE]  - M-major (row_major for WMMA A-fragments)
//   Bs[BLOCK_K][BN_STRIDE]  - K-major (row_major for WMMA B-fragments)

template<
    bool QUANT_ACT, bool HAS_BIAS,
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int WARPS_M, int WARPS_N,
    int L2_SWIZZLE_GROUP = 8
>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 2)
conv3d_implicit_gemm_wmma(
    const float* __restrict__ x_pad,
    const float* __restrict__ w_flat,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const float* __restrict__ act_amax,
    int Cin, int Dp, int Hp, int Wp,
    int Cout, int OD, int OH, int OW,
    int kD, int kH, int kW,
    int sd, int sh, int sw,
    int dd, int dh, int dw,
    int M, int K
) {
    // Derived constants
    constexpr int NUM_WARPS = WARPS_M * WARPS_N;
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int WARP_M = BLOCK_M / WARPS_M;   // 32
    constexpr int WARP_N = BLOCK_N / WARPS_N;    // 16
    constexpr int WARP_TILES_M = WARP_M / 16;    // 2
    constexpr int WARP_TILES_N = WARP_N / 16;    // 1

    // BF16 shared memory strides with padding to avoid bank conflicts
    // Pad by 8 BF16 elements (16 bytes) — keeps 16-byte alignment while breaking conflicts
    constexpr int BK_STRIDE = BLOCK_K + 8;
    constexpr int BN_STRIDE = BLOCK_N + 8;

    // Thread/warp indices
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / WARPS_N;    // which M-warp (0..WARPS_M-1)
    const int warp_n = warp_id % WARPS_N;    // which N-warp (0..WARPS_N-1)

    // L2-friendly block scheduling (swizzle)
    int bm, bn;
    {
        const int pid = blockIdx.x;
        constexpr int GS = L2_SWIZZLE_GROUP;
        const int grid_n = (Cout + BLOCK_N - 1) / BLOCK_N;
        const int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
        const int tiles_per_group = GS * grid_n;

        const int group_row = pid / tiles_per_group;
        const int group_rem = pid % tiles_per_group;
        bn = group_rem / GS;
        const int swizzle_lane = group_rem % GS;
        bm = group_row * GS + swizzle_lane;

        if (bm >= grid_m || bn >= grid_n) return;
    }

    // Dynamic shared memory — BF16 tiles
    extern __shared__ char smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    // As: [BLOCK_M][BK_STRIDE]  — M-major
    constexpr int A_SMEM_ELEMS = BLOCK_M * BK_STRIDE;
    __nv_bfloat16* Bs = As + A_SMEM_ELEMS;
    // Bs: [BLOCK_K][BN_STRIDE]  — K-major

    // WMMA accumulators — FP32
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; wn++) {
            wmma::fill_fragment(acc[wm][wn], 0.0f);
        }
    }

    // Global scale for FP4 quantization
    float global_scale = 1.0f;
    if constexpr (QUANT_ACT) {
        global_scale = act_amax[0] / (6.0f * 448.0f);
    }

    // Precompute spatial constants
    const int HpWp = Hp * Wp;
    const int DpHpWp = Dp * HpWp;
    const int kHW = kH * kW;
    const int kDHW = kD * kHW;
    const int OHW = OH * OW;
    const int ODHW = OD * OHW;

    const int m_start = bm * BLOCK_M;
    const int n_start = bn * BLOCK_N;
    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // Total elements to load cooperatively
    constexpr int A_ELEMS = BLOCK_M * BLOCK_K;
    constexpr int B_ELEMS = BLOCK_K * BLOCK_N;

    // Main loop over K tiles
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start_tile = k_tile * BLOCK_K;

        // =====================================================================
        // Load A tile into BF16 shared memory (M-major layout)
        // As[m][k] stored at As[m * BK_STRIDE + k]
        // =====================================================================
        if constexpr (QUANT_ACT) {
            // Fused FP4 quantization: each warp handles M-rows
            constexpr int ELEMS_PER_LANE = (BLOCK_K + 31) / 32;

            for (int m = warp_id; m < BLOCK_M; m += NUM_WARPS) {
                int m_idx = m_start + m;

                int n_batch, od_val, oh_val, ow_val;
                if (m_idx < M) {
                    n_batch = m_idx / ODHW;
                    int rem = m_idx % ODHW;
                    od_val = rem / OHW;
                    rem = rem % OHW;
                    oh_val = rem / OW;
                    ow_val = rem % OW;
                } else {
                    n_batch = 0; od_val = 0; oh_val = 0; ow_val = 0;
                }

                float local_max = 0.0f;
                float vals[ELEMS_PER_LANE];

                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++) {
                    int k = lane_id + i * 32;
                    float val = 0.0f;
                    if (k < BLOCK_K && m_idx < M) {
                        int k_idx = k_start_tile + k;
                        if (k_idx < K) {
                            int c = k_idx / kDHW;
                            int remk = k_idx % kDHW;
                            int kd_v = remk / kHW;
                            remk = remk % kHW;
                            int kh_v = remk / kW;
                            int kw_v = remk % kW;

                            int id = od_val * sd + kd_v * dd;
                            int ih = oh_val * sh + kh_v * dh;
                            int iw = ow_val * sw + kw_v * dw;

                            val = x_pad[n_batch * Cin * DpHpWp + c * DpHpWp + id * HpWp + ih * Wp + iw];
                        }
                    }
                    vals[i] = val;
                    local_max = fmaxf(local_max, fabsf(val));
                }

                float block_max = warp_reduce_max(local_max);
                float scale = quantize_scale_fp8(block_max, global_scale);
                if (scale < 1e-5f) scale = 1.0f;
                float inv_scale = 1.0f / scale;

                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++) {
                    int k = lane_id + i * 32;
                    if (k < BLOCK_K) {
                        float val = vals[i];
                        float sign = (val >= 0.0f) ? 1.0f : -1.0f;
                        float q = fp4_quantize_value(fabsf(val) * inv_scale);
                        float result = sign * q * scale;
                        // M-major: As[m * BK_STRIDE + k]
                        As[m * BK_STRIDE + k] = __float2bfloat16(result);
                    }
                }
            }
        } else {
            // Non-quantized: cooperative load, store as BF16 in M-major
            #pragma unroll 4
            for (int i = tid; i < A_ELEMS; i += NUM_THREADS) {
                int local_m = i / BLOCK_K;
                int local_k = i % BLOCK_K;
                int m_idx = m_start + local_m;
                int k_idx = k_start_tile + local_k;

                float val = 0.0f;
                if (m_idx < M && k_idx < K) {
                    int n_batch = m_idx / ODHW;
                    int rem = m_idx % ODHW;
                    int od_val = rem / OHW;
                    rem = rem % OHW;
                    int oh_val = rem / OW;
                    int ow_val = rem % OW;

                    int c = k_idx / kDHW;
                    int remk = k_idx % kDHW;
                    int kd_v = remk / kHW;
                    remk = remk % kHW;
                    int kh_v = remk / kW;
                    int kw_v = remk % kW;

                    int id = od_val * sd + kd_v * dd;
                    int ih = oh_val * sh + kh_v * dh;
                    int iw = ow_val * sw + kw_v * dw;

                    val = x_pad[n_batch * Cin * DpHpWp + c * DpHpWp + id * HpWp + ih * Wp + iw];
                }
                // M-major: As[m * BK_STRIDE + k]
                As[local_m * BK_STRIDE + local_k] = __float2bfloat16(val);
            }
        }

        // =====================================================================
        // Load B tile into BF16 shared memory (K-major layout)
        // Bs[k][n] stored at Bs[k * BN_STRIDE + n]
        // =====================================================================
        #pragma unroll 4
        for (int i = tid; i < B_ELEMS; i += NUM_THREADS) {
            int local_k = i / BLOCK_N;
            int local_n = i % BLOCK_N;
            int k_idx = k_start_tile + local_k;
            int n_idx = n_start + local_n;

            float val = 0.0f;
            if (k_idx < K && n_idx < Cout) {
                val = w_flat[k_idx * Cout + n_idx];
            }
            Bs[local_k * BN_STRIDE + local_n] = __float2bfloat16(val);
        }

        __syncthreads();

        // =====================================================================
        // WMMA Compute: iterate over K in steps of 16 (WMMA K-dim)
        // =====================================================================
        constexpr int K_STEPS = BLOCK_K / 16;

        #pragma unroll
        for (int kk = 0; kk < K_STEPS; kk++) {
            // Load A and B fragments from shared memory
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[WARP_TILES_M];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[WARP_TILES_N];

            // Load A fragments: each from As[(warp_m * WARP_M + wm*16) * BK_STRIDE + kk*16]
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; wm++) {
                int a_row = warp_m * WARP_M + wm * 16;
                int a_col = kk * 16;
                wmma::load_matrix_sync(a_frag[wm], &As[a_row * BK_STRIDE + a_col], BK_STRIDE);
            }

            // Load B fragments: each from Bs[(kk*16) * BN_STRIDE + (warp_n * WARP_N + wn*16)]
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; wn++) {
                int b_row = kk * 16;
                int b_col = warp_n * WARP_N + wn * 16;
                wmma::load_matrix_sync(b_frag[wn], &Bs[b_row * BN_STRIDE + b_col], BN_STRIDE);
            }

            // MMA: acc[wm][wn] += a_frag[wm] * b_frag[wn]
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; wn++) {
                    wmma::mma_sync(acc[wm][wn], a_frag[wm], b_frag[wn], acc[wm][wn]);
                }
            }
        }

        __syncthreads();
    }

    // =========================================================================
    // Store results: use shared memory as FP32 staging buffer
    // Each warp stores its accumulator fragments, then all threads cooperatively
    // copy to global memory with bounds checking and bias addition.
    // =========================================================================

    // Reinterpret shared memory as FP32 for output staging
    // We need BLOCK_M * BLOCK_N floats = 64 * 32 * 4 = 8192 bytes
    // This fits within our shared memory (>= 27KB)
    float* out_smem = reinterpret_cast<float*>(smem_raw);
    // out_smem layout: [BLOCK_M][BLOCK_N], row-major

    // Each warp stores its accumulator fragments to shared memory
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; wn++) {
            int out_row = warp_m * WARP_M + wm * 16;
            int out_col = warp_n * WARP_N + wn * 16;
            // Store to out_smem[out_row][out_col] with stride BLOCK_N
            wmma::store_matrix_sync(&out_smem[out_row * BLOCK_N + out_col], acc[wm][wn], BLOCK_N, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // Cooperatively copy from shared memory to global memory
    constexpr int OUT_ELEMS = BLOCK_M * BLOCK_N;
    #pragma unroll 4
    for (int i = tid; i < OUT_ELEMS; i += NUM_THREADS) {
        int local_m = i / BLOCK_N;
        int local_n = i % BLOCK_N;
        int m_idx = m_start + local_m;
        int n_idx = n_start + local_n;

        if (m_idx < M && n_idx < Cout) {
            float result = out_smem[i];
            if constexpr (HAS_BIAS) {
                result += bias[n_idx];
            }
            y[m_idx * Cout + n_idx] = result;
        }
    }
}

// =============================================================================
// Python Binding
// =============================================================================

torch::Tensor conv3d_implicit_gemm_cuda(
    torch::Tensor x_pad,
    torch::Tensor w_flat,
    torch::Tensor bias,
    torch::Tensor act_amax,
    int N_batch, int Cin, int Dp, int Hp, int Wp,
    int Cout, int OD, int OH, int OW,
    int kD, int kH, int kW,
    int sd, int sh, int sw,
    int dd, int dh, int dw,
    int M, int K,
    bool quant_act, bool has_bias,
    int fp4_block_size
) {
    auto y = torch::zeros({M, Cout}, x_pad.options());

    // Helper to compute padded 1D grid size for L2 swizzle
    constexpr int GS = 8;  // L2_SWIZZLE_GROUP
    auto compute_grid = [&](int BM, int BN) -> dim3 {
        int grid_m = (M + BM - 1) / BM;
        int grid_n = (Cout + BN - 1) / BN;
        int num_m_groups = (grid_m + GS - 1) / GS;
        int total_blocks = num_m_groups * GS * grid_n;
        return dim3(total_blocks, 1);
    };

    // Macro to dispatch kernel with all 4 template specializations
    #define LAUNCH_WMMA_KERNEL(BM, BN, BK, WM, WN) \
    { \
        constexpr int BK_S = BK + 8; \
        constexpr int BN_S = BN + 8; \
        constexpr size_t smem_a = BM * BK_S * sizeof(__nv_bfloat16); \
        constexpr size_t smem_b = BK * BN_S * sizeof(__nv_bfloat16); \
        constexpr size_t smem = smem_a + smem_b; \
        \
        dim3 block(WM * WN * 32); \
        dim3 grid = compute_grid(BM, BN); \
        \
        auto set_smem = [](auto kernel) { \
            constexpr size_t s_a = BM * (BK + 8) * sizeof(__nv_bfloat16); \
            constexpr size_t s_b = BK * (BN + 8) * sizeof(__nv_bfloat16); \
            constexpr size_t s = s_a + s_b; \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, s); \
        }; \
        \
        if (quant_act && has_bias) { \
            auto kern = conv3d_implicit_gemm_wmma<true, true, BM, BN, BK, WM, WN>; \
            set_smem(kern); \
            kern<<<grid, block, smem>>>( \
                x_pad.data_ptr<float>(), w_flat.data_ptr<float>(), \
                bias.data_ptr<float>(), y.data_ptr<float>(), \
                act_amax.data_ptr<float>(), \
                Cin, Dp, Hp, Wp, Cout, OD, OH, OW, kD, kH, kW, \
                sd, sh, sw, dd, dh, dw, M, K); \
        } else if (quant_act) { \
            auto kern = conv3d_implicit_gemm_wmma<true, false, BM, BN, BK, WM, WN>; \
            set_smem(kern); \
            kern<<<grid, block, smem>>>( \
                x_pad.data_ptr<float>(), w_flat.data_ptr<float>(), \
                bias.data_ptr<float>(), y.data_ptr<float>(), \
                act_amax.data_ptr<float>(), \
                Cin, Dp, Hp, Wp, Cout, OD, OH, OW, kD, kH, kW, \
                sd, sh, sw, dd, dh, dw, M, K); \
        } else if (has_bias) { \
            auto kern = conv3d_implicit_gemm_wmma<false, true, BM, BN, BK, WM, WN>; \
            set_smem(kern); \
            kern<<<grid, block, smem>>>( \
                x_pad.data_ptr<float>(), w_flat.data_ptr<float>(), \
                bias.data_ptr<float>(), y.data_ptr<float>(), \
                act_amax.data_ptr<float>(), \
                Cin, Dp, Hp, Wp, Cout, OD, OH, OW, kD, kH, kW, \
                sd, sh, sw, dd, dh, dw, M, K); \
        } else { \
            auto kern = conv3d_implicit_gemm_wmma<false, false, BM, BN, BK, WM, WN>; \
            set_smem(kern); \
            kern<<<grid, block, smem>>>( \
                x_pad.data_ptr<float>(), w_flat.data_ptr<float>(), \
                bias.data_ptr<float>(), y.data_ptr<float>(), \
                act_amax.data_ptr<float>(), \
                Cin, Dp, Hp, Wp, Cout, OD, OH, OW, kD, kH, kW, \
                sd, sh, sw, dd, dh, dw, M, K); \
        } \
    }

    if (fp4_block_size == 128) {
        // BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, WARPS_M=2, WARPS_N=4
        // 8 warps = 256 threads -> faster cooperative loading
        // WARP_M=32, WARP_N=16, WARP_TILES_M=2, WARP_TILES_N=1 -> 2 mma per warp per K-step
        // Shared: 64*(128+8)*2 + 128*(64+8)*2 = 17,408 + 18,432 = 35,840 bytes (~35KB)
        LAUNCH_WMMA_KERNEL(64, 64, 128, 2, 4)
    } else {
        // BLOCK_M=64, BLOCK_N=64, BLOCK_K=256, WARPS_M=2, WARPS_N=4
        // 8 warps = 256 threads -> faster cooperative loading
        // Shared: 64*(256+8)*2 + 256*(64+8)*2 = 33,792 + 36,864 = 70,656 bytes (~69KB)
        LAUNCH_WMMA_KERNEL(64, 64, 256, 2, 4)
    }

    #undef LAUNCH_WMMA_KERNEL

    return y;
}
"""

# Compile the CUDA kernel
_cuda_module = None


def _get_cuda_module():
    """Get or compile the CUDA module."""
    global _cuda_module
    if _cuda_module is None:
        from torch.utils.cpp_extension import load_inline

        _cuda_module = load_inline(
            name="conv3d_implicit_gemm_cuda_v19_wmma",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_KERNEL_SOURCE,
            functions=["conv3d_implicit_gemm_cuda"],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "--ptxas-options=-v",
                "-std=c++17",
            ],
        )
    return _cuda_module


def _triple(v) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    assert len(v) == 3
    return (int(v[0]), int(v[1]), int(v[2]))


def _pad6(padding) -> tuple[int, int, int, int, int, int]:
    if isinstance(padding, int):
        p = int(padding)
        return (p, p, p, p, p, p)
    if len(padding) == 3:
        pd, ph, pw = map(int, padding)
        return (pw, pw, ph, ph, pd, pd)
    assert len(padding) == 6
    return tuple(map(int, padding))  # type: ignore[return-value]


@torch.no_grad()
def conv3d_implicit_gemm_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    act_amax: torch.Tensor | None = None,
    quant_act: bool = False,
    fp4_block_size: int = 256,
) -> torch.Tensor:
    """Optimized CUDA-based Conv3D via implicit GEMM with BF16 WMMA tensor cores.

    Args:
        x: Input tensor [N, Cin, D, H, W]
        w: Weight tensor [Cout, Cin, kD, kH, kW]
        bias: Optional bias tensor [Cout]
        stride: Convolution stride (D, H, W)
        padding: Convolution padding (D, H, W)
        dilation: Convolution dilation (D, H, W)
        act_amax: Activation max value for FP4 quantization
        quant_act: Whether to apply FP4 quantization to activations
        fp4_block_size: FP4 quantization block size (128 or 256)

    Returns:
        Output tensor [N, Cout, OD, OH, OW]
    """
    cuda_mod = _get_cuda_module()

    assert x.ndim == 5 and w.ndim == 5
    n_batch, cin, d, h, w_in = x.shape
    cout, cin_w, kd, kh, kw = w.shape
    assert cin_w == cin

    sd, sh, sw = _triple(stride)
    dd, dh, dw = _triple(dilation)
    pad_wl, pad_wr, pad_hl, pad_hr, pad_dl, pad_dr = _pad6(padding)

    x_pad = F.pad(x, (pad_wl, pad_wr, pad_hl, pad_hr, pad_dl, pad_dr))
    dp = d + pad_dl + pad_dr
    hp = h + pad_hl + pad_hr
    wp = w_in + pad_wl + pad_wr

    od = (dp - (dd * (kd - 1) + 1)) // sd + 1
    oh = (hp - (dh * (kh - 1) + 1)) // sh + 1
    ow = (wp - (dw * (kw - 1) + 1)) // sw + 1

    m = n_batch * od * oh * ow
    k = cin * kd * kh * kw

    w_flat = w.reshape(cout, k).transpose(0, 1).contiguous()

    x_pad = x_pad.float().contiguous()
    w_flat = w_flat.float().contiguous()

    has_bias = bias is not None
    bias_t = bias.float().contiguous() if has_bias else torch.empty(0, device=x.device)  # type: ignore[union-attr]

    do_quant = quant_act and act_amax is not None
    amax_t = act_amax.float().contiguous() if do_quant else torch.empty(0, device=x.device)  # type: ignore[union-attr]

    y_flat = cuda_mod.conv3d_implicit_gemm_cuda(
        x_pad,
        w_flat,
        bias_t,
        amax_t,
        n_batch,
        cin,
        dp,
        hp,
        wp,
        cout,
        od,
        oh,
        ow,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        dd,
        dh,
        dw,
        m,
        k,
        do_quant,
        has_bias,
        fp4_block_size,
    )

    y = y_flat.view(n_batch, od, oh, ow, cout).permute(0, 4, 1, 2, 3).contiguous()
    return y.to(x.dtype)
