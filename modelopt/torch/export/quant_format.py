# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The quantization and KV-cache format names shared by every export backend.

Backend-specific names live with their backend: the TensorRT-LLM checkpoint layout
constants, for example, are in :mod:`modelopt.torch.export.trtllm.model_config`.
"""

QUANTIZATION_NONE = None
QUANTIZATION_FP8 = "fp8"
QUANTIZATION_INT8_SQ = "int8_sq"
QUANTIZATION_INT8_WO = "int8_wo"
QUANTIZATION_INT4_AWQ = "int4_awq"
QUANTIZATION_W4A8_AWQ = "w4a8_awq"
QUANTIZATION_NVFP4 = "nvfp4"
QUANTIZATION_NVFP4_SVDQUANT = "nvfp4_svdquant"
QUANTIZATION_W4A8_NVFP4_FP8 = "w4a8_nvfp4_fp8"
QUANTIZATION_MXFP4 = "mxfp4"
QUANTIZATION_MXFP8 = "mxfp8"
QUANTIZATION_W4A8_MXFP4_FP8 = "w4a8_mxfp4_fp8"
QUANTIZATION_W4A16_NVFP4 = "w4a16_nvfp4"
QUANTIZATION_NVFP4_AWQ = "nvfp4_awq"
QUANTIZATION_FP8_PB_REAL = "fp8_pb_real"
QUANTIZATION_FP8_PB_WO = "fp8_pb_wo"
QUANTIZATION_FP8_PC_PT = "fp8_pc_pt"
QUANTIZATION_IQ1_S = "iq1_s"
QUANTIZATION_IQ2_XS = "iq2_xs"

# Formats whose scales are purely per-module, so export never merges them across the q/k/v
# and gate/up groups that share an input. Every other format unifies input_amax (and, for
# NVFP4, weight_scale_2) across such a group, which only a whole-model forward can discover.
FUSION_FREE_FORMATS = frozenset(
    {
        QUANTIZATION_FP8,
        QUANTIZATION_IQ1_S,
        QUANTIZATION_IQ2_XS,
        QUANTIZATION_NONE,
        QUANTIZATION_FP8_PB_REAL,
    }
)

KV_CACHE_FP8 = "FP8"
KV_CACHE_INT8 = "INT8"
KV_CACHE_NVFP4 = "NVFP4"
KV_CACHE_NVFP4_AFFINE = "NVFP4_AFFINE"
