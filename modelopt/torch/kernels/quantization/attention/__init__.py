# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Quantization-specific attention kernel pieces.

``p_qdq.py`` holds the softmax-P (``p_bmm_quantizer``) quant-dequant
``@triton.jit`` helpers invoked by the unified flash-attention kernel in
``common/attention/triton_fa.py`` under its ``P_QDQ`` constexpr guard.
Only NVFP4 needs a P-specific helper (tiling and block-amax policy on top of
``quantization/common/nvfp4_quant.py``); the FP8 mode uses
``quantization/common/fp8_quant.fp8_scalar_qdq`` directly.
"""
