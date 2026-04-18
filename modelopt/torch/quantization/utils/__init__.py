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

# ruff: noqa: F405
"""Quantization utilities."""

from .core_utils import *
from .layerwise_calib import LayerActivationCollector

__all__ = [
    "EXPORT_MODE",
    "convert_quantization_axis_to_reduce_axis",
    "export_torch_mode",
    "is_quantized",
    "is_quantized_column_parallel_linear",
    "is_quantized_linear",
    "is_quantized_row_parallel_linear",
    "reduce_amax",
    "reduce_sum",
    "replace_function",
    "update_quant_cfg_with_kv_cache_quant",
    "weight_attr_names",
]
