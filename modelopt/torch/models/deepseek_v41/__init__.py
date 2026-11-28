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


"""DeepSeek V4.1 metadata and lazily imported differentiable text components."""

from importlib import import_module

from . import specs

_EXPORTS = {
    "DeepseekV41Config": "configuration_hf",
    "DeepseekV41DecoderLayer": "modeling_hf",
    "DeepseekV41ForCausalLM": "modeling_hf",
    "DeepseekV41Model": "modeling_hf",
    "DeepseekV41PreTrainedModel": "modeling_hf",
    "DeepseekV41RecomputeCache": "modeling_hf",
    "DeepseekV41MoE": "moe",
    "DeepseekV41MoEConfig": "moe",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
