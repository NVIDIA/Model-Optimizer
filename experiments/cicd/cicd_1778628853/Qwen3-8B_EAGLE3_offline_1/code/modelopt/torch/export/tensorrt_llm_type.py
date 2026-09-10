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

"""Code from TRT-LLM that export optimized models to the TensorRT-LLM checkpoint."""

from enum import IntEnum


# These clasess are directly copied from TRT-LLM to relex TRT-LLM dependency for checkpoint export
class LayerNormType(IntEnum):
    """LayerNormType from tensorrt_llm.functional."""

    LayerNorm = 0
    RmsNorm = 1
    GroupNorm = 2


class LayerNormPositionType(IntEnum):
    """LayerNormPositionType from tensorrt_llm.functional."""

    pre_layernorm = 0
    post_layernorm = 1


class MLPType(IntEnum):
    """MLPType from tensorrt_llm.functional."""

    MLP = 0
    GatedMLP = 1
    FusedGatedMLP = 2
