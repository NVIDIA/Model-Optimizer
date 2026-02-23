#!/usr/bin/env python3
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

"""Megatron Bridge for Puzzletron Qwen3-based AnyModel heterogeneous checkpoints."""

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Qwen3ForCausalLM

from modelopt.torch.puzzletron.export.mbridge.base import HeterogeneousBridgeMixin


@MegatronModelBridge.register_bridge(source=Qwen3ForCausalLM, target=GPTModel)
class PuzzletronQwen3AnyModelBridge(HeterogeneousBridgeMixin, Qwen3Bridge):
    """
    Megatron Bridge for Puzzletron Qwen3-based AnyModel checkpoints.

    Extends Qwen3Bridge with support for heterogeneous layer architectures (block_configs).
    All Qwen3-specific settings are inherited from Qwen3Bridge.
    """

    # provider_bridge() is inherited from HeterogeneousBridgeMixin
    # It automatically reuses Qwen3Bridge.provider_bridge() and adds heterogeneous config
    # mapping_registry() is inherited from Qwen3Bridge
