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

"""Predefined LoRA configurations for common Megatron-Core model architectures.

These configs are designed to be passed directly to
:func:`modelopt.torch.peft.update_model` or used as the ``--lora-config``
argument in the PTQ script.

Config name strings map to entries in :data:`LORA_CFG_CHOICES`.
"""

import torch.nn.init as init

__all__ = ["DENSE_LORA_CFG", "LORA_CFG_CHOICES", "MOE_LORA_CFG"]

# ---------------------------------------------------------------------------
# Dense (non-MoE) model config
# ---------------------------------------------------------------------------
# Targets the four linear projections that are standard in every transformer
# decoder layer:
#   - linear_qkv  : fused Q/K/V projection  (ColumnParallelLinear)
#   - linear_proj : attention output projection (RowParallelLinear)
#   - linear_fc1  : MLP gate/up projection  (ColumnParallelLinear)
#   - linear_fc2  : MLP down projection     (RowParallelLinear)
#
# All other modules are excluded via the wildcard ``"*": {"enable": False}``
# fallback (later patterns override earlier ones).
DENSE_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_cfg": {
        "*": {"enable": False},
        "*linear_qkv*": {"rank": 64, "enable": True},
        "*linear_proj*": {"rank": 64, "enable": True},
        "*linear_fc1*": {"rank": 64, "enable": True},
        "*linear_fc2*": {"rank": 64, "enable": True},
    },
}

# ---------------------------------------------------------------------------
# MoE model config
# ---------------------------------------------------------------------------
# Apply LoRA adapter per-layer in each local_expert
MOE_LORA_CFG = {
    "adapter_type": "lora",
    "freeze_base_model": False,
    "freeze_base_layers": True,
    "adapter_cfg": {
        "*": {"enable": False},
        "*local_experts*linear_fc1*": {"rank": 64, "enable": True},
        "*local_experts*linear_fc2*": {"rank": 64, "enable": True},
    },
}
MOE_LORA_RANDOM_INIT_CFG = {
    "adapter_type": "lora",
    "freeze_base_model": False,
    "freeze_base_layers": True,
    "adapter_cfg": {
        "*": {"enable": False},
        "*local_experts*linear_fc1*": {
            "rank": 64,
            "enable": True,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
        },
        "*local_experts*linear_fc2*": {
            "rank": 64,
            "enable": True,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
        },
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
LORA_CFG_CHOICES: dict[str, dict] = {
    "dense": DENSE_LORA_CFG,
    "moe": MOE_LORA_CFG,
    "moe_random": MOE_LORA_RANDOM_INIT_CFG,
}
