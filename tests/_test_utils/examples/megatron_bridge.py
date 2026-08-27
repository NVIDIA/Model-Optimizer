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
"""Shared helpers for Megatron-Bridge example tests."""


def qwen35_moe_bridge_supported() -> bool:
    """Whether MBridge supports Qwen3.5-MoE per-expert weight assembly, i.e. nemo:26.08+.

    Mount an updated MBridge to run these on 26.06.
    """
    try:
        from megatron.bridge.models.conversion import model_bridge

        return hasattr(model_bridge, "_fuse_per_expert_hf_weight")
    except Exception:
        return False
