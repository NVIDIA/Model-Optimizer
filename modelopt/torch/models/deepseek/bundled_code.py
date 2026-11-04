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

"""Loading support shared by the DeepSeek-V3-architecture types."""

__all__ = ["prefers_bundled_code"]


def prefers_bundled_code(hf_config, checkpoint_path, config_kwargs) -> bool:
    """Load these checkpoints through the modeling code they ship.

    A DeepSeek-V3 architecture is named ``DeepseekV3ForCausalLM`` under every model type that uses
    it, and transformers ships a class of that name. Kimi-K2 is the one it would be wrong for: the
    name alone picks transformers' class for a checkpoint that comes with its own.
    """
    return True
