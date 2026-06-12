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

"""Plugins for pruning for Transformers Attention."""

# import nas plugin to check if it is enabled else raises an Exception
from modelopt.torch.nas.plugins.transformers import *  # noqa: F403

from ..fastnas import FastNASConfig
from ..gradnas import GradNASConfig


def _n_heads_config():
    return {"n_heads_ratio": None, "n_heads_divisor": 1}


FastNASConfig.register_default(
    {
        "hf.BertAttention": _n_heads_config(),
        "hf.GPTJAttention": _n_heads_config(),
    }
)

GradNASConfig.register_default(
    {
        "hf.BertAttention": _n_heads_config(),
        "hf.GPTJAttention": _n_heads_config(),
    }
)
