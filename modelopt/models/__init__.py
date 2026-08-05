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

"""Per-model descriptors, one module per HF model type.

Module names mirror the ``transformers.models`` layout; trust-remote-code models
(``arctic``, ``deepseek``) use their config ``model_type``. Each model module
registers one global ``ModelSpec`` (see ``specs.py``); importing this package
registers them all. Consumers resolve a spec via the registry lookups
(``get_spec`` / ``match_moe_block``) and read its fields.
"""

from .registry import *
from .specs import *

# Importing the model modules registers every spec as a side effect.
from . import (  # isort: skip
    arctic,
    dbrx,
    deepseek,
    gemma,
    gemma4,
    gpt_oss,
    llama,
    mixtral,
    nemotron,
    nemotron_h,
    qwen2_moe,
    qwen3,
    qwen3_5_moe,
    qwen3_moe,
    qwen3_next,
)
