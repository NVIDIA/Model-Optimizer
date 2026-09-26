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

"""Default DFlash architecture config.

Model-specific settings (hidden_size, num_attention_heads, rope_*, etc.)
are inherited from the base model in HFDFlashModel.modify(). Static
defaults that don't depend on the base model are set here, similar to
``eagle/default_config.py``.
"""

default_dflash_config = {
    # DFlash-specific
    "num_hidden_layers": 5,
    # Architecture defaults (overridable by user config)
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "initializer_range": 0.02,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "_attn_implementation": "sdpa",
}
