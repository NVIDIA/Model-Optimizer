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
are inherited from the base model in HFDFlashModel.modify(). Only
DFlash-specific defaults are set here.
"""

default_dflash_config = {
    "num_hidden_layers": 5,
    "rms_norm_eps": 1e-06,
    "attention_bias": False,
    "attention_dropout": 0.0,
}
