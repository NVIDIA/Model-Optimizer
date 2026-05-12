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

from types import SimpleNamespace

from modelopt.torch.puzzletron.pruning.pruning_utils import _lm_head_dim


def test_lm_head_dim_uses_explicit_nested_head_dim():
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(head_dim=96, hidden_size=3072, num_attention_heads=32)
    )
    assert _lm_head_dim(cfg) == 96


def test_lm_head_dim_falls_back_to_hidden_size_over_heads():
    cfg = SimpleNamespace(text_config=SimpleNamespace(hidden_size=3072, num_attention_heads=32))
    assert _lm_head_dim(cfg) == 96
