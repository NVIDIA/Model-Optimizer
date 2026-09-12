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

"""Unit tests for modelopt.torch.export.layer_utils — expert naming."""

import torch.nn as nn

from modelopt.torch.export.layer_utils import get_expert_linear_names

# ---------------------------------------------------------------------------
# get_expert_linear_names tests
# ---------------------------------------------------------------------------


class Gemma4TextDecoderLayer(nn.Module):
    pass


class MixtralSparseMoeBlock(nn.Module):
    pass


class NemotronHMOE(nn.Module):
    pass


def test_get_expert_linear_names_gemma4():
    assert get_expert_linear_names(Gemma4TextDecoderLayer(), "gemma4") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]


def test_get_expert_linear_names_mixtral():
    assert get_expert_linear_names(MixtralSparseMoeBlock(), "mixtral") == ["w1", "w2", "w3"]


def test_get_expert_linear_names_nemotron():
    assert get_expert_linear_names(NemotronHMOE(), "nemotron_h") == ["up_proj", "down_proj"]
