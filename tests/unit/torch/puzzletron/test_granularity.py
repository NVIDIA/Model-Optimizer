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

import pytest

from modelopt.torch.puzzletron.granularity import resolve_granularity


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("depth", "subblock"),
        ("vllm_stats", "block"),
        ("scoring", "block"),
        ("bypass", "block"),
    ],
)
def test_stage_granularity_defaults(stage, expected):
    assert resolve_granularity(stage, {}) == expected


def test_explicit_granularity_overrides_default():
    assert resolve_granularity("scoring", {"granularity": "subblock"}) == "subblock"


def test_invalid_granularity_is_rejected_at_config_boundary():
    with pytest.raises(ValueError, match="block.*subblock"):
        resolve_granularity("depth", {"granularity": "layer"})


def test_unknown_stage_requires_an_explicit_granularity():
    with pytest.raises(ValueError, match="no granularity default"):
        resolve_granularity("unknown", {})
