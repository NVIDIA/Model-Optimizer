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
"""Unit tests for ``examples/llm_ptq/example_utils.py``."""

from types import SimpleNamespace

import pytest
from _test_utils.examples.llm_ptq_example_utils import example_utils


@pytest.mark.parametrize(
    ("num_nextn", "num_hidden", "expected"),
    [
        (0, 80, []),
        (1, 78, ["model.layers.78"]),
        (3, 80, ["model.layers.80", "model.layers.81", "model.layers.82"]),
    ],
)
def test_get_inlined_mtp_prefixes_returns_expected_prefixes(num_nextn, num_hidden, expected):
    """Pure config -> prefix list. Documents the inlined-MTP detection contract."""
    cfg = SimpleNamespace(num_nextn_predict_layers=num_nextn, num_hidden_layers=num_hidden)
    assert example_utils.get_inlined_mtp_prefixes(cfg) == expected


def test_get_inlined_mtp_prefixes_missing_field_returns_empty():
    """Configs without num_nextn_predict_layers (non-MTP architectures) yield []."""
    cfg = SimpleNamespace(num_hidden_layers=32)
    assert example_utils.get_inlined_mtp_prefixes(cfg) == []
