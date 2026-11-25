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

from modelopt import env


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (" ", False),
        ("1", True),
        (" TRUE ", True),
        ("yes", True),
        ("On", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        ("off", False),
    ],
)
def test_yes_no_variable_reads_each_spelling_and_blank_as_its_default(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MODELOPT_SKIP_WEIGHT_QUANT_CHECK", raising=False)
    else:
        monkeypatch.setenv("MODELOPT_SKIP_WEIGHT_QUANT_CHECK", value)
    assert env.skip_weight_quant_check() is expected


def test_yes_no_variable_refuses_any_other_value(monkeypatch):
    """A value the switch does not know raises, rather than quietly reading as on or off."""
    monkeypatch.setenv("MODELOPT_NVFP4_TRITON_SWEEP", "enabled")
    with pytest.raises(ValueError, match="MODELOPT_NVFP4_TRITON_SWEEP='enabled'"):
        env.nvfp4_triton_sweep()
