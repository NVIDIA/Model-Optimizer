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

from modelopt.torch.puzzletron.tools.hydra_utils import _warmup_steps_resolver, warmup_steps


def test_warmup_steps_casts_inputs_before_computing():
    assert warmup_steps("100", "10", "2", "5", "0.5") == 1


def test_warmup_steps_preserves_legacy_defaults():
    assert warmup_steps("1000", "10", "2") == 2
    assert _warmup_steps_resolver("1000", "10", "2") == 2
    assert _warmup_steps_resolver("1000", "10", "2", "0.5") == 25
    assert _warmup_steps_resolver("1000", "10", "2", "5", "0.5") == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tokens": -1, "block": 1, "mbs": 1, "grad_accum": 1, "pct": 0.1}, "tokens"),
        ({"tokens": 1, "block": 0, "mbs": 1, "grad_accum": 1, "pct": 0.1}, "block"),
        ({"tokens": 1, "block": 1, "mbs": 0, "grad_accum": 1, "pct": 0.1}, "mbs"),
        ({"tokens": 1, "block": 1, "mbs": 1, "grad_accum": 0, "pct": 0.1}, "grad_accum"),
        ({"tokens": 1, "block": 1, "mbs": 1, "grad_accum": 1, "pct": 1.1}, "pct"),
    ],
)
def test_warmup_steps_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        warmup_steps(**kwargs)
