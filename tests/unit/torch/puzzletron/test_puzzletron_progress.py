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

"""Tests for progress numbering with the optional bypass stage."""

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.puzzletron_nas_plugin import _progress_step


def _cfg_with_bypass():
    return OmegaConf.create({"bypass": {"experiment_dir": "/tmp/x"}})


def _cfg_without_bypass():
    return OmegaConf.create({"some_other_key": True})


def _cfg_with_null_bypass():
    return OmegaConf.create({"bypass": None})


@pytest.mark.parametrize(
    ("cfg", "stage", "expected_step"),
    [
        (_cfg_without_bypass(), "mip", (7, 8)),
        (_cfg_with_null_bypass(), "mip", (7, 8)),
        (_cfg_with_bypass(), "bypass", (5, 9)),
        (_cfg_with_bypass(), "mip", (8, 9)),
    ],
)
def test_progress_step_accounts_for_optional_bypass(
    cfg, stage: str, expected_step: tuple[int, int]
):
    assert _progress_step(cfg, stage) == expected_step


@pytest.mark.parametrize(
    ("cfg", "stage"),
    [
        (_cfg_without_bypass(), "bypass"),
        (_cfg_with_bypass(), "definitely_not_a_real_stage"),
    ],
)
def test_progress_step_rejects_unreachable_stages(cfg, stage: str):
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        _progress_step(cfg, stage)
