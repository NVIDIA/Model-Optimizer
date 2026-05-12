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

"""Unit tests for ``_total_steps`` / ``_progress_step`` in ``puzzletron_nas_plugin``.

These two helpers are the single source of truth for the user-facing
``Puzzletron Progress N/T`` log lines emitted by ``convert_puzzletron_model``
and ``PuzzletronSearcher.run_search``. A regression that drops or reorders a
stage silently misnumbers every progress message; worse, an off-by-one would
hide which stage the pipeline crashed in.
"""

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.puzzletron_nas_plugin import (
    _STAGE_ORDER,
    _progress_step,
    _total_steps,
)


def _cfg_with_bypass():
    return OmegaConf.create({"bypass": {"experiment_dir": "/tmp/x"}})


def _cfg_without_bypass():
    return OmegaConf.create({"some_other_key": True})


def _cfg_with_null_bypass():
    return OmegaConf.create({"bypass": None})


def test_total_steps_with_bypass_is_nine():
    assert _total_steps(_cfg_with_bypass()) == 9


def test_total_steps_without_bypass_key_is_eight():
    assert _total_steps(_cfg_without_bypass()) == 8


def test_total_steps_with_null_bypass_is_eight():
    """``bypass: null`` (typical override-to-disable) must read as 'no bypass'."""
    assert _total_steps(_cfg_with_null_bypass()) == 8


def test_progress_step_walks_eight_stages_without_bypass():
    cfg = _cfg_without_bypass()
    expected_no_bypass = [s for s in _STAGE_ORDER if s != "bypass"]
    seen = []
    for stage in expected_no_bypass:
        step, total = _progress_step(cfg, stage)
        seen.append((stage, step, total))
    assert seen == [
        ("start", 1, 8),
        ("convert", 2, 8),
        ("score_activations", 3, 8),
        ("prune", 4, 8),
        ("build_library", 5, 8),
        ("score_blocks", 6, 8),
        ("mip", 7, 8),
        ("complete", 8, 8),
    ]


def test_progress_step_walks_nine_stages_with_bypass():
    cfg = _cfg_with_bypass()
    seen = [(stage, *_progress_step(cfg, stage)) for stage in _STAGE_ORDER]
    assert seen == [
        ("start", 1, 9),
        ("convert", 2, 9),
        ("score_activations", 3, 9),
        ("prune", 4, 9),
        ("bypass", 5, 9),
        ("build_library", 6, 9),
        ("score_blocks", 7, 9),
        ("mip", 8, 9),
        ("complete", 9, 9),
    ]


def test_progress_step_bypass_stage_unknown_when_absent():
    """Asking for the bypass stage when bypass isn't configured is a programming
    error — must raise, not silently return 0/8."""
    cfg = _cfg_without_bypass()
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        _progress_step(cfg, "bypass")


def test_progress_step_unknown_stage_raises():
    cfg = _cfg_with_bypass()
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        _progress_step(cfg, "definitely_not_a_real_stage")


def test_mip_step_shifts_when_bypass_added_or_removed():
    """Removing bypass must shift MIP from 8/9 to 7/8 — pinned by the docstring
    on _progress_step which calls this out explicitly."""
    assert _progress_step(_cfg_with_bypass(), "mip") == (8, 9)
    assert _progress_step(_cfg_without_bypass(), "mip") == (7, 8)
