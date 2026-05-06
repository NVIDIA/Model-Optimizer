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

"""Unit tests for replacement-library checkpoint discovery + bypass priority.

The ``build_replacement_library`` module is responsible for two correctness-critical
behaviors after a bypass run:

1. ``_get_last_checkpoint_from_each_experiment`` must surface every valid
   checkpoint under ``puzzle_dir/ckpts/``, including those that live there only
   as symlinks (which is exactly how bypass writes its results).
2. When a bypass-trained subblock and a Truncate-init subblock would produce
   the same architectural identifier, the bypass-trained one must be preferred
   by the downstream ``drop_duplicates(keep="first")``. This is enforced by a
   tuple-sort closure inside ``_build_subblocks_df`` that gives bypass paths
   priority 0 and everything else priority 1.

A regression in either path silently discards bypass-trained weights — exactly
the kind of bug that's invisible in normal CI runs.
"""

from pathlib import Path

import pytest

from modelopt.torch.puzzletron.replacement_library import build_replacement_library as brl


# ---------------------------------------------------------------------------
# Filesystem fixture: tiny puzzle_dir with three checkpoints
# ---------------------------------------------------------------------------


def _write_minimal_config(checkpoint_dir: Path) -> None:
    """Write a placeholder config.json so the discovery rglob finds the dir.

    The actual config contents don't matter — these tests monkeypatch
    ``is_valid_decilm_checkpoint`` so no real config parsing happens.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text("{}")


@pytest.fixture
def puzzle_dir_with_three_ckpts(tmp_path: Path, monkeypatch) -> Path:
    """Build a puzzle_dir tree mirroring a real post-bypass post-prune layout.

    Layout::

        puzzle_dir/
          ckpts/
            teacher/                       # real dir
              config.json
            bypass_ffn_256_heads_4 -> ../bypass/bypass_runs/.../iter-000010-ckpt
            pruned_intermediate_256 -> ../pruning/pruned_intermediate_256
          bypass/bypass_runs/bypass_ffn_256_heads_4/iter-000010-ckpt/
            config.json
          pruning/pruned_intermediate_256/
            config.json

    The two non-teacher entries under ``ckpts/`` are symlinks — that is how
    ``puzzletron_nas_plugin.realize_bypass_checkpoints`` and the pruning
    pipeline actually write them. ``_get_last_checkpoint_from_each_experiment``
    must `.resolve()` these to see the real path under ``bypass/bypass_runs/``
    or ``pruning/`` — that resolution is what the priority sort later keys on.
    """
    puzzle_dir = tmp_path / "puzzle_dir"
    ckpts = puzzle_dir / "ckpts"
    ckpts.mkdir(parents=True)

    # Teacher: real directory directly under ckpts.
    _write_minimal_config(ckpts / "teacher")

    # Bypass: real dir under bypass/bypass_runs/, symlinked from ckpts/.
    bypass_real = puzzle_dir / "bypass" / "bypass_runs" / "bypass_ffn_256_heads_4" / "iter-000010-ckpt"
    _write_minimal_config(bypass_real)
    (ckpts / "bypass_ffn_256_heads_4").symlink_to(bypass_real, target_is_directory=True)

    # Truncate-pruned: real dir under pruning/, symlinked from ckpts/.
    pruning_real = puzzle_dir / "pruning" / "pruned_intermediate_256"
    _write_minimal_config(pruning_real)
    (ckpts / "pruned_intermediate_256").symlink_to(pruning_real, target_is_directory=True)

    # Make every config.json look "valid" without parsing — load_model_config
    # would otherwise try to load these as real HF configs.
    monkeypatch.setattr(brl, "is_valid_decilm_checkpoint", lambda *a, **kw: True)

    return puzzle_dir


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_get_last_checkpoint_from_each_experiment_finds_all_three(
    puzzle_dir_with_three_ckpts: Path,
):
    discovered = brl._get_last_checkpoint_from_each_experiment(puzzle_dir_with_three_ckpts)
    discovered_names = {p.name for p in discovered}
    assert discovered_names == {"teacher", "iter-000010-ckpt", "pruned_intermediate_256"}


def test_get_last_checkpoint_from_each_experiment_resolves_symlinks(
    puzzle_dir_with_three_ckpts: Path,
):
    """The resolved paths must reflect the real filesystem location.

    This is what makes the bypass-priority sort work — the closure inside
    ``_build_subblocks_df`` checks ``"bypass" in p.parts and "bypass_runs"
    in p.parts``, which only succeeds on the resolved path.
    """
    discovered = brl._get_last_checkpoint_from_each_experiment(puzzle_dir_with_three_ckpts)
    bypass_path = next(p for p in discovered if p.name == "iter-000010-ckpt")
    assert "bypass" in bypass_path.parts
    assert "bypass_runs" in bypass_path.parts
    # And the pruning entry must NOT pick up "bypass" anywhere in its parts.
    pruning_path = next(p for p in discovered if p.name == "pruned_intermediate_256")
    assert "bypass" not in pruning_path.parts


def test_get_last_checkpoint_skips_invalid_checkpoints(
    puzzle_dir_with_three_ckpts: Path, monkeypatch
):
    """Only checkpoints that pass ``is_valid_decilm_checkpoint`` should appear.

    A regression where a malformed config.json silently slips through would
    later raise inside ``_construct_subblock_rows_from_current_checkpoint``
    with a much less helpful traceback.
    """

    def _only_teacher_is_valid(checkpoint_dir, trust_remote_code=False):
        return Path(checkpoint_dir).name == "teacher"

    monkeypatch.setattr(brl, "is_valid_decilm_checkpoint", _only_teacher_is_valid)
    discovered = brl._get_last_checkpoint_from_each_experiment(puzzle_dir_with_three_ckpts)
    assert {p.name for p in discovered} == {"teacher"}


# ---------------------------------------------------------------------------
# Bypass-priority sort
# ---------------------------------------------------------------------------


def _bypass_priority(p: Path) -> tuple[int, str]:
    """Re-implementation of the closure inside ``_build_subblocks_df``.

    Kept identical to ``modelopt/torch/puzzletron/replacement_library/
    build_replacement_library.py:222-225``. If that closure is changed,
    update this test mirror; this is intentional duplication so the unit
    test stays cheap (no need to build an end-to-end DataFrame just to
    verify a 3-line priority function).
    """
    is_bypass = "bypass" in p.parts and "bypass_runs" in p.parts
    return (0 if is_bypass else 1, str(p))


def test_bypass_priority_orders_bypass_before_pruning(puzzle_dir_with_three_ckpts: Path):
    """The same input set the real code receives must sort bypass first."""
    discovered = brl._get_last_checkpoint_from_each_experiment(puzzle_dir_with_three_ckpts)
    teacher = next(p for p in discovered if p.name == "teacher")
    non_teacher_sorted = sorted(discovered - {teacher}, key=_bypass_priority)

    # Bypass must come first; pruning must come second.
    assert non_teacher_sorted[0].name == "iter-000010-ckpt"
    assert non_teacher_sorted[1].name == "pruned_intermediate_256"


def test_bypass_priority_is_stable_for_two_bypass_checkpoints(tmp_path: Path):
    """Multiple bypass checkpoints must sort deterministically by string.

    Without this, ``set`` iteration order changes the picked-first checkpoint
    across Python invocations, defeating the whole point of the priority sort.
    """
    p1 = tmp_path / "puzzle/bypass/bypass_runs/bypass_a/iter-000010-ckpt"
    p2 = tmp_path / "puzzle/bypass/bypass_runs/bypass_b/iter-000020-ckpt"
    paths = {p2, p1}  # insert in non-sorted order
    out = sorted(paths, key=_bypass_priority)
    assert [p.name for p in out] == ["iter-000010-ckpt", "iter-000020-ckpt"]
    # Repeated runs hit the same order.
    assert sorted({p1, p2}, key=_bypass_priority) == out
