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

"""Tests for bypass checkpoint metadata consumed by replacement-library extraction."""

import json
from pathlib import Path

import pandas as pd
import pytest

from modelopt.torch.puzzletron.block_config import FFNConfig
from modelopt.torch.puzzletron.replacement_library import build_replacement_library as brl


@pytest.mark.parametrize("metadata_file", ["bypass_config.json", "args.json"])
def test_infer_subblocks_to_extract_accepts_bypass_keys(tmp_path: Path, metadata_file: str):
    for i, (keys_to_learn, expected_subblocks) in enumerate(
        [
            ("entire_block", ["block"]),
            ("subblock_ffn", ["ffn"]),
            ("subblock_attention", ["attention"]),
            ("subblock_mamba", ["attention"]),
            (["subblock_attention", "subblock_ffn"], ["attention", "ffn"]),
        ]
    ):
        checkpoint_dir = tmp_path / f"checkpoint_{i}"
        checkpoint_dir.mkdir()
        metadata = (
            {"keys_to_learn": keys_to_learn}
            if metadata_file == "bypass_config.json"
            else {"model_factory": {"keys_to_learn": keys_to_learn}}
        )
        (checkpoint_dir / metadata_file).write_text(json.dumps(metadata))

        assert brl._infer_subblocks_to_extract(checkpoint_dir, []) == expected_subblocks


@pytest.mark.parametrize("metadata_file", ["bypass_config.json", "args.json"])
def test_infer_subblocks_to_extract_rejects_legacy_keys(tmp_path: Path, metadata_file: str):
    for i, keys_to_learn in enumerate(["mlp", "attn", ["mlp", "attn"]]):
        checkpoint_dir = tmp_path / f"legacy_checkpoint_{i}"
        checkpoint_dir.mkdir()
        metadata = (
            {"keys_to_learn": keys_to_learn}
            if metadata_file == "bypass_config.json"
            else {"model_factory": {"keys_to_learn": keys_to_learn}}
        )
        (checkpoint_dir / metadata_file).write_text(json.dumps(metadata))

        with pytest.raises(ValueError, match="keys_to_learn"):
            brl._infer_subblocks_to_extract(checkpoint_dir, [])


def test_get_last_checkpoint_from_each_experiment_resolves_ckpts_symlinks(
    tmp_path: Path, monkeypatch
):
    puzzle_dir = tmp_path / "puzzle_dir"
    ckpts_dir = puzzle_dir / "ckpts"
    ckpts_dir.mkdir(parents=True)

    teacher_dir = ckpts_dir / "teacher"
    bypass_dir = puzzle_dir / "bypass" / "bypass_runs" / "bypass_ffn" / "step-000010-ckpt"
    pruned_dir = puzzle_dir / "pruning" / "pruned_ffn"
    for checkpoint_dir in (teacher_dir, bypass_dir, pruned_dir):
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}")

    (ckpts_dir / "bypass_ffn").symlink_to(bypass_dir, target_is_directory=True)
    (ckpts_dir / "pruned_ffn").symlink_to(pruned_dir, target_is_directory=True)
    monkeypatch.setattr(brl, "is_valid_decilm_checkpoint", lambda *args, **kwargs: True)

    discovered = brl._get_last_checkpoint_from_each_experiment(puzzle_dir)

    assert discovered == {teacher_dir.resolve(), bypass_dir.resolve(), pruned_dir.resolve()}


def test_build_subblocks_df_prefers_bypass_rows_over_pruned_duplicates(tmp_path: Path, monkeypatch):
    puzzle_dir = tmp_path / "puzzle_dir"
    teacher_dir = puzzle_dir / "ckpts" / "teacher"
    bypass_dir = puzzle_dir / "bypass" / "bypass_runs" / "bypass_ffn" / "step-000010-ckpt"
    pruned_dir = puzzle_dir / "pruning" / "pruned_ffn"

    monkeypatch.setattr(
        brl,
        "_get_last_checkpoint_from_each_experiment",
        lambda *args, **kwargs: {teacher_dir, pruned_dir, bypass_dir},
    )

    def _construct_rows(checkpoint_dir: Path, *args, **kwargs):
        if checkpoint_dir == teacher_dir:
            return []
        return [
            {
                "attention_checkpoint_dir": None,
                "ffn_checkpoint_dir": str(checkpoint_dir),
                "block_config": None,
                "attention_config": None,
                "ffn_config": FFNConfig(intermediate_size=256),
                "block_idx": 0,
                "block_repr": None,
                "attention_repr": None,
                "ffn_repr": None,
            }
        ]

    monkeypatch.setattr(brl, "_construct_subblock_rows_from_current_checkpoint", _construct_rows)

    subblocks_df = brl._build_subblocks_df(
        master_puzzle_dir=puzzle_dir,
        teacher_checkpoint_dir=teacher_dir,
        add_ffn_no_ops=False,
        add_attention_no_ops=False,
    )

    assert len(subblocks_df) == 1
    assert subblocks_df["ffn_checkpoint_dir"].item() == str(bypass_dir)
    assert not pd.isna(subblocks_df["ffn_repr"].item())
