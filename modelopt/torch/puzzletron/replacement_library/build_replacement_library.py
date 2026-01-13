# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""This module constructs the replacement library JSON files from a puzzle directory containing
multiple trained model checkpoints. It analyzes checkpoints to extract unique block and subblock
configurations, builds a library of available replacements, and generates solutions for layer
replacement in compressed models. The resulting replacement library can then be used by
ReplacementLibrary to efficiently load models with mixed teacher/student layers.
"""
# mypy: ignore-errors

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
)
from modelopt.torch.puzzletron.replacement_library.replacement_utils import (
    is_replacement_identical_to_teacher,
    replacement_is_teacher,
    sort_replacements,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import (
    SAFETENSORS_SUBBLOCKS_DIR_NAME,
    is_valid_decilm_checkpoint,
    load_model_config,
)
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.robust_json import json_dump
from modelopt.torch.puzzletron.utils.parsing import format_global_config
from modelopt.torch.puzzletron.utils.utils import block_config_to_str, subblock_config_to_str

UNIQUE_SUBBLOCK_IDENTIFIER = ["block_config", "attention_config", "ffn_config", "block_idx"]
CHECKPOINTS_DIR_NAME = "ckpts"


def build_replacement_library(
    master_puzzle_dir: Path | str,
    teacher_checkpoint_dir: Path | str | None = None,
    add_ffn_no_ops: bool = True,
    add_attention_no_ops: bool = True,
) -> None:
    """For normal puzzle runs, use default values.
    For advanced use cases, see the Usage section.
    """
    master_puzzle_dir = Path(master_puzzle_dir)
    (master_puzzle_dir / "ckpts").mkdir(exist_ok=True)
    teacher_checkpoint_dir = infer_teacher_dir(master_puzzle_dir, teacher_checkpoint_dir)
    subblocks_df = _build_subblocks_df(
        master_puzzle_dir,
        teacher_checkpoint_dir,
        add_ffn_no_ops,
        add_attention_no_ops,
    )
    block_library_df = _build_block_library_from_subblocks(subblocks_df)

    layer_replacements = _build_layer_replacements(
        block_library_df, master_puzzle_dir, teacher_checkpoint_dir
    )

    single_sequence_replacement_solutions = _build_single_sequence_replacement_solutions(
        layer_replacements, teacher_checkpoint_dir
    )

    json_dump(block_library_df.to_dict(orient="records"), master_puzzle_dir / "block_library.json")
    json_dump(subblocks_df.to_dict(orient="records"), master_puzzle_dir / "subblock_library.json")
    json_dump(layer_replacements, master_puzzle_dir / "replacement_library.json")
    json_dump(
        single_sequence_replacement_solutions,
        master_puzzle_dir / "single_sequence_replacement_solutions.json",
    )
    mprint("done")


def launch_build_replacement_library(cfg: DictConfig) -> None:
    """Launch the build replacement library function with Hydra configuration."""
    mprint(f"Building replacement library for puzzle directory: {cfg.puzzle_dir}")
    mprint(f"Teacher directory: {cfg.teacher_dir}")
    mprint(
        f"Build replacement library config: {format_global_config(cfg.build_replacement_library, title='Build replacement library')}"
    )

    build_replacement_library(
        master_puzzle_dir=cfg.puzzle_dir,
        teacher_checkpoint_dir=cfg.teacher_dir,
        add_ffn_no_ops=cfg.build_replacement_library.add_ffn_no_ops,
        add_attention_no_ops=cfg.build_replacement_library.add_attention_no_ops,
    )


def infer_teacher_dir(
    master_puzzle_dir: Path | str,
    teacher_checkpoint_dir: Path | str | None = None,
) -> Path:
    if teacher_checkpoint_dir is None:
        teacher_checkpoint_dir = Path(master_puzzle_dir) / CHECKPOINTS_DIR_NAME / "teacher"
        if not teacher_checkpoint_dir.exists():
            raise ValueError(
                "You must either provide the --teacher_checkpoint_dir argument, or create a link to the "
                "teacher dir under '{PUZZLE_DIR}/ckpts'."
            )
    teacher_checkpoint_dir = Path(teacher_checkpoint_dir).resolve().absolute()
    return teacher_checkpoint_dir


def _build_block_library_from_subblocks(subblocks_df: pd.DataFrame) -> pd.DataFrame:
    joint_blocks_df = subblocks_df.dropna(subset=["block_config"]).copy()
    constructed_blocks_df = _construct_blocks_from_subblocks(subblocks_df)

    is_constructed_block_has_joint_variant = pd.Series(
        map(tuple, constructed_blocks_df[["block_config", "block_idx"]].values)
    ).isin(pd.Series(map(tuple, joint_blocks_df[["block_config", "block_idx"]].values)))
    constructed_blocks_df = constructed_blocks_df[~is_constructed_block_has_joint_variant]

    block_library_df = pd.concat([joint_blocks_df, constructed_blocks_df])
    block_library_df["block_repr"] = block_library_df["block_config"].apply(block_config_to_str)

    dups = block_library_df.loc[
        block_library_df[["block_config", "block_idx"]].duplicated()
    ].sort_values(by=["block_config", "block_idx"])
    if len(dups) > 0:
        mprint(f"Found {len(dups)} duplicate blocks in the block library. Here are some examples:")
        dup_block_idx = dups["block_idx"].iloc[0]
        dups_with_same_block_idx = dups[dups["block_idx"] == dup_block_idx]
        for _, row in dups_with_same_block_idx.head(10).iterrows():
            mprint(row.to_dict())
        json_dump(block_library_df.to_dict(orient="records"), "ERROR_block_library.json")
        json_dump(subblocks_df.to_dict(orient="records"), "ERROR_subblock_library.json")
        raise ValueError(
            f"Found {len(dups)} duplicate blocks in the block library. See ERROR_block_library.json and ERROR_subblock_library.json for more details."
        )

    return block_library_df


def _construct_blocks_from_subblocks(subblocks_df: pd.DataFrame) -> pd.DataFrame:
    columns = subblocks_df.columns
    decomp_blocks_df = subblocks_df[subblocks_df["block_config"].isna()].drop(
        columns=columns[columns.str.contains("block_config|joint|block_repr")]
    )

    attention_df = decomp_blocks_df.dropna(subset="attention_config").drop(
        columns=columns[columns.str.contains("ffn")]
    )
    ffn_df = decomp_blocks_df.dropna(subset="ffn_config").drop(
        columns=columns[columns.str.contains("attention")]
    )
    constructed_blocks_df = pd.merge(attention_df, ffn_df, on="block_idx")

    constructed_blocks_df["block_config"] = constructed_blocks_df.apply(
        lambda row: BlockConfig(ffn=row["ffn_config"], attention=row["attention_config"]), axis=1
    )

    return constructed_blocks_df


def _build_subblocks_df(
    master_puzzle_dir: Path | str,
    teacher_checkpoint_dir: Path | str,
    add_ffn_no_ops: bool,
    add_attention_no_ops: bool,
) -> pd.DataFrame:
    teacher_checkpoint_dir = Path(teacher_checkpoint_dir)
    checkpoint_dirs = _get_last_checkpoint_from_each_experiment(master_puzzle_dir)
    checkpoint_dirs = [teacher_checkpoint_dir] + list(checkpoint_dirs - {teacher_checkpoint_dir})
    checkpoints_to_split = [teacher_checkpoint_dir]

    subblock_rows = []
    for checkpoint_dir in checkpoint_dirs:
        subblocks_to_extract = _infer_subblocks_to_extract(checkpoint_dir, checkpoints_to_split)
        if len(subblocks_to_extract) > 0:
            subblock_rows_from_current_checkpoint = (
                _construct_subblock_rows_from_current_checkpoint(
                    checkpoint_dir, subblocks_to_extract
                )
            )
            subblock_rows.extend(subblock_rows_from_current_checkpoint)

    subblocks_df = pd.DataFrame(subblock_rows)

    subblocks_df = _drop_duplicates_of_decomp_no_op(subblocks_df)
    assert subblocks_df.duplicated().sum() == 0

    if add_ffn_no_ops or add_attention_no_ops:
        subblocks_df = _add_no_op_subblock_rows(subblocks_df, add_ffn_no_ops, add_attention_no_ops)

    subblocks_df = _drop_duplicates_of_teacher(subblocks_df, teacher_checkpoint_dir)

    subblocks_that_have_multiple_sources = list(
        subblocks_df[subblocks_df.duplicated(UNIQUE_SUBBLOCK_IDENTIFIER, keep=False)].groupby(
            UNIQUE_SUBBLOCK_IDENTIFIER, dropna=False
        )
    )
    if len(subblocks_that_have_multiple_sources) > 0:
        mprint(
            f"Found {len(subblocks_that_have_multiple_sources)} subblock types with multiple sources. Dropping duplicates..."
        )
        for subblock_identifier, duplicates_df in subblocks_that_have_multiple_sources:
            mprint("\n================================")
            mprint(dict(zip(UNIQUE_SUBBLOCK_IDENTIFIER, subblock_identifier)))
            for _, row in duplicates_df.iterrows():
                mprint(row.to_dict())

        # Drop duplicates, keeping the first occurrence (which should be from teacher)
        mprint(f"Dropping duplicates. Original count: {len(subblocks_df)}")
        subblocks_df = subblocks_df.drop_duplicates(subset=UNIQUE_SUBBLOCK_IDENTIFIER, keep="first")
        mprint(f"After dropping duplicates: {len(subblocks_df)}")

    subblocks_df["ffn_repr"] = subblocks_df["ffn_config"].apply(subblock_config_to_str)
    subblocks_df["attention_repr"] = subblocks_df["attention_config"].apply(subblock_config_to_str)
    subblocks_df["block_repr"] = subblocks_df["block_config"].apply(block_config_to_str)

    return subblocks_df


def _drop_duplicates_of_teacher(
    subblocks_df: pd.DataFrame,
    teacher_checkpoint_dir: Path | str,
) -> pd.DataFrame:
    orig_subblocks_df = subblocks_df.copy()

    attention_is_teacher = subblocks_df["attention_checkpoint_dir"] == str(teacher_checkpoint_dir)
    ffn_is_teacher = subblocks_df["ffn_checkpoint_dir"] == str(teacher_checkpoint_dir)
    is_joint_teacher = attention_is_teacher & ffn_is_teacher

    is_decomp_attention = subblocks_df["ffn_config"].isna()
    is_decomp_ffn = subblocks_df["attention_config"].isna()
    is_joint_block = ~is_decomp_attention & ~is_decomp_ffn

    student_indices_that_have_teacher_dups = []

    for current_subset, is_teacher in [
        (is_decomp_attention, attention_is_teacher),
        (is_decomp_ffn, ffn_is_teacher),
        (is_joint_block, is_joint_teacher),
    ]:
        subblocks_df = orig_subblocks_df.copy().loc[current_subset]

        subblocks_df["is_student"] = ~is_teacher.loc[current_subset]

        def get_student_indices_that_have_teacher_dups(grouped_is_student: pd.Series) -> list:
            if grouped_is_student.all():
                return []
            return grouped_is_student.index[grouped_is_student].tolist()

        current_student_indices_that_have_teacher_dups = [
            dup_index
            for dup_list in subblocks_df.groupby(UNIQUE_SUBBLOCK_IDENTIFIER, dropna=False)[
                "is_student"
            ].apply(get_student_indices_that_have_teacher_dups)
            for dup_index in dup_list
        ]
        student_indices_that_have_teacher_dups.extend(
            current_student_indices_that_have_teacher_dups
        )

    dedup_subblocks_df = orig_subblocks_df.drop(index=student_indices_that_have_teacher_dups)
    return dedup_subblocks_df


def _drop_duplicates_of_decomp_no_op(subblocks_df: pd.DataFrame) -> pd.DataFrame:
    is_decomp = subblocks_df["block_config"].isna()
    is_ffn_no_op = subblocks_df["ffn_config"].apply(lambda conf: conf is not None and conf.no_op)
    is_attention_no_op = subblocks_df["attention_config"].apply(
        lambda conf: conf is not None and conf.no_op
    )
    is_duplicated = subblocks_df.duplicated(subset=UNIQUE_SUBBLOCK_IDENTIFIER, keep="first")
    is_dup_of_decomp_no_op = is_duplicated & is_decomp & (is_ffn_no_op | is_attention_no_op)
    subblocks_df = subblocks_df[~is_dup_of_decomp_no_op]
    return subblocks_df


def _construct_subblock_rows_from_current_checkpoint(
    checkpoint_dir: Path, subblocks_to_extract: list[str]
) -> list[dict[str, Any]]:
    subblock_rows_from_current_checkpoint = []
    model_config = load_model_config(checkpoint_dir)
    for block_idx, block_config in enumerate(model_config.block_configs):
        for subblock_to_extract in subblocks_to_extract:
            subblock_row = _init_empty_subblock_row(block_idx)

            if subblock_to_extract == "block":
                subblock_row["block_config"] = block_config
                subblock_row["attention_config"] = block_config.attention
                subblock_row["attention_checkpoint_dir"] = (
                    str(checkpoint_dir) if not block_config.attention.no_op else None
                )
                subblock_row["ffn_config"] = block_config.ffn
                subblock_row["ffn_checkpoint_dir"] = (
                    str(checkpoint_dir) if not block_config.ffn.no_op else None
                )
            elif subblock_to_extract == "ffn":
                subblock_row["ffn_config"] = block_config.ffn
                subblock_row["ffn_checkpoint_dir"] = (
                    str(checkpoint_dir) if not block_config.ffn.no_op else None
                )
            elif subblock_to_extract == "attention":
                subblock_row["attention_config"] = block_config.attention
                subblock_row["attention_checkpoint_dir"] = (
                    str(checkpoint_dir) if not block_config.attention.no_op else None
                )
            else:
                raise ValueError()

            subblock_rows_from_current_checkpoint.append(subblock_row)
    return subblock_rows_from_current_checkpoint


def _add_no_op_subblock_rows(
    subblocks_df: pd.DataFrame,
    add_ffn_no_op: bool,
    add_attention_no_op: bool,
) -> pd.DataFrame:
    n_layer = subblocks_df["block_idx"].max() + 1

    no_op_subblocks = []
    if add_ffn_no_op:
        no_op_subblocks.append("ffn")
    if add_attention_no_op:
        no_op_subblocks.append("attention")

    additional_no_op_rows = []
    for no_op_subblock in no_op_subblocks:
        rows_with_no_op_subblock, subblock_cls = _get_rows_with_no_op_subblock(
            subblocks_df, no_op_subblock
        )
        existing_no_op_indices = rows_with_no_op_subblock["block_idx"].values
        missing_no_op_indices = list(set(range(n_layer)) - set(existing_no_op_indices))
        for block_idx in missing_no_op_indices:
            no_op_subblock_row = {
                **_init_empty_subblock_row(block_idx),
                f"{no_op_subblock}_config": subblock_cls(no_op=True),
            }
            additional_no_op_rows.append(no_op_subblock_row)

    subblocks_df = pd.concat([subblocks_df, pd.DataFrame(additional_no_op_rows)])

    for no_op_subblock in no_op_subblocks:
        rows_with_no_op_subblock, _ = _get_rows_with_no_op_subblock(subblocks_df, no_op_subblock)
        assert len(rows_with_no_op_subblock) == n_layer, (
            f"Got {len(rows_with_no_op_subblock)} rows with {no_op_subblock}=no_op, but we have {n_layer} layers"
        )
    return subblocks_df


def _get_rows_with_no_op_subblock(
    subblocks_df: pd.DataFrame, no_op_subblock: str
) -> tuple[pd.DataFrame, type[AttentionConfig] | type[FFNConfig]]:
    other_subblock = "ffn" if no_op_subblock == "attention" else "attention"
    subblock_cls = AttentionConfig if no_op_subblock == "attention" else FFNConfig
    no_op_subblock_config = subblock_cls(no_op=True)
    rows_with_no_op_subblock = subblocks_df[
        (subblocks_df[f"{no_op_subblock}_config"] == no_op_subblock_config)
        & subblocks_df[f"{other_subblock}_config"].isna()
    ]
    return rows_with_no_op_subblock, subblock_cls


def _get_last_checkpoint_from_each_experiment(master_puzzle_dir: Path | str) -> set[Path]:
    master_puzzle_dir = Path(master_puzzle_dir)
    master_checkpoints_dir = master_puzzle_dir / CHECKPOINTS_DIR_NAME
    subdirs_of_master_checkpoints_dir = [
        p.resolve() for p in master_checkpoints_dir.iterdir() if p.is_dir()
    ]
    checkpoint_dirs = [
        p.parent
        for subdir in subdirs_of_master_checkpoints_dir
        for p in subdir.rglob("config.json")
    ]

    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir == master_checkpoints_dir:
            raise ValueError(
                f"We need at least 1 hierarchy level under the '{CHECKPOINTS_DIR_NAME}' dir. "
                "Name your checkpoints, preferably with meaningful names. "
                "If you are Ido Galil, tell Tomer that you got this exception ;) "
            )

    # Filter out non-DeciLM checkpoints (e.g., unconverted Llama checkpoints)
    valid_checkpoint_dirs = [cp for cp in checkpoint_dirs if is_valid_decilm_checkpoint(cp)]

    experiment_dirs = [
        p if (p in subdirs_of_master_checkpoints_dir) else p.parent for p in valid_checkpoint_dirs
    ]

    deduped_checkpoint_dirs = set(
        pd.DataFrame({"checkpoint_dir": valid_checkpoint_dirs, "experiment_dir": experiment_dirs})
        .sort_values("checkpoint_dir")
        .drop_duplicates(subset="experiment_dir", keep="last")["checkpoint_dir"]
        .tolist()
    )
    return deduped_checkpoint_dirs


def _infer_subblocks_to_extract(
    checkpoint_dir: Path,
    checkpoints_to_split: list[Path],
) -> list[str]:
    if (checkpoint_dir / "replacement_library.json").exists():
        return []
    bypass_config_path = checkpoint_dir / "bypass_config.json"
    if (checkpoint_dir in checkpoints_to_split) or (not bypass_config_path.exists()):
        subblocks_to_extract = ["block", "attention", "ffn"]
    else:
        bypass_config = json.loads(bypass_config_path.read_text())
        keys_to_learn = bypass_config.get("keys_to_learn", "entire_block")
        if keys_to_learn == "entire_block":
            subblocks_to_extract = ["block"]
        elif "mlp" in keys_to_learn and "attn" not in keys_to_learn:
            subblocks_to_extract = ["ffn"]
        elif "attn" in keys_to_learn and "mlp" not in keys_to_learn:
            subblocks_to_extract = ["attention"]
        else:
            raise ValueError(f"Unrecognized {keys_to_learn=}")
    return subblocks_to_extract


def _init_empty_subblock_row(block_idx: int) -> dict[str, Any]:
    return {
        "attention_checkpoint_dir": None,
        "ffn_checkpoint_dir": None,
        "block_config": None,
        "attention_config": None,
        "ffn_config": None,
        "block_idx": block_idx,
        "block_repr": None,
        "attention_repr": None,
        "ffn_repr": None,
    }


def _build_layer_replacements(
    block_library_df: pd.DataFrame,
    master_puzzle_dir: Path,
    teacher_checkpoint_dir: Path,
) -> list[dict]:
    layer_replacements_from_blocks = _build_layer_replacements_from_block_library(block_library_df)
    layer_replacements_from_checkpoints = _gather_layer_replacements_from_checkpoints(
        master_puzzle_dir
    )
    layer_replacements = layer_replacements_from_blocks + layer_replacements_from_checkpoints
    layer_replacements = _filter_duplicate_teacher_replacements(
        layer_replacements, teacher_checkpoint_dir
    )
    return layer_replacements


def _build_layer_replacements_from_block_library(block_library_df: pd.DataFrame) -> list[dict]:
    layer_replacements = []
    for _, row in block_library_df.iterrows():
        block_idx = row["block_idx"]
        block_config = row["block_config"]
        weight_paths = []
        for subblock_name in ["attention", "ffn"]:
            checkpoint_dir = row[f"{subblock_name}_checkpoint_dir"]
            if checkpoint_dir is not None:
                subblock_path = (
                    Path(checkpoint_dir)
                    / SAFETENSORS_SUBBLOCKS_DIR_NAME
                    / f"block_{block_idx}_{subblock_name}.safetensors"
                )
                weight_paths.append(subblock_path)
        weight_paths = sorted(set(weight_paths))
        layer_replacement = {
            "parent_layer_indices": [block_idx],
            "child_block_configs": [block_config],
            "weight_paths": weight_paths,
        }
        layer_replacements.append(layer_replacement)
    return layer_replacements


def _gather_layer_replacements_from_checkpoints(master_puzzle_dir: str | Path) -> list[dict]:
    gathered_layer_replacements = []
    checkpoint_dirs = _get_last_checkpoint_from_each_experiment(master_puzzle_dir)
    for checkpoint_dir in checkpoint_dirs:
        if (layer_replacements_path := checkpoint_dir / "replacement_library.json").exists():
            layer_replacements = json.loads(layer_replacements_path.read_text())
            for layer_replacement in layer_replacements:
                layer_replacement["child_block_configs"] = [
                    BlockConfig(**block_config_dict)
                    for block_config_dict in layer_replacement["child_block_configs"]
                ]
                layer_replacement["weight_paths"] = sorted(
                    set(Path(p) for p in layer_replacement["weight_paths"])
                )
            gathered_layer_replacements.extend(layer_replacements)
    return gathered_layer_replacements


def _filter_duplicate_teacher_replacements(
    layer_replacements: list[dict],
    teacher_checkpoint_dir: Path,
) -> list[dict]:
    teacher_model_config = load_model_config(teacher_checkpoint_dir)
    filtered_layer_replacements = []
    for layer_replacement in layer_replacements:
        if replacement_is_teacher(
            layer_replacement, teacher_model_config, teacher_checkpoint_dir
        ) or not is_replacement_identical_to_teacher(layer_replacement, teacher_model_config):
            filtered_layer_replacements.append(layer_replacement)
    return filtered_layer_replacements


def _build_single_sequence_replacement_solutions(
    layer_replacements: list[dict],
    teacher_checkpoint_dir: Path,
) -> list[dict]:
    teacher_model_config = load_model_config(teacher_checkpoint_dir)
    n_layer = teacher_model_config.num_hidden_layers

    teacher_replacements = dict()
    student_replacements = []
    for layer_replacement in layer_replacements:
        if replacement_is_teacher(layer_replacement, teacher_model_config, teacher_checkpoint_dir):
            block_idx = layer_replacement["parent_layer_indices"][0]
            teacher_replacements[block_idx] = layer_replacement
        else:
            student_replacements.append(layer_replacement)

    teacher_indices_represented_in_replacements = sorted(teacher_replacements.keys())
    assert teacher_indices_represented_in_replacements == list(range(n_layer)), (
        f"{n_layer=}, {teacher_indices_represented_in_replacements=}"
    )

    student_replacements = sort_replacements(student_replacements)

    solutions = []
    for layer_replacement in student_replacements:
        block_indices_not_represented_in_replacement = sorted(
            set(range(n_layer)) - set(layer_replacement["parent_layer_indices"])
        )
        chosen_replacements = sort_replacements(
            [layer_replacement]
            + [
                teacher_replacements[block_idx]
                for block_idx in block_indices_not_represented_in_replacement
            ]
        )

        block_configs = [
            block_config
            for replacement in chosen_replacements
            for block_config in replacement["child_block_configs"]
        ]

        solutions.append(
            {
                "single_sequence_replacement": layer_replacement,
                "chosen_replacements": chosen_replacements,
                "block_configs": block_configs,
            }
        )

    return solutions
