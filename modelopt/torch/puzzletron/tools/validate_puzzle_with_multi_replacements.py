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

"""Validates puzzle solutions by applying layer replacements and evaluating model performance.

TODO: Consider moving this a separate module dedicated for scoring
"""

# mypy: ignore-errors

import gc
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist

from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..replacement_library.library import ReplacementLibrary
from ..replacement_library.replacement_utils import parse_layer_replacement
from ..utils.parsing import get_nested_key
from .common import resolve_torch_dtype

__all__ = ["validate_puzzle_solutions", "load_puzzle_solutions"]

"""
Usage Example:
==============

Validate single_block_replacement_solutions by calling validate_puzzle_solutions() directly
with an args object containing the required attributes. See the function docstring for details.

"""


@torch.no_grad()
def validate_puzzle_solutions(args: DictConfig, hydra_cfg: DictConfig | None = None) -> None:
    """Validate puzzle solutions by applying layer replacements and evaluating model performance.

    Args:
        args: Configuration object containing the following attributes:

            Puzzle Configuration (Required):
            - replacement_library_path (Path): Path to the replacement library JSON file.
            - solutions_path (Path): Path to puzzle solutions JSON file or directory containing solution files.
            - solutions_to_validate (list[int], optional): Indices of specific solutions to validate. Validates all solutions if None.
            - sort_solutions_by (str, optional): JSON field path to sort solutions by before validation.
            - bigger_is_better (bool): If True, sort solutions in descending order. Used with sort_solutions_by.
            - skip_validation (bool): If True, skip model validation and only save models if requested.
            - save_models (bool): If True, save realized model checkpoints for each solution.

            Teacher/Tokenizer Configuration:
            - teacher_dir (Path, optional): Path to teacher model directory. Auto-inferred if not provided.
            - tokenizer_name (str, optional): Tokenizer name/path. Uses teacher_dir if not specified.

            Model Configuration (Required if skip_validation=False):
            - model_dtype (str or torch.dtype): Model data type (e.g., "torch.bfloat16", torch.float16).
            - autocast_dtype (str or torch.dtype): Autocast data type for mixed precision.

            Dataset Configuration (Required if skip_validation=False):
            - dataset_path (str): Path to the validation dataset.
            - data_column (str): Column name in dataset containing text data.
            - block_size (int): Maximum sequence length for tokenization.
            - eval_samples (int, optional): Number of samples to evaluate.
            - val_dataset_name (str): Name of validation dataset split.
            - source_datasets_to_discard (list[str], optional): List of source datasets to exclude.
            - load_dataset_fn (callable, optional): Custom function to load the dataset.

            Data Processing (Required if skip_validation=False):
            - micro_batch_size (int): Batch size for evaluation.
            - seed (int): Random seed for reproducibility.
            - shuffle_seed (int, optional): Seed for shuffling data.
            - varlen (bool): Enable variable-length sequences.
            - bos_rate (float): Rate of adding BOS token.
            - fim_rate (float): Fill-in-the-middle rate for code completion tasks.
            - fim_spm_rate (float): SPM-based fill-in-the-middle rate.

            Output Configuration:
            - output_dir (Path, optional): Directory to save validation results. Auto-generated from solutions_path if not provided.

            Execution Options (Optional if skip_validation=False):
            - calc_losses_on_cpu (bool): Calculate losses on CPU to avoid OOM.
            - write_results (bool): Write validation results to file.
            - activations_log_dir (str, optional): Directory to log activation scores.
            - activation_hooks_kwargs (str or dict, optional): Arguments for activation hooks.

    Returns:
        None. Saves validation results and optionally model checkpoints to disk.
    """
    descriptor = ModelDescriptorFactory.get(args.descriptor)

    puzzle_solutions = load_puzzle_solutions(
        args.solutions_path, args.sort_solutions_by, args.bigger_is_better
    )
    if args.solutions_to_validate is None:
        args.solutions_to_validate = list(range(len(puzzle_solutions)))
    puzzle_solutions = [puzzle_solutions[i] for i in args.solutions_to_validate]

    output_dir = (
        args.output_dir
        if getattr(args, "output_dir", None) is not None
        else args.solutions_path.with_name(f"{args.solutions_path.stem}--validation")
    )

    replacement_library = ReplacementLibrary(
        args.replacement_library_path,
        descriptor=descriptor,
        model_config_overrides={"use_cache": False},
    )

    realized_checkpoints: list[tuple[int, Path, dict]] = []
    for i_solution, puzzle_solution in tqdm(
        list(zip(args.solutions_to_validate, puzzle_solutions)), desc="Validating solutions"
    ):
        layer_replacements = _extract_layer_replacements_from_puzzle_solution(puzzle_solution)
        model_config = replacement_library.create_model_config(layer_replacements)
        checkpoint_dir = None

        if args.save_models or not args.skip_validation:
            checkpoint_dir = (
                args.solutions_path.with_name(f"{args.solutions_path.stem}--checkpoints")
                / f"solution_{i_solution}"
            )

            model_config.dtype = resolve_torch_dtype(getattr(args, "model_dtype", "torch.bfloat16"))
            if dist.is_master():
                from ..identity import stable_hash

                replacement_library.materialize_checkpoint(
                    layer_replacements,
                    checkpoint_dir,
                    model_config=model_config,
                    overwrite=bool(getattr(args, "overwrite", False)),
                    solution_identity=stable_hash(puzzle_solution, prefix="mip_solution"),
                )
            realized_checkpoints.append((i_solution, checkpoint_dir, puzzle_solution))

        dist.barrier()

    if not args.skip_validation:
        if hydra_cfg is None:
            raise ValueError(
                "AutoModel solution validation requires the full Hydra config; "
                "call validate_puzzle_solutions(args, hydra_cfg=cfg)"
            )
        from ..plugins.automodel.validation import validate_realized_checkpoints_automodel

        validate_realized_checkpoints_automodel(
            hydra_cfg,
            args,
            realized_checkpoints,
            output_dir,
        )


def can_realize_as_symlinks(layer_replacements: list[dict]) -> bool:
    for layer_replacement in layer_replacements:
        num_parent_layers = len(layer_replacement["parent_layer_indices"])
        num_child_layers = len(layer_replacement["child_block_configs"])
        if num_parent_layers != num_child_layers or num_parent_layers != 1:
            return False
    return True


def _load_tokenizer(args: DictConfig, trust_remote_code: bool = False) -> PreTrainedTokenizerBase:
    tokenizer = None
    if (tokenizer_name := getattr(args, "tokenizer_name", None)) is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
    elif args.teacher_dir is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.teacher_dir, trust_remote_code=trust_remote_code
            )
        except Exception:
            pass
    if tokenizer is None:
        warnings.warn("Couldn't find a tokenizer, trying to continue without one")
    return tokenizer


def _extract_layer_replacements_from_puzzle_solution(
    puzzle_solution: dict,
) -> list[dict]:
    puzzle_solution = puzzle_solution.get("puzzle_solution", puzzle_solution)
    layer_replacements = [
        parse_layer_replacement(rep) for rep in puzzle_solution["chosen_replacements"]
    ]
    return layer_replacements


def load_puzzle_solutions(
    solutions_path: Path,
    sort_solutions_by: Optional[str],
    bigger_is_better: bool,
) -> list[dict]:
    assert solutions_path.exists(), f"{solutions_path=} does not exist"

    if solutions_path.is_file():
        puzzle_solutions = json.loads(solutions_path.read_text())
        if isinstance(puzzle_solutions, dict):
            puzzle_solutions = [puzzle_solutions]
    else:
        puzzle_solutions = [
            json.loads(p.read_text()) for p in solutions_path.glob("*solution*.json")
        ]

    if len(puzzle_solutions) == 0:
        raise ValueError(f"No solutions under {solutions_path=}")

    if sort_solutions_by is not None:
        puzzle_solutions = sorted(
            puzzle_solutions, key=partial(get_nested_key, field=sort_solutions_by)
        )
        if bigger_is_better:
            puzzle_solutions = puzzle_solutions[::-1]
        vals = [get_nested_key(sol, sort_solutions_by) for sol in puzzle_solutions]
        print(f"sorted solutions by {sort_solutions_by}. {vals[:10]=} {vals[-10:]=}")

    return puzzle_solutions
