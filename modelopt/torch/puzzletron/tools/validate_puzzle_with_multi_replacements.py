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

from ..anymodel.converter import Converter
from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..replacement_library.library import ReplacementLibrary
from ..replacement_library.replacement_utils import parse_layer_replacement
from ..utils.parsing import get_nested_key
from ..utils.validate_runtime_pipeline import perform_pipeline_stitches
from . import validate_model
from .checkpoint_utils import copy_tokenizer
from .checkpoint_utils_hf import save_checkpoint, save_checkpoint_from_shards
from .common import resolve_torch_dtype
from .sharded_checkpoint_utils import load_and_shard_model
from .validation_utils import (
    validate_model_and_extract_hidden_states,
    validate_model_with_teacher_similarity_metrics,
)

__all__ = ["validate_puzzle_solutions", "load_puzzle_solutions"]

"""
Usage Example:
==============

Validate single_block_replacement_solutions by calling validate_puzzle_solutions() directly
with an args object containing the required attributes. See the function docstring for details.

"""


@torch.no_grad()
def validate_puzzle_solutions(args: DictConfig) -> None:
    """
    Validate and (optionally) save realized models for a collection of puzzle solutions.
    
    Loads puzzle solutions and a replacement library, applies each solution's layer replacements to realize a model, optionally evaluates realized models (including optional teacher-based hidden-state similarity metrics), and optionally saves realized model checkpoints and tokenizer files to disk.
    
    Parameters:
        args (DictConfig): Configuration with fields used by this routine. Key fields:
            - replacement_library_path (Path): Path to the replacement library JSON.
            - solutions_path (Path): File or directory containing puzzle solution JSON(s).
            - solutions_to_validate (list[int], optional): Indices of solutions to process; all solutions if None.
            - skip_validation (bool): If True, skip model validation steps.
            - save_models (bool): If True, save realized model checkpoints and tokenizer files.
            - teacher_dir (Path, optional): Path to a teacher model for hidden-state comparisons.
            - tokenizer_name (str, optional): Tokenizer name or path; teacher_dir is used if unset.
            - output_dir (Path, optional): Directory to write validation outputs; auto-derived from solutions_path if unset.
            - model_dtype (str or torch.dtype, optional): Dtype to set on saved model configs.
            - (Other dataset/validation options may be read from args when validation is enabled.)
    
    Returns:
        None
    """
    descriptor = ModelDescriptorFactory.get(args.descriptor)

    puzzle_solutions = load_puzzle_solutions(
        args.solutions_path, args.sort_solutions_by, args.bigger_is_better
    )
    if args.solutions_to_validate is None:
        args.solutions_to_validate = list(range(len(puzzle_solutions)))
    puzzle_solutions = [puzzle_solutions[i] for i in args.solutions_to_validate]

    tokenizer = _load_tokenizer(args, trust_remote_code=descriptor.requires_trust_remote_code())
    if not args.skip_validation:
        val_dataloader = (
            validate_model.prepare_dataloader(args, tokenizer) if dist.is_master() else None
        )

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

    teacher_hidden_states = None
    if (args.teacher_dir is not None) and (not args.skip_validation):
        teacher_model = load_and_shard_model(
            checkpoint_path=args.teacher_dir, descriptor=descriptor
        )
        teacher_model.cuda(dist.local_rank())
        stitched_model = perform_pipeline_stitches(teacher_model, descriptor=descriptor)
        teacher_hidden_states = validate_model_and_extract_hidden_states(
            args,
            stitched_model,
            tokenizer,
            output_dir,
            model_name="teacher",
            val_dataloader=val_dataloader,
        )

        # Properly release CUDA memory after teacher validation
        teacher_model.cpu()
        stitched_model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()

    for i_solution, puzzle_solution in tqdm(
        list(zip(args.solutions_to_validate, puzzle_solutions)), desc="Validating solutions"
    ):
        layer_replacements = _extract_layer_replacements_from_puzzle_solution(puzzle_solution)
        realizable_as_symlinks = can_realize_as_symlinks(layer_replacements)
        # realizable_as_symlinks = False
        model_config = replacement_library.create_model_config(layer_replacements)
        if (args.save_models and not realizable_as_symlinks) or (not args.skip_validation):
            model = replacement_library.load_model(layer_replacements)
            model_config = model.config

        if args.save_models:
            checkpoint_dir = (
                args.solutions_path.with_name(f"{args.solutions_path.stem}--checkpoints")
                / f"solution_{i_solution}"
            )

            model_config.dtype = resolve_torch_dtype(getattr(args, "model_dtype", "torch.bfloat16"))
            Converter.copy_checkpoint_files(args.teacher_dir, checkpoint_dir)
            if realizable_as_symlinks:
                if dist.is_master():
                    # TODO: Loo into internal Puzzleron code to see how to save as symlinks
                    # save_checkpoint_as_symlinks is currently not supported
                    pass
            save_checkpoint_from_shards(model, checkpoint_dir, descriptor)

            copy_tokenizer(
                args.tokenizer_name,
                checkpoint_dir,
                trust_remote_code=descriptor.requires_trust_remote_code(),
            )

        dist.barrier()

        if not args.skip_validation:
            model.cuda(dist.local_rank())
            stitched_model = perform_pipeline_stitches(model, descriptor=descriptor)
            validate_model_with_teacher_similarity_metrics(
                args,
                stitched_model,
                tokenizer,
                teacher_hidden_states,
                output_dir,
                model_name=f"solution_{i_solution}",
                extra_payload={"i_solution": i_solution, "puzzle_solution": puzzle_solution},
                val_dataloader=val_dataloader,
            )

            # Properly release CUDA memory after solution validation
            model.cpu()
            stitched_model.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        dist.barrier()


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
