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

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.replacement_library.replacement_library import ReplacementLibrary
from modelopt.torch.puzzletron.replacement_library.replacement_utils import parse_layer_replacement
from modelopt.torch.puzzletron.tools import validate_model
from modelopt.torch.puzzletron.tools.checkpoint_utils import (
    SAFETENSORS_SUBBLOCKS_DIR_NAME,
    copy_tokenizer,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import (
    copy_deci_lm_hf_code,
    save_checkpoint,
    save_safetensors_index,
)
from modelopt.torch.puzzletron.tools.validation_utils import (
    validate_model_and_extract_hidden_states,
    validate_model_with_teacher_similarity_metrics,
)
from modelopt.torch.puzzletron.utils.parsing import get_nested_key
from modelopt.torch.puzzletron.utils.validate_runtime_pipeline import perform_pipeline_stitches

"""
Usage Example:
==============

Validate single_block_replacement_solutions by calling validate_puzzle_solutions() directly
with an args object containing the required attributes. See the function docstring for details.

"""


@torch.no_grad()
def validate_puzzle_solutions(args: DictConfig) -> None:
    """Validate puzzle solutions by applying layer replacements and evaluating model performance.

    Args:
        args: Configuration object containing the following attributes:

            Puzzle Configuration (Required) attributes:

            - ``replacement_library_path`` (Path): Path to the replacement library JSON file.
            - ``solutions_path`` (Path): Path to puzzle solutions JSON file or directory containing solution files.
            - ``solutions_to_validate`` (list[int], optional): Indices of specific solutions to validate.
              Validates all solutions if None.
            - ``sort_solutions_by`` (str, optional): JSON field path to sort solutions by before validation.
            - ``bigger_is_better`` (bool): If True, sort solutions in descending order. Used with sort_solutions_by.
            - ``skip_validation`` (bool): If True, skip model validation and only save models if requested.
            - ``save_models`` (bool): If True, save realized model checkpoints for each solution.

            Teacher/Tokenizer Configuration attributes:

            - ``teacher_dir`` (Path, optional): Path to teacher model directory. Auto-inferred if not provided.
            - ``tokenizer_name`` (str, optional): Tokenizer name/path. Uses teacher_dir if not specified.

            Model Configuration (Required if skip_validation=False) attributes:

            - ``model_dtype`` (str or torch.dtype): Model data type (e.g., "torch.bfloat16", torch.float16).
            - ``autocast_dtype`` (str or torch.dtype): Autocast data type for mixed precision.

            Dataset Configuration (Required if skip_validation=False) attributes:

            - ``dataset_path`` (str): Path to the validation dataset.
            - ``data_column`` (str): Column name in dataset containing text data.
            - ``block_size`` (int): Maximum sequence length for tokenization.
            - ``eval_samples`` (int, optional): Number of samples to evaluate.
            - ``val_dataset_name`` (str): Name of validation dataset split.
            - ``source_datasets_to_discard`` (list[str], optional): List of source datasets to exclude.
            - ``load_dataset_fn`` (callable, optional): Custom function to load the dataset.

            Data Processing (Required if skip_validation=False) attributes:

            - ``micro_batch_size`` (int): Batch size for evaluation.
            - ``seed`` (int): Random seed for reproducibility.
            - ``shuffle_seed`` (int, optional): Seed for shuffling data.
            - ``varlen`` (bool): Enable variable-length sequences.
            - ``bos_rate`` (float): Rate of adding BOS token.
            - ``fim_rate`` (float): Fill-in-the-middle rate for code completion tasks.
            - ``fim_spm_rate`` (float): SPM-based fill-in-the-middle rate.

            Output Configuration attributes:

            - ``output_dir`` (Path, optional): Directory to save validation results.
              Auto-generated from solutions_path if not provided.

            Execution Options (Optional if skip_validation=False) attributes:

            - ``calc_losses_on_cpu`` (bool): Calculate losses on CPU to avoid OOM.
            - ``write_results`` (bool): Write validation results to file.
            - ``activations_log_dir`` (str, optional): Directory to log activation scores.
            - ``activation_hooks_kwargs`` (str or dict, optional): Arguments for activation hooks.

    Returns:
        None. Saves validation results and optionally model checkpoints to disk.
    """
    puzzle_solutions = load_puzzle_solutions(
        args.solutions_path, args.sort_solutions_by, args.bigger_is_better
    )
    if args.solutions_to_validate is None:
        args.solutions_to_validate = list(range(len(puzzle_solutions)))
    puzzle_solutions = [puzzle_solutions[i] for i in args.solutions_to_validate]

    tokenizer = _load_tokenizer(args)
    if not args.skip_validation:
        val_dataloader = (
            validate_model.prepare_dataloader(args, tokenizer) if dist.is_master() else None
        )

    output_dir = (
        args.output_dir
        if getattr(args, "output_dir", None) is not None
        else args.solutions_path.with_name(f"{args.solutions_path.stem}--validation")
    )

    replacement_library = ReplacementLibrary(args.replacement_library_path)

    teacher_hidden_states = None
    if (args.teacher_dir is not None) and (not args.skip_validation):
        teacher_model = replacement_library.load_checkpoint(args.teacher_dir)
        teacher_model.cuda(dist.local_rank())
        stitched_model = perform_pipeline_stitches(teacher_model)
        teacher_hidden_states = validate_model_and_extract_hidden_states(
            args,
            stitched_model,
            tokenizer,
            output_dir,
            model_name="teacher",
            pipeline_parallel=True,
            val_dataloader=val_dataloader,
        )

    for i_solution, puzzle_solution in tqdm(
        list(zip(args.solutions_to_validate, puzzle_solutions)), desc="Validating solutions"
    ):
        layer_replacements = _extract_layer_replacements_from_puzzle_solution(puzzle_solution)
        # realizable_as_symlinks = can_realize_as_symlinks(layer_replacements)
        realizable_as_symlinks = False
        model_config = replacement_library.create_model_config(layer_replacements)
        if (args.save_models and not realizable_as_symlinks) or (not args.skip_validation):
            model = replacement_library.load_model(layer_replacements)
            model_config = model.config

        if args.save_models:
            checkpoint_dir = (
                args.solutions_path.with_name(f"{args.solutions_path.stem}--checkpoints")
                / f"solution_{i_solution}"
            )

            model_config.dtype = args.model_dtype
            model_config.architectures = ["DeciLMForCausalLM"]
            if realizable_as_symlinks:
                if dist.is_master():
                    save_checkpoint_as_symlinks(
                        layer_replacements, model_config, checkpoint_dir, replacement_library
                    )
            else:
                save_checkpoint(model, checkpoint_dir)

            copy_tokenizer(args.tokenizer_name, checkpoint_dir)
            copy_deci_lm_hf_code(checkpoint_dir)

        dist.barrier()

        if not args.skip_validation:
            model.cuda(dist.local_rank())
            stitched_model = perform_pipeline_stitches(model)
            validate_model_with_teacher_similarity_metrics(
                args,
                stitched_model,
                tokenizer,
                teacher_hidden_states,
                output_dir,
                model_name=f"solution_{i_solution}",
                extra_payload={"i_solution": i_solution, "puzzle_solution": puzzle_solution},
                pipeline_parallel=True,
                val_dataloader=val_dataloader,
            )

        dist.barrier()


def can_realize_as_symlinks(layer_replacements: list[dict]) -> bool:
    for layer_replacement in layer_replacements:
        num_parent_layers = len(layer_replacement["parent_layer_indices"])
        num_child_layers = len(layer_replacement["child_block_configs"])
        if num_parent_layers != num_child_layers or num_parent_layers != 1:
            return False
    return True


def force_create_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def save_checkpoint_as_symlinks(
    layer_replacements: list[dict],
    model_config: DeciLMConfig,
    checkpoint_dir: Path,
    replace_library: ReplacementLibrary,
) -> None:
    model_config.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "subblocks_safetensors").mkdir(parents=True, exist_ok=True)
    save_safetensors_index(model_config, checkpoint_dir)

    for layer_replacement in layer_replacements:
        for weight_path in layer_replacement["weight_paths"]:
            force_create_symlink(
                weight_path, checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME / weight_path.name
            )

    lm_head_path = replace_library.get_teacher_lm_head_path()
    force_create_symlink(
        lm_head_path, checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME / lm_head_path.name
    )

    embedding_path = replace_library.get_teacher_embedding_path()
    force_create_symlink(
        embedding_path, checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME / embedding_path.name
    )


def _load_tokenizer(args: DictConfig) -> PreTrainedTokenizerBase:
    tokenizer = None
    if (tokenizer_name := getattr(args, "tokenizer_name", None)) is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    elif args.teacher_dir is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, trust_remote_code=True)
        except:
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
    sort_solutions_by: str | None,
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
