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

TODO: Consider moving this a separate module dedicated for scoring.
"""

# mypy: ignore-errors

import argparse
import json
import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch._compress.replacement_library.build_replacement_library import infer_teacher_dir
from modelopt.torch._compress.replacement_library.replacement_library import ReplacementLibrary
from modelopt.torch._compress.replacement_library.replacement_utils import parse_layer_replacement
from modelopt.torch._compress.tools import validate_model
from modelopt.torch._compress.tools.checkpoint_utils import (
    SAFETENSORS_SUBBLOCKS_DIR_NAME,
    copy_tokenizer,
)
from modelopt.torch._compress.tools.checkpoint_utils_hf import (
    save_checkpoint,
    save_safetensors_index,
)
from modelopt.torch._compress.tools.validation_utils import (
    validate_model_and_extract_hidden_states,
    validate_model_with_teacher_similarity_metrics,
)
from modelopt.torch._compress.utils.parsing import get_nested_key, parse_path
from modelopt.torch._compress.utils.validate_runtime_pipeline import perform_pipeline_stitches

"""
Usage:
======

Validate single_block_replacement_solutions
===========================================

(
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True";
PUZZLE_DIR=".../Llama-3_2-1B-Instruct/parallel_puzzle";

torchrun --nproc-per-node=8  \
  -m  modelopt.torch._compress.tools.validate_puzzle_with_multi_replacements  \
  --replacement_library_path  ${PUZZLE_DIR}/replacement_library.json  \
  --solutions_path  ${PUZZLE_DIR}/single_sequence_replacement_solutions.json  \
  --solutions_to_validate  0  \
  \
  --dataset_path .../v0.4/valid  \
  --data_column  conversation  --block_size 8192  --seed 42  --shuffle_seed 444  --bos_rate 0.5  \
  --eval_samples 32  --micro_batch_size 1  \
  \
  --save_models  \

)


"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement_library_path", type=parse_path, required=True)
    parser.add_argument("--solutions_path", type=parse_path, required=True)
    parser.add_argument("--teacher_dir", type=parse_path, default=None)
    parser.add_argument("--solutions_to_validate", type=int, nargs="+", default=None)
    parser.add_argument("--sort_solutions_by", type=str, default=None)
    parser.add_argument("--bigger_is_better", action="store_true")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    args, unknown_args = parser.parse_known_args()
    if not args.skip_validation:
        validation_args = validate_model.build_arg_parser().parse_args(unknown_args)
        args = argparse.Namespace(
            **{**validation_args.__dict__, **args.__dict__}
        )  # if arg names overlap, the latter one wins
    else:
        args.block_size = None

    args.teacher_dir = _try_infer_teacher_dir(args.replacement_library_path, args.teacher_dir)

    args.tokenizer_name = getattr(args, "tokenizer_name", None)
    if args.tokenizer_name is None:
        args.tokenizer_name = args.teacher_dir

    return args


@torch.no_grad()
def validate_puzzle_solutions(args: argparse.Namespace) -> None:
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
        teacher_model.to(torch.device(dist.local_rank()))
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

            model_config.dtype = "bfloat16"
            model_config.architectures = ["DeciLMForCausalLM"]
            if realizable_as_symlinks:
                if dist.is_master():
                    save_checkpoint_as_symlinks(
                        layer_replacements, model_config, checkpoint_dir, replacement_library
                    )
            else:
                save_checkpoint(model, checkpoint_dir)

            copy_tokenizer(args.tokenizer_name, checkpoint_dir)
            copy_hf_code(checkpoint_dir)

            dist.barrier()

        dist.barrier()

        if not args.skip_validation:
            model.to(torch.device(dist.local_rank()))
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


def copy_hf_code(checkpoint_dir: Path) -> None:
    code_dir = Path(__file__).parent / "deci_lm_hf_code"
    print(f"copying hf code from {code_dir} ")
    for file in code_dir.glob("*.py"):
        shutil.copy(file, checkpoint_dir / file.name)


def _try_infer_teacher_dir(
    replacement_library_path: str | Path,
    teacher_dir: str | Path | None,
) -> Path | None:
    if teacher_dir is not None:
        return teacher_dir

    try:
        teacher_dir = infer_teacher_dir(
            master_puzzle_dir=Path(replacement_library_path).parent, teacher_checkpoint_dir=None
        )
        return teacher_dir
    except:
        return None


def _load_tokenizer(args: argparse.Namespace) -> PreTrainedTokenizerBase:
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


if __name__ == "__main__":
    validate_puzzle_solutions(args=parse_args())
