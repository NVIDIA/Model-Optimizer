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
import json
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import transformers
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.misc import set_seed
from _test_utils.torch.puzzletron.utils import PUZZLETRON_FAMILIES, setup_test_model_and_data
from packaging.version import Version

import modelopt.torch.puzzletron as mtpz
import modelopt.torch.utils.distributed as dist

# The e2e test to compress a model based on Local Neural Architecture Search (Mixed Integer Programing NAS search)
# using a one-click command.
#
# Note: Bypass is disabled now in the test.
#

SEED = 1234


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    PUZZLETRON_FAMILIES,
)
def test_puzzletron(
    project_root_path: Path,
    tmp_path: Path,
    num_gpus,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str,
    has_moe_layers: bool,
):
    if "Qwen3-VL" in hf_model_name and Version(transformers.__version__) < Version("4.57.0"):
        pytest.skip("Qwen3-VL is not supported with transformers < 4.57.0")

    if "Nemotron" in hf_model_name:
        pytest.importorskip("mamba_ssm", reason="mamba_ssm required for Nemotron tests")

    spawn_multiprocess_job(
        size=num_gpus,
        job=partial(
            _test_puzzletron_multiprocess_job,
            project_root_path,
            tmp_path,
            hf_model_name,
            converter,
            hybrid_override_pattern,
            has_moe_layers,
        ),
        backend="nccl",
    )


def _test_puzzletron_multiprocess_job(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str,
    has_moe_layers: bool,
    rank: int,
    size: int,
):
    # Set seed BEFORE dist.setup() to ensure reproducibility across all processes
    set_seed(SEED)
    dist.setup(timeout=timedelta(minutes=10))

    # Setup the test model and data.
    puzzle_dir, hf_checkpoint_path, dataset_path = setup_test_model_and_data(
        tmp_path, rank, hf_model_name, hybrid_override_pattern
    )
    hydra_config_dir = project_root_path / "tests/gpu/torch/puzzletron/resources/configs"
    model_basename = hf_model_name.split("/")[1]
    hydra_config_name = f"{hf_model_name}/{model_basename}"

    # Convert the model using AnyModel converter.
    if rank == 0:
        mtpz.anymodel.convert_model(
            input_dir=str(hf_checkpoint_path),
            output_dir=str(puzzle_dir / "ckpts/teacher"),
            converter=converter,
        )
    dist.barrier()

    # Compress the model using a one-click approach
    hydra_cfg = mtpz.entrypoint.puzzletron(
        str(hydra_config_dir), hydra_config_name, str(puzzle_dir), str(dataset_path)
    )

    #
    # Check assertions (collect all failures, report at the end)
    #
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    if rank == 0:
        if has_moe_layers:
            # assertions for the score_pruning_activations step 1 (MoE models only)
            rank_filepath = (
                f"pruning/pruning_scores/expert_removal/10samples_diverse_mini/rank_{rank}.pth"
            )
            check((puzzle_dir / rank_filepath).is_file(), f"Expected {rank_filepath} to exist")

            # assertions for the pruning_ckpts step 2
            check(
                (puzzle_dir / "ckpts/num_experts_8").exists(),
                "Expected ckpts/num_experts_8 to exist",
            )

            # assertions for the mip_and_realize_models step 6
            # Find the MIP solution directory dynamically (e.g., stats_num_local_experts_*)
            mip_solutions_dir = puzzle_dir / "mip/puzzle_solutions"
            solution_dirs = [
                d
                for d in mip_solutions_dir.iterdir()
                if d.is_dir() and d.name.startswith("stats_num_local_experts_")
            ]
            check(
                len(solution_dirs) == 1,
                f"Expected exactly one stats_num_local_experts_* directory, found: {[d.name for d in solution_dirs]}",
            )
            if len(solution_dirs) == 1:
                solution_dir = solution_dirs[0]
                solution_0_ckpt_config_path = (
                    solution_dir / "solutions--checkpoints/solution_0/config.json"
                )
                check(
                    solution_0_ckpt_config_path.exists(),
                    f"Expected {solution_0_ckpt_config_path} to exist",
                )
                check(
                    (solution_dir / "solutions.json").exists(),
                    f"Expected {solution_dir / 'solutions.json'} to exist",
                )

            # Validate lm_loss — wider tolerance for MoE expert-routing non-determinism
            errors.extend(_check_lm_loss(puzzle_dir, hf_model_name))
        else:
            # assertions for the score_pruning_activations step 1 (FFN pruning)
            rank_filepath = (
                f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
            )
            check((puzzle_dir / rank_filepath).is_file(), f"Expected {rank_filepath} to exist")

            # assertions for the pruning_ckpts step 2
            check(
                (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists(),
                "Expected ckpts/ffn_256_attn_no_op to exist",
            )

            # assertions for the mip_and_realize_models step 6
            errors.extend(_check_mip_solutions(puzzle_dir, hf_model_name))

        # assertions for the build_library_and_stats step 4
        check(
            (puzzle_dir / "replacement_library.json").is_file(),
            "Expected replacement_library.json to exist",
        )
        errors.extend(_check_subblock_stats_anymodel(hf_model_name, hydra_cfg))

        # assertions for the scoring step 5
        solution_0_filepath = (
            puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
        )
        check(solution_0_filepath.exists(), f"Expected {solution_0_filepath} to exist")

    dist.cleanup()

    if errors:
        pytest.fail(
            f"{len(errors)} assertion(s) failed for {hf_model_name}:\n  - " + "\n  - ".join(errors)
        )


def _check_subblock_stats_anymodel(hf_model_name: str, hydra_cfg) -> list[str]:
    """Minimal subblock_stats checks and teacher memory / param regression values."""
    errors: list[str] = []
    if not (Path(hydra_cfg.puzzle_dir) / "subblock_stats.json").is_file():
        errors.append("Expected subblock_stats.json to exist")
        return errors
    teacher_mem_mib = mtpz.mip.get_teacher_memory_from_subblock_stats(hydra_cfg)
    teacher_num_params = mtpz.mip.get_teacher_num_params_from_subblock_stats(hydra_cfg)

    if abs(teacher_mem_mib - EXPECTED_TEACHER_MEMORY_MIB[hf_model_name]) >= 1e-2:
        errors.append(
            f"Teacher memory mismatch for {hf_model_name}: "
            f"expected {EXPECTED_TEACHER_MEMORY_MIB[hf_model_name]}, got {teacher_mem_mib}"
        )
    if teacher_num_params != EXPECTED_TEACHER_NUM_PARAMS[hf_model_name]:
        errors.append(
            f"Teacher num_params mismatch for {hf_model_name}: "
            f"expected {EXPECTED_TEACHER_NUM_PARAMS[hf_model_name]}, got {teacher_num_params}"
        )
    return errors


def _check_lm_loss(puzzle_dir: Path, hf_model_name: str, tolerance: float = 0.15) -> list[str]:
    """Validate lm_loss for a model solution.

    Tolerance is 0.15: enough headroom for cross-GPU drift between Ada and Blackwell
    (empirically up to ~0.1 even at fp32, because cuBLAS picks different fused-kernel
    paths on different architectures). Hybrid mamba+MoE models like Nemotron-3-Nano
    drift even more and are excluded from EXPECTED_LM_LOSS.
    """
    errors: list[str] = []
    solution_0_path = (
        puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
    )
    if not solution_0_path.exists():
        errors.append(f"Expected {solution_0_path} to exist for lm_loss check")
        return errors
    with open(solution_0_path) as f:
        validation = json.load(f)

    actual_lm_loss = validation["lm_loss"]["avg"]
    expected_lm_loss = EXPECTED_LM_LOSS.get(hf_model_name)
    if expected_lm_loss is not None:
        if abs(actual_lm_loss - expected_lm_loss) >= tolerance:
            errors.append(f"lm_loss mismatch: expected {expected_lm_loss}, got {actual_lm_loss}")
    elif hf_model_name != "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16":
        errors.append(
            f"Expected lm_loss values not found for {hf_model_name}! Observed value: {actual_lm_loss}"
        )
    return errors


def _check_mip_solutions(puzzle_dir: Path, hf_model_name: str) -> list[str]:
    """Assertions for the mip_and_realize_models step."""
    errors: list[str] = []
    mip_dir = puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB"

    if not (mip_dir / "solutions.json").exists():
        errors.append(f"Expected {mip_dir / 'solutions.json'} to exist")
    if not (mip_dir / "solutions--checkpoints/solution_0/config.json").exists():
        errors.append(
            f"Expected {mip_dir / 'solutions--checkpoints/solution_0/config.json'} to exist"
        )

    # Validate lm_loss
    errors.extend(_check_lm_loss(puzzle_dir, hf_model_name))
    return errors


# Expected lm_loss values per model.
EXPECTED_LM_LOSS = {
    "meta-llama/Llama-3.1-8B-Instruct": 4.830112,
    "meta-llama/Llama-3.2-3B-Instruct": 4.861398,
    "mistralai/Mistral-Small-24B-Instruct-2501": 4.830872,
    # Nemotron-3-Nano-30B-A3B (hybrid mamba+MoE) intentionally excluded — see _check_lm_loss.
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": 5.034576,
    "openai/gpt-oss-20b": 4.898024,
    "Qwen/Qwen2.5-7B-Instruct": 4.865112,
    "Qwen/Qwen3-8B": 4.832244,
    # Qwen3-VL lm_loss is 5.03125 on transformers 5.6+ but ~4.92 on 4.57 (known cross-major drift)
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 5.03125,
}


# Expected teacher memory from subblock_stats (MiB)
EXPECTED_TEACHER_MEMORY_MIB = {
    "meta-llama/Llama-3.1-8B-Instruct": 395.63,
    "meta-llama/Llama-3.2-3B-Instruct": 395.63,
    "mistralai/Mistral-Small-24B-Instruct-2501": 395.63,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": 432.81,
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": 197.63,
    "openai/gpt-oss-20b": 437.33,
    "Qwen/Qwen2.5-7B-Instruct": 386.25,
    "Qwen/Qwen3-8B": 395.63,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 406.14,
}


# Expected total teacher params from subblock_stats
EXPECTED_TEACHER_NUM_PARAMS = {
    "meta-llama/Llama-3.1-8B-Instruct": 6096128,
    "meta-llama/Llama-3.2-3B-Instruct": 6096128,
    "mistralai/Mistral-Small-24B-Instruct-2501": 6096128,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": 126255872,
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": 2949888,
    "openai/gpt-oss-20b": 27959168,
    "Qwen/Qwen2.5-7B-Instruct": 1181696,
    "Qwen/Qwen3-8B": 6096640,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 11609856,
}
