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
import torch
import transformers
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.misc import set_seed
from _test_utils.torch.puzzletron.utils import setup_test_model_and_data
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
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "llama", None, False),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama", None, False),
        ("mistralai/Mistral-Small-24B-Instruct-2501", "mistral_small", None, False),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "nemotron_h", "*E", True),
        ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", "nemotron_h_v2", "*-", False),
        ("openai/gpt-oss-20b", "gpt_oss", None, True),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2", None, False),
        ("Qwen/Qwen3-8B", "qwen3", None, False),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen3_vl", None, True),
    ],
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
    # Check assertions
    #
    if rank == 0:
        if has_moe_layers:
            # assertions for the score_pruning_activations step 1 (MoE models only)
            rank_filepath = (
                f"pruning/pruning_scores/expert_removal/10samples_diverse_mini/rank_{rank}.pth"
            )
            assert (puzzle_dir / rank_filepath).is_file(), f"Expected {rank_filepath} to exist"

            # assertions for the pruning_ckpts step 2
            assert (puzzle_dir / "ckpts/num_experts_8").exists()

            # assertions for the mip_and_realize_models step 6
            # Find the MIP solution directory dynamically (e.g., stats_num_local_experts_*)
            mip_solutions_dir = puzzle_dir / "mip/puzzle_solutions"
            solution_dirs = [
                d
                for d in mip_solutions_dir.iterdir()
                if d.is_dir() and d.name.startswith("stats_num_local_experts_")
            ]
            assert len(solution_dirs) == 1, (
                f"Expected exactly one stats_num_local_experts_* directory, found: {[d.name for d in solution_dirs]}"
            )
            solution_dir = solution_dirs[0]

            solution_0_ckpt_config_path = (
                solution_dir / "solutions--checkpoints/solution_0/config.json"
            )
            assert solution_0_ckpt_config_path.exists()
            assert (solution_dir / "solutions.json").exists()

            # Validate lm_loss
            _assert_lm_loss(puzzle_dir, hf_model_name, tolerance=0.01)
        else:
            # assertions for the score_pruning_activations step 1 (FFN pruning)
            _assert_score_pruning_activations(puzzle_dir, hf_model_name)

            # assertions for the pruning_ckpts step 2
            assert (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists()

            # assertions for the mip_and_realize_models step 6
            _assert_mip_solutions(puzzle_dir, hf_model_name)

        # assertions for the build_library_and_stats step 4
        assert (puzzle_dir / "replacement_library.json").is_file()
        _assert_subblock_stats_anymodel(hf_model_name, hydra_cfg)

        # assertions for the scoring step 5
        solution_0_filepath = (
            puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
        )
        assert solution_0_filepath.exists()

    dist.cleanup()


def _assert_subblock_stats_anymodel(hf_model_name: str, hydra_cfg) -> None:
    """Minimal subblock_stats checks and teacher memory / param regression values."""
    assert (Path(hydra_cfg.puzzle_dir) / "subblock_stats.json").is_file()
    teacher_mem_mib = mtpz.mip.get_teacher_memory_from_subblock_stats(hydra_cfg)
    teacher_num_params = mtpz.mip.get_teacher_num_params_from_subblock_stats(hydra_cfg)

    assert abs(teacher_mem_mib - EXPECTED_TEACHER_MEMORY_MIB[hf_model_name]) < 1e-2, (
        f"Teacher memory mismatch for {hf_model_name}: "
        f"expected {EXPECTED_TEACHER_MEMORY_MIB[hf_model_name]}, got {teacher_mem_mib}"
    )
    assert teacher_num_params == EXPECTED_TEACHER_NUM_PARAMS[hf_model_name], (
        f"Teacher num_params mismatch for {hf_model_name}: "
        f"expected {EXPECTED_TEACHER_NUM_PARAMS[hf_model_name]}, got {teacher_num_params}"
    )


def _assert_score_pruning_activations(puzzle_dir: Path, hf_model_name: str):
    """Assertions for the score_pruning_activations step 1."""
    rank = dist.rank()
    rank_filepath = f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
    assert (puzzle_dir / rank_filepath).is_file()

    pruning_scores = torch.load(puzzle_dir / rank_filepath)
    layer_names = list(pruning_scores.keys())
    expected = EXPECTED_FFN_PRUNING_VALUES[hf_model_name]
    size = dist.size()

    if expected is not None:
        # In multi-GPU: layers are distributed across ranks
        # Each rank processes len(expected) // size layers
        expected_layers_per_rank = len(expected) // size
        assert len(layer_names) == expected_layers_per_rank, (
            f"Expected {expected_layers_per_rank} FFN layers on rank {rank}/{size}, got {len(layer_names)}"
        )
        # Check each layer's values
        for i, layer_name in enumerate(layer_names):
            layer_data = pruning_scores[layer_name]
            # Calculate global layer index from rank and local index
            global_idx = rank * expected_layers_per_rank + i
            assert layer_data["score"][0].item() == expected[global_idx]["score"], (
                layer_name,
                layer_data["score"][0].item(),
                expected[global_idx]["score"],
                global_idx,
            )
            assert (
                layer_data["channels_importance_ascending"][0].item()
                == expected[global_idx]["channels"]
            )
    else:
        observed_values = []
        for layer_name in layer_names:
            layer_data = pruning_scores[layer_name]
            observed_values.append(
                {
                    "score": layer_data["score"][0].item(),
                    "channels": layer_data["channels_importance_ascending"][0].item(),
                }
            )
        pytest.fail(f"Expected pruning values not found for {hf_model_name}!\n{observed_values=}")


def _assert_lm_loss(puzzle_dir: Path, hf_model_name: str, tolerance: float = 0.01):
    """Validate lm_loss for a model solution."""
    solution_0_path = (
        puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
    )
    with open(solution_0_path) as f:
        validation = json.load(f)

    actual_lm_loss = validation["lm_loss"]["avg"]
    expected_lm_loss = EXPECTED_LM_LOSS.get(hf_model_name)
    if expected_lm_loss is not None:
        assert abs(actual_lm_loss - expected_lm_loss) < tolerance, (
            f"lm_loss mismatch: expected {expected_lm_loss}, got {actual_lm_loss}"
        )
    # TODO: not reproducible in CI, skipping for now
    elif hf_model_name != "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16":
        pytest.fail(
            f"Expected lm_loss values not found for {hf_model_name}! Observed value: {actual_lm_loss}"
        )


def _assert_mip_solutions(puzzle_dir: Path, hf_model_name: str):
    """Assertions for the mip_and_realize_models step."""
    mip_dir = puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB"

    assert (mip_dir / "solutions.json").exists()
    assert (mip_dir / "solutions--checkpoints/solution_0/config.json").exists()

    # Validate lm_loss
    _assert_lm_loss(puzzle_dir, hf_model_name)


# Expected pruning activation values per model
# Each model has a list of (score[0], channels[0]) tuples for each FFN layer
EXPECTED_FFN_PRUNING_VALUES = {
    "meta-llama/Llama-3.1-8B-Instruct": [
        {"score": 435, "channels": 94},
        {"score": 82, "channels": 338},
    ],
    "meta-llama/Llama-3.2-3B-Instruct": [
        {"score": 440, "channels": 94},
        {"score": 88, "channels": 338},
    ],
    "mistralai/Mistral-Small-24B-Instruct-2501": [
        {"score": 410, "channels": 94},
        {"score": 82, "channels": 338},
    ],
    # NemotronH with pattern "*-" has only 1 FFN layer (the "-" layer)
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": [
        {"score": 469, "channels": 81},
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        {"score": 374, "channels": 205},
        # NOTE: below score differs as per GPU: set as per CI's RTX Pro 6000 BW. Getting 100 on RTX 6000 Ada
        {"score": 102, "channels": 317},
    ],
    "Qwen/Qwen3-8B": [
        {"score": 405, "channels": 173},
        {"score": 48, "channels": 376},
    ],
}


# Expected lm_loss values per model
EXPECTED_LM_LOSS = {
    "meta-llama/Llama-3.1-8B-Instruct": 4.913641,
    "meta-llama/Llama-3.2-3B-Instruct": 4.885118,
    "mistralai/Mistral-Small-24B-Instruct-2501": 4.913618,
    # TODO: not reproducible in CI, skipping for now
    # "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": 5.068373,
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": 4.987095,
    "openai/gpt-oss-20b": 4.898407,
    "Qwen/Qwen2.5-7B-Instruct": 4.890478,
    "Qwen/Qwen3-8B": 4.927514,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 5.0625,  # 4.828125 for transformers v4.57
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
