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
"""Tests for distill_hf.py script."""

from pathlib import Path

from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.distributed.utils import get_free_port
from _test_utils.torch.puzzletron.utils import create_and_save_small_hf_model, create_tokenizer

from modelopt.torch.puzzletron.anymodel import convert_model


def test_distill_hf(project_root_path: Path, tmp_path: Path):
    """Integration test for distill_hf.py.

    Creates Llama models programmatically, converts them to heterogeneous format (AnyModel),
    and runs mbridge distillation. The models are created with reduced size for faster testing.
    Models are converted to include block_configs.
    """
    # Prepare student and teacher models
    student_hf_path, teacher_hf_path = _prepare_student_and_teacher_models(
        project_root_path, tmp_path
    )

    # Prepare output directory
    output_dir = tmp_path / "distill_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare HF export directory
    hf_export_dir = tmp_path / "hf_export"
    hf_export_dir.mkdir(parents=True, exist_ok=True)

    # Build command-line arguments for distill_hf.py
    # Use torchrun for distributed execution (single GPU for testing)
    nproc_per_node = 1
    tp_size = 1
    train_iters = 5

    cmd_parts = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "--master-addr",
        "127.0.0.1",  # Explicitly set master address
        "--master-port",
        str(get_free_port()),  # Pass port directly to torchrun to avoid conflicts
        "distill_hf.py",
        "--student_hf_path",
        student_hf_path,
        "--teacher_hf_path",
        teacher_hf_path,
        "--output_dir",
        str(output_dir),
        "--hf-export-path",
        str(hf_export_dir),  # Export to HuggingFace format
        "--hf-model",
        "meta-llama/Llama-3.1-8B-Instruct",  # Note: uses hyphen, not underscore
        "--tp_size",
        str(tp_size),
        "--pp_size",
        "1",
        "--seq_length",
        "128",  # Reduced for faster forward/backward passes
        "--use_mock_data",  # Use mock data to avoid disk I/O overhead
        "--split",
        "99,1,0",
        "--mbs",
        "1",
        "--gbs",
        "4",  # Global batch size
        "--train_iters",
        str(train_iters),  # Minimal iterations for smoke test
        "--lr",
        "0.0001",
        "--min_lr",
        "1e-5",
        "--lr_warmup_iters",
        "2",  # Reduced warmup iterations
        "--eval_interval",
        "100",  # Disable evaluation (set to > train_iters)
        "--eval_iters",
        "0",  # No evaluation iterations
        "--log_interval",
        "5",  # Reduced logging frequency
    ]

    run_example_command(
        cmd_parts,
        example_path="puzzletron/mbridge_distillation",
    )

    # Check that distillation checkpoint contains run_config.yaml
    run_config_path = output_dir / "checkpoints" / f"iter_{train_iters:07d}" / "run_config.yaml"
    assert run_config_path.exists(), f"Expected run_config.yaml to exist at: {run_config_path}"

    # Verify HF export was created
    config_path = hf_export_dir / "config.json"
    assert config_path.exists(), f"HF export should contain config.json at: {config_path}"

    # Check for model weights index file
    weight_index_path = hf_export_dir / "model.safetensors.index.json"
    assert weight_index_path.exists(), (
        f"HF export should contain model.safetensors.index.json at: {weight_index_path}"
    )

    print(
        f"PYTEST SUMMARY: test_distill_hf test has finished successfully. "
        f"Output directory: {output_dir}, HF export: {hf_export_dir}"
    )


def _prepare_student_and_teacher_models(project_root_path: Path, tmp_path: Path) -> tuple[str, str]:
    """Prepare student and teacher models for distillation.

    Creates Llama models programmatically, converts them to heterogeneous format (AnyModel),
    and returns the paths to the converted checkpoints.

    Args:
        project_root_path: Path to the project root directory
        tmp_path: Temporary directory for test artifacts

    Returns:
        Tuple of (student_hf_path, teacher_hf_path) as strings
    """

    # Create temporary directories for models
    student_hf_dir = tmp_path / "student_hf"
    teacher_hf_dir = tmp_path / "teacher_hf"

    # Create tokenizer (uses local tokenizer from test resources)
    tokenizer = create_tokenizer(project_root_path)

    # Create student model using utility function
    # This uses local config files and preserves model-specific settings
    create_and_save_small_hf_model(
        output_path=str(student_hf_dir),
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        hf_config_name="llama_3_1_8b_instruct",
        hybrid_override_pattern=None,
    )

    # Create teacher model (same as student for testing)
    create_and_save_small_hf_model(
        output_path=str(teacher_hf_dir),
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        hf_config_name="llama_3_1_8b_instruct",
        hybrid_override_pattern=None,
    )

    # Convert models to AnyModel format BEFORE distillation
    # This is needed as converted checkpoints will be used as input for distillation later
    student_anymodel_dir = tmp_path / "student_anymodel"
    teacher_anymodel_dir = tmp_path / "teacher_anymodel"

    convert_model(
        input_dir=str(student_hf_dir),
        output_dir=str(student_anymodel_dir),
        converter="llama",
    )

    convert_model(
        input_dir=str(teacher_hf_dir),
        output_dir=str(teacher_anymodel_dir),
        converter="llama",
    )
    print("Models converted to AnyModel format:")
    print(f"  Student AnyModel: {student_anymodel_dir}")
    print(f"  Teacher AnyModel: {teacher_anymodel_dir}")

    return student_anymodel_dir, teacher_anymodel_dir
