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

import torch
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.distributed.utils import get_free_port
from _test_utils.torch.puzzletron.utils import create_and_save_small_hf_model
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from transformers import AutoModelForCausalLM

from modelopt.torch.puzzletron.anymodel import convert_model


def test_distill_hf(project_root_path: Path, tmp_path: Path):
    """Integration test for distill_hf.py.

    Creates Qwen3 models programmatically, converts them to heterogeneous format (AnyModel),
    and runs mbridge distillation. The models are created with reduced size for faster testing.
    Models are converted to include block_configs.
    """
    # Prepare student and teacher models
    student_hf_dir, student_anymodel_dir, _, teacher_anymodel_dir = (
        _prepare_student_and_teacher_models(project_root_path, tmp_path)
    )

    output_dir = tmp_path / "distill_output"
    hf_export_dir = tmp_path / "hf_export"

    # Build command-line arguments for distill_hf.py
    nproc_per_node = torch.cuda.device_count()
    tp_size = nproc_per_node
    train_iters = 5

    cmd_parts = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(get_free_port()),
        "distill_hf.py",
        "--use_mock_data",
    ]
    extend_cmd_parts(
        cmd_parts,
        student_hf_path=student_anymodel_dir,
        teacher_hf_path=teacher_anymodel_dir,
        output_dir=output_dir,
        tp_size=tp_size,
        pp_size=1,
        seq_length=128,
        split="99,1,0",
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr=0.0001,
        min_lr=1e-5,
        lr_warmup_iters=2,
        eval_interval=100,
        eval_iters=0,
        log_interval=5,
        hf_export_path=hf_export_dir,
        hf_model=student_hf_dir,
    )

    run_example_command(cmd_parts, example_path="puzzletron/mbridge_distillation")

    # Check that distillation checkpoint contains run_config.yaml
    run_config_path = output_dir / "checkpoints" / f"iter_{train_iters:07d}" / "run_config.yaml"
    assert run_config_path.exists(), f"Expected run_config.yaml to exist at: {run_config_path}"

    # Verify that the distilled model can be loaded in HuggingFace format
    model = AutoModelForCausalLM.from_pretrained(hf_export_dir)
    assert model is not None, "Failed to load distilled model with AutoModelForCausalLM"


def _prepare_student_and_teacher_models(
    project_root_path: Path, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    """Prepare student and teacher models for distillation.

    Creates Qwen3 models programmatically, converts them to heterogeneous format (AnyModel),
    and returns the paths to the converted checkpoints.

    """

    # Create temporary directories for models
    student_hf_dir = tmp_path / "student_hf"
    teacher_hf_dir = tmp_path / "teacher_hf"

    # Create tokenizer (uses local tokenizer from test resources)
    tokenizer = get_tiny_tokenizer()

    # Create student model using utility function (loads config from Hub).
    # TODO: Make the student model using different ffn sizes across layers.
    create_and_save_small_hf_model(
        output_path=str(student_hf_dir),
        tokenizer=tokenizer,
        hf_model_name="Qwen/Qwen3-0.6B",
        hybrid_override_pattern=None,
    )

    # Create teacher model (same as student for testing)
    create_and_save_small_hf_model(
        output_path=str(teacher_hf_dir),
        tokenizer=tokenizer,
        hf_model_name="Qwen/Qwen3-0.6B",
        hybrid_override_pattern=None,
    )

    # Convert models to AnyModel format BEFORE distillation
    # This is needed as converted checkpoints will be used as input for distillation later
    student_anymodel_dir = tmp_path / "student_anymodel"
    teacher_anymodel_dir = tmp_path / "teacher_anymodel"

    convert_model(
        input_dir=str(student_hf_dir), output_dir=str(student_anymodel_dir), converter="qwen3"
    )

    convert_model(
        input_dir=str(teacher_hf_dir), output_dir=str(teacher_anymodel_dir), converter="qwen3"
    )
    print("Models converted to AnyModel format:")
    print(f"  Student AnyModel: {student_anymodel_dir}")
    print(f"  Teacher AnyModel: {teacher_anymodel_dir}")

    return student_hf_dir, student_anymodel_dir, teacher_hf_dir, teacher_anymodel_dir
