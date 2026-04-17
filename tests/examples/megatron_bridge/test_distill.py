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
"""Tests for prune_minitron.py and distill.py scripts."""

import subprocess
from pathlib import Path

from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.puzzletron.utils import create_and_save_small_hf_model
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir, get_tiny_tokenizer

from modelopt.torch.puzzletron.anymodel import convert_model


def test_distill_and_convert(tmp_path: Path, num_gpus):
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    train_iters = 5
    distill_output_dir = tmp_path / "distill_output"
    distill_cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=teacher_hf_path,
        teacher_hf_path=teacher_hf_path,
        output_dir=distill_output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=32,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=2,
        eval_interval=5,
        eval_iters=1,
        log_interval=1,
    )
    run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    megatron_ckpt_path = distill_output_dir / f"checkpoints/iter_{train_iters:07d}"
    assert megatron_ckpt_path.exists()

    # Convert distilled Megatron checkpoint back to HF format
    distilled_hf_path = tmp_path / "distilled_hf"
    subprocess.run(
        [
            "python",
            "/opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py",
            "export",
            "--hf-model",
            str(teacher_hf_path),
            "--megatron-path",
            str(megatron_ckpt_path),
            "--hf-path",
            str(distilled_hf_path),
        ],
        check=True,
    )
    assert (distilled_hf_path / "config.json").exists()


def test_distill_puzzletron_anymodel(tmp_path: Path, num_gpus):
    """Integration test for distill.py with Puzzletron AnyModel (heterogeneous) checkpoints.

    Creates Qwen3 models, converts the student to Puzzletron AnyModel format
    (heterogeneous layer architectures), runs mbridge distillation, and exports
    the distilled checkpoint to HuggingFace format via --hf_export_path.
    """
    student_hf_dir, student_anymodel_dir, teacher_hf_dir = (
        _prepare_puzzletron_anymodel_student_and_teacher(tmp_path)
    )

    train_iters = 5
    output_dir = tmp_path / "distill_output"
    hf_export_path = tmp_path / "distilled_anymodel_hf"
    cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=student_anymodel_dir,
        teacher_hf_path=teacher_hf_dir,
        output_dir=output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=32,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=2,
        eval_interval=5,
        eval_iters=1,
        log_interval=1,
        hf_export_path=hf_export_path,
        student_hf_model=student_hf_dir,
    )
    run_example_command(cmd_parts, example_path="megatron_bridge")

    run_config_path = output_dir / "checkpoints" / f"iter_{train_iters:07d}" / "run_config.yaml"
    assert run_config_path.exists(), f"Expected run_config.yaml at: {run_config_path}"

    assert (hf_export_path / "config.json").exists(), (
        f"Expected HF export at: {hf_export_path}/config.json"
    )


def _prepare_puzzletron_anymodel_student_and_teacher(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create Qwen3 models and convert student to Puzzletron AnyModel format."""
    student_hf_dir = tmp_path / "student_hf"
    teacher_hf_dir = tmp_path / "teacher_hf"

    tokenizer = get_tiny_tokenizer()

    create_and_save_small_hf_model(
        output_path=str(student_hf_dir), tokenizer=tokenizer, hf_model_name="Qwen/Qwen3-0.6B"
    )

    create_and_save_small_hf_model(
        output_path=str(teacher_hf_dir), tokenizer=tokenizer, hf_model_name="Qwen/Qwen3-0.6B"
    )

    student_anymodel_dir = tmp_path / "student_anymodel"
    convert_model(
        input_dir=str(student_hf_dir), output_dir=str(student_anymodel_dir), converter="qwen3"
    )

    return student_hf_dir, student_anymodel_dir, teacher_hf_dir
