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

import json
from pathlib import Path

from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.puzzletron.utils import create_and_save_small_hf_model
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir, get_tiny_tokenizer

from modelopt.torch.puzzletron.anymodel import convert_model
from modelopt.torch.utils.plugins.megatron_preprocess_data import megatron_preprocess_data


def test_distill_and_convert(tmp_path: Path, num_gpus):
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    train_iters = 3
    distill_output_dir = tmp_path / "distill_output"
    distilled_hf_path = tmp_path / "distilled_hf"
    validation_exports = tmp_path / "validation_exports"
    distill_cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=teacher_hf_path,
        teacher_hf_path=teacher_hf_path,
        output_dir=distill_output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=1,
        eval_interval=1,
        eval_iters=1,
        log_interval=1,
        hf_export_path=distilled_hf_path,
        hf_validation_export_path=validation_exports,
        hf_validation_export_interval=2,
        student_hf_model=teacher_hf_path,
    )
    run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    assert (distill_output_dir / f"checkpoints/iter_{train_iters:07d}").exists()
    assert (distilled_hf_path / "config.json").exists()
    assert (validation_exports / "iter_0000002/config.json").exists()
    assert (validation_exports / "iter_0000003/config.json").exists()
    assert not (validation_exports / "iter_0000001").exists()


def test_distill_validate_only(tmp_path: Path, num_gpus, capfd):
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    training_data_path = _create_megatron_dataset(tmp_path, "training", teacher_hf_path)
    validation_data_path = _create_megatron_dataset(tmp_path, "validation", teacher_hf_path)
    output_dir = tmp_path / "validation_output"
    validation_exports = tmp_path / "validation_exports"
    cmd_parts = extend_cmd_parts(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "distill.py",
            "--validate_only",
            "--data_paths",
            "1.0",
            training_data_path,
            "--target_validation_data_paths",
            "1.0",
            validation_data_path,
        ],
        student_hf_path=teacher_hf_path,
        teacher_hf_path=teacher_hf_path,
        output_dir=output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=1,
        eval_iters=1,
        hf_validation_export_path=validation_exports,
        hf_validation_export_interval=1,
        student_hf_model=teacher_hf_path,
    )
    run_example_command(cmd_parts, example_path="megatron_bridge")

    # capfd captures stdout and stderr from the torchrun subprocess.
    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "validation loss at iteration 0 on validation set" in output
    assert "total loss value:" in output
    assert "lm loss value:" in output
    assert "target validation loss at iteration 0" in output
    assert "target total loss validation:" in output
    assert "target lm loss validation:" in output
    assert (validation_exports / "iter_0000000/config.json").exists()
    assert not (output_dir / "checkpoints/iter_0000001").exists()


def test_distill_puzzletron_anymodel(tmp_path: Path, num_gpus):
    """Integration test for distill.py with Puzzletron AnyModel (heterogeneous) checkpoints.

    Creates Qwen3 models, converts the student to Puzzletron AnyModel format
    (heterogeneous layer architectures), runs mbridge distillation, and exports
    the distilled checkpoint to HuggingFace format via --hf_export_path.
    """
    student_hf_dir, student_anymodel_dir, teacher_hf_dir = (
        _prepare_puzzletron_anymodel_student_and_teacher(tmp_path)
    )

    train_iters = 2
    output_dir = tmp_path / "distill_output"
    hf_export_path = tmp_path / "distilled_anymodel_hf"
    cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=student_anymodel_dir,
        teacher_hf_path=teacher_hf_dir,
        output_dir=output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=1,
        eval_interval=train_iters,
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


def _create_megatron_dataset(tmp_path: Path, name: str, tokenizer: Path) -> str:
    """Create a small local Megatron dataset for a distillation test."""
    jsonl_path = tmp_path / f"{name}.jsonl"
    documents = ({"text": f"{name} document {index}. " * 32} for index in range(256))
    jsonl_path.write_text(
        "".join(f"{json.dumps(document)}\n" for document in documents), encoding="utf-8"
    )
    return megatron_preprocess_data(
        jsonl_paths=jsonl_path,
        output_dir=tmp_path / f"{name}_data",
        tokenizer_name_or_path=tokenizer,
        json_keys="text",
        append_eod=True,
        max_sequence_length=256,
        workers=1,
    )[0]
