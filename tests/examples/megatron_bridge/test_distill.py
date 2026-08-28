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
"""Tests for distill.py and export_distilled_megatron_to_hf.py scripts."""

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.puzzletron.utils import create_and_save_small_hf_model
from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_5_vl_dir,
    create_tiny_qwen3_dir,
    get_tiny_tokenizer,
)
from transformers import AutoModelForImageTextToText

from modelopt.torch.puzzletron.anymodel import convert_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "megatron_bridge"))

from export_distilled_megatron_to_hf import _get_checkpoint_export_paths


@pytest.mark.timeout(360)  # distill + inline HF export + export of every iteration
def test_distill_llm_hf_export(tmp_path, num_gpus):
    """Distill with an inline HF export, then re-export every saved iteration standalone."""
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    train_iters = 2
    distill_output_dir = tmp_path / "distill_output"
    distilled_hf_path = tmp_path / "distilled_hf"
    all_exports = tmp_path / "all_exports"
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
        eval_interval=1,  # save_interval defaults to it, so every iteration is checkpointed
        eval_iters=1,
        log_interval=1,
        checkpoint_keep_last=-1,
        hf_export_path=distilled_hf_path,
    )
    run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    checkpoint_root = distill_output_dir / "checkpoints"
    assert (checkpoint_root / f"iter_{train_iters:07d}").exists()
    assert (distilled_hf_path / "config.json").exists()

    # export_distilled_megatron_to_hf.py converts any saved iteration, not just the final one.
    all_export_cmd_parts = extend_cmd_parts(
        ["torchrun", "--nproc_per_node=1", "export_distilled_megatron_to_hf.py"],
        student_hf_path=teacher_hf_path,
        megatron_path=checkpoint_root,
        hf_export_path=all_exports,
        export_iterations="all",
    )
    run_example_command(all_export_cmd_parts, example_path="megatron_bridge")

    for it in range(1, train_iters + 1):
        assert (all_exports / f"iter_{it:07d}/config.json").exists()


def test_distill_llm_sft(tmp_path, num_gpus):
    """--sft distills from prompt-completion jsonl instead of pre-tokenized --data_paths."""
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    train_iters = 2
    gbs = 4
    dataset_root = tmp_path / "sft_data"
    dataset_root.mkdir()
    # More records than train_iters * gbs so the sampler does not run dry.
    records = [{"input": f"Q: what follows {i}?\nA:", "output": f" {i + 1}"} for i in range(64)]
    for split in ("training", "validation"):
        (dataset_root / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

    distill_output_dir = tmp_path / "distill_output"
    distill_cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--sft"],
        student_hf_path=teacher_hf_path,
        teacher_hf_path=teacher_hf_path,
        sft_dataset_root=dataset_root,
        output_dir=distill_output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=64,
        mbs=1,
        gbs=gbs,
        train_iters=train_iters,
        lr_warmup_iters=1,
        eval_interval=train_iters,
        eval_iters=1,
        log_interval=1,
    )
    run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    assert (distill_output_dir / f"checkpoints/iter_{train_iters:07d}").exists()


def test_distill_validate_only(tmp_path, num_gpus):
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    train_iters = 2
    output_dir = tmp_path / "distill_output"
    hf_export_path = tmp_path / "distilled_hf"
    distill_cmd_parts = extend_cmd_parts(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "distill.py",
            "--use_mock_data",
            "--validate_only",
        ],
        student_hf_path=teacher_hf_path,
        teacher_hf_path=teacher_hf_path,
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
    )
    output = run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    assert "skipping training ..." in output
    assert "iteration 0 on validation set" in output
    assert not (output_dir / f"checkpoints/iter_{train_iters:07d}").exists()
    assert not (hf_export_path / "config.json").exists()


# NOTE: Qwen3.5-VL-MoE covered by test_qad.py
@pytest.mark.timeout(360)  # sometimes times out in CI env with default 300s timeout
def test_distill_vlm(tmp_path, num_gpus):
    # Self-distillation of a tiny VLM: only the language model is distilled; the vision tower and the
    # vision->language projector must be left byte-for-byte untouched.
    tp_size = num_gpus
    pp_size = 1
    vlm_hf_path, teacher_model = create_tiny_qwen3_5_vl_dir(
        tmp_path,
        with_tokenizer=True,
        return_model=True,
        num_hidden_layers=2,
        intermediate_size=128,
    )

    # The language model spans ``model.language_model.*`` plus the (possibly untied) output head
    # ``lm_head`` -- both are distilled and may change. Everything else (vision tower + projector)
    # must stay byte-for-byte identical.
    def _is_language_model(name: str) -> bool:
        return name.startswith(("model.language_model.", "lm_head"))

    teacher_non_lm = {
        name: param.detach().clone()
        for name, param in teacher_model.named_parameters()
        if not _is_language_model(name)
    }
    assert teacher_non_lm, (
        "Expected non-language-model params (vision tower / projector) in the VLM."
    )

    train_iters = 2
    distill_output_dir = tmp_path / "distill_output"
    distilled_hf_path = tmp_path / "distilled_hf"
    distill_cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=vlm_hf_path,
        teacher_hf_path=vlm_hf_path,
        output_dir=distill_output_dir,
        tp_size=tp_size,
        pp_size=pp_size,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=1,
        eval_interval=train_iters,
        eval_iters=1,
        log_interval=1,
    )
    run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    megatron_ckpt = distill_output_dir / f"checkpoints/iter_{train_iters:07d}"
    assert megatron_ckpt.exists()

    # Separately convert the distilled Megatron checkpoint to HF with the standalone export script
    export_cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "export_distilled_megatron_to_hf.py"],
        student_hf_path=vlm_hf_path,
        megatron_path=megatron_ckpt,
        hf_export_path=distilled_hf_path,
        tp_size=tp_size,
        pp_size=pp_size,
    )
    run_example_command(export_cmd_parts, example_path="megatron_bridge")

    assert (distilled_hf_path / "config.json").exists()

    # from_pretrained (default strict load) verifies the saved weights match the (unchanged) config.
    distilled_model = AutoModelForImageTextToText.from_pretrained(distilled_hf_path)
    assert hasattr(distilled_model.config, "vision_config")
    distilled_params = dict(distilled_model.named_parameters())
    # Everything outside the language model is identical to the teacher (vision tower untouched).
    for name, teacher_param in teacher_non_lm.items():
        assert name in distilled_params, (
            f"Missing non-language-model param after distillation: {name}"
        )
        assert torch.equal(distilled_params[name].float(), teacher_param.float()), (
            f"Non-language-model param changed during LM-only distillation: {name}"
        )


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


def test_checkpoint_export_paths_for_selected_iterations(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "iter_0000001").mkdir()
    (checkpoint_root / "iter_0000002").mkdir()
    hf_export_root = tmp_path / "hf_validation"

    args = argparse.Namespace(
        megatron_path=str(checkpoint_root),
        hf_export_path=str(hf_export_root),
        export_iterations=["2", "1", "2"],
    )

    assert _get_checkpoint_export_paths(args) == [
        (checkpoint_root / "iter_0000001", hf_export_root / "iter_0000001"),
        (checkpoint_root / "iter_0000002", hf_export_root / "iter_0000002"),
    ]


def test_checkpoint_export_paths_for_all_iterations(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "iter_0000002").mkdir()
    (checkpoint_root / "not_an_iteration").mkdir()
    (checkpoint_root / "iter_0000001").mkdir()
    hf_export_root = tmp_path / "hf_validation"

    args = argparse.Namespace(
        megatron_path=str(checkpoint_root),
        hf_export_path=str(hf_export_root),
        export_iterations=["all"],
    )

    assert _get_checkpoint_export_paths(args) == [
        (checkpoint_root / "iter_0000001", hf_export_root / "iter_0000001"),
        (checkpoint_root / "iter_0000002", hf_export_root / "iter_0000002"),
    ]
