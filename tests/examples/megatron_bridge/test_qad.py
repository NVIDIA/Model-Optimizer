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
"""End-to-end test for Quantization Aware Distillation (QAD): quantize + distill + export."""

from pathlib import Path

import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.export.unified_checkpoint import assert_exported_checkpoint_matches
from _test_utils.torch.transformers_models import (
    create_tiny_gemma3vl_dir,
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_dir,
    create_tiny_qwen3vl_dir,
)


@pytest.mark.timeout(720)  # Multiple steps in one test hence takes longer than the default timeout
@pytest.mark.parametrize(
    ("create_student", "exports_hf", "no_moe_grouped_gemm"),
    [
        (lambda tmp_path: create_tiny_qwen3_dir(tmp_path, with_tokenizer=True), True, False),
        # Qwen3-VL is the only VLM architecture in the Megatron HF export mapping, so it is the one
        # VLM case that runs the export step end-to-end.
        (lambda tmp_path: create_tiny_qwen3vl_dir(tmp_path, with_tokenizer=True), True, False),
        # Dense-VLM QAD path; the MoE VLM below covers it in CI, so run this one on demand only.
        pytest.param(
            lambda tmp_path: create_tiny_gemma3vl_dir(
                tmp_path,
                with_processor=True,
                num_hidden_layers=2,
                intermediate_size=128,
                max_position_embeddings=512,
            ),
            False,
            False,
            marks=pytest.mark.manual,
        ),
        pytest.param(
            lambda tmp_path: create_tiny_qwen3_5_moe_vl_dir(
                tmp_path, with_processor=True, num_hidden_layers=2
            ),
            True,
            # Gated MoE experts are only exportable as SequentialMLP; grouped GEMM raises.
            True,
        ),
    ],
    ids=["qwen3", "qwen3vl", "gemma3vl", "qwen3_5_moe_vl"],
)
def test_qad(tmp_path: Path, num_gpus, create_student, exports_hf, no_moe_grouped_gemm):
    """Quantize a tiny model, run QAD from the quantized student, and export the result.

    For VLMs only the language model is quantized and distilled (vision tower / projector untouched),
    and a text calibration dataset infers text-only LM calibration. Quantized-HF export needs the
    architecture in the Megatron export mapping, so a case missing there (Gemma3-VL) stops at the
    distilled Megatron checkpoint and only verifies the ModelOpt (quantize) state survived.
    """
    hf_model_path = create_student(tmp_path)
    moe_flag = ["--no_moe_grouped_gemm"] if no_moe_grouped_gemm else []
    quantized_megatron_path = tmp_path / "quantized_megatron"
    distill_output_dir = tmp_path / "qad_output"
    train_iters = 3
    early_exit_iter = 2

    # Step 1: PTQ the (language) model to FP8 and save a Megatron checkpoint carrying the ModelOpt state.
    quantize_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate", *moe_flag],
        hf_model_name_or_path=hf_model_path,
        recipe="general/ptq/fp8_default-kv_fp8",
        tp_size=num_gpus,
        pp_size=1,
        calib_dataset_name="cnn_dailymail",  # text dataset -> (for VLMs) text-only LM calibration
        calib_num_samples=8,
        calib_batch_size=2,
        seq_length=16,
        export_megatron_path=quantized_megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert list(quantized_megatron_path.rglob("modelopt_state")), (
        "Expected modelopt_state in the quantized Megatron checkpoint"
    )

    # Step 2: QAD -- load the quantized student from the Megatron checkpoint (restoring the ModelOpt
    # quantizers) and distill from the (unquantized) HF teacher. The distilled checkpoint must keep the
    # ModelOpt state so the quantizers survive distillation.
    distill_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data", *moe_flag],
        student_hf_path=hf_model_path,
        student_megatron_path=quantized_megatron_path,
        teacher_hf_path=hf_model_path,
        output_dir=distill_output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=2,
        eval_interval=early_exit_iter,
        eval_iters=1,
        save_interval=1,
        log_interval=1,
        exit_interval=early_exit_iter,
        exit_duration_in_mins=10,
    )
    run_example_command(distill_cmd, example_path="megatron_bridge", setup_free_port=True)
    distilled_megatron_path = distill_output_dir / "checkpoints"
    tracker = distilled_megatron_path / "latest_checkpointed_iteration.txt"
    assert tracker.read_text(encoding="utf-8").strip() == str(early_exit_iter)
    assert (distilled_megatron_path / "iter_0000001").is_dir()
    assert list(distilled_megatron_path.rglob("modelopt_state")), (
        "Expected modelopt_state to be preserved in the distilled (QAD) checkpoint"
    )

    if not exports_hf:
        return  # architecture missing from the export mapping; stop at the distilled checkpoint

    # Step 3: export the distilled quantized checkpoint to a unified HF checkpoint. hf_quant_config.json
    # is only written for a quantized model, so its presence confirms the quantizers survived QAD.
    hf_export_path = tmp_path / "qad_fp8_hf"
    export_cmd = extend_cmd_parts(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "export_quantized_megatron_to_hf.py",
            *moe_flag,
        ],
        hf_model_name_or_path=hf_model_path,
        megatron_path=distilled_megatron_path,
        export_unified_hf_path=hf_export_path,
        pp_size=num_gpus,
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    # QAD trains the student, so weights drift from the reference: names/shapes only.
    assert_exported_checkpoint_matches(hf_export_path, hf_model_path, check_values=False)
