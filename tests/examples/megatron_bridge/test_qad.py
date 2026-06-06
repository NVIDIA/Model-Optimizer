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
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir


@pytest.mark.timeout(720)  # Multiple steps in one test hence takes longer than default timeout
def test_qad(tmp_path: Path, num_gpus):
    """Quantize a tiny Qwen3, run QAD from the quantized student, and export to a unified HF ckpt."""
    hf_model_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    quantized_megatron_path = tmp_path / "qwen3_fp8_megatron"
    distill_output_dir = tmp_path / "qad_output"
    hf_export_path = tmp_path / "qwen3_qad_fp8_hf"
    train_iters = 2

    # Step 1: PTQ the model to FP8 and save a Megatron checkpoint carrying the ModelOpt state.
    quantize_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        recipe="general/ptq/fp8_default-kv_fp8",
        tp_size=num_gpus,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=16,
        calib_batch_size=4,
        seq_length=16,
        export_megatron_path=quantized_megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert list(quantized_megatron_path.rglob("modelopt_state")), (
        "Expected modelopt_state in the quantized Megatron checkpoint"
    )

    # Step 2: QAD -- load the quantized student from the Megatron checkpoint (restoring the
    # ModelOpt quantizers) and distill from the (unquantized) HF teacher. The distilled checkpoint
    # must keep the ModelOpt state so it can later be exported as a quantized HF checkpoint.
    distill_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=hf_model_path,
        student_megatron_path=quantized_megatron_path,
        teacher_hf_path=hf_model_path,
        output_dir=distill_output_dir,
        tp_size=num_gpus,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=2,
        eval_interval=train_iters,
        eval_iters=1,
        log_interval=1,
    )
    run_example_command(distill_cmd, example_path="megatron_bridge", setup_free_port=True)
    distilled_megatron_path = distill_output_dir / "checkpoints"
    assert (distilled_megatron_path / "latest_checkpointed_iteration.txt").exists()
    assert list(distilled_megatron_path.rglob("modelopt_state")), (
        "Expected modelopt_state to be preserved in the distilled (QAD) checkpoint"
    )

    # Step 3: export the distilled quantized checkpoint to a unified HF checkpoint. hf_quant_config.json
    # is only written for a quantized model, so its presence confirms the quantizers survived QAD.
    export_cmd = extend_cmd_parts(
        ["torchrun", "--nproc_per_node=1", "export.py"],
        hf_model_name_or_path=hf_model_path,
        megatron_path=distilled_megatron_path,
        export_unified_hf_path=hf_export_path,
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    assert list(hf_export_path.glob("*.safetensors")), "Expected exported safetensors weights"
