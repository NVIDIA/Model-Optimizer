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
"""Tests for quantize.py and export_quantized_megatron_to_hf.py scripts."""

from pathlib import Path

import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.export.unified_checkpoint import assert_exported_checkpoint_matches
from _test_utils.torch.megatron.modelopt_state import assert_has_modelopt_state
from _test_utils.torch.transformers_models import (
    create_tiny_nemotron_h_dir,
    create_tiny_qwen3_dir,
    create_tiny_qwen3_moe_dir,
    create_tiny_qwen3vl_dir,
)

# Per-architecture export *mappings* are covered in-process by
# tests/gpu_megatron/torch/export/test_unified_export_megatron.py; these cases cover the script
# wiring (CLI, recipe, checkpoint hand-off) that only running quantize.py + the exporter exercises.
# Use a vLLM-friendly head_dim (64): the default tiny config (head_dim=2) is unsupported.
_DENSE_KWARGS = {
    "hidden_size": 128,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 256,
    "max_position_embeddings": 512,
}


@pytest.mark.parametrize(
    ("create_model", "model_kwargs"),
    [
        (create_tiny_qwen3_dir, _DENSE_KWARGS),
        # MoE: routed experts used to be dropped silently from the export.
        (create_tiny_qwen3_moe_dir, _DENSE_KWARGS),
        # Dense VLM: only the language model is quantized, vision is copied through.
        (create_tiny_qwen3vl_dir, {}),
        # Mamba hybrid + MoE, and the one architecture that keeps grouped-GEMM experts.
        (create_tiny_nemotron_h_dir, {}),
    ],
    ids=["qwen3", "qwen3_moe", "qwen3vl", "nemotron_h"],
)
@pytest.mark.timeout(360)  # quantize + export in one test; 1-gpu CI exceeds the default 300s
def test_quantize_and_export(tmp_path: Path, num_gpus, create_model, model_kwargs):
    """Quantize a tiny model via a YAML recipe and export it to a unified HF checkpoint."""
    hf_model_path = create_model(tmp_path, with_tokenizer=True, **model_kwargs)
    megatron_path = tmp_path / "fp8_megatron"
    hf_export_path = tmp_path / "fp8_hf"

    # Step 1: quantize and save a Megatron checkpoint
    quantize_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        recipe="general/ptq/fp8_default-kv_fp8",
        tp_size=num_gpus,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=4,
        calib_batch_size=2,
        seq_length=16,
        export_megatron_path=megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (megatron_path / "latest_checkpointed_iteration.txt").exists()
    assert_has_modelopt_state(megatron_path)

    # Step 2: export to HF
    export_cmd = extend_cmd_parts(
        ["torchrun", "--nproc_per_node=1", "export_quantized_megatron_to_hf.py"],
        hf_model_name_or_path=hf_model_path,
        megatron_path=megatron_path,
        export_unified_hf_path=hf_export_path,
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    assert_exported_checkpoint_matches(hf_export_path, hf_model_path)

    # The exported unified checkpoint should be loadable and runnable by vLLM. The deployment check below
    # is disabled because it takes too long in CI (likely because of first run)
    #
    # import vllm
    # llm = vllm.LLM(
    #     model=str(hf_export_path),
    #     tensor_parallel_size=1,
    #     enforce_eager=True,
    #     gpu_memory_utilization=0.4,
    #     max_model_len=128,
    #     dtype="bfloat16",
    # )
    # outputs = llm.generate(["Hello!"], vllm.SamplingParams(max_tokens=4))
    # assert outputs and outputs[0].outputs and outputs[0].outputs[0].text
