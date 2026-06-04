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
"""Tests for quantize.py and export.py scripts."""

from pathlib import Path

from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir


def test_quantize_and_export(tmp_path: Path, num_gpus):
    """Quantize a tiny Qwen3 via a YAML recipe and export it to a unified HF checkpoint."""
    # Use a vLLM-friendly head_dim (64) since the default tiny config (head_dim=2) is unsupported.
    hf_model_path = create_tiny_qwen3_dir(
        tmp_path,
        with_tokenizer=True,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
    )
    megatron_path = tmp_path / "qwen3_fp8_megatron"
    hf_export_path = tmp_path / "qwen3_fp8_hf"

    # Step 1: quantize (tensor parallelism is supported here) and save a Megatron checkpoint. The
    # checkpoint must carry the ModelOpt state so it can be reloaded (for export or further QAT/QAD).
    quantize_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        recipe="general/ptq/fp8_default-kv_fp8",
        tp_size=num_gpus,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=4,
        calib_batch_size=1,
        seq_length=16,
        export_megatron_path=megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (megatron_path / "latest_checkpointed_iteration.txt").exists()
    assert list(megatron_path.rglob("modelopt_state")), (
        "Expected modelopt_state in the Megatron checkpoint"
    )

    # Step 2: export to HF (re-shards to TP=1) on a single rank. export.py reloads the quantized
    # Megatron checkpoint (restoring the ModelOpt quantizers) before converting to HF.
    export_cmd = extend_cmd_parts(
        ["torchrun", "--nproc_per_node=1", "export.py"],
        hf_model_name_or_path=hf_model_path,
        megatron_path=megatron_path,
        export_unified_hf_path=hf_export_path,
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)

    # HF (unified) quantized checkpoint exists with the exported quantization config + weights.
    # hf_quant_config.json is only written when the reloaded model is actually quantized, so its
    # presence also confirms export.py restored the ModelOpt quantizers from the checkpoint.
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    assert list(hf_export_path.glob("*.safetensors")), "Expected exported safetensors weights"

    # The exported unified checkpoint should be loadable and runnable by vLLM. The deployment check below
    # is disabled because it hangs in CI; to validate deployment locally in nemo container, uncomment it
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
