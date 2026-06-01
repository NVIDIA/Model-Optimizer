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

import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir

# Standalone helper (run under torchrun) that reloads a quantized Megatron checkpoint and asserts
# the ModelOpt quantizers were restored.
_MEGATRON_LOADER = Path(__file__).parent / "_load_megatron_quantized.py"


def test_quantize_export_and_vllm_deployment(tmp_path: Path, num_gpus):
    """Quantize a tiny Qwen3 via a YAML recipe, export to HF with export.py, and load it with vLLM."""
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

    # Step 1: quantize (tensor parallelism is supported here) and save a Megatron checkpoint.
    quantize_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        recipe="general/ptq/fp8_default-kv_fp8",
        tp_size=num_gpus,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=16,
        calib_batch_size=1,
        seq_length=32,
        export_path=megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (megatron_path / "latest_checkpointed_iteration.txt").exists()

    # Step 2: export to HF (re-shards to TP=1) on a single rank.
    export_cmd = extend_cmd_parts(
        ["torchrun", "--nproc_per_node=1", "export.py"],
        hf_model_name_or_path=hf_model_path,
        megatron_path=megatron_path,
        export_path=hf_export_path,
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)

    # HF (unified) quantized checkpoint exists with the exported quantization config + weights.
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    assert list(hf_export_path.glob("*.safetensors")), "Expected exported safetensors weights"

    # The exported unified checkpoint should be loadable and runnable by vLLM. Skip only this
    # deployment step (the quantization + export above is already validated) if vLLM is absent.
    vllm = pytest.importorskip("vllm")
    llm = vllm.LLM(
        model=str(hf_export_path),
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=128,
        dtype="bfloat16",
    )
    outputs = llm.generate(["Hello!"], vllm.SamplingParams(max_tokens=4))
    assert outputs and outputs[0].outputs[0].text is not None


def test_quantize_megatron_checkpoint_reload(tmp_path: Path, num_gpus):
    """Quantize a tiny Qwen3 to FP8, save in Megatron format, and reload to check quantizers."""
    hf_model_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    megatron_path = tmp_path / "qwen3_fp8_megatron"

    cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        quant_cfg="fp8",
        tp_size=num_gpus,
        pp_size=1,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=16,
        calib_batch_size=1,
        seq_length=32,
        export_path=megatron_path,
    )
    run_example_command(cmd_parts, example_path="megatron_bridge", setup_free_port=True)

    # Megatron checkpoint exists and carries the quantization (modelopt) state.
    assert (megatron_path / "latest_checkpointed_iteration.txt").exists()
    assert list(megatron_path.rglob("modelopt_state")), (
        "Expected modelopt_state in the Megatron checkpoint"
    )

    # Reload the checkpoint via the bridge and assert the quantizers were restored. The loader
    # asserts internally, so a non-zero exit (raised by run_example_command) fails this test.
    run_example_command(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(_MEGATRON_LOADER),
            str(hf_model_path),
            str(megatron_path),
            str(num_gpus),
        ],
        example_path="megatron_bridge",
        setup_free_port=True,
    )
