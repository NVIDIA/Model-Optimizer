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
import torch
import yaml
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.export.unified_checkpoint import (
    assert_exported_checkpoint_matches,
    load_safetensors_dir,
)
from _test_utils.torch.megatron.modelopt_state import assert_has_modelopt_state
from _test_utils.torch.transformers_models import (
    create_tiny_nemotron_h_dir,
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
    (
        "create_model",
        "model_kwargs",
        "recipe",
        "quantize_parallelism",
        "export_parallelism",
        "fakequant_export_parallelism",
    ),
    [
        # MoE: routed experts used to be dropped silently from the export. Calibrates with context
        # parallelism (not on the VLM: M-RoPE CP needs nemo:26.10).
        (
            create_tiny_qwen3_moe_dir,
            _DENSE_KWARGS,
            "general/ptq/nvfp4_default-kv_fp8",
            "cp_size",
            "pp_size",
            # Per-expert HF names, each EP rank owning some of them.
            "ep_size",
        ),
        # Dense VLM: only the language model is quantized, vision is copied through. Keeps the
        # FP8 script-to-checkpoint coverage.
        (
            create_tiny_qwen3vl_dir,
            {},
            "general/ptq/fp8_default-kv_fp8",
            "tp_size",
            "pp_size",
            "tp_size",
        ),
        # Mamba hybrid + MoE with grouped-GEMM experts: exported with its experts sharded
        # across ranks (EP), resharded from the TP-quantized checkpoint.
        (
            create_tiny_nemotron_h_dir,
            {},
            "general/ptq/nvfp4_default-kv_fp8",
            "tp_size",
            "ep_size",
            "pp_size",
        ),
    ],
    ids=["qwen3_moe", "qwen3vl", "nemotron_h"],
)
@pytest.mark.timeout(420)  # quantize + 2 exports in one test; 1-gpu CI exceeds the default 300s
def test_quantize_and_export(
    tmp_path: Path,
    num_gpus,
    create_model,
    model_kwargs,
    recipe,
    quantize_parallelism,
    export_parallelism,
    fakequant_export_parallelism,
    fake_hub_checkpoint,
):
    """Quantize a tiny model via a YAML recipe and export it to a unified HF checkpoint.

    The model is passed as a Hub ID (see ``fake_hub_checkpoint``): both steps load and export from
    its local copy, and the Megatron checkpoint records the Hub ID as its tokenizer.
    """
    hf_model_path = create_model(tmp_path, with_tokenizer=True, **model_kwargs)
    model_arg = fake_hub_checkpoint.serve(
        hf_model_path, "quantize.py", "export_quantized_megatron_to_hf.py", nproc=num_gpus
    )
    megatron_path = tmp_path / "quantized_megatron"
    hf_export_path = tmp_path / "quantized_hf"

    # Step 1: quantize and save a Megatron checkpoint
    quantize_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=model_arg,
        recipe=recipe,
        **{quantize_parallelism: num_gpus},
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=4,
        calib_batch_size=2,
        seq_length=16,
        export_megatron_path=megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (megatron_path / "latest_checkpointed_iteration.txt").exists()
    assert_has_modelopt_state(megatron_path)
    if fake_hub_checkpoint.active:
        assert (
            fake_hub_checkpoint.saved_tokenizer_model(megatron_path)
            == fake_hub_checkpoint.hub_model_id
        )

    # Step 2: export to HF
    export_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "export_quantized_megatron_to_hf.py"],
        hf_model_name_or_path=model_arg,
        megatron_path=megatron_path,
        export_unified_hf_path=hf_export_path,
        **{export_parallelism: num_gpus},
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    assert_exported_checkpoint_matches(hf_export_path, hf_model_path)
    if fake_hub_checkpoint.active:
        assert (hf_export_path / fake_hub_checkpoint.marker).exists()

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

    # Step 3: export folded weights + quantizer sidecars through Megatron-Bridge for vLLM fake quant
    fakequant_path = tmp_path / "vllm_fakequant"
    fakequant_cmd = extend_cmd_parts(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "export_quantized_megatron_to_hf.py",
            "--vllm_fakequant_export",
        ],
        hf_model_name_or_path=model_arg,
        megatron_path=megatron_path,
        export_unified_hf_path=fakequant_path,
        **{fakequant_export_parallelism: num_gpus},
    )
    run_example_command(fakequant_cmd, example_path="megatron_bridge", setup_free_port=True)
    is_vlm = "vision_config" in (hf_model_path / "config.json").read_text()
    assert_exported_checkpoint_matches(
        fakequant_path,
        hf_model_path,
        check_values=False,
        bit_exact_prefixes=("model.visual.",) if is_vlm else (),
    )
    assert not any("quantizer" in name for name in load_safetensors_dir(fakequant_path))
    recipe = yaml.safe_load((fakequant_path / "quant_recipe.yaml").read_text())
    quantizers = recipe["quantizers"]
    state = torch.load(fakequant_path / "quantizer_state.pth", weights_only=True)
    assert any(name.endswith("k_bmm_quantizer") for name in quantizers)
    assert {key.rpartition(".")[0] for key in state} <= quantizers.keys()
