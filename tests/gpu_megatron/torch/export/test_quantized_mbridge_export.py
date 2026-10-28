# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Quantized HF export through Megatron-Bridge matches ModelOpt's Megatron exporter."""

import fnmatch
import json
import os
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch.export.unified_checkpoint import load_safetensors_dir
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_nemotron_h_dir,
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_dir,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf
from modelopt.torch.utils.plugins.mbridge import (
    get_language_model,
    load_mbridge_model_from_hf,
    use_moe_grouped_gemm,
)
from modelopt.torch.utils.plugins.megatron_generate import megatron_prefill

QUANT_CFGS = {
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
}


def _load(export_dir: Path):
    return load_safetensors_dir(export_dir), json.loads((export_dir / "config.json").read_text())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


def _excluded(patterns, layers):
    return {layer for layer in layers if any(fnmatch.fnmatch(layer, p) for p in patterns)}


def _quantize(source, quant_cfg_name, size):
    # Models in a reused worker pick their attention backend through process-wide variables.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(name, None)
    bridge, _, model, unwrapped, _ = load_mbridge_model_from_hf(
        hf_model_name_or_path=str(source),
        provider_overrides={"tensor_model_parallel_size": size, "pipeline_dtype": torch.bfloat16},
        moe_grouped_gemm=use_moe_grouped_gemm(str(source)),
    )
    language_model, is_vlm = get_language_model(unwrapped)
    quant_cfg = QUANT_CFGS[quant_cfg_name]
    if is_vlm:
        # As in quantize.py, the vision tower stays unquantized.
        disable_vision = {"quantizer_name": "vision_model.*", "enable": False}
        quant_cfg = {**quant_cfg, "quant_cfg": [*quant_cfg["quant_cfg"], disable_vision]}
    torch.manual_seed(0)
    input_ids = torch.randint(0, language_model.vocab_size, (2, 16), device="cuda")
    mtq.quantize(unwrapped, quant_cfg, lambda m: megatron_prefill(language_model, input_ids))
    return bridge, model, unwrapped


def _export_with_megatron_exporter(source, quant_cfg_name, export_dir, rank, size):
    _, _, unwrapped = _quantize(source, quant_cfg_name, size)
    export_mcore_gpt_to_hf(unwrapped, str(source), dtype=torch.bfloat16, export_dir=str(export_dir))


def _export_with_bridge(source, quant_cfg_name, export_dir, rank, size):
    bridge, model, _ = _quantize(source, quant_cfg_name, size)
    bridge.save_hf_pretrained_modelopt(model, str(export_dir))


@pytest.fixture
def bridge_workers(request):
    # TP=2 on multi-GPU hosts, so the bridge also gathers sharded weights and scales. TP stays <= 2,
    # the fewest KV heads of these models. A fixture starts the pool before the test timeout.
    return request.getfixturevalue(f"dist_workers_size_{min(torch.cuda.device_count(), 2)}")


@pytest.mark.parametrize(
    ("create_source", "quant_cfg_name"),
    [
        (partial(create_tiny_llama_dir, with_tokenizer=True), "nvfp4"),
        (partial(create_tiny_qwen3_dir, with_tokenizer=True), "fp8"),
        # The bridge can't export quantized Mamba or GatedDeltaNet projections yet.
        (partial(create_tiny_nemotron_h_dir, with_tokenizer=True), "nvfp4_mlp_only"),
        (partial(create_tiny_qwen3_5_moe_vl_dir, with_processor=True), "nvfp4_mlp_only"),
    ],
    ids=["llama-nvfp4", "qwen3-fp8", "nemotron_h-nvfp4_mlp_only", "qwen3_5_moe_vl-nvfp4_mlp_only"],
)
def test_matches_megatron_exporter(
    dist_workers_size_1, bridge_workers, tmp_path, create_source, quant_cfg_name
):
    source = create_source(tmp_path)
    reference_dir, export_dir = tmp_path / "megatron_exporter", tmp_path / "bridge"
    # The Megatron exporter needs TP=1.
    dist_workers_size_1.run(
        partial(_export_with_megatron_exporter, source, quant_cfg_name, reference_dir)
    )
    bridge_workers.run(partial(_export_with_bridge, source, quant_cfg_name, export_dir))

    ref_weights, ref_config = _load(reference_dir)
    weights, config = _load(export_dir)
    assert weights.keys() == ref_weights.keys()
    for name, weight in weights.items():
        reference = ref_weights[name]
        # The Megatron exporter keeps per-tensor scales 1-d; both hold the same scalar.
        if reference.numel() == weight.numel() == 1:
            reference = reference.reshape(weight.shape)
        assert weight.shape == reference.shape, name
        assert weight.dtype == reference.dtype, name
        if name.endswith("input_scale") and torch.cuda.device_count() > 1:
            # Activations reduced across TP ranks calibrate to slightly different amaxes.
            torch.testing.assert_close(weight, reference, rtol=2e-2, atol=0, msg=name)
        else:
            assert torch.equal(_bits(weight), _bits(reference)), name

    quant, ref_quant = config.pop("quantization_config"), ref_config.pop("quantization_config")
    assert config == ref_config
    assert {k: v for k, v in quant.items() if k != "ignore"} == {
        k: v for k, v in ref_quant.items() if k != "ignore"
    }
    # The bridge must exclude exactly the unquantized linear layers. The Megatron exporter's
    # patterns spell them differently and miss some, e.g. Qwen3.5's mlp.shared_expert_gate.
    layers = {
        name.rpartition(".")[0]
        for name, weight in weights.items()
        if name.endswith(".weight") and weight.dim() == 2
    }
    unquantized = {layer for layer in layers if f"{layer}.weight_scale" not in weights}
    assert _excluded(quant["ignore"], layers) == unquantized
    assert _excluded(ref_quant["ignore"], layers) <= unquantized
