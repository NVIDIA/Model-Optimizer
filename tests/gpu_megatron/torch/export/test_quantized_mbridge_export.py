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
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_qwen3_dir,
    create_tiny_qwen3_moe_dir,
)
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf
from modelopt.torch.utils.plugins.mbridge import load_mbridge_model_from_hf, use_moe_grouped_gemm
from modelopt.torch.utils.plugins.megatron_generate import megatron_prefill

QUANT_CFGS = {
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "w4a16_nvfp4": mtq.W4A16_NVFP4_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
}


def _load(export_dir: Path):
    weights = {}
    for shard in sorted(export_dir.glob("*.safetensors")):
        weights.update(load_file(shard))
    return weights, json.loads((export_dir / "config.json").read_text())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


def _excluded(patterns, layers):
    return {layer for layer in layers if any(fnmatch.fnmatch(layer, p) for p in patterns)}


def _test_matches_megatron_exporter(source, quant_cfg_name, tmp_path, rank, size):
    # Models in a reused worker pick their attention backend through process-wide variables.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(name, None)
    bridge, _, model, unwrapped, _ = load_mbridge_model_from_hf(
        hf_model_name_or_path=str(source),
        provider_overrides={"pipeline_dtype": torch.bfloat16},
        moe_grouped_gemm=use_moe_grouped_gemm(str(source)),
    )
    torch.manual_seed(0)
    input_ids = torch.randint(0, unwrapped.vocab_size, (2, 16), device="cuda")
    mtq.quantize(unwrapped, QUANT_CFGS[quant_cfg_name], lambda m: megatron_prefill(m, input_ids))
    reference_dir, export_dir = tmp_path / "megatron_exporter", tmp_path / "bridge"
    export_mcore_gpt_to_hf(
        unwrapped, str(source), dtype=torch.bfloat16, export_dir=str(reference_dir)
    )
    bridge.save_hf_pretrained_modelopt(model, str(export_dir))
    if rank != 0:
        return

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
        assert torch.equal(_bits(weight), _bits(reference)), name

    quant, ref_quant = config.pop("quantization_config"), ref_config.pop("quantization_config")
    # The Megatron exporter also writes head_dim where the source config leaves it implied.
    if "head_dim" not in config:
        implied = config["hidden_size"] // config["num_attention_heads"]
        assert ref_config.pop("head_dim", implied) == implied
    assert config == ref_config
    assert {k: v for k, v in quant.items() if k != "ignore"} == {
        k: v for k, v in ref_quant.items() if k != "ignore"
    }
    # The exclusion patterns differ in spelling but must exclude the same linear layers.
    layers = {
        name.rpartition(".")[0]
        for name, weight in weights.items()
        if name.endswith(".weight") and weight.dim() == 2
    }
    assert _excluded(quant["ignore"], layers) == _excluded(ref_quant["ignore"], layers)


@pytest.mark.parametrize(
    ("create_source", "quant_cfg_name"),
    [
        (create_tiny_llama_dir, "nvfp4"),
        (create_tiny_qwen3_dir, "fp8"),
        (create_tiny_qwen3_dir, "w4a16_nvfp4"),
        (create_tiny_qwen3_moe_dir, "nvfp4_mlp_only"),
    ],
)
def test_matches_megatron_exporter(dist_workers_size_1, tmp_path, create_source, quant_cfg_name):
    source = create_source(tmp_path, with_tokenizer=True)
    dist_workers_size_1.run(
        partial(_test_matches_megatron_exporter, source, quant_cfg_name, tmp_path)
    )
