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
"""vLLM fake-quant export through Megatron-Bridge matches ModelOpt's Megatron exporter."""

import os
from functools import partial
from pathlib import Path

import pytest
import torch
import yaml
from _test_utils.torch.transformers_models import create_tiny_nemotron_h_dir, create_tiny_qwen3_dir
from megatron.core.utils import unwrap_model
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq
from modelopt.torch.export.plugins.vllm_fakequant_hf import (
    QUANTIZER_RECIPE_FORMAT,
    QUANTIZER_RECIPE_VERSION,
)
from modelopt.torch.export.plugins.vllm_fakequant_megatron import (
    check_mcore_vllm_fakequant_quantizers,
)
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils.plugins.mbridge import load_mbridge_model_from_hf
from modelopt.torch.utils.plugins.megatron_generate import megatron_prefill


def _load_export(export_dir: Path):
    weights = {}
    for shard in sorted(export_dir.glob("*.safetensors")):
        weights.update(load_file(shard))
    state = torch.load(export_dir / "quantizer_state.pth", weights_only=True)
    recipe = yaml.safe_load((export_dir / "quant_recipe.yaml").read_text())
    assert recipe["format"] == QUANTIZER_RECIPE_FORMAT
    assert recipe["version"] == QUANTIZER_RECIPE_VERSION
    return weights, state, recipe["quantizers"]


QUANT_CFGS = {
    "nvfp4-kv_fp8": load_recipe("general/ptq/nvfp4_default-kv_fp8").quantize.model_dump(),
    # INT4 + FP8 weights: a SequentialQuantizer chain the export folds as a whole.
    "w4a8": {**mtq.W4A8_AWQ_BETA_CFG, "algorithm": "max"},
}


def _test_matches_megatron_exporter(source, quant_cfg_name, tmp_path, rank, size):
    # Models in a reused worker pick their attention backend through process-wide variables.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(name, None)
    bridge, _, model, unwrapped, _ = load_mbridge_model_from_hf(
        hf_model_name_or_path=str(source), provider_overrides={"pipeline_dtype": torch.bfloat16}
    )
    torch.manual_seed(0)
    input_ids = torch.randint(0, unwrapped.vocab_size, (2, 16), device="cuda")
    mtq.quantize(unwrapped, QUANT_CFGS[quant_cfg_name], lambda m: megatron_prefill(m, input_ids))
    reference_dir, export_dir = tmp_path / "megatron_exporter", tmp_path / "bridge"
    export_mcore_gpt_to_hf_vllm_fq(unwrapped, str(source), export_dir=str(reference_dir))
    for chunk in model:
        check_mcore_vllm_fakequant_quantizers(unwrap_model(chunk))
        mtq.fold_weight(unwrap_model(chunk))
    bridge.save_hf_pretrained(model, str(export_dir))
    if rank != 0:
        return

    ref_weights, ref_state, ref_recipe = _load_export(reference_dir)
    weights, state, quantizers = _load_export(export_dir)
    assert weights.keys() == ref_weights.keys()
    for name, weight in weights.items():
        torch.testing.assert_close(weight, ref_weights[name], rtol=0, atol=0, msg=name)

    enabled = {name for name, entry in ref_recipe.items() if not entry["_disabled"]}
    assert quantizers.keys() == enabled
    assert any(name.endswith("k_bmm_quantizer") for name in quantizers) == quant_cfg_name.endswith(
        "kv_fp8"
    )
    to_cfg = TensorQuantizer.attribute_config_from_modelopt_state
    for name, entry in quantizers.items():
        assert to_cfg(ref_recipe[name]).items() <= to_cfg(entry).items(), name

    assert state.keys() == {key for key in ref_state if key.rpartition(".")[0] in quantizers}
    for key, value in state.items():
        torch.testing.assert_close(value, ref_state[key], rtol=0, atol=0, msg=key)


@pytest.mark.parametrize(
    ("create_source", "quant_cfg_name"),
    [(create_tiny_nemotron_h_dir, "nvfp4-kv_fp8"), (create_tiny_qwen3_dir, "w4a8")],
)
def test_matches_megatron_exporter(dist_workers_size_1, tmp_path, create_source, quant_cfg_name):
    source = create_source(tmp_path, with_tokenizer=True)
    dist_workers_size_1.run(
        partial(_test_matches_megatron_exporter, source, quant_cfg_name, tmp_path)
    )
