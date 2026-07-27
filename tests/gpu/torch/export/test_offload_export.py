# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU integration tests for offload-aware unified HF export.

Tests the full round-trip:
  tiny LLaMA (CPU-offloaded via accelerate)
  → FP8 layerwise calibration (calib_mutates_weights=False)
  → export_hf_checkpoint
  → assert no meta tensors in output safetensors
  → assert hf_quant_config.json present with fp8 format
"""

import copy
import json

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def _make_cpu_offloaded_model(tmp_path, num_hidden_layers=3):
    """Tiny LLaMA with first decoder layer offloaded to CPU, rest on GPU."""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_hidden_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # First layer on CPU to exercise the offload path; lm_head / embed on GPU.
    device_map = {}
    for n, _m in model.named_modules():
        if "layers" not in n or n.split("layers.")[-1].isdigit():
            device_map[n] = 0
    device_map["model.layers.0"] = "cpu"

    model = load_checkpoint_and_dispatch(model, tiny_llama_dir, device_map=device_map)
    return model, config, tiny_llama_dir


def _layerwise_fp8_cfg():
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    algo = cfg.get("algorithm", "max")
    method = algo if isinstance(algo, str) else algo.get("method", "max")
    # calib_mutates_weights is a field of LayerwiseConfig (nested), not of the algorithm.
    cfg["algorithm"] = {"method": method, "layerwise": {"calib_mutates_weights": False}}
    return cfg


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG, _layerwise_fp8_cfg()])
def test_export_hf_checkpoint_cpu_offloaded(tmp_path, quant_cfg):
    """export_hf_checkpoint must succeed on a CPU-offloaded model and produce valid weights.

    Regression guard against the pre-fix bug where remove_hook_from_module was called
    before weight materialization, causing meta tensors to be serialized as empty safetensors.
    """
    num_hidden_layers = 3
    model, _config, _llama_dir = _make_cpu_offloaded_model(
        tmp_path / "offloaded", num_hidden_layers=num_hidden_layers
    )
    model.eval()

    def forward_loop(m):
        ids = torch.randint(0, m.config.vocab_size, (1, 32)).cuda()
        with torch.no_grad():
            m(ids)

    model = mtq.quantize(model, quant_cfg, forward_loop)

    export_dir = tmp_path / "hf_export"
    export_dir.mkdir()
    export_hf_checkpoint(model, export_dir=str(export_dir))

    # --- Assertions ---

    # 1. hf_quant_config.json must exist and declare fp8
    quant_config_path = export_dir / "hf_quant_config.json"
    assert quant_config_path.exists(), "hf_quant_config.json not written"
    with open(quant_config_path) as f:
        quant_config = json.load(f)
    assert quant_config["quantization"]["quant_algo"] == "FP8", (
        f"Expected FP8, got {quant_config['quantization'].get('quant_algo')}"
    )

    # 2. All tensors in safetensors shards must be non-empty (no meta serialized as zeros)
    safetensor_files = list(export_dir.glob("*.safetensors"))
    assert safetensor_files, "No safetensors files written"

    for st_file in safetensor_files:
        with safe_open(str(st_file), framework="pt") as st:
            for key in list(st.keys()):
                tensor = st.get_tensor(key)
                assert tensor.numel() > 0, f"Zero-numel tensor for key '{key}' in {st_file.name}"
                assert not tensor.is_meta, f"Meta tensor for key '{key}' in {st_file.name}"
                # Weight tensors (not scales) must have non-zero norm — guards against all-zeros
                # from meta serialization
                if "weight" in key and "scale" not in key and "quantizer" not in key:
                    assert tensor.float().abs().sum() > 0, (
                        f"All-zero weight tensor '{key}' in {st_file.name} — "
                        "possible meta tensor serialization bug"
                    )
