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


def _make_cpu_offloaded_model(tmp_path, num_hidden_layers=3, tiny_llama_dir=None):
    """Tiny LLaMA with first decoder layer offloaded to CPU, rest on GPU."""
    if tiny_llama_dir is None:
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


def _read_shards(export_dir):
    """Map every exported tensor key to its (shape, dtype), unioned across shards."""
    tensors = {}
    for st_file in sorted(export_dir.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt") as st:
            for key in st.keys():  # noqa: SIM118
                t = st.get_tensor(key)
                assert key not in tensors, f"Duplicate key '{key}' across shards"
                tensors[key] = (tuple(t.shape), t.dtype)

    index_path = export_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        assert set(weight_map) == set(tensors), "index weight_map disagrees with shard contents"
        for shard_name in set(weight_map.values()):
            assert (export_dir / shard_name).exists(), f"Indexed shard '{shard_name}' missing"
    return tensors


@pytest.mark.parametrize("max_shard_size", ["10GB", "5KB"])
def test_streaming_export_matches_batch_export(tmp_path, max_shard_size):
    """The offloaded streaming path must emit the same tensors as the resident batch path.

    Every other assertion in this file iterates only the keys that were written, so none
    of them can see a tensor going *missing* -- which is how the dropped ``mtp.*`` tensors
    were found. The streaming exporter reimplements much of ``_export_transformers_checkpoint``
    plus ``postprocess_state_dict`` per tensor, so the two can drift silently.

    Both models load the same checkpoint, so key/shape/dtype parity is exact. Values are
    not compared: layer 0 calibrates on CPU in the offloaded model and on GPU in the
    resident one, and that alone can shift an amax in the last bits.

    The ``5KB`` case additionally drives multi-shard index generation through the real
    export path (``_StreamingShardWriter`` is otherwise only tested in isolation). The
    whole tiny export is ~28KB, so anything larger stays single-shard and would silently
    skip that coverage.
    """
    llama_dir = create_tiny_llama_dir(tmp_path / "src", num_hidden_layers=3)

    def forward_loop(m):
        ids = torch.randint(0, m.config.vocab_size, (1, 32)).cuda()
        with torch.no_grad():
            m(ids)

    def _export(model, subdir):
        model.eval()
        mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
        export_dir = tmp_path / subdir
        export_dir.mkdir()
        export_hf_checkpoint(model, export_dir=str(export_dir), max_shard_size=max_shard_size)
        return _read_shards(export_dir)

    offloaded, _cfg, _dir = _make_cpu_offloaded_model(
        tmp_path / "offloaded", tiny_llama_dir=llama_dir
    )
    offloaded_tensors = _export(offloaded, "export_offloaded")

    resident = AutoModelForCausalLM.from_pretrained(llama_dir).cuda()
    resident_tensors = _export(resident, "export_resident")

    missing = set(resident_tensors) - set(offloaded_tensors)
    extra = set(offloaded_tensors) - set(resident_tensors)
    assert not missing, f"Streaming export dropped {len(missing)} tensor(s): {sorted(missing)}"
    assert not extra, f"Streaming export emitted {len(extra)} unexpected tensor(s): {sorted(extra)}"

    mismatched = {
        k: (resident_tensors[k], offloaded_tensors[k])
        for k in resident_tensors
        if resident_tensors[k] != offloaded_tensors[k]
    }
    assert not mismatched, f"shape/dtype drift between export paths: {mismatched}"
