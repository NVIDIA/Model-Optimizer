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

"""Exercise native V4 naming and real IQ2 packing without requiring CUDA/MCore."""

import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.plugins.vllm_fakequant_megatron import VllmFqGPTModelExporter
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight


def _exporter(folded=False, source_keys=()):
    cls = VllmFqGPTModelExporter if folded else GPTModelExporter
    exporter = cls.__new__(cls)
    exporter.dtype = torch.bfloat16
    exporter.moe_router_dtype = None
    exporter.arch = "DeepseekV4ForCausalLM"
    exporter._state_dict = OrderedDict()
    exporter.exclude_modules = []
    exporter.layer_config_dict = {}
    exporter._deepseek_v4_source_keys = set(source_keys)
    exporter.rules = exporter._populate_rule_book()[exporter.arch]
    if folded:
        exporter._quantizer_recipe_markers = []
        exporter._quantizer_validation_failure = ""
        exporter._packing_experts = False
        exporter._grouped_fq_modules = set()
    return exporter


def _compressor():
    module = nn.Module()
    module.linear_wkv = nn.Linear(256, 256, bias=False).bfloat16()
    module.linear_wgate = nn.Linear(256, 256, bias=False).bfloat16()
    module.norm = nn.LayerNorm(256, bias=False).bfloat16()
    module.ape = nn.Parameter(torch.randn(4, 256))
    return module


def _attention(compression):
    attention = nn.Module()
    for name in ("linear_q_down_proj", "linear_q_up_proj", "linear_kv_proj", "linear_proj"):
        setattr(attention, name, nn.Linear(256, 256, bias=False).bfloat16())
    attention.q_layernorm = nn.LayerNorm(256, bias=False).bfloat16()
    attention.kv_layernorm = nn.LayerNorm(256, bias=False).bfloat16()
    attention.linear_o_group_proj = nn.Parameter(torch.randn(2, 256, 256).bfloat16())
    attention.core_attention = nn.Module()
    core = attention.core_attention
    core.attn_sink = nn.Parameter(torch.randn(2))
    if compression:
        core.compressor = _compressor()
    if compression == "csa":
        core.indexer = nn.Module()
        core.indexer.linear_wq_b = nn.Linear(256, 256, bias=False).bfloat16()
        core.indexer.linear_weights_proj = nn.Linear(256, 2, bias=False).bfloat16()
        core.indexer.compressor = _compressor()
    return attention


@pytest.mark.parametrize("compression", [None, "csa", "hca"])
@pytest.mark.parametrize("legacy", [False, True])
def test_v4_attention_exports_compressors_and_optional_indexer(compression, legacy):
    prefix = "layers.3.attn."
    scorer = prefix + (
        "indexer.weights_proj.weight" if legacy else "indexer.scorer.weights_proj.weight"
    )
    exporter = _exporter(source_keys=[scorer])
    attention = _attention(compression)
    exporter._get_deepseek_v4_attention_state_dict(attention, 3)
    state = exporter._state_dict
    assert len(state) == {None: 8, "hca": 12, "csa": 18}[compression]
    assert torch.equal(state[prefix + "wo_a.weight"], attention.linear_o_group_proj)
    assert state[prefix + "attn_sink"].dtype == torch.float32
    if compression:
        assert state[prefix + "compressor.ape"].dtype == torch.float32
    if compression == "csa":
        assert scorer in state
        assert state[scorer].dtype == torch.bfloat16


@pytest.mark.parametrize("source_bias", [False, True])
def test_v4_router_preserves_integer_hash_table_and_source_bias_policy(source_bias):
    prefix = "layers.0.ffn.gate."
    exporter = _exporter(source_keys=[prefix + "bias"] if source_bias else [])
    router = nn.Linear(256, 4, bias=False).bfloat16()
    router.register_buffer("expert_bias", torch.arange(4, dtype=torch.float32))
    router.register_buffer("tid2eid", torch.tensor([[0, 1], [2, 3]], dtype=torch.int32))
    exporter._get_deepseek_v4_router_state_dict(router, 0)
    assert (prefix + "bias" in exporter._state_dict) == source_bias
    assert torch.equal(exporter._state_dict[prefix + "tid2eid"], router.tid2eid)
    assert exporter._state_dict[prefix + "tid2eid"].dtype == torch.int32


@pytest.mark.parametrize("folded", [False, True])
def test_v4_expert_export_splits_gate_up_and_reuses_packed_codes(folded):
    torch.manual_seed(6105)
    module = nn.Linear(256, 512, bias=False).bfloat16()
    module.config = SimpleNamespace(ffn_hidden_size=256)
    mtq.quantize(
        module,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_xxs", "backend": "ggml", "block_sizes": {-1: 256}},
                },
            ],
            "algorithm": None,
        },
    )
    expected = module.weight_quantizer(module.weight).detach()
    fmt = GGML_FORMAT_REGISTRY["iq2_xxs"]
    packed = fmt.pack(module.weight, module.weight_quantizer).reshape(512, 1, 66)
    exporter = _exporter(folded=folded)
    exporter.rules["local_experts.linear_fc1"](module, 2, 17)
    state = exporter._state_dict
    if folded:
        recipe = exporter._extract_quantizer_recipe_markers({0: state})
        assert len(recipe) == 6  # input/weight/output quantizers for each split projection
        assert not exporter._quantizer_validation_failure
    assert set(state) == {"layers.2.ffn.experts.17.w1.weight", "layers.2.ffn.experts.17.w3.weight"}
    for key, start in (("w1", 0), ("w3", 256)):
        weight = state[f"layers.2.ffn.experts.17.{key}.weight"]
        if folded:
            assert weight.dtype == torch.bfloat16
            assert torch.equal(weight, expected[start : start + 256])
        else:
            assert weight.dtype == torch.uint8
            assert torch.equal(weight.reshape(256, 1, 66), packed[start : start + 256])


@pytest.mark.parametrize("folded", [False, True])
def test_v4_expert_export_matches_forward_when_only_up_weights_change(folded):
    """An unchanged gate must follow the fused layer's updated quantization decision."""
    torch.manual_seed(41)
    fmt = GGML_FORMAT_REGISTRY["iq2_xxs"]
    packed, shape = fmt.quantize(torch.randn(4, 256, dtype=torch.bfloat16))
    module = nn.Linear(256, 4, bias=False).bfloat16()
    module.config = SimpleNamespace(ffn_hidden_size=2)
    mtq.quantize(
        module,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_xxs", "backend": "ggml", "block_sizes": {-1: 256}},
                },
            ],
            "algorithm": None,
        },
    )
    with torch.no_grad():
        module.weight.copy_(fmt.dequantize(packed, shape, dtype=torch.bfloat16))
        module.weight[2:].add_(0.125)
    pin_packed_weight(module.weight_quantizer, "iq2_xxs", packed)
    expected = module.weight_quantizer(module.weight).detach()
    exporter = _exporter(folded=folded)
    exporter.rules["local_experts.linear_fc1"](module, 0, 0)
    weights = [
        exporter._state_dict[f"layers.0.ffn.experts.0.{part}.weight"] for part in ("w1", "w3")
    ]
    if not folded:
        weights = [
            fmt.dequantize(value, torch.tensor([2, 256]), dtype=torch.bfloat16) for value in weights
        ]
    assert torch.equal(torch.cat(weights), expected)


@pytest.mark.parametrize("deepseek_v4", [False, True])
def test_mhc_export_preserves_v4_fp32_projection_and_existing_dtype_policy(deepseek_v4):
    exporter = _exporter()
    exporter.rules["deepseek_v4"] = deepseek_v4
    projection = torch.tensor([[1.000001, 0.1234567]], dtype=torch.float32)
    layer = SimpleNamespace(
        inner_layer=SimpleNamespace(),
        hyper_connection=SimpleNamespace(
            mapping_proj=SimpleNamespace(weight=projection),
            bias=torch.tensor([0.1234567]),
            alpha_pre=torch.tensor([1.0]),
            alpha_post=torch.tensor([2.0]),
            alpha_res=torch.tensor([3.0]),
        ),
    )
    exporter._get_hyper_connection_state_dict(layer, 0)
    state = exporter._state_dict
    expected = projection if deepseek_v4 else projection.bfloat16()
    assert state["layers.0.hc_ffn_fn"].dtype == expected.dtype
    assert torch.equal(state["layers.0.hc_ffn_fn"], expected)
    assert state["layers.0.hc_ffn_base"].dtype == torch.float32
    assert state["layers.0.hc_ffn_scale"].dtype == torch.float32


def test_v4_export_preserves_fp32_head_controls():
    exporter = _exporter()
    controls = {
        name: torch.tensor([[1.000001, 0.1234567]], dtype=torch.float32)
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
    }
    exporter.model = SimpleNamespace(
        pre_process=False,
        post_process=True,
        decoder=SimpleNamespace(layers=[], **controls),
    )
    exporter._get_state_dict()
    for name, expected in controls.items():
        assert exporter._state_dict[name].dtype == torch.float32
        assert torch.equal(exporter._state_dict[name], expected)


def test_v4_source_coverage_checks_root_layer_names(tmp_path):
    source = tmp_path / "source"
    exported = tmp_path / "exported"
    source.mkdir()
    exported.mkdir()
    missing = "layers.0.attn.indexer.compressor.wkv.weight"
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {missing: "x.safetensors"}})
    )
    (exported / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.attn.wkv.weight": "x.safetensors"}})
    )
    exporter = _exporter()
    exporter._hf_text_config = SimpleNamespace(num_hidden_layers=1)
    exporter._src_num_hidden_layers = 1
    with pytest.raises(RuntimeError, match="Export dropped 1 tensor"):
        exporter._verify_exported_keys(exported, source)


@pytest.mark.parametrize(
    "missing", ["hc_head_scale", "layers.0.hc_attn_base", "layers.0.ffn.gate.tid2eid"]
)
def test_v4_source_coverage_checks_raw_parameters(tmp_path, missing):
    source = tmp_path / "source"
    exported = tmp_path / "exported"
    source.mkdir()
    exported.mkdir()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {missing: "x.safetensors"}})
    )
    (exported / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.attn.wkv.weight": "x.safetensors"}})
    )
    exporter = _exporter()
    with pytest.raises(RuntimeError, match="Export dropped V4 parameter"):
        exporter._verify_exported_keys(exported, source)


@pytest.mark.parametrize(("num_layers", "mtp"), [(3, 0), (4, 1)])
def test_v4_export_rejects_unpaired_layers_and_mtp(num_layers, mtp):
    exporter = _exporter()
    exporter.model = SimpleNamespace(
        config=SimpleNamespace(num_layers=num_layers, mtp_num_layers=mtp)
    )
    exporter._hf_text_config = SimpleNamespace(num_nextn_predict_layers=mtp)
    with pytest.raises((ValueError, NotImplementedError), match=r"paired|main-only"):
        exporter._init_deepseek_v4_export()
