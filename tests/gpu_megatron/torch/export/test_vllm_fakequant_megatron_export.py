# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial

import pytest
import torch
import yaml
from _test_utils.torch.megatron.models import get_mcore_gpt_model, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import run_mcore_inference
from _test_utils.torch.transformers_models import create_tiny_llama_dir, create_tiny_nemotron_h_dir
from megatron.core.parallel_state import is_pipeline_last_stage
from safetensors import safe_open

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq
from modelopt.torch.export.plugins.vllm_fakequant_megatron import (
    VllmFqGPTModelExporter,
    gather_mcore_vllm_fq_quantized_state_dict,
    gather_mcore_vllm_fq_quantizer_recipe,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@contextmanager
def _assert_weight_qdq_once(model, prefix=""):
    quantizers = [
        module
        for name, module in model.named_modules()
        if name.startswith(prefix)
        and name.endswith("weight_quantizer")
        and isinstance(module, TensorQuantizer)
        and module.is_enabled
    ]
    assert quantizers or (prefix and not is_pipeline_last_stage())
    calls = Counter()

    def count_qdq(module, args, output):
        calls[module] += 1

    handles = [quantizer.register_forward_hook(count_qdq) for quantizer in quantizers]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
    assert all(calls[quantizer] == 1 for quantizer in quantizers), calls


def _assert_exported_quantizers(export_dir, expected_names):
    state = torch.load(export_dir / "quantizer_state.pth", weights_only=True, map_location="cpu")
    recipe = yaml.safe_load((export_dir / "quant_recipe.yaml").read_text())
    assert expected_names <= recipe.keys()
    assert {name + "._amax" for name in expected_names} <= state.keys()
    for name in expected_names:
        assert not recipe[name]["_disabled"]
        amax = state[name + "._amax"]
        assert amax.dtype == torch.float32
        torch.testing.assert_close(amax, torch.full_like(amax, 1.001), rtol=0, atol=0)
    assert {key.rsplit(".", 1)[0] for key in state} <= recipe.keys()
    assert not any(key.endswith("._quant_recipe_marker") for key in state)
    assert not any(key.endswith("._quant_recipe_marker") for key in recipe)
    assert not (export_dir / "hf_quant_config.json").exists()
    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(weight_map.values()):
        with safe_open(export_dir / shard, framework="pt") as f:
            shard_keys = f.keys()
            assert not any(
                "quantizer" in key or "._quant_recipe_marker" in key for key in shard_keys
            )
    return state, recipe, weight_map


def _test_mcore_vllm_export(tmp_path, quant_cfg, rank, size, prebuild=False):
    """Test megatron-core model export for vLLM with fake quantization."""
    # Create a tiny mcore GPT model
    num_layers = 2
    hidden_size = 64
    num_attention_heads = 8
    num_query_groups = 1
    ffn_hidden_size = 128
    max_sequence_length = 32
    vocab_size = 64

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="swiglu",
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()
    model.eval()

    # Quantize the model
    def forward_loop(model):
        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            run_mcore_inference(model, input_ids)

    model = mtq.quantize(model, quant_cfg, forward_loop)
    # Preserve calibration precision even when the exported weights are BF16.
    for name, quantizer in model.named_modules():
        if isinstance(quantizer, TensorQuantizer) and "input_quantizer" in name:
            if getattr(quantizer, "_amax", None) is not None:
                quantizer.float()
                quantizer.amax = torch.full_like(quantizer.amax, 1.001)
    # Create HF config for export
    pretrained_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_size,
        "max_position_embeddings": max_sequence_length,
        "model_type": "llama",
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_query_groups,
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
    }

    stale_quantizer = "stale_source.input_quantizer"
    if rank == 0:
        with open(tmp_path / "config.json", "w") as f:
            json.dump(pretrained_config, f)
        torch.save(
            {stale_quantizer + "._amax": torch.tensor(42.0)}, tmp_path / "quantizer_state.pth"
        )
        with open(tmp_path / "quant_recipe.yaml", "w") as f:
            yaml.safe_dump({stale_quantizer: {"_disabled": True}}, f)
    torch.distributed.barrier()

    export_dir = tmp_path / "vllm_export"
    with _assert_weight_qdq_once(model):
        if prebuild:
            exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
            _ = exporter.state_dict
            assert exporter.layer_state_dicts
            exporter.save_pretrained(str(export_dir), tmp_path)
        else:
            export_mcore_gpt_to_hf_vllm_fq(
                model,
                pretrained_model_name_or_path=tmp_path,
                dtype=torch.bfloat16,
                export_dir=str(export_dir),
            )

    expected_names = {
        *(
            f"model.layers.{i}.self_attn.{proj}_proj.input_quantizer"
            for i in range(num_layers)
            for proj in ("q", "k", "v", "o")
        ),
        *(
            f"model.layers.{i}.mlp.{proj}_proj.input_quantizer"
            for i in range(num_layers)
            for proj in ("gate", "up", "down")
        ),
    }
    state, recipe, weight_map = _assert_exported_quantizers(export_dir, expected_names)
    assert stale_quantizer + "._amax" not in state
    assert stale_quantizer not in recipe
    assert {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        *(f"model.layers.{i}.self_attn.q_proj.weight" for i in range(num_layers)),
    } <= weight_map.keys()


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("prebuild", [False, True], ids=["direct", "cached"])
def test_mcore_vllm_export(request, tmp_path, quant_cfg, pp_size, prebuild):
    """Preserve fresh sidecars and weights across PP stages, including cached shard access."""
    workers = request.getfixturevalue(f"dist_workers_size_{pp_size}")
    workers.run(partial(_test_mcore_vllm_export, tmp_path, quant_cfg, prebuild=prebuild))


def _test_mcore_vllm_export_mtp(tmp_path, rank, size):
    model = get_mcore_hybrid_model(
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=4,
        hybrid_layer_pattern="M*EE/*E",
        num_query_groups=4,
        max_sequence_length=32,
        vocab_size=32,
        mamba_num_heads=8,
        num_moe_experts=4,
        normalization="RMSNorm",
        mtp_num_layers=1,
    ).cuda()
    model.eval()

    def forward_loop(model):
        with torch.no_grad():
            run_mcore_inference(model, torch.randint(0, 32, (1, 32), device="cuda"))

    quant_cfg = deepcopy(mtq.FP8_DEFAULT_CFG)
    # The default preset excludes MTP; this regression exercises a quantized live head.
    quant_cfg["quant_cfg"] = [
        entry for entry in quant_cfg["quant_cfg"] if entry.get("quantizer_name") != "mtp.*"
    ]
    model = mtq.quantize(model, quant_cfg, forward_loop)
    mtp_quantizers = {
        name: quantizer
        for name, quantizer in model.named_modules()
        if name.startswith("mtp.") and isinstance(quantizer, TensorQuantizer)
    }
    if is_pipeline_last_stage():
        assert mtp_quantizers
        for name, quantizer in mtp_quantizers.items():
            if quantizer.is_enabled:
                assert getattr(quantizer, "_amax", None) is not None
                if name.endswith("input_quantizer"):
                    quantizer.float()
                    quantizer.amax = torch.full_like(quantizer.amax, 1.001)

    source = tmp_path / "tiny_nemotron_h"
    if rank == 0:
        create_tiny_nemotron_h_dir(
            tmp_path,
            num_hidden_layers=4,
            hybrid_override_pattern="M*EE",
            n_routed_experts=4,
            num_nextn_predict_layers=1,
        )
    torch.distributed.barrier()

    export_dir = tmp_path / "mtp_export"
    with _assert_weight_qdq_once(model, prefix="mtp."):
        export_mcore_gpt_to_hf_vllm_fq(
            model,
            pretrained_model_name_or_path=str(source),
            dtype=torch.bfloat16,
            export_dir=str(export_dir),
        )
    expected_names = {
        "mtp.layers.0.eh_proj.input_quantizer",
        *(f"mtp.layers.0.mixer.{proj}_proj.input_quantizer" for proj in ("q", "k", "v", "o")),
        *(
            f"mtp.layers.1.mixer.experts.{expert}.{proj}_proj.input_quantizer"
            for expert in range(4)
            for proj in ("up", "down")
        ),
        *(
            f"mtp.layers.1.mixer.shared_experts.{proj}_proj.input_quantizer"
            for proj in ("up", "down")
        ),
    }
    _, _, weight_map = _assert_exported_quantizers(export_dir, expected_names)
    assert "mtp.layers.0.eh_proj.weight" in weight_map


@pytest.mark.parametrize("pp_size", [1, 2])
def test_mcore_vllm_export_mtp(request, tmp_path, pp_size):
    """Live Nemotron MTP quantizers reach sidecars and never leak into weight shards."""
    workers = request.getfixturevalue(f"dist_workers_size_{pp_size}")
    workers.run(partial(_test_mcore_vllm_export_mtp, tmp_path))


def _test_mcore_vllm_export_unsupported_setting(tmp_path, attribute_cfg, rank, size):
    model = get_mcore_gpt_model(
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()
    quant_cfg = deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg["algorithm"] = None
    model = mtq.quantize(model, quant_cfg)
    if rank == size - 1:
        quantizer = next(
            quantizer
            for name, quantizer in model.named_modules()
            if name.endswith("input_quantizer") and quantizer.is_enabled
        )
        if "if_quant" in attribute_cfg:
            quantizer.disable_quant()
        elif "enable_pre_quant_scale" in attribute_cfg:
            quantizer._enable_pre_quant_scale = False
        else:
            quantizer.set_from_attribute_config(attribute_cfg)

    source = tmp_path / "tiny_llama"
    if rank == 0:
        create_tiny_llama_dir(tmp_path)
    torch.distributed.barrier()
    export_dir = tmp_path / "unsupported_export"
    setting = next(key for key in attribute_cfg if key != "enable")
    with pytest.raises(ValueError, match=f"Unsupported.*input_quantizer: {setting}"):
        export_mcore_gpt_to_hf_vllm_fq(model, source, export_dir=str(export_dir))
    assert not export_dir.exists()


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize(
    "attribute_cfg",
    [
        pytest.param({"unsigned": True}, id="unsigned"),
        pytest.param({"narrow_range": True}, id="narrow_range"),
        pytest.param({"rotate": True}, id="rotate"),
        pytest.param({"rotate": {"enable": True, "rotate_fp32": True}}, id="rotate_config"),
        pytest.param({"enable": False, "rotate": True}, id="disabled_rotation"),
        pytest.param({"fake_quant": False}, id="real_quant"),
        pytest.param({"type": "dynamic"}, id="dynamic"),
        pytest.param({"bias": {-1: None}}, id="bias"),
        pytest.param({"backend": "custom"}, id="backend"),
        pytest.param({"use_constant_amax": True}, id="use_constant_amax"),
        pytest.param({"if_quant": False}, id="quant_disabled"),
        pytest.param({"enable_pre_quant_scale": False}, id="pre_quant_scale_disabled"),
    ],
)
def test_mcore_vllm_export_unsupported_setting(request, tmp_path, attribute_cfg, pp_size):
    """An unsupported setting on the final stage rejects export on every rank."""
    workers = request.getfixturevalue(f"dist_workers_size_{pp_size}")
    workers.run(partial(_test_mcore_vllm_export_unsupported_setting, tmp_path, attribute_cfg))


def _test_cross_rank_recipe_merge(tmp_path, conflicting, rank, size):
    assert size == 2
    name = "model.layers.0.self_attn.q_proj.input_quantizer"
    recipe = {"_num_bits": 8 if rank == 0 or not conflicting else 4}
    with (
        pytest.raises(ValueError, match="Conflicting quantizer recipes")
        if conflicting
        else nullcontext()
    ):
        gather_mcore_vllm_fq_quantizer_recipe({name: recipe}, tmp_path)
    if not conflicting:
        torch.distributed.barrier()
        with open(tmp_path / "quant_recipe.yaml") as f:
            assert yaml.safe_load(f) == {name: recipe}


@pytest.mark.parametrize("conflicting", [False, True])
def test_cross_rank_recipe_merge(dist_workers_size_2, tmp_path, conflicting):
    """Matching TP/EP recipes merge, while conflicting ranks fail together."""
    dist_workers_size_2.run(partial(_test_cross_rank_recipe_merge, tmp_path, conflicting))


def _test_cross_rank_tensor_merge(tmp_path, conflicting, rank, size):
    assert size == 2
    name = "model.layers.0.self_attn.q_proj.input_quantizer._amax"
    tensor = torch.tensor([1.0 + rank if conflicting else 1.0])
    with (
        pytest.raises(ValueError, match="Conflicting quantizer tensors")
        if conflicting
        else nullcontext()
    ):
        gather_mcore_vllm_fq_quantized_state_dict(None, {1: {name: tensor}}, tmp_path)
    if not conflicting:
        torch.distributed.barrier()
        state = torch.load(tmp_path / "quantizer_state.pth", weights_only=True)
        torch.testing.assert_close(state[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("conflicting", [False, True])
def test_cross_rank_tensor_merge(dist_workers_size_2, tmp_path, conflicting):
    """Equal duplicate tensors merge, while conflicting ranks fail together."""
    dist_workers_size_2.run(partial(_test_cross_rank_tensor_merge, tmp_path, conflicting))
