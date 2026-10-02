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
from contextlib import nullcontext
from functools import partial

import pytest
import torch
import yaml
from _test_utils.torch.megatron.models import get_mcore_gpt_model, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import initialize_for_megatron, run_mcore_inference
from safetensors import safe_open

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq
from modelopt.torch.export.plugins.vllm_fakequant_megatron import (
    VllmFqGPTModelExporter,
    gather_mcore_vllm_fq_quantized_state_dict,
    gather_mcore_vllm_fq_quantizer_recipe,
)
from modelopt.torch.quantization.nn import GroupedQuantizer, TensorQuantizer


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

    if rank == 0:
        with open(tmp_path / "config.json", "w") as f:
            json.dump(pretrained_config, f)
    torch.distributed.barrier()

    # Export directory
    export_dir = tmp_path / "vllm_export"
    export_dir.mkdir(exist_ok=True)

    quantizers = [
        module
        for name, module in model.named_modules()
        if name.endswith("weight_quantizer")
        and isinstance(module, TensorQuantizer)
        and module.is_enabled
    ]
    assert quantizers
    calls = Counter()

    def count_qdq(module, args, output):
        calls[module] += 1

    handles = [quantizer.register_forward_hook(count_qdq) for quantizer in quantizers]
    try:
        if prebuild:
            exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
            assert exporter.state_dict
            assert exporter.layer_state_dicts
            exporter.save_pretrained(str(export_dir), tmp_path)
        else:
            export_mcore_gpt_to_hf_vllm_fq(
                model,
                pretrained_model_name_or_path=tmp_path,
                dtype=torch.bfloat16,
                export_dir=str(export_dir),
            )
    finally:
        for handle in handles:
            handle.remove()

    assert all(calls[quantizer] == 1 for quantizer in quantizers), calls

    # check if quant_amax.pth file exists
    quant_amax_file = export_dir / "quantizer_state.pth"
    assert quant_amax_file.exists(), f"quantizer_state.pth file should be created in {export_dir}"

    # Recipes take the same export mapping path as quantizer tensors. Every tensor-side
    # quantizer must therefore have a recipe at its final exported module path, and the
    # temporary routing markers must not leak into either sidecar.
    quantizer_state = torch.load(quant_amax_file, weights_only=True, map_location="cpu")
    input_amaxes = [
        value
        for key, value in quantizer_state.items()
        if "input_quantizer" in key and key.endswith("._amax")
    ]
    assert input_amaxes
    for amax in input_amaxes:
        assert amax.dtype == torch.float32
        torch.testing.assert_close(amax, torch.full_like(amax, 1.001), rtol=0, atol=0)
    quantizer_recipe_file = export_dir / "quant_recipe.yaml"
    assert quantizer_recipe_file.exists()
    with open(quantizer_recipe_file) as f:
        quantizer_recipe = yaml.safe_load(f)

    marker_suffix = "._quant_recipe_marker"
    assert not any(key.endswith(marker_suffix) for key in quantizer_state)
    assert not any(key.endswith(marker_suffix) for key in quantizer_recipe)

    state_quantizer_names = {key.rsplit(".", 1)[0] for key in quantizer_state if "quantizer" in key}
    missing_recipe_names = state_quantizer_names - quantizer_recipe.keys()
    assert not missing_recipe_names, (
        "Exported quantizer tensors are missing matching recipe entries: "
        f"{sorted(missing_recipe_names)}"
    )

    # make sure hf_quant_config.json file does not exist
    hf_quant_config_file = export_dir / "hf_quant_config.json"
    assert not hf_quant_config_file.exists(), (
        f"hf_quant_config.json file should not be created in {export_dir}"
    )

    with open(export_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    assert {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        *(f"model.layers.{i}.self_attn.q_proj.weight" for i in range(num_layers)),
    } <= weight_map.keys()
    for shard in set(weight_map.values()):
        with safe_open(export_dir / shard, framework="pt") as f:
            shard_keys = f.keys()
            assert not any("quantizer" in key or marker_suffix in key for key in shard_keys)


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_mcore_vllm_export(request, tmp_path, quant_cfg, pp_size):
    """Export each PP stage once and retain weights and sidecars from every stage."""
    workers = request.getfixturevalue(f"dist_workers_size_{pp_size}")
    workers.run(partial(_test_mcore_vllm_export, tmp_path, quant_cfg))


def test_mcore_vllm_export_after_state_dict_access(dist_workers_size_1, tmp_path):
    """Cached shards retain recipe markers when export begins after state inspection."""
    dist_workers_size_1.run(
        partial(_test_mcore_vllm_export, tmp_path, mtq.FP8_DEFAULT_CFG, prebuild=True)
    )


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


def _test_mcore_vllm_grouped_export(tmp_path, quant_cfg, device, rank, size, prebuild=False):
    model = (
        get_mcore_hybrid_model(
            initialize_megatron=True,
            num_layers=1,
            hybrid_layer_pattern="E",
            hidden_size=64,
            num_attention_heads=8,
            num_query_groups=8,
            ffn_hidden_size=128,
            max_sequence_length=16,
            vocab_size=64,
            normalization="RMSNorm",
            transformer_impl="transformer_engine",
            moe_grouped_gemm=True,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
        )
        .cuda()
        .eval()
    )

    def forward_loop(model):
        with torch.no_grad():
            run_mcore_inference(model, torch.arange(16, device="cuda").unsqueeze(0))

    mtq.quantize(model, quant_cfg, forward_loop)
    experts = model.decoder.layers[0].mlp.experts
    grouped_modules = [experts.linear_fc1, experts.linear_fc2]
    expected_weights = {}
    for module, projection in zip(grouped_modules, ("up_proj", "down_proj")):
        assert isinstance(module.weight_quantizer, GroupedQuantizer)
        module.weight_quantizer[-1].disable()
        for i, quantizer in enumerate(module.weight_quantizer):
            weight = getattr(module, f"weight{i}")
            with torch.no_grad():
                expected = quantizer(weight).cpu()
            if quantizer.is_enabled:
                assert not torch.equal(expected, weight.cpu())
            expected_weights[f"backbone.layers.0.mixer.experts.{i}.{projection}.weight"] = (
                expected.clone()
            )

    model.to(device)
    original_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    original_hooks = {module: dict(module._state_dict_hooks) for module in grouped_modules}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 32,
                "hybrid_override_pattern": "E",
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 8,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "vocab_size": 64,
                "torch_dtype": "bfloat16",
            },
            f,
        )

    def assert_model_unchanged():
        for module in grouped_modules:
            assert dict(module._state_dict_hooks) == original_hooks[module]
            assert isinstance(module.weight_quantizer, GroupedQuantizer)
            assert not hasattr(module, "weight")
            assert not module.weight_quantizer[-1].is_enabled
        current_state = {
            key: value
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        assert current_state.keys() == original_state.keys()
        for key, value in original_state.items():
            torch.testing.assert_close(current_state[key], value, rtol=0, atol=0)

    # Fail after the first grouped linear has been processed, then retry with a fresh exporter.
    def fail_quantization(module, args):
        raise RuntimeError("injected grouped QDQ failure")

    failure_hook = experts.linear_fc2.weight_quantizer[0].register_forward_pre_hook(
        fail_quantization
    )
    try:
        exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="injected grouped QDQ failure"):
            exporter.save_pretrained(str(tmp_path / "failed_export"), tmp_path)
    finally:
        failure_hook.remove()
    assert_model_unchanged()

    calls = Counter()

    def count_qdq(module, args, output):
        calls[module] += 1

    handles = []
    try:
        for module in grouped_modules:
            for quantizer in module.weight_quantizer:
                handle = quantizer.register_forward_hook(count_qdq)
                handles.append(handle)
        export_dir = tmp_path / "grouped_export"
        if prebuild:
            exporter = VllmFqGPTModelExporter(model, tmp_path, dtype=torch.bfloat16)
            assert exporter.layer_state_dicts
            exporter.save_pretrained(str(export_dir), tmp_path)
        else:
            export_mcore_gpt_to_hf_vllm_fq(
                model, tmp_path, dtype=torch.bfloat16, export_dir=str(export_dir)
            )
    finally:
        for handle in handles:
            handle.remove()

    assert_model_unchanged()
    for module in grouped_modules:
        for quantizer in module.weight_quantizer:
            assert calls[quantizer] == int(quantizer.is_enabled)

    with open(export_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    for key, expected in expected_weights.items():
        with safe_open(export_dir / weight_map[key], framework="pt") as f:
            torch.testing.assert_close(f.get_tensor(key), expected, rtol=0, atol=0)

    quantizer_state = torch.load(export_dir / "quantizer_state.pth", weights_only=True)
    with open(export_dir / "quant_recipe.yaml") as f:
        recipe = yaml.safe_load(f)
    assert not any("weight_quantizer" in key for key in quantizer_state)
    assert {key.rsplit(".", 1)[0] for key in quantizer_state} <= recipe.keys()
    assert not any("{}" in key for key in recipe)


@pytest.mark.parametrize(
    "quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG], ids=["fp8", "nvfp4"]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mcore_vllm_grouped_export(dist_workers_size_1, tmp_path, quant_cfg, device):
    """Grouped export applies QDQ once, preserves the model, and removes temporary hooks."""
    dist_workers_size_1.run(partial(_test_mcore_vllm_grouped_export, tmp_path, quant_cfg, device))


def test_mcore_vllm_grouped_export_after_state_dict_access(dist_workers_size_1, tmp_path):
    """Cached grouped shards retain folded weights when state is inspected before save."""
    dist_workers_size_1.run(
        partial(_test_mcore_vllm_grouped_export, tmp_path, mtq.FP8_DEFAULT_CFG, "cpu"),
        prebuild=True,
    )


def _test_mcore_vllm_grouped_ep_export(tmp_path, rank, size):
    """Every EP rank contributes its local folded experts to one checkpoint."""
    initialize_for_megatron(expert_model_parallel_size=size)
    model = (
        get_mcore_hybrid_model(
            initialize_megatron=False,
            expert_model_parallel_size=size,
            num_layers=1,
            hybrid_layer_pattern="E",
            hidden_size=64,
            num_attention_heads=8,
            num_query_groups=8,
            ffn_hidden_size=128,
            max_sequence_length=16,
            vocab_size=64,
            normalization="RMSNorm",
            transformer_impl="transformer_engine",
            moe_grouped_gemm=True,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
        )
        .cuda()
        .eval()
    )

    def forward_loop(model):
        with torch.no_grad():
            run_mcore_inference(model, torch.arange(16, device="cuda").unsqueeze(0))

    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    experts = model.decoder.layers[0].mlp.experts
    expected_local = {}
    for module, projection in (
        (experts.linear_fc1, "up_proj"),
        (experts.linear_fc2, "down_proj"),
    ):
        assert isinstance(module.weight_quantizer, GroupedQuantizer)
        for local_id in range(module.num_gemms):
            global_id = rank * module.num_gemms + local_id
            weight = getattr(module, f"weight{local_id}")
            with torch.no_grad():
                expected = module.weight_quantizer[local_id](weight.to(torch.bfloat16))
            expected_local[f"backbone.layers.0.mixer.experts.{global_id}.{projection}.weight"] = (
                expected.cpu()
            )

    all_expected = [None] * size
    torch.distributed.all_gather_object(all_expected, expected_local)
    if rank == 0:
        with open(tmp_path / "config.json", "w") as f:
            json.dump(
                {
                    "architectures": ["NemotronHForCausalLM"],
                    "model_type": "nemotron_h",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "moe_intermediate_size": 64,
                    "moe_shared_expert_intermediate_size": 32,
                    "hybrid_override_pattern": "E",
                    "num_hidden_layers": 1,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "head_dim": 8,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "vocab_size": 64,
                    "torch_dtype": "bfloat16",
                },
                f,
            )
    torch.distributed.barrier()

    export_dir = tmp_path / "grouped_ep_export"
    export_mcore_gpt_to_hf_vllm_fq(
        model, tmp_path, dtype=torch.bfloat16, export_dir=str(export_dir)
    )
    torch.distributed.barrier()
    if rank == 0:
        with open(export_dir / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        for per_rank in all_expected:
            for key, expected in per_rank.items():
                with safe_open(export_dir / weight_map[key], framework="pt") as f:
                    torch.testing.assert_close(f.get_tensor(key), expected, rtol=0, atol=0)


def test_mcore_vllm_grouped_ep_export(dist_workers_size_2, tmp_path):
    dist_workers_size_2.run(partial(_test_mcore_vllm_grouped_ep_export, tmp_path))
