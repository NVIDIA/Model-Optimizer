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
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import run_mcore_inference
from safetensors import safe_open

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq
from modelopt.torch.export.plugins.vllm_fakequant_megatron import (
    gather_mcore_vllm_fq_quantized_state_dict,
    gather_mcore_vllm_fq_quantizer_state,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def _test_mcore_vllm_export(tmp_path, quant_cfg, rank, size):
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


def _test_cross_rank_recipe_merge(tmp_path, conflicting, rank, size):
    assert size == 2
    name = "model.layers.0.self_attn.q_proj.input_quantizer"
    recipe = {"_num_bits": 8 if rank == 0 or not conflicting else 4}
    with (
        pytest.raises(ValueError, match="Conflicting quantizer recipes")
        if conflicting
        else nullcontext()
    ):
        gather_mcore_vllm_fq_quantizer_state({name: recipe}, tmp_path)
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
