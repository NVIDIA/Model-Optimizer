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

import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import transformers
from _test_utils.torch.export.unified_checkpoint import assert_exported_checkpoint_matches
from _test_utils.torch.megatron.models import get_mcore_gpt_model, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import get_forward
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_nemotron_dir,
    create_tiny_nemotron_h_dir,
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_moe_dir,
    create_tiny_qwen3vl_dir,
)
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

import modelopt.torch.export.unified_export_megatron as uem
import modelopt.torch.quantization as mtq
import modelopt.torch.speculative as mtsp
from modelopt.torch.export import KV_CACHE_FP8, export_mcore_gpt_to_hf, import_mcore_gpt_from_hf
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.speculative.eagle.default_config import default_eagle_config
from modelopt.torch.speculative.plugins.megatron_eagle import _DynamicEagleGPTModel
from modelopt.torch.speculative.plugins.megatron_medusa import _DynamicMedusaGPTModel


def _verify_model_quant_config(
    export_dir: Path, quant_config: str | None = None, kv_cache_quant_cfg: str | None = None
):
    """Verify config.json and hf_quant_config.json"""
    config_dict = json.load(open(export_dir / "config.json"))
    hf_quant_config_dict = json.load(open(export_dir / "hf_quant_config.json"))
    # Make sure config.json and hf_quant_config.json are consistent
    assert (
        config_dict["quantization_config"]["quant_algo"]
        == hf_quant_config_dict["quantization"]["quant_algo"]
    )
    assert (
        config_dict["quantization_config"]["ignore"]
        == hf_quant_config_dict["quantization"]["exclude_modules"]
    )

    # Verify config.json
    if kv_cache_quant_cfg:
        assert config_dict["quantization_config"]["kv_cache_scheme"]["num_bits"] == 8

    # Verify hf_quant_config.json
    if quant_config:
        quant_config_dict = hf_quant_config_dict["quantization"]
        quant_type = quant_config_dict["quant_algo"]
        assert (
            quant_type in quant_config
        )  # quant config str is subset of quant config e.g. NVFP4 -> NVFP4_DEFAULT_CFG
        assert len(quant_config_dict["exclude_modules"]) > 1  # Dynamically added exclude modules
        if quant_type == "NVFP4":
            assert quant_config_dict["group_size"] == 16

        if kv_cache_quant_cfg:
            assert quant_config_dict["kv_cache_quant_algo"] == KV_CACHE_FP8


def _test_unified_export_megatron(
    tmp_path,
    model_type,
    extra_module,
    quant_config,
    kv_cache_quant_cfg,
    rank,
    size,
    model_dir=None,
):
    if model_type == "nemotron_h":
        config = transformers.AutoConfig.from_pretrained(model_dir)
        model = get_mcore_hybrid_model(
            tensor_model_parallel_size=size,
            pipeline_model_parallel_size=1,
            initialize_megatron=True,
            num_layers=config.num_hidden_layers,
            hybrid_layer_pattern=config.hybrid_override_pattern,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            ffn_hidden_size=config.intermediate_size,
            max_sequence_length=config.max_position_embeddings,
            vocab_size=config.vocab_size,
            mamba_state_dim=config.ssm_state_size,
            mamba_num_heads=config.mamba_num_heads,
            mamba_head_dim=config.mamba_head_dim,
            mamba_num_groups=config.n_groups,
            num_moe_experts=config.n_routed_experts,
            moe_ffn_hidden_size=config.moe_intermediate_size,
            moe_shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
            # NemotronH is the only arch that exports fused grouped-GEMM experts.
            moe_grouped_gemm=True,
            # NemotronH norms are RMSNorm; the builder defaults to LayerNorm, whose biases have no
            # counterpart in the HF checkpoint.
            normalization="RMSNorm",
        ).cuda()
    elif model_type == "qwen3vl":
        config = transformers.AutoConfig.from_pretrained(model_dir)
        text_cfg = config.text_config
        num_layers = text_cfg.num_hidden_layers
        hidden_size = text_cfg.hidden_size
        num_attention_heads = text_cfg.num_attention_heads
        num_query_groups = text_cfg.num_key_value_heads
        ffn_hidden_size = text_cfg.intermediate_size
        max_sequence_length = text_cfg.max_position_embeddings
        vocab_size = text_cfg.vocab_size
        extra_kwargs = {"kv_channels": text_cfg.head_dim, "qk_layernorm": True}
    elif model_type in {"qwen3_5_moe_vl_grouped", "qwen3_5_moe_vl_sequential"}:
        text_cfg = transformers.AutoConfig.from_pretrained(model_dir).text_config
        num_layers = text_cfg.num_hidden_layers
        hidden_size = text_cfg.hidden_size
        num_attention_heads = text_cfg.num_attention_heads
        num_query_groups = text_cfg.num_key_value_heads
        ffn_hidden_size = text_cfg.intermediate_size
        max_sequence_length = text_cfg.max_position_embeddings
        vocab_size = text_cfg.vocab_size
        # Hybrid GatedDeltaNet + gated attention, with routed experts stored packed.
        extra_kwargs = {
            "kv_channels": text_cfg.head_dim,
            "qk_layernorm": True,
            "experimental_attention_variant": "gated_delta_net",
            "num_moe_experts": text_cfg.num_experts,
            "moe_ffn_hidden_size": text_cfg.moe_intermediate_size,
            "moe_shared_expert_intermediate_size": text_cfg.shared_expert_intermediate_size,
            "moe_shared_expert_gate": True,
            # Match the HF layer_types pattern (every Nth layer is full attention, rest GDN).
            "linear_attention_freq": len(text_cfg.layer_types),
            # Both layouts must reach the same per-expert HF tensors, via
            # GroupedGatedMLPSlicing (TEGroupedMLP) and GatedMLPSlicing (SequentialMLP).
            "moe_grouped_gemm": model_type.endswith("grouped"),
        }
    elif model_type == "qwen3_moe":
        config = transformers.AutoConfig.from_pretrained(model_dir)
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_query_groups = config.num_key_value_heads
        ffn_hidden_size = config.intermediate_size
        max_sequence_length = config.max_position_embeddings
        vocab_size = config.vocab_size
        # SequentialMLP: the Qwen3 rules only cover per-expert (``local_experts``) MoE.
        extra_kwargs = {
            "kv_channels": config.hidden_size // config.num_attention_heads,
            "qk_layernorm": True,
            "num_moe_experts": config.num_experts,
            "moe_ffn_hidden_size": config.moe_intermediate_size,
            "moe_grouped_gemm": False,
        }
    elif model_type in {"llama", "nemotron"}:
        config = transformers.AutoConfig.from_pretrained(model_dir)
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_query_groups = config.num_key_value_heads
        ffn_hidden_size = config.intermediate_size
        max_sequence_length = config.max_position_embeddings
        vocab_size = config.vocab_size
        extra_kwargs = {}
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    activation_func = "squared_relu" if model_type == "nemotron" else "swiglu"
    normalization = "LayerNorm" if model_type == "nemotron" else "RMSNorm"

    model = (
        model
        if model_type == "nemotron_h"
        else get_mcore_gpt_model(
            tensor_model_parallel_size=size,
            pipeline_model_parallel_size=1,
            initialize_megatron=True,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            ffn_hidden_size=ffn_hidden_size,
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size,
            activation_func=activation_func,
            normalization=normalization,
            transformer_impl="modelopt",
            **extra_kwargs,
        ).cuda()
    )

    if quant_config:
        quant_config_dict = getattr(mtq, quant_config)
        if kv_cache_quant_cfg:
            kv_quant_cfg = getattr(mtq, kv_cache_quant_cfg)["quant_cfg"]
            quant_config_dict = mtq.utils.update_quant_cfg_with_kv_cache_quant(
                quant_config_dict, kv_quant_cfg
            )
        forward = get_forward(model)
        model = mtq.quantize(model, quant_config_dict, forward)

    if extra_module == "medusa":
        config = {
            "medusa_num_heads": 1,
            "medusa_num_layers": 1,
        }
        model = mtsp.convert(model, [("medusa", config)])
        assert isinstance(model, _DynamicMedusaGPTModel)
    elif extra_module == "eagle":
        config = {"eagle_architecture_config": deepcopy(default_eagle_config)}
        model = mtsp.convert(model, [("eagle", config)])
        assert isinstance(model, _DynamicEagleGPTModel)

    hf_config_dir = model_dir

    tmp_export_dir = tmp_path / "export"
    export_mcore_gpt_to_hf(
        model,
        hf_config_dir,
        dtype=torch.bfloat16,
        export_dir=str(tmp_export_dir),
    )

    if quant_config:
        _verify_model_quant_config(tmp_export_dir, quant_config, kv_cache_quant_cfg)

    if rank == 0 and extra_module is None:
        # Names / shapes only: these Megatron weights are random, not loaded from model_dir.
        allow_missing = ()
        if model_type.startswith("qwen3_5_moe_vl"):
            # The BF16 source packs routed experts; quantized export writes them per expert,
            # as the released NVFP4 checkpoints do.
            allow_missing = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
        assert_exported_checkpoint_matches(
            tmp_export_dir,
            model_dir,
            check_values=False,
            allow_missing=allow_missing,
            # get_mcore_gpt_model always enables the MoE router bias; tiny HF configs have none.
            allow_unexpected=("mlp.gate.expert_bias",),
        )

    if model_type == "qwen3vl" and rank == 0:
        # try to load the model and run a forward pass
        vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            tmp_export_dir, torch_dtype=torch.bfloat16
        ).cuda()
        input_ids = torch.zeros(1, 4, dtype=torch.long).cuda()
        with torch.no_grad():
            out = vl_model(input_ids=input_ids)
        assert out.logits.shape[-1] == vl_model.config.text_config.vocab_size


@pytest.mark.parametrize(
    ("model_type", "extra_module", "quant_config", "kv_cache_quant_cfg"),
    [
        ("nemotron", None, None, None),
        # NemotronH (Mamba + attention + grouped-GEMM MoE) is the stronger quantized case, but it
        # routes no NVFP4 weight through the dense-MLP rules, so keep one dense NVFP4 param too.
        ("nemotron", None, "NVFP4_DEFAULT_CFG", None),
        ("nemotron_h", None, "NVFP4_DEFAULT_CFG", None),
        ("nemotron_h", None, "NVFP4_DEFAULT_CFG", "FP8_KV_CFG"),
        ("nemotron", "eagle", None, None),
        ("nemotron", "medusa", None, None),
        ("llama", None, None, None),
        ("llama", None, "FP8_DEFAULT_CFG", None),
        ("llama", None, "FP8_DEFAULT_CFG", "FP8_KV_CFG"),
        ("llama", "eagle", None, None),
        ("llama", "medusa", None, None),
        ("qwen3vl", None, None, None),
        ("qwen3vl", None, "FP8_DEFAULT_CFG", None),
        # Regression guard: routed experts used to be dropped silently from the export.
        ("qwen3_moe", None, None, None),
        ("qwen3_moe", None, "FP8_DEFAULT_CFG", None),
        # Packed routed experts (Qwen3.5). NVFP4 keeps per-expert block scales while FP8 merges a
        # single scale, so the packing rules only get full coverage across both formats.
        ("qwen3_5_moe_vl_grouped", None, "NVFP4_DEFAULT_CFG", None),
        ("qwen3_5_moe_vl_grouped", None, "FP8_DEFAULT_CFG", None),
        ("qwen3_5_moe_vl_sequential", None, "NVFP4_DEFAULT_CFG", None),
    ],
)
def test_unified_export_megatron(
    dist_workers_size_1, tmp_path, model_type, extra_module, quant_config, kv_cache_quant_cfg
):
    if model_type == "llama":
        model_dir = create_tiny_llama_dir(tmp_path)
    elif model_type == "qwen3vl":
        model_dir = create_tiny_qwen3vl_dir(tmp_path)
    elif model_type == "nemotron":
        model_dir = create_tiny_nemotron_dir(tmp_path)
    elif model_type == "nemotron_h":
        model_dir = create_tiny_nemotron_h_dir(tmp_path)
    elif model_type == "qwen3_moe":
        model_dir = create_tiny_qwen3_moe_dir(tmp_path)
    elif model_type.startswith("qwen3_5_moe_vl"):
        model_dir = create_tiny_qwen3_5_moe_vl_dir(tmp_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    # TODO: Fix TP>1 failures
    dist_workers_size_1.run(
        partial(
            _test_unified_export_megatron,
            tmp_path,
            model_type,
            extra_module,
            quant_config,
            kv_cache_quant_cfg,
            model_dir=model_dir,
        ),
    )


def _test_unified_import_megatron(model_dir, rank, size, model_type="llama"):
    config = transformers.AutoConfig.from_pretrained(model_dir)

    if model_type == "qwen3vl":
        cfg = config.text_config
        extra_kwargs = {
            "kv_channels": cfg.head_dim,
            "transformer_impl": "modelopt",
            "qk_layernorm": True,
        }
    else:
        cfg = config
        extra_kwargs = {}

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_query_groups=cfg.num_key_value_heads,
        ffn_hidden_size=cfg.intermediate_size,
        max_sequence_length=cfg.max_position_embeddings,
        vocab_size=cfg.vocab_size,
        activation_func="swiglu",
        normalization="RMSNorm",
        **extra_kwargs,
    ).cuda()

    import_mcore_gpt_from_hf(model, model_dir)


@pytest.mark.parametrize("model_type", ["llama", "qwen3vl"])
def test_unified_import_megatron(dist_workers, tmp_path, model_type):
    num_gpus = torch.cuda.device_count()
    if model_type == "llama":
        model_dir = create_tiny_llama_dir(tmp_path, num_key_value_heads=num_gpus)
    elif model_type == "qwen3vl":
        model_dir = create_tiny_qwen3vl_dir(
            tmp_path,
            num_attention_heads=num_gpus,
            num_key_value_heads=num_gpus,
            hidden_size=num_gpus * 8,  # head_dim=8
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    dist_workers.run(partial(_test_unified_import_megatron, model_dir, model_type=model_type))


def _test_qkv_slicing_gqa_tp2(tmp_path, rank, size):
    """Export a GQA model with TP=2 to verify _qkv_slicing handles sharded weights."""
    num_layers = 2
    hidden_size = 64
    num_attention_heads = 8
    num_query_groups = 2  # GQA: fewer KV heads than Q heads; both divisible by TP=2
    ffn_hidden_size = 128
    max_sequence_length = 32
    vocab_size = 64

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
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

    # Quantize with FP8 to also exercise the per-channel weight_scale reshape in _qkv_slicing
    forward = get_forward(model)
    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward)

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
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(pretrained_config, f)

    export_dir = tmp_path / "export"
    export_mcore_gpt_to_hf(
        model,
        tmp_path,
        dtype=torch.bfloat16,
        export_dir=str(export_dir),
    )

    # Verify Q/K/V projections were exported (collect keys from all shard files)
    if rank == 0:
        safetensors_files = list(export_dir.glob("*.safetensors"))
        assert safetensors_files, "no safetensors files found in export dir"
        keys = []
        for sf in safetensors_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                keys.extend(f.keys())
        assert any("q_proj.weight" in k for k in keys), "q_proj.weight missing from export"
        assert any("k_proj.weight" in k for k in keys), "k_proj.weight missing from export"
        assert any("v_proj.weight" in k for k in keys), "v_proj.weight missing from export"


def test_qkv_slicing_gqa_tp2(dist_workers_size_2, tmp_path):
    """Export with TP=2 on a GQA model should not raise a reshape error in _qkv_slicing."""
    dist_workers_size_2.run(partial(_test_qkv_slicing_gqa_tp2, tmp_path))


def _test_export_pp2_mtp_metadata_matches_shards(tmp_path, model_dir, rank, size):
    """With PP>1, per-shard JSON keys should exist in the referenced safetensors shard."""
    config = transformers.AutoConfig.from_pretrained(model_dir)

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        ffn_hidden_size=config.intermediate_size,
        max_sequence_length=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        activation_func="swiglu",
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()

    export_dir = tmp_path / "export_pp2"
    original_get_mtp_state_dict = GPTModelExporter._get_mtp_state_dict

    # Simulate stage-local MTP tensors (only on the last PP rank).
    def _fake_get_mtp_state_dict(self):
        if rank != size - 1:
            return {}
        return {f"mtp.injected.rank{rank}.weight": torch.ones(8, dtype=torch.bfloat16).cpu()}

    GPTModelExporter._get_mtp_state_dict = _fake_get_mtp_state_dict

    try:
        export_mcore_gpt_to_hf(
            model,
            model_dir,
            dtype=torch.bfloat16,
            export_dir=str(export_dir),
        )
    finally:
        GPTModelExporter._get_mtp_state_dict = original_get_mtp_state_dict

    if rank == 0:
        shard_json_files = sorted(export_dir.glob("model-*.json"))
        assert shard_json_files, "no per-shard metadata json files found"

        shard_keys_cache = {}
        all_weight_map_keys = set()
        for shard_json_file in shard_json_files:
            with open(shard_json_file) as f:
                shard_meta = json.load(f)
            for key, shard_file in shard_meta["weight_map"].items():
                all_weight_map_keys.add(key)
                if shard_file not in shard_keys_cache:
                    with safe_open(
                        str(export_dir / shard_file), framework="pt", device="cpu"
                    ) as sf:
                        shard_keys_cache[shard_file] = set(sf.keys())
                assert key in shard_keys_cache[shard_file], (
                    f"key '{key}' from {shard_json_file.name} missing in {shard_file}"
                )

        assert any(key.startswith("mtp.injected.") for key in all_weight_map_keys), (
            "expected injected mtp.* key missing from shard metadata/index map"
        )


def test_unified_export_megatron_pp2_mtp_metadata_matches_shards(dist_workers_size_2, tmp_path):
    model_dir = create_tiny_llama_dir(tmp_path)
    dist_workers_size_2.run(
        partial(_test_export_pp2_mtp_metadata_matches_shards, tmp_path, model_dir)
    )


def test_qkv_slicing_records_hf_excludes_for_unquantized_fused_qkv():
    """Unquantized fused MCore linear_qkv should become HF q/k/v excludes."""
    exporter = object.__new__(GPTModelExporter)
    exporter.dtype = torch.bfloat16
    exporter.exclude_modules = ["backbone.layers.0.mixer"]
    exporter.layer_config_dict = {}
    exporter._state_dict = {}

    hidden_size = 8
    head_size = 4
    num_attention_heads = 2
    num_query_groups = 1
    qkv_dim = num_attention_heads + 2 * num_query_groups
    weight = torch.arange(qkv_dim * head_size * hidden_size, dtype=torch.bfloat16).reshape(
        qkv_dim * head_size, hidden_size
    )

    module = torch.nn.Module()
    module.config = type(
        "Config",
        (),
        {
            "hidden_size": hidden_size,
            "num_query_groups": num_query_groups,
            "num_attention_heads": num_attention_heads,
            "kv_channels": head_size,
        },
    )()
    exporter._get_quantized_state = lambda *args, **kwargs: ({"weight": weight}, None, 0)

    exporter._qkv_slicing(module, "backbone.layers.0.mixer.")

    assert "backbone.layers.0.mixer" not in exporter.exclude_modules
    assert "backbone.layers.0.mixer.q_proj" in exporter.exclude_modules
    assert "backbone.layers.0.mixer.k_proj" in exporter.exclude_modules
    assert "backbone.layers.0.mixer.v_proj" in exporter.exclude_modules


def _make_exporter_for_mtp(model_dir: Path) -> GPTModelExporter:
    """Create a minimal GPTModelExporter instance for testing _get_mtp_state_dict."""
    exporter = object.__new__(GPTModelExporter)
    exporter._hf_pretrained_model_name = str(model_dir)
    exporter._state_dict = {}  # MTP keys are absent — they should be picked up
    exporter.exclude_modules = []
    return exporter


def test_mtp_state_dict_single_safetensors(tmp_path):
    """MTP weights are collected from a model with a single model.safetensors file."""
    model_dir = tmp_path / "fake_hf_model"
    model_dir.mkdir()

    tensors = {
        "model.embed_tokens.weight": torch.zeros(64, 32),
        "mtp.0.enorm.weight": torch.ones(32),
        "mtp.0.hnorm.weight": torch.full((32,), 2.0),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))

    exporter = _make_exporter_for_mtp(model_dir)
    mtp_state_dict = exporter._get_mtp_state_dict()

    assert "mtp.0.enorm.weight" in mtp_state_dict
    assert "mtp.0.hnorm.weight" in mtp_state_dict
    assert "model.embed_tokens.weight" not in mtp_state_dict, "non-MTP key should not be included"
    assert torch.allclose(mtp_state_dict["mtp.0.enorm.weight"], torch.ones(32))
    assert torch.allclose(mtp_state_dict["mtp.0.hnorm.weight"], torch.full((32,), 2.0))
    assert "mtp*" in exporter.exclude_modules


def test_mtp_state_dict_no_mtp_keys(tmp_path):
    """_get_mtp_state_dict returns empty dict and leaves exclude_modules unchanged when no MTP keys."""
    model_dir = tmp_path / "fake_hf_model"
    model_dir.mkdir()

    tensors = {"model.embed_tokens.weight": torch.zeros(64, 32)}
    save_file(tensors, str(model_dir / "model.safetensors"))

    exporter = _make_exporter_for_mtp(model_dir)
    mtp_state_dict = exporter._get_mtp_state_dict()

    assert mtp_state_dict == {}
    assert exporter.exclude_modules == []


def test_mtp_state_dict_index_file(tmp_path):
    """MTP weights are collected from a sharded checkpoint (index file path)."""
    model_dir = tmp_path / "fake_hf_model"
    model_dir.mkdir()

    shard1 = {
        "model.embed_tokens.weight": torch.zeros(64, 32),
        "mtp.0.enorm.weight": torch.ones(32),
    }
    shard2 = {"mtp.0.hnorm.weight": torch.full((32,), 3.0)}
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "mtp.0.enorm.weight": "model-00001-of-00002.safetensors",
            "mtp.0.hnorm.weight": "model-00002-of-00002.safetensors",
        }
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    exporter = _make_exporter_for_mtp(model_dir)
    mtp_state_dict = exporter._get_mtp_state_dict()

    assert "mtp.0.enorm.weight" in mtp_state_dict
    assert "mtp.0.hnorm.weight" in mtp_state_dict
    assert torch.allclose(mtp_state_dict["mtp.0.hnorm.weight"], torch.full((32,), 3.0))
    assert "mtp*" in exporter.exclude_modules


class _FakeTEGroupedMLP:
    """Minimal TEGroupedMLP stand-in exposing num_gemms, weight{i}, and state_dict()."""

    def __init__(self, num_gemms: int, hidden: int = 8, ffn: int = 16, local_expert_indices=None):
        self.num_gemms = num_gemms
        self._weights = {
            f"weight{i}": torch.randn(ffn, hidden, dtype=torch.bfloat16) for i in range(num_gemms)
        }
        for k, v in self._weights.items():
            setattr(self, k, v)
        if local_expert_indices is not None:
            self.local_expert_indices = local_expert_indices

    def state_dict(self):
        return dict(self._weights)


def _make_exporter_for_grouped_mlp() -> GPTModelExporter:
    exporter = object.__new__(GPTModelExporter)
    exporter.dtype = torch.bfloat16
    exporter._state_dict = {}
    exporter._get_quantized_state = lambda *a, **k: ({}, None, 0)
    exporter._get_weight_scales = lambda *a, **k: (None, None)
    exporter._record_layer_quant_config = lambda *a, **k: None
    exporter.exclude_modules = []
    return exporter


def test_grouped_mlp_slicing_maps_local_to_global_expert_ids():
    """EP>1 fix: without global remapping, every EP rank would write experts.0..N-1 and
    collide on the writer's state_dict.
    """
    exporter = _make_exporter_for_grouped_mlp()
    # Simulate EP rank 2 of an EP=4 job: this rank owns global experts 4 and 5.
    module = _FakeTEGroupedMLP(num_gemms=2, local_expert_indices=[4, 5])

    exporter._grouped_mlp_slicing(module, "experts.{}.gate_up_proj")

    assert "experts.4.gate_up_proj.weight" in exporter._state_dict
    assert "experts.5.gate_up_proj.weight" in exporter._state_dict
    # Local indices 0/1 must NOT leak into the exported state_dict.
    assert "experts.0.gate_up_proj.weight" not in exporter._state_dict
    assert "experts.1.gate_up_proj.weight" not in exporter._state_dict


def test_grouped_mlp_slicing_normalizes_tensor_local_expert_indices():
    """local_expert_indices may arrive as a torch.Tensor (Megatron path). It must be
    normalized to list[int] -- a naive `bool(tensor)` on a multi-element tensor raises.
    """
    exporter = _make_exporter_for_grouped_mlp()
    module = _FakeTEGroupedMLP(
        num_gemms=2, local_expert_indices=torch.tensor([6, 7], dtype=torch.long)
    )

    exporter._grouped_mlp_slicing(module, "experts.{}.gate_up_proj")

    assert "experts.6.gate_up_proj.weight" in exporter._state_dict
    assert "experts.7.gate_up_proj.weight" in exporter._state_dict


def test_grouped_mlp_slicing_splits_gate_up_per_expert():
    """Quantized Qwen3.5 export writes gate_proj / up_proj / down_proj per expert."""
    exporter = _make_exporter_for_grouped_mlp()
    module = _FakeTEGroupedMLP(num_gemms=2, hidden=8, ffn=16, local_expert_indices=[0, 1])

    exporter._grouped_mlp_slicing(
        module, "experts.{}", gate_proj_name="gate_proj", up_proj_name="up_proj"
    )

    for expert in (0, 1):
        gate = exporter._state_dict[f"experts.{expert}.gate_proj.weight"]
        up = exporter._state_dict[f"experts.{expert}.up_proj.weight"]
        assert gate.shape == (8, 8) and up.shape == (8, 8)
        torch.testing.assert_close(gate, module.state_dict()[f"weight{expert}"][:8])
        torch.testing.assert_close(up, module.state_dict()[f"weight{expert}"][8:])


def test_grouped_mlp_slicing_splits_per_block_scale_and_replicates_scale_2():
    """A per-block weight_scale is sliced with the weight; the scalar weight_scale_2 is shared."""
    exporter = _make_exporter_for_grouped_mlp()
    module = _FakeTEGroupedMLP(num_gemms=1, hidden=8, ffn=16, local_expert_indices=[0])
    weight_scale = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    weight_scale_2 = torch.tensor(0.5)
    exporter._get_weight_scales = lambda *a, **k: (weight_scale, weight_scale_2)
    exporter._get_quantized_state = lambda *a, **k: ({"weight": None}, "NVFP4", 16)
    recorded: list[str] = []
    exporter._record_layer_quant_config = lambda prefix, *a, **k: recorded.append(prefix)

    with patch(
        "modelopt.torch.export.unified_export_megatron.to_quantized_weight",
        side_effect=lambda w, *a, **k: w,
    ):
        exporter._grouped_mlp_slicing(
            module, "experts.{}", gate_proj_name="gate_proj", up_proj_name="up_proj"
        )

    torch.testing.assert_close(
        exporter._state_dict["experts.0.gate_proj.weight_scale"], weight_scale[:8]
    )
    torch.testing.assert_close(
        exporter._state_dict["experts.0.up_proj.weight_scale"], weight_scale[8:]
    )
    for proj in ("gate_proj", "up_proj"):
        torch.testing.assert_close(
            exporter._state_dict[f"experts.0.{proj}.weight_scale_2"], weight_scale_2
        )
    assert set(recorded) == {"experts.0.gate_proj.", "experts.0.up_proj."}


def test_grouped_mlp_slicing_scalar_scale_is_shared_by_both_shards():
    """A 0-dim scale has no output dim to slice, so both projections reuse it."""
    exporter = _make_exporter_for_grouped_mlp()
    module = _FakeTEGroupedMLP(num_gemms=1, hidden=8, ffn=16, local_expert_indices=[0])
    scale = torch.tensor(0.25)
    exporter._get_weight_scales = lambda *a, **k: (scale, None)
    exporter._get_quantized_state = lambda *a, **k: ({"weight": None}, "FP8", 0)

    with patch(
        "modelopt.torch.export.unified_export_megatron.to_quantized_weight",
        side_effect=lambda w, *a, **k: w,
    ):
        exporter._grouped_mlp_slicing(
            module, "experts.{}", gate_proj_name="gate_proj", up_proj_name="up_proj"
        )

    for proj in ("gate_proj", "up_proj"):
        torch.testing.assert_close(exporter._state_dict[f"experts.0.{proj}.weight_scale"], scale)


def test_grouped_mlp_slicing_records_unquantized_experts_as_excluded():
    """exclude_modules is an explicit list: unquantized experts must be named in it, or a
    mixed-precision checkpoint leaves the runtime no signal for them.
    """
    exporter = _make_exporter_for_grouped_mlp()  # stub yields qformat None
    module = _FakeTEGroupedMLP(num_gemms=2, local_expert_indices=[0, 1])

    exporter._grouped_mlp_slicing(
        module, "experts.{}", gate_proj_name="gate_proj", up_proj_name="up_proj"
    )

    assert set(exporter.exclude_modules) == {
        "experts.0.gate_proj",
        "experts.0.up_proj",
        "experts.1.gate_proj",
        "experts.1.up_proj",
    }


def test_grouped_mlp_slicing_collects_all_missing_expert_weights():
    """New collect-then-raise behavior: the error message must name every missing
    weight{i}, not just the first one hit.
    """
    exporter = _make_exporter_for_grouped_mlp()
    module = _FakeTEGroupedMLP(num_gemms=3)
    # Drop weight0 AND weight2; only weight1 remains.
    module.state_dict = lambda: {"weight1": module.weight1}

    with pytest.raises(ValueError) as exc_info:
        exporter._grouped_mlp_slicing(module, "experts.{}.gate_up_proj")

    msg = str(exc_info.value)
    assert "weight0" in msg and "weight2" in msg, (
        f"error should list all missing weights, got: {msg}"
    )


def test_is_sidecar_writer_rank_pins_to_dp0_ep0(monkeypatch):
    """DP>1 fix predicate: only the DP0/EP0 rank among is_last_stage_main_rank writes
    sidecar files. Guards the predicate used at three sites in save_pretrained.
    """
    # is_last_stage_main_rank=False is never a writer, regardless of DP/EP.
    monkeypatch.setattr(uem, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(uem, "get_expert_model_parallel_rank", lambda: 0)
    assert GPTModelExporter._is_sidecar_writer_rank(False) is False

    # DP0/EP0 is the writer.
    assert GPTModelExporter._is_sidecar_writer_rank(True) is True

    # DP rank != 0 loses the writer role even if is_last_stage_main_rank.
    monkeypatch.setattr(uem, "get_data_parallel_rank", lambda: 1)
    monkeypatch.setattr(uem, "get_expert_model_parallel_rank", lambda: 0)
    assert GPTModelExporter._is_sidecar_writer_rank(True) is False

    # EP rank != 0 loses the writer role.
    monkeypatch.setattr(uem, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(uem, "get_expert_model_parallel_rank", lambda: 1)
    assert GPTModelExporter._is_sidecar_writer_rank(True) is False


def _make_exporter_for_key_check(num_layers: int) -> GPTModelExporter:
    exporter = object.__new__(GPTModelExporter)
    exporter.model = SimpleNamespace(config=SimpleNamespace(num_layers=num_layers))
    return exporter


def _write_index(dir_path: Path, keys) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "model-00001.safetensors")})
    )


@pytest.mark.parametrize(
    ("exported", "raises"),
    [
        # Packed source expanded into per-expert entries: not a drop.
        (["model.layers.0.mlp.experts.0.gate_proj.weight"], False),
        # Module genuinely absent: nothing lives under its prefix.
        (["model.layers.0.self_attn.q_proj.weight"], True),
    ],
)
def test_verify_exported_keys_accepts_expansion_but_still_catches_drops(tmp_path, exported, raises):
    """The self-check must tolerate one source module expanding into several exported ones."""
    source, export = tmp_path / "src", tmp_path / "exp"
    _write_index(source, ["model.layers.0.mlp.experts.gate_up_proj"])
    _write_index(export, exported)
    exporter = _make_exporter_for_key_check(num_layers=1)

    if raises:
        with pytest.raises(RuntimeError, match="Export dropped"):
            exporter._verify_exported_keys(str(export), str(source))
    else:
        exporter._verify_exported_keys(str(export), str(source))
