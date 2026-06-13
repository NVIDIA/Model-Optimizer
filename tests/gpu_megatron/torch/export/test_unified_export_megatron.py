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

import pytest
import torch
import transformers
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import get_forward
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_nemotron_dir,
    create_tiny_qwen3vl_dir,
)
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

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
    if model_type == "qwen3vl":
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
        activation_func=activation_func,
        normalization=normalization,
        transformer_impl="modelopt",
        **extra_kwargs,
    ).cuda()

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

    if model_type == "qwen3vl" and rank == 0:
        # sanity check that vision weights were merged by export_mcore_gpt_to_hf
        keys = []
        for sf in sorted(tmp_export_dir.glob("*.safetensors")):
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                keys.extend(f.keys())
        # every decoder layer should be present, not just some
        for i in range(num_layers):
            assert any(k.startswith(f"model.language_model.layers.{i}.") for k in keys), (
                f"language model layer {i} keys missing from export"
            )
        assert any(k.startswith("model.visual.") for k in keys), (
            "vision encoder keys missing from export"
        )
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
        ("nemotron", None, "NVFP4_DEFAULT_CFG", None),
        ("nemotron", None, "NVFP4_DEFAULT_CFG", "FP8_KV_CFG"),
        ("nemotron", "eagle", None, None),
        ("nemotron", "medusa", None, None),
        ("llama", None, None, None),
        ("llama", None, "FP8_DEFAULT_CFG", None),
        ("llama", None, "FP8_DEFAULT_CFG", "FP8_KV_CFG"),
        ("llama", "eagle", None, None),
        ("llama", "medusa", None, None),
        ("qwen3vl", None, None, None),
        ("qwen3vl", None, "FP8_DEFAULT_CFG", None),
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


def _make_minimal_exporter() -> GPTModelExporter:
    exporter = object.__new__(GPTModelExporter)
    exporter.dtype = torch.bfloat16
    exporter.exclude_modules = []
    exporter.layer_config_dict = {}
    exporter._state_dict = {}
    return exporter


def test_mamba_conv1d_remapping_exports_direct_params():
    """New MCore Mamba direct conv params should export to HF conv1d keys."""
    exporter = _make_minimal_exporter()
    mixer = torch.nn.Module()
    mixer.conv1d_weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 1, 4))
    mixer.conv1d_bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float32))

    exporter._mamba_conv1d_remapping(mixer, "backbone.layers.0.mixer.conv1d.")

    torch.testing.assert_close(
        exporter._state_dict["backbone.layers.0.mixer.conv1d.weight"],
        mixer.conv1d_weight.to(torch.bfloat16).cpu(),
    )
    torch.testing.assert_close(
        exporter._state_dict["backbone.layers.0.mixer.conv1d.bias"],
        mixer.conv1d_bias.to(torch.bfloat16).cpu(),
    )
    assert exporter.exclude_modules == ["backbone.layers.0.mixer.conv1d"]


def test_mamba_conv1d_remapping_preserves_old_conv_module_export():
    """Old MCore Mamba nn.Conv1d modules should still use normal name remapping."""
    exporter = _make_minimal_exporter()
    mixer = torch.nn.Module()
    mixer.conv1d = torch.nn.Conv1d(3, 3, 4, groups=3)

    exporter._mamba_conv1d_remapping(mixer, "backbone.layers.0.mixer.conv1d.")

    torch.testing.assert_close(
        exporter._state_dict["backbone.layers.0.mixer.conv1d.weight"],
        mixer.conv1d.weight.to(torch.bfloat16).cpu(),
    )
    torch.testing.assert_close(
        exporter._state_dict["backbone.layers.0.mixer.conv1d.bias"],
        mixer.conv1d.bias.to(torch.bfloat16).cpu(),
    )


def test_mamba_layer_state_dict_routes_conv1d_mapping_to_mixer():
    """Mamba layer export should route the full mixer into the conv1d rule."""
    exporter = _make_minimal_exporter()
    exporter.rules = exporter._populate_rule_book()["NemotronHForCausalLM"]

    layer = SimpleNamespace(
        norm=torch.nn.LayerNorm(3),
        mixer=SimpleNamespace(
            norm=torch.nn.LayerNorm(3),
            A_log=torch.ones(2),
            D=torch.ones(2),
            dt_bias=torch.ones(2),
            conv1d_weight=torch.nn.Parameter(
                torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
            ),
            conv1d_bias=torch.nn.Parameter(torch.arange(3, dtype=torch.float32)),
            in_proj=torch.nn.Linear(3, 3),
            out_proj=torch.nn.Linear(3, 3),
        ),
    )

    exporter._get_mamba_layer_state_dict(layer, layer_id=0)

    assert "backbone.layers.0.mixer.conv1d.weight" in exporter._state_dict
    assert "backbone.layers.0.mixer.conv1d.bias" in exporter._state_dict


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
