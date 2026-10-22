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

"""``load_pretrained`` must record exactly the checkpoint tensors the model has no parameter for.

These drive a real ``from_pretrained`` on CPU, because what is under test is which checkpoint
tensors the Transformers loader consumed. The record is what the export carries: a tensor wrongly
called unplaced lands in the export beside the weight it became, and one wrongly called placed is
dropped from it.
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    FineGrainedFP8Config,
    GptOssConfig,
    GptOssForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Mxfp4Config,
)

from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    load_pretrained,
    parallel_load_and_prepare_fsdp2,
)

# A layer past ``num_hidden_layers``: genuinely stray, so it must always be recorded.
STRAY = "model.layers.1.input_layernorm.weight"


def _tiny_llama_config(**overrides):
    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "tie_word_embeddings": False,
        "architectures": ["LlamaForCausalLM"],
    }
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def _save(path, config, state, quantization_config=None):
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    config_json = config.to_dict()
    if quantization_config is not None:
        config_json["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(config_json))


def _load(path, auto_class=None, **kwargs):
    model = load_pretrained(str(path), auto_class, **kwargs)
    if getattr(model, "_weight_conversions", None) is None:
        pytest.skip("this transformers does not record the conversions from_pretrained ran")
    return model


_MTP_AND_AUX = {"mtp.fc.weight": torch.ones(2), "aux_tower.blocks.0.weight": torch.ones(2)}


@pytest.mark.parametrize(
    ("extra", "subfolder"),
    [({}, None), (_MTP_AND_AUX, None), (_MTP_AND_AUX, "nested")],
    ids=["nothing-stray", "mtp-and-aux-tower", "in-a-subfolder"],
)
def test_records_the_stray_tensors_and_the_checkpoint(tmp_path, extra, subfolder):
    """Nothing keys off a name such as 'mtp': an auxiliary tower is recorded by the same rule. An
    empty record is an answer too, so the export does not derive the list again. A subfolder is
    where the checkpoint is: the config and the record come from it, as the weights do."""
    config = _tiny_llama_config()
    state = LlamaForCausalLM(config).to(torch.bfloat16).state_dict()
    checkpoint = tmp_path / subfolder if subfolder else tmp_path
    checkpoint.mkdir(exist_ok=True)
    _save(checkpoint, config, {**state, **extra})

    options = {"subfolder": subfolder} if subfolder else {}
    model = _load(tmp_path, dtype=torch.bfloat16, **options)

    assert model._modelopt_unplaced_source_keys == sorted(extra), "sorted, for a stable export"
    assert model._modelopt_source_checkpoint == str(checkpoint)
    assert model.dtype == torch.bfloat16, "from_pretrained kwargs must pass through"


class _Recorder:
    """Stands in for a Transformers auto class: records each ``from_pretrained``'s options."""

    def __init__(self, auto_class):
        self.auto_class, self.options = auto_class, []

    def from_pretrained(self, path, **kwargs):
        self.options.append(kwargs)
        return self.auto_class.from_pretrained(path, **kwargs)


def test_the_hub_options_pick_the_snapshot_and_reach_the_loads_from_it(tmp_path, monkeypatch):
    """``revision``, ``token``, ``cache_dir`` and ``local_files_only`` choose what a Hub ID resolves
    to. The config and model loads from the snapshot take all but ``revision`` too: an ``auto_map``
    entry can name custom code in another repo, which they fetch, offline or private, with them."""
    config = _tiny_llama_config()
    _save(tmp_path, config, LlamaForCausalLM(config).state_dict())
    resolved = []

    def ensure_local_checkpoint(model_name_or_path, **hub_kwargs):
        resolved.append((model_name_or_path, hub_kwargs))
        return model_name_or_path, str(tmp_path)

    configs, models = _Recorder(AutoConfig), _Recorder(AutoModelForCausalLM)
    monkeypatch.setattr(model_load_utils, "ensure_local_checkpoint", ensure_local_checkpoint)
    monkeypatch.setattr(model_load_utils, "AutoConfig", configs)
    hub_kwargs = {
        "revision": "abc123",
        "token": "hf_x",
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
    }

    model = load_pretrained("org/model", models, **hub_kwargs)

    assert resolved[0] == ("org/model", hub_kwargs)
    # Anything resolved after that is the local snapshot.
    assert all(path == str(tmp_path) for path, _ in resolved[1:])
    code_options = {k: v for k, v in hub_kwargs.items() if k != "revision"}
    loads = configs.options + models.options
    assert len(loads) == 2
    assert all({k: load[k] for k in hub_kwargs if k in load} == code_options for load in loads)
    assert model._modelopt_source_checkpoint == str(tmp_path)


def test_a_tied_name_the_checkpoint_stores_is_placed(tmp_path):
    """Qwen3-0.6B stores lm_head.weight beside the embedding it is tied to. named_parameters()
    lists a tied parameter under its first name only, but the other name is the model's own
    parameter too: neither the whole-model nor the FSDP2 load may record it for the export to carry
    over beside the weight it is."""
    config = _tiny_llama_config(tie_word_embeddings=True)
    _save(
        tmp_path, config, {k: v.clone() for k, v in LlamaForCausalLM(config).state_dict().items()}
    )
    assert "lm_head.weight" in load_file(str(tmp_path / "model.safetensors"))

    assert _load(tmp_path)._modelopt_unplaced_source_keys == []
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(
            "gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'group'}"
        )
    try:
        sharded = parallel_load_and_prepare_fsdp2(str(tmp_path), torch.device("cpu"), 0, 1)
    finally:
        if owns_group:
            torch.distributed.destroy_process_group()
    assert sharded._modelopt_unplaced_source_keys == []


@pytest.mark.parametrize(
    ("option", "refused"),
    [({"output_loading_info": True}, "output_loading_info"), ({"variant": "fp16"}, "variant")],
)
def test_an_option_it_cannot_honor_is_refused_before_anything_loads(option, refused):
    """``output_loading_info`` would return ``(model, loading_info)``, where a model is expected; a
    ``variant`` would load weight files the record and the export do not read.

    Resolving this path, neither a directory nor a valid Hub ID, raises too, so the refusal must
    come before anything runs.
    """
    with pytest.raises(ValueError, match=refused):
        load_pretrained("no/such/checkpoint", **option)


def test_a_config_passed_for_the_load_needs_none_on_disk(tmp_path):
    """The load uses the config it is passed, as ``from_pretrained`` does, and so does the choice of
    its loading hook: a directory without a config.json loads."""
    config = _tiny_llama_config()
    _save(tmp_path, config, {**LlamaForCausalLM(config).state_dict(), STRAY: torch.ones(32)})
    (tmp_path / "config.json").unlink()
    (tmp_path / "generation_config.json").write_text("{}")  # from_pretrained reads it instead

    model = _load(tmp_path, config=config)

    assert model._modelopt_unplaced_source_keys == [STRAY]


def test_an_ignore_rule_does_not_hide_an_unplaced_tensor(tmp_path):
    """Qwen3-Next ignores ``^mtp.*``, DeepSeek-V3 and GLM their MTP layers.

    Transformers drops what those rules match from its own ``unexpected_keys``, so a list taken
    from the loader came back empty for exactly the MTP heads the export exists to carry.
    """

    class IgnoresMtp(LlamaForCausalLM):
        _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    config = _tiny_llama_config()
    _save(
        tmp_path, config, {**LlamaForCausalLM(config).state_dict(), "mtp.fc.weight": torch.ones(2)}
    )

    model = _load(tmp_path, IgnoresMtp)

    assert model._modelopt_unplaced_source_keys == ["mtp.fc.weight"]


def test_a_tensor_the_index_omits_is_recorded(tmp_path):
    """The index is not an inventory: a shard can hold a tensor it does not list, such as an MTP
    head stored in a main shard. The loader still sees it, and so must the record."""
    config = _tiny_llama_config()
    state = LlamaForCausalLM(config).state_dict()
    shard = "model-00001-of-00001.safetensors"
    save_file({**state, "mtp.fc.weight": torch.ones(2)}, str(tmp_path / shard))
    index = {"metadata": {}, "weight_map": dict.fromkeys(state, shard)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))

    model = _load(tmp_path)

    assert model._modelopt_unplaced_source_keys == ["mtp.fc.weight"]


def test_experts_of_a_missing_layer_are_recorded_by_their_source_names(tmp_path):
    """The record names the tensors on disk, not what converting them would have produced.

    Per-expert tensors of a layer the model does not build (an MTP layer, say) convert to one fused
    ``...experts.gate_up_proj``, which no shard holds; the export can only carry the sources.
    """
    config = AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        tie_word_embeddings=False,
        architectures=["Qwen3MoeForCausalLM"],
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path)
    shapes = {"gate_proj": (16, 32), "up_proj": (16, 32), "down_proj": (32, 16)}
    stray = {
        f"model.layers.1.mlp.experts.{e}.{proj}.weight": torch.ones(shape)
        for e in range(2)
        for proj, shape in shapes.items()
    }
    weights = tmp_path / "model.safetensors"
    save_file({**load_file(str(weights)), **stray}, str(weights), metadata={"format": "pt"})

    model = _load(tmp_path)

    assert model._modelopt_unplaced_source_keys == sorted(stray)


def test_fp8_block_scales_folded_in_at_load_are_placed(tmp_path):
    """DeepSeek-V3 style FP8: dequantizing folds every ``weight_scale_inv`` into its weight."""
    # Every projection a whole number of 128x128 blocks: transformers before 5.10.1 cannot
    # dequantize a partial block (a 2-head k_proj would be 64x128).
    config = _tiny_llama_config(hidden_size=128, intermediate_size=256, num_key_value_heads=4)
    state = {STRAY: torch.ones(config.hidden_size, dtype=torch.bfloat16)}
    for name, tensor in LlamaForCausalLM(config).to(torch.bfloat16).state_dict().items():
        if name.endswith("proj.weight"):
            # One 128x128 block per 128-wide tile; the values only need to be valid FP8.
            blocks = (-(-tensor.shape[0] // 128), -(-tensor.shape[1] // 128))
            state[name] = tensor.to(torch.float8_e4m3fn)
            state[name.removesuffix("weight") + "weight_scale_inv"] = torch.ones(blocks)
        else:
            state[name] = tensor
    fp8 = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
    _save(tmp_path, config, state, {**fp8, "weight_block_size": [128, 128]})

    model = _load(
        tmp_path, quantization_config=FineGrainedFP8Config(dequantize=True), dtype=torch.bfloat16
    )

    assert model._modelopt_unplaced_source_keys == [STRAY]


def test_mxfp4_expert_blocks_folded_in_at_load_are_placed(tmp_path):
    """gpt-oss, loaded as examples/hf_ptq loads it: ``Mxfp4Config(dequantize=True)`` folds each
    expert's ``*_blocks`` and ``*_scales`` into a bf16 ``gate_up_proj`` / ``down_proj``."""
    hidden, inter, experts = 64, 64, 4
    config = GptOssConfig(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=experts,
        num_experts_per_tok=2,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
    )
    packed = {"gate_up_proj": (2 * inter, hidden), "down_proj": (hidden, inter)}
    state = {STRAY: torch.ones(hidden, dtype=torch.bfloat16)}
    for name, tensor in GptOssForCausalLM(config).to(torch.bfloat16).state_dict().items():
        proj = name.rsplit(".", 1)[-1]
        if ".experts." in name and proj in packed:
            rows, cols = packed[proj]
            grid = (experts, rows, cols // 32)  # one E8M0 scale per 32-value block
            state[f"{name}_blocks"] = torch.zeros(*grid, 16, dtype=torch.uint8)
            state[f"{name}_scales"] = torch.full(grid, 127, dtype=torch.uint8)
        else:
            state[name] = tensor
    # The modules gpt-oss ships unquantized.
    bf16 = [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ]
    _save(tmp_path, config, state, {"quant_method": "mxfp4", "modules_to_not_convert": bf16})

    model = _load(tmp_path, quantization_config=Mxfp4Config(dequantize=True), dtype=torch.bfloat16)

    assert model._modelopt_unplaced_source_keys == [STRAY]
