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

"""The meta-device + torch-DCP checkpoint loader in ``modelopt.torch.utils.distributed``.

``_meta_dcp_load`` materializes a model built on ``meta`` and fills it from HF safetensors through
``torch.distributed.checkpoint``; ``_map_ckpt_keys_to_model`` bridges checkpoint names to the built
model's ``state_dict`` names; ``_fsdp2_wrap_modules`` picks the blocks FSDP2 shards. The loads run
for real on CPU, from tiny transformers models into hand-written safetensors files. DCP loads in a
single process without a process group; the two tests that need the distributed path start a
one-rank gloo group.
"""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import get_tiny_llama
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText, Qwen2_5_VLConfig

from modelopt.torch.utils.plugins.model_load_utils import (
    _fsdp2_wrap_modules,
    _map_ckpt_keys_to_model,
    _meta_dcp_load,
)


def _save(ckpt_dir, tensors):
    """Write ``tensors`` as a one-file HF safetensors checkpoint in ``ckpt_dir``; return its path."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().clone() for k, v in tensors.items()}
    save_file(tensors, str(ckpt_dir / "model.safetensors"))
    return str(ckpt_dir)


def _meta_twin(build, **kwargs):
    """``build(**kwargs)`` with params on ``meta`` and buffers real, as the loader's callers hold it."""
    with init_empty_weights():
        return build(**kwargs)


def _tiny_qwen2_5_vl():
    """A tiny Qwen2.5-VL: the LM nests at ``model.language_model``, the vision tower at ``model.visual``."""
    cfg = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "max_position_embeddings": 16,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "fullatt_block_indexes": [1],
        },
        tie_word_embeddings=False,
    )
    return AutoModelForImageTextToText.from_config(cfg)


def _flat_vl_name(key):
    """The name a Qwen2.5-VL checkpoint stores ``key`` under: ``model.layers.*`` and ``visual.*``."""
    if key.startswith("model.language_model."):
        return "model." + key.removeprefix("model.language_model.")
    return key.removeprefix("model.") if key.startswith("model.visual.") else key


# ---------------------------------------------------------------- checkpoint -> model key mapping


def test_exact_name_match_takes_precedence():
    """A checkpoint name the model has verbatim maps to itself before any conversion pattern runs."""
    model = nn.Module()
    model._checkpoint_conversion_mapping = {r"^model\.": "model.language_model."}
    model_keys = {"model.norm.weight", "model.language_model.norm.weight"}

    key_map = _map_ckpt_keys_to_model(model, {"model.norm.weight"}, model_keys)

    assert key_map == {"model.norm.weight": "model.norm.weight"}


def test_conversion_mapping_applies_the_first_pattern_that_fires():
    """Patterns are tried in order and only the first that fires rewrites a name (no chaining); the
    rewrite counts only if the model has the result, otherwise the name falls through to suffix match.
    """
    model = nn.Module()
    model._checkpoint_conversion_mapping = {
        r"^model\.(?!language_model\.)": "model.language_model.",
        r"^model\.language_model\.": "chained.",
        r"^visual\.": "model.vision_tower.",  # this model calls it ``model.visual``
    }
    model_keys = {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "chained.layers.0.mlp.down_proj.weight",
        "model.visual.proj.weight",
    }
    ckpt_keys = {"model.layers.0.mlp.down_proj.weight", "visual.proj.weight"}

    assert _map_ckpt_keys_to_model(model, ckpt_keys, model_keys) == {
        "model.layers.0.mlp.down_proj.weight": "model.language_model.layers.0.mlp.down_proj.weight",
        "visual.proj.weight": "model.visual.proj.weight",
    }


def test_suffix_match_resolves_the_vlm_prefix_nesting():
    """transformers>=5 nests a VLM's LM and vision tower one level below its checkpoint names, and the
    model carries no ``_checkpoint_conversion_mapping``. The longest component suffix naming exactly
    one model key bridges that: ``layers.0`` tells LM layer 0 from vision block 0, which shares the rest.
    """
    model_keys = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.1.mlp.down_proj.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.mlp.down_proj.weight",
        "lm_head.weight",
    }
    expected = {
        "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
        "model.layers.0.mlp.down_proj.weight": "model.language_model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight": "model.language_model.layers.1.mlp.down_proj.weight",
        "model.norm.weight": "model.language_model.norm.weight",
        "visual.blocks.0.mlp.down_proj.weight": "model.visual.blocks.0.mlp.down_proj.weight",
        "lm_head.weight": "lm_head.weight",
    }

    assert _map_ckpt_keys_to_model(nn.Module(), expected.keys(), model_keys) == expected


def test_ambiguous_or_unknown_suffixes_stay_unmapped():
    """A name whose suffixes match several model keys (two towers both have ``layers.0``) or none (a
    quantization scale with no slot) is left for the coverage guard, never assigned to a guess.
    """
    model_keys = {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.audio_tower.layers.0.mlp.down_proj.weight",
    }
    ckpt_keys = {
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale",
    }

    assert _map_ckpt_keys_to_model(nn.Module(), ckpt_keys, model_keys) == {}


def test_a_key_the_model_has_no_place_for_never_displaces_a_match():
    """DeepSeek-V3's MTP layer and the layers a truncated model lacks have no slot in the model, yet
    their suffixes name a key uniquely -- one another checkpoint key already names exactly. Mapped,
    they would overwrite that key's tensor; they must stay unmapped.
    """
    model_keys = {"model.norm.weight", "model.layers.0.input_layernorm.weight"}
    ckpt_keys = model_keys | {
        "model.layers.61.shared_head.norm.weight",
        "model.layers.1.input_layernorm.weight",
    }

    assert _map_ckpt_keys_to_model(nn.Module(), ckpt_keys, model_keys) == {
        key: key for key in model_keys
    }


def test_two_keys_claiming_one_model_key_map_neither():
    """Two checkpoint names reaching the same model key by suffix are equally plausible; loading
    either would let set order pick the tensor, so neither is mapped."""
    model_keys = {"model.language_model.norm.weight"}

    assert (
        _map_ckpt_keys_to_model(nn.Module(), {"model.norm.weight", "mtp.norm.weight"}, model_keys)
        == {}
    )


# -------------------------------------------------------------------------- FSDP2 wrap boundaries


def test_wrap_modules_are_the_decoder_layers():
    """A causal LM is wrapped at its ``_no_split_modules`` blocks: the decoder layers, in order."""
    model = _meta_twin(get_tiny_llama)

    assert _fsdp2_wrap_modules(model) == list(model.model.layers)


def test_vlm_wraps_only_its_language_layers():
    """Qwen2.5-VL's wrapper also lists the vision block in ``_no_split_modules``, but only the language
    layers may be sharded: sharded vision weights deadlock calibration, so the tower stays replicated.
    """
    model = _meta_twin(_tiny_qwen2_5_vl)
    assert type(model.model.visual.blocks[0]).__name__ in model._no_split_modules

    assert _fsdp2_wrap_modules(model) == list(model.model.language_model.layers)


@pytest.mark.parametrize("no_split_modules", [None, ["NoSuchBlock"]])
def test_wrap_modules_fall_back_to_decoder_layer_detection(no_split_modules):
    """Without usable ``_no_split_modules`` (absent, or naming no module present), modelopt's
    decoder-layer auto-detection still finds the blocks.
    """
    model = _meta_twin(get_tiny_llama)
    model._no_split_modules = no_split_modules

    assert _fsdp2_wrap_modules(model) == list(model.model.layers)


def test_wrap_modules_refuse_a_model_without_blocks():
    """With neither source of boundaries matching, fail loudly instead of guessing a layout."""
    with pytest.raises(RuntimeError, match="Could not determine FSDP2 wrap boundaries"):
        _fsdp2_wrap_modules(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)))


# --------------------------------------------------------------------------------- meta + DCP load


def test_load_fills_a_meta_model_keeps_its_buffers_and_reties_it(tmp_path):
    """Whole-model load: every checkpoint tensor lands, the real rotary buffers survive ``to_empty``,
    and the tied ``lm_head`` the checkpoint omits is re-tied instead of left as empty memory.
    """
    src = get_tiny_llama(tie_word_embeddings=True)
    ckpt = {k: v for k, v in src.state_dict().items() if k != "lm_head.weight"}
    ckpt_dir = _save(tmp_path, ckpt)
    model = _meta_twin(get_tiny_llama, tie_word_embeddings=True)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    assert buffers and all(p.is_meta for p in model.parameters())

    _meta_dcp_load(model, ckpt_dir, "cpu")

    loaded = model.state_dict()
    for name, value in ckpt.items():
        assert torch.equal(loaded[name], value), name
    for name, buf in model.named_buffers():
        assert torch.equal(buf, buffers[name]), name
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_flat_vlm_checkpoint_names_reach_the_nested_params(tmp_path):
    """Qwen2.5-VL's checkpoint says ``model.layers.*`` / ``visual.*`` where the built model has
    ``model.language_model.*`` / ``model.visual.*``; a by-name load would fill none of them.
    """
    src = _tiny_qwen2_5_vl()
    ref = src.state_dict()
    ckpt = {_flat_vl_name(k): v for k, v in ref.items()}
    assert ckpt.keys() & ref.keys() == {"lm_head.weight"}  # every other name differs
    model = _meta_twin(_tiny_qwen2_5_vl)

    _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu")

    loaded = model.state_dict()
    assert loaded.keys() == ref.keys()
    for name, value in ref.items():
        assert torch.equal(loaded[name], value), name


def test_load_names_leave_already_real_params_untouched(tmp_path):
    """EP-dense: with ``load_names`` only those meta params are materialized and loaded. The MLPs stand
    in for experts ``from_pretrained`` already placed; the checkpoint holds other values for them, and
    they must keep their own (same tensors, neither re-emptied nor reloaded).
    """
    src = get_tiny_llama()
    ckpt_dir = _save(tmp_path, src.state_dict())
    model = _meta_twin(get_tiny_llama)
    experts = {}
    for i, layer in enumerate(model.model.layers):
        layer.mlp.to_empty(device="cpu")
        for name, param in layer.mlp.named_parameters():
            nn.init.constant_(param, -7.0)
            experts[f"model.layers.{i}.mlp.{name}"] = param
    dense = {n for n, _ in model.named_parameters()} - experts.keys()

    _meta_dcp_load(model, ckpt_dir, "cpu", load_names=dense)

    params = dict(model.named_parameters())
    for name, param in experts.items():
        assert params[name] is param and bool((param == -7.0).all()), name
    ref = src.state_dict()
    for name in dense:
        assert torch.equal(params[name], ref[name]), name


class _Router(nn.Module):
    """A router whose bias correction is a param ``from_pretrained`` loads, beside a meta weight."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))
        self.e_score_correction_bias = nn.Parameter(torch.arange(4.0))


def test_a_real_param_beside_a_meta_one_keeps_its_value(tmp_path):
    """Materializing the router re-empties all of its params, but only ``weight`` is reloaded."""
    model = nn.Module()
    model.gate = _Router()
    ckpt = {"gate.weight": torch.randn(4, 4), "gate.e_score_correction_bias": torch.zeros(4)}

    _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu", load_names={"gate.weight"})

    assert torch.equal(model.gate.weight, ckpt["gate.weight"])
    assert torch.equal(model.gate.e_score_correction_bias, torch.arange(4.0))


class _DenseAndExpert(nn.Module):
    """A dense param the EP-dense load fills and an expert one ``from_pretrained`` already placed."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=False)
        self.expert = nn.Linear(4, 4, bias=False)


def test_load_names_judge_suffixes_against_the_whole_model(tmp_path):
    """Among the dense names alone ``weight`` names only ``dense.weight``, so a checkpoint key with no
    slot in the model (a per-expert ``experts.0.weight``) would land on it. Across the whole model
    ``weight`` also names the expert, so that key stays unmapped and the dense param gets its own.
    """
    ckpt = {"dense.weight": torch.randn(4, 4), "experts.0.weight": torch.randn(4, 4)}
    model = _meta_twin(_DenseAndExpert)
    model.expert.to_empty(device="cpu")

    _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu", load_names={"dense.weight"})

    assert torch.equal(model.dense.weight, ckpt["dense.weight"])


def test_a_truncated_model_keeps_its_own_layers(tmp_path):
    """A 1-layer model loading a 2-layer checkpoint gets the checkpoint's layer 0. Layer 1's names
    reach layer 0's keys by suffix alone and must not overwrite them."""
    src = get_tiny_llama()
    model = _meta_twin(get_tiny_llama, num_hidden_layers=1)

    _meta_dcp_load(model, _save(tmp_path, src.state_dict()), "cpu")

    ref = src.state_dict()
    for name, value in model.state_dict().items():
        assert torch.equal(value, ref[name]), name


class _WithBuffers(nn.Module):
    """A linear, a persistent buffer (checkpoints carry it) and a non-persistent one (built at init)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("table", torch.arange(4.0), persistent=False)


def test_persistent_buffers_load_from_the_checkpoint_and_others_survive(tmp_path):
    """``to_empty`` wipes buffers too: one the checkpoint carries is loaded from it, and one no
    checkpoint can carry (non-persistent) gets its pre-load value back.
    """
    ckpt = {
        "proj.weight": torch.randn(4, 4),
        "proj.bias": torch.randn(4),
        "scale": torch.full((4,), 3.0),
    }
    with init_empty_weights():
        model = _WithBuffers()

    _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu")

    assert torch.equal(model.scale, ckpt["scale"])
    assert torch.equal(model.table, torch.arange(4.0))
    assert torch.equal(model.proj.weight, ckpt["proj.weight"])


class _OneBigManySmall(nn.Module):
    """One param holding ~99% of the weight beside twelve tiny ones."""

    def __init__(self):
        super().__init__()
        self.big = nn.Linear(64, 64, bias=False)
        self.small = nn.ModuleList(nn.Linear(2, 2) for _ in range(6))


_SMALL = {f"small.{i}.{kind}" for i in range(6) for kind in ("weight", "bias")}


def _guard_case(ckpt_dir, missing):
    """A meta ``_OneBigManySmall`` and a checkpoint with all of its tensors except ``missing``."""
    with init_empty_weights():
        model = _OneBigManySmall()
    ckpt = {k: torch.randn(v.shape) for k, v in model.state_dict().items() if k not in missing}
    return model, ckpt, _save(ckpt_dir, ckpt)


def test_coverage_guard_weighs_unloaded_params_by_numel(tmp_path):
    """Missing the one ~99% param fails loudly while missing the twelve tiny ones (~1%) loads. A
    count-based guard gets both backwards, and the fused-expert miss it exists for is a few huge params.
    """
    model, _, ckpt_dir = _guard_case(tmp_path / "big_missing", missing={"big.weight"})
    with pytest.raises(RuntimeError, match="matched no checkpoint key"):
        _meta_dcp_load(model, ckpt_dir, "cpu")

    model, ckpt, ckpt_dir = _guard_case(tmp_path / "small_missing", missing=_SMALL)
    _meta_dcp_load(model, ckpt_dir, "cpu")
    assert torch.equal(model.big.weight, ckpt["big.weight"])


def test_skip_names_exempt_params_filled_by_another_path(tmp_path):
    """Params a separate loader fills (fused grouped experts) are left out of the guard's tally."""
    model, ckpt, ckpt_dir = _guard_case(tmp_path, missing={"big.weight"})

    _meta_dcp_load(model, ckpt_dir, "cpu", skip_names={"big.weight"})

    assert torch.equal(model.small[0].weight, ckpt["small.0.weight"])


def test_max_unloaded_frac_sets_the_guard_threshold(tmp_path):
    """The ~1% shortfall the default tolerates fails once the tolerance is zero."""
    model, _, ckpt_dir = _guard_case(tmp_path, missing=_SMALL)

    with pytest.raises(RuntimeError, match="matched no checkpoint key"):
        _meta_dcp_load(model, ckpt_dir, "cpu", max_unloaded_frac=0.0)


def test_checkpoint_without_tensors_is_rejected(tmp_path):
    """A weightless (config-only) snapshot fails with its own message, not as an all-unloaded model."""
    (tmp_path / "config.json").write_text("{}")
    with init_empty_weights():
        model = _OneBigManySmall()

    with pytest.raises(RuntimeError, match="has NO tensors"):
        _meta_dcp_load(model, str(tmp_path), "cpu")


def test_load_request_follows_the_model_state_dict_order(tmp_path, monkeypatch):
    """``dcp.load`` gets the checkpoint names in the model's ``state_dict`` order, which every rank
    shares. Key-map or set order is hash-seeded per process, and differing orders across ranks
    deadlock the collective load planner.
    """
    src = _tiny_qwen2_5_vl()
    ckpt_dir = _save(tmp_path, {_flat_vl_name(k): v for k, v in src.state_dict().items()})
    model = _meta_twin(_tiny_qwen2_5_vl)
    requests = []
    real_load = dcp.load

    def recording_load(state_dict, **kwargs):
        requests.append(list(state_dict))
        return real_load(state_dict, **kwargs)

    monkeypatch.setattr(dcp, "load", recording_load)
    _meta_dcp_load(model, ckpt_dir, "cpu")

    assert requests == [[_flat_vl_name(k) for k in model.state_dict()]]


# ------------------------------------------------------------------- under a process group


@pytest.fixture
def gloo_world_of_one(skip_on_windows, tmp_path_factory):
    """A one-rank gloo process group, so the loader takes its distributed path."""
    store = tmp_path_factory.mktemp("pg") / "store"
    torch.distributed.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
    yield
    torch.distributed.destroy_process_group()


def test_rank0_reports_unloaded_params_and_loaded_buffers(tmp_path, gloo_world_of_one):
    """In a distributed run rank 0 names the params the checkpoint left unfilled (below the guard's
    bar) and the persistent buffers it filled.
    """
    ckpt = {"proj.weight": torch.randn(4, 4), "scale": torch.full((4,), 3.0)}
    with init_empty_weights():
        model = _WithBuffers()

    with pytest.warns(UserWarning) as record:
        _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu")

    messages = [str(w.message) for w in record]
    assert any("1 param(s) not loaded" in m and "'proj.bias'" in m for m in messages), messages
    assert any("1 persistent buffer(s)" in m and "'scale'" in m for m in messages), messages


def test_fsdp2_sharded_params_are_filled_in_place(tmp_path, gloo_world_of_one):
    """FSDP2-full: the blocks and the root are ``fully_shard``-ed on meta before the load. The params
    must stay sharded ``DTensor``s holding the checkpoint values, and the buffers must survive the
    root ``to_empty``.
    """
    src = get_tiny_llama()
    ckpt_dir = _save(tmp_path, src.state_dict())
    model = _meta_twin(get_tiny_llama)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    mesh = init_device_mesh("cpu", (1,))
    for block in _fsdp2_wrap_modules(model):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)

    _meta_dcp_load(model, ckpt_dir, "cpu")

    ref = src.state_dict()
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), name
        assert torch.equal(param.full_tensor(), ref[name]), name
    for name, buf in model.named_buffers():
        assert torch.equal(buf, buffers[name]), name
