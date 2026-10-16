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
model's ``state_dict`` names. The loads run for real on CPU, from tiny transformers models into
hand-written safetensors files. DCP loads in a single process without a process group; the tests that
need the distributed path start a one-rank gloo group.
"""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import get_tiny_llama
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText, Qwen2_5_VLConfig

from modelopt.torch.utils.plugins.model_load_utils import (
    _copy_tied_values,
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


def _every_name(model):
    """Every ``state_dict`` name of ``model``: what a load into a model with nothing placed fills."""
    return set(model.state_dict())


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


# --------------------------------------------------------------------------------- meta + DCP load


def test_load_fills_a_meta_model_keeps_its_buffers_and_reties_it(tmp_path):
    """Every checkpoint tensor lands, the real rotary buffers survive ``to_empty``, and the tied
    ``lm_head`` the checkpoint omits is re-tied instead of left as empty memory.
    """
    src = get_tiny_llama(tie_word_embeddings=True)
    ckpt = {k: v for k, v in src.state_dict().items() if k != "lm_head.weight"}
    ckpt_dir = _save(tmp_path, ckpt)
    model = _meta_twin(get_tiny_llama, tie_word_embeddings=True)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    assert buffers and all(p.is_meta for p in model.parameters())

    _meta_dcp_load(model, ckpt_dir, "cpu", load_names=_every_name(model))

    loaded = model.state_dict()
    for name, value in ckpt.items():
        assert torch.equal(loaded[name], value), name
    for name, buf in model.named_buffers():
        assert torch.equal(buf, buffers[name]), name
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize("stored", ["model.embed_tokens.weight", "lm_head.weight"])
def test_a_tie_keeps_the_checkpoint_values_whichever_alias_holds_them(
    tmp_path, monkeypatch, stored
):
    """A tied checkpoint stores one alias; tying keeps the embedding's parameter. When the checkpoint
    stored only ``lm_head.weight``, the kept parameter must take its values rather than keep the memory
    ``to_empty`` gave it, as ``from_pretrained`` leaves it."""
    src = get_tiny_llama(tie_word_embeddings=True)
    tied = {"model.embed_tokens.weight", "lm_head.weight"}
    ckpt = {k: v for k, v in src.state_dict().items() if k not in tied - {stored}}
    ckpt_dir = _save(tmp_path, ckpt)
    model = _meta_twin(get_tiny_llama, tie_word_embeddings=True)
    # Fresh storage holds NaN, so a parameter nothing wrote to shows.
    empty_like = torch.empty_like
    monkeypatch.setattr(
        torch, "empty_like", lambda t, **kw: empty_like(t, **kw).fill_(float("nan"))
    )

    _meta_dcp_load(model, ckpt_dir, "cpu", load_names=_every_name(model))

    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(model.lm_head.weight, ckpt[stored])


def test_flat_vlm_checkpoint_names_reach_the_nested_params(tmp_path):
    """Qwen2.5-VL's checkpoint says ``model.layers.*`` / ``visual.*`` where the built model has
    ``model.language_model.*`` / ``model.visual.*``; a by-name load would fill none of them.
    """
    src = _tiny_qwen2_5_vl()
    ref = src.state_dict()
    ckpt = {_flat_vl_name(k): v for k, v in ref.items()}
    assert ckpt.keys() & ref.keys() == {"lm_head.weight"}  # every other name differs
    model = _meta_twin(_tiny_qwen2_5_vl)

    _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu", load_names=_every_name(model))

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


def test_target_params_that_already_had_values_need_no_checkpoint_tensor(tmp_path):
    """A target param that was real before the load -- beside a meta one, so restored after
    ``to_empty``, or in a module the load never re-empties -- keeps its value when the checkpoint
    lacks it, instead of failing as unfilled."""
    model = nn.Module()
    model.gate = _Router()
    model.expert = nn.Linear(2, 2, bias=False)
    expert = model.expert.weight.detach().clone()
    ckpt = {"gate.weight": torch.randn(4, 4)}

    _meta_dcp_load(
        model,
        _save(tmp_path, ckpt),
        "cpu",
        load_names={"gate.weight", "gate.e_score_correction_bias", "expert.weight"},
    )

    assert torch.equal(model.gate.weight, ckpt["gate.weight"])
    assert torch.equal(model.gate.e_score_correction_bias, torch.arange(4.0))
    assert torch.equal(model.expert.weight, expert)


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

    _meta_dcp_load(model, _save(tmp_path, src.state_dict()), "cpu", load_names=_every_name(model))

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
    checkpoint can carry (non-persistent) gets its pre-load value back, even one that requires grad.
    """
    ckpt = {
        "proj.weight": torch.randn(4, 4),
        "proj.bias": torch.randn(4),
        "scale": torch.full((4,), 3.0),
    }
    with init_empty_weights():
        model = _WithBuffers()
    model.table.requires_grad_(True)  # restoring it in place must not be recorded by autograd

    _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu", load_names=_every_name(model))

    assert torch.equal(model.scale, ckpt["scale"])
    assert torch.equal(model.table, torch.arange(4.0))
    assert torch.equal(model.proj.weight, ckpt["proj.weight"])


def test_an_untied_param_the_checkpoint_lacks_fails_the_load(tmp_path):
    """``proj.bias`` has no checkpoint tensor and tying fills nothing: it still holds the memory
    ``to_empty`` gave it, so the load fails naming it instead of returning the model."""
    ckpt = {"proj.weight": torch.randn(4, 4), "scale": torch.full((4,), 3.0)}
    with init_empty_weights():
        model = _WithBuffers()

    with pytest.raises(
        RuntimeError, match=r"1 param\(s\) have no checkpoint tensor.*\['proj.bias'\]"
    ):
        _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu", load_names=_every_name(model))


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


def test_coverage_guard_weighs_unloaded_params_by_numel(tmp_path, monkeypatch):
    """Missing the one ~99% param fails before the load; missing the twelve tiny ones (~1%) gets past
    that guard and fails after the load instead. A count-based guard gets both backwards, and the
    fused-expert miss it exists for is a few huge params.
    """
    loads = []
    real_load = dcp.load

    def counting_load(state_dict, **kwargs):
        loads.append(len(state_dict))
        return real_load(state_dict, **kwargs)

    monkeypatch.setattr(dcp, "load", counting_load)
    model, _, ckpt_dir = _guard_case(tmp_path / "big_missing", missing={"big.weight"})
    with pytest.raises(RuntimeError, match="matched no checkpoint key"):
        _meta_dcp_load(model, ckpt_dir, "cpu", load_names=_every_name(model))
    assert not loads

    model, _, ckpt_dir = _guard_case(tmp_path / "small_missing", missing=_SMALL)
    with pytest.raises(RuntimeError, match="12 param\\(s\\) have no checkpoint tensor"):
        _meta_dcp_load(model, ckpt_dir, "cpu", load_names=_every_name(model))
    assert loads == [1]


def test_checkpoint_without_tensors_is_rejected(tmp_path):
    """A weightless (config-only) snapshot fails with its own message, not as an all-unloaded model."""
    (tmp_path / "config.json").write_text("{}")
    with init_empty_weights():
        model = _OneBigManySmall()

    with pytest.raises(RuntimeError, match="has NO tensors"):
        _meta_dcp_load(model, str(tmp_path), "cpu", load_names=_every_name(model))


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
    _meta_dcp_load(model, ckpt_dir, "cpu", load_names=_every_name(model))

    assert requests == [[_flat_vl_name(k) for k in model.state_dict()]]


# ------------------------------------------------------------------- under a process group


@pytest.fixture
def gloo_world_of_one(skip_on_windows, tmp_path_factory):
    """A one-rank gloo process group, so the loader takes its distributed path."""
    store = tmp_path_factory.mktemp("pg") / "store"
    torch.distributed.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
    yield
    torch.distributed.destroy_process_group()


def test_rank0_reports_the_persistent_buffers_it_loaded(tmp_path, gloo_world_of_one):
    """In a distributed run rank 0 names the persistent buffers the checkpoint filled."""
    ckpt = {"proj.weight": torch.randn(4, 4), "proj.bias": torch.randn(4), "scale": torch.ones(4)}
    with init_empty_weights():
        model = _WithBuffers()

    with pytest.warns(UserWarning, match=r"1 persistent buffer\(s\).*'scale'"):
        _meta_dcp_load(model, _save(tmp_path, ckpt), "cpu", load_names=_every_name(model))


def test_fsdp2_sharded_params_are_filled_in_place(tmp_path, gloo_world_of_one):
    """The EP hybrid's layout: the decoder layers are ``fully_shard``-ed on meta before the load, the
    root is not. The layers' params must stay sharded ``DTensor``s holding the checkpoint values, the
    root's plain params must hold theirs, and the buffers must survive ``to_empty``.
    """
    src = get_tiny_llama()
    ckpt_dir = _save(tmp_path, src.state_dict())
    model = _meta_twin(get_tiny_llama)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    mesh = init_device_mesh("cpu", (1,))
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)

    _meta_dcp_load(model, ckpt_dir, "cpu", load_names=_every_name(model))

    ref = src.state_dict()
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor) == name.startswith("model.layers."), name
        loaded = param.full_tensor() if isinstance(param, DTensor) else param
        assert torch.equal(loaded, ref[name]), name
    for name, buf in model.named_buffers():
        assert torch.equal(buf, buffers[name]), name


def _tie_pair(layout):
    """A kept param and the filled one it replaced, laid out as ``layout`` names."""
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    kept, values = torch.zeros(4, 4), torch.arange(16.0).reshape(4, 4)
    if layout == "dtensor-vs-plain":
        return distribute_tensor(kept, mesh, [Shard(0)]), values
    if layout == "plain-vs-dtensor":
        return kept, distribute_tensor(values, mesh, [Shard(0)])
    if layout == "other-placements":
        return distribute_tensor(kept, mesh, [Shard(0)]), distribute_tensor(
            values, mesh, [Replicate()]
        )
    if layout == "other-mesh":
        other = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
        return distribute_tensor(kept, mesh, [Shard(0)]), distribute_tensor(
            values, other, [Shard(0)]
        )
    return kept, values.reshape(2, 8)  # other-shape


@pytest.mark.parametrize(
    "layout",
    ["dtensor-vs-plain", "plain-vs-dtensor", "other-placements", "other-mesh", "other-shape"],
)
def test_a_tie_laid_out_differently_is_refused_and_left_as_it_was(gloo_world_of_one, layout):
    """Carrying a tied param's values over is a local copy. A pair it would have to redistribute is
    refused rather than counted as loaded, and the kept param keeps what it held."""
    kept, src = _tie_pair(layout)
    local = kept.to_local() if isinstance(kept, DTensor) else kept
    before = local.clone()

    with pytest.raises(RuntimeError, match="laid out differently"):
        _copy_tied_values(kept, src, "lm_head.weight")

    assert torch.equal(local, before)
