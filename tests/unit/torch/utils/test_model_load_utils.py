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

"""Tests for ``modelopt.torch.utils.plugins.model_load_utils``.

Nothing here needs a GPU. The tests of ``resolve_distribution_plan`` and of the loads
``load_pretrained`` routes a plan to answer the process group's rank and size as a torchrun launch
would, and serve tiny real transformers configs from memory: the decision builds their real models on
the meta device, and stand-ins for the sharded loaders record how they were called. Only the last
tests of each part use a process group, two gloo ranks.
"""

import inspect
import math
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from _test_utils.torch.moe_models import model_with_decoder_at
from safetensors.torch import save_file

pytest.importorskip("accelerate")
transformers = pytest.importorskip("transformers")
from _test_utils.torch.transformers_models import (
    _TinyQwen35Tokenizer,
    get_tiny_qwen3_5_moe_vl,
    get_tiny_qwen3_5_vl,
)
from packaging.version import Version
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    LlamaConfig,
    LlamaForCausalLM,
    Phi4MultimodalConfig,
    Qwen2_5_VLConfig,
    Qwen3MoeConfig,
    T5Config,
    WhisperConfig,
)

import modelopt.torch.models.hf as models_hf
import modelopt.torch.utils.distributed as dist
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.distributed import DistributionPlan, distribution_plan
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    _auto_class,
    _conversion_plan,
    _convert_keys,
    _resolve_target,
    build_meta_causal_lm,
    load_pretrained,
    record_unplaced_source_keys,
    resolve_distribution_plan,
    unplaced_source_keys,
)


def _build_tiny_qwen3_moe():
    """A tiny meta-init Qwen3-MoE (fused ``gate_up_proj`` + ``down_proj`` experts) for converter tests."""
    pytest.importorskip("transformers")
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3_moe",
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        max_position_embeddings=16,
    )
    try:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:  # modeling class unavailable in this transformers build
        pytest.skip(f"qwen3_moe modeling unavailable: {e}")
    return model, cfg


def test_checkpoint_key_converter_multisource_expert_fusion():
    """A multi-source fused MoE (Qwen3: gate_proj+up_proj -> gate_up_proj) converts correctly.

    Exercises the multi-source path (gate/up concatenated after expert stacking) AND the
    single-source path (down_proj is a plain expert stack) in one model.
    """
    model, cfg = _build_tiny_qwen3_moe()
    plan = _conversion_plan(model)
    assert plan is not None

    names = dict(model.named_parameters())
    gname = next(n for n in names if n.endswith("mlp.experts.gate_up_proj"))
    prefix = gname[: -len("mlp.experts.gate_up_proj")]
    n_exp, inter, hidden = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size

    gate = [torch.randn(inter, hidden) for _ in range(n_exp)]
    up = [torch.randn(inter, hidden) for _ in range(n_exp)]
    down = [torch.randn(hidden, inter) for _ in range(n_exp)]
    state = {}
    for e in range(n_exp):
        state[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = gate[e]
        state[f"{prefix}mlp.experts.{e}.up_proj.weight"] = up[e]
        state[f"{prefix}mlp.experts.{e}.down_proj.weight"] = down[e]

    out = _convert_keys(plan, state)

    # Multi-source: experts stacked (dim 0) then gate|up concatenated (dim 1), gate first.
    gate_up = out[f"{prefix}mlp.experts.gate_up_proj"]
    assert gate_up.shape == (n_exp, 2 * inter, hidden) == tuple(names[gname].shape)
    assert torch.equal(gate_up, torch.cat([torch.stack(gate), torch.stack(up)], dim=1))

    # Single-source (regression): down_proj is a plain expert stack.
    down_proj = out[f"{prefix}mlp.experts.down_proj"]
    assert down_proj.shape == (n_exp, hidden, inter)
    assert torch.equal(down_proj, torch.stack(down))

    # Name-only mapping routes every source key to the fused target.
    for e in range(n_exp):
        assert _resolve_target(plan, f"{prefix}mlp.experts.{e}.gate_proj.weight")[0] == gname
        assert _resolve_target(plan, f"{prefix}mlp.experts.{e}.up_proj.weight")[0] == gname


@pytest.mark.parametrize("factory", [get_tiny_qwen3_5_vl, get_tiny_qwen3_5_moe_vl])
def test_unplaced_keys_preserve_registered_qwen_vlm_parameters(tmp_path, factory):
    model = factory(tokenizer=_TinyQwen35Tokenizer())
    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    state["mtp.eh_proj.weight"] = torch.zeros(2, 2)
    save_file(state, tmp_path / "model.safetensors")

    assert unplaced_source_keys(model, str(tmp_path)) == ["mtp.eh_proj.weight"]


# --- recording what the load could not place -----------------------------------------------------


def test_record_unplaced_source_keys_stores_sorted_keys_and_provenance():
    model = torch.nn.Linear(2, 2)
    keys = record_unplaced_source_keys(model, "/ckpt/path", ["z.weight", "a.weight"])

    assert keys == ["a.weight", "z.weight"]
    assert model._modelopt_unplaced_source_keys == ["a.weight", "z.weight"]
    assert model._modelopt_source_checkpoint == "/ckpt/path"


def test_record_unplaced_source_keys_accepts_any_iterable():
    model = torch.nn.Linear(2, 2)
    assert record_unplaced_source_keys(model, "/c", iter(["b", "a"])) == ["a", "b"]


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_meta_qwen_vlm_preserves_architecture_and_vision(model_type):
    try:
        config = AutoConfig.for_model(model_type)
    except ValueError:
        pytest.skip("Qwen3.5 requires a recent Transformers")
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["linear_attention", "full_attention"]
    config.vision_config.depth = 1
    model = build_meta_causal_lm("unused", False, "eager", config)
    assert type(model).__name__.endswith("ForConditionalGeneration")
    assert hasattr(model.model, "visual")
    assert (
        len(LayerActivationCollector.get_decoder_layers(model))
        == config.text_config.num_hidden_layers
    )
    assert all(parameter.is_meta for parameter in model.parameters())


# -------------------------------------------------------------- deciding how a model is distributed


_TINY = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
}

needs_qwen3_moe_ep_plan = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"),
    reason="qwen3_moe's base_model_ep_plan ships in transformers 5.7",
)


def _qwen3_moe(num_experts=8):
    """Qwen3-MoE, whose base_model_ep_plan ships with transformers itself."""
    return Qwen3MoeConfig(num_experts=num_experts, moe_intermediate_size=8, **_TINY)


def _bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _meta(config):
    """``config``'s model as the decision builds it, in the same dtype."""
    return model_load_utils.build_meta_causal_lm(None, False, None, config)


@pytest.fixture(autouse=True)
def _no_distribution_knobs(monkeypatch):
    """A shell that exports the knobs would otherwise steer every decision below."""
    monkeypatch.delenv("MODELOPT_EP_DEGREE", raising=False)
    monkeypatch.delenv("MODELOPT_FSDP_SIZE", raising=False)


def _gathered_alike(gathered, mine, group=None):
    """Answer a gather as if every rank handed it what this rank did."""
    gathered[:] = [mine] * len(gathered)


@pytest.fixture
def run_as(monkeypatch):
    """Run as rank ``rank`` of a ``world_size``-rank launch, deciding for itself.

    Every rank acts as its group's first, and reads the checkpoint as this one does. How the ranks
    agree on one plan, and on one load, is tested on two gloo ranks at the end of each part.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _gathered_alike)
    broadcast_from = dist.broadcast_from

    def broadcast_as_the_first(label, fn, src=0):
        # broadcast_from sends from dist.rank(), which process_group sets for the FSDP2 load.
        with monkeypatch.context() as sending:
            sending.setattr(dist, "rank", lambda group=None: src)
            return broadcast_from(label, fn, src)

    monkeypatch.setattr(dist, "broadcast_from", broadcast_as_the_first)

    def enter(world_size, rank=0):
        monkeypatch.setattr(dist, "is_distributed_run", lambda: world_size > 1)
        monkeypatch.setattr(dist, "size", lambda group=None: world_size)
        monkeypatch.setattr(dist, "is_master", lambda group=None: rank == 0)
        monkeypatch.setattr(dist, "local_rank", lambda: rank)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)

    return enter


@pytest.fixture
def serve(monkeypatch):
    """Answer every config read with ``config`` rather than a checkpoint on disk."""

    def serve(config):
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: config)
        return config

    return serve


@pytest.fixture
def gpu_memory(monkeypatch):
    """Give this rank a GPU of ``size`` bytes."""

    def give(size):
        monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: size)

    return give


# ------------------------------------------------------------------------------ model class


def _vlm_config(**kwargs):
    return SimpleNamespace(text_config=SimpleNamespace(), architectures=[], **kwargs)


def _tiny_qwen2_5_vl():
    return Qwen2_5_VLConfig(
        text_config={**_TINY, "max_position_embeddings": 16},
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "fullatt_block_indexes": [0],
        },
    )


_REMOTE_VLM = {"AutoModelForCausalLM": "m.KimiK25ForConditionalGeneration"}
_REMOTE_IMAGE_TEXT_VLM = {
    **_REMOTE_VLM,
    "AutoModelForImageTextToText": "m.KimiK25ForConditionalGeneration",
}


@pytest.mark.parametrize(
    ("config", "require_lm_head", "trust_remote_code", "expected"),
    [
        (SimpleNamespace(architectures=["LlamaForCausalLM"]), True, False, AutoModelForCausalLM),
        (_tiny_qwen2_5_vl(), False, False, AutoModel),
        (_vlm_config(auto_map=_REMOTE_VLM), True, True, AutoModelForCausalLM),
        (_vlm_config(auto_map=_REMOTE_VLM), True, False, AutoModel),
        (_vlm_config(auto_map=_REMOTE_IMAGE_TEXT_VLM), True, True, AutoModelForImageTextToText),
        (_tiny_qwen2_5_vl(), True, False, AutoModelForImageTextToText),
        (Phi4MultimodalConfig(), True, False, AutoModelForCausalLM),
        (_vlm_config(), True, False, AutoModel),
        (
            T5Config(architectures=["T5ForConditionalGeneration"]),
            True,
            False,
            AutoModelForSeq2SeqLM,
        ),
        (WhisperConfig(), True, False, AutoModelForSpeechSeq2Seq),
        (T5Config(architectures=["T5ForConditionalGeneration"]), False, False, AutoModel),
        (_vlm_config(auto_map=_REMOTE_VLM), False, True, AutoModelForCausalLM),
        (
            _vlm_config(auto_map={**_REMOTE_VLM, "AutoModel": "m.KimiK25Model"}),
            False,
            True,
            AutoModel,
        ),
    ],
    ids=[
        "causal-lm",
        "vlm-backbone",
        "remote-vlm-with-head",
        "remote-vlm-untrusted",
        "remote-image-text-vlm",
        "native-vlm-with-head",
        "native-multimodal-causal-lm",
        "unknown-vlm",
        "encoder-decoder",
        "speech-encoder-decoder",
        "encoder-decoder-backbone",
        "remote-vlm-without-a-base-class",
        "remote-vlm-backbone",
    ],
)
def test_the_class_that_builds_the_whole_model(
    config, require_lm_head, trust_remote_code, expected
):
    """A VLM's AutoModel leaves the LM head out: with the head required it builds through the class
    that carries it, keeping the headless backbone only when no class recognizes the config. A remote
    class counts only when the checkpoint's code is trusted. An encoder-decoder model's architecture
    reads like a VLM's, but its seq2seq class carries its head, also for a config that names no
    architecture."""
    assert _auto_class(config, require_lm_head, trust_remote_code=trust_remote_code) is expected


# -------------------------------------------------------------------------------- the decision


@pytest.mark.parametrize(
    "knob",
    [
        None,
        ("MODELOPT_EP_DEGREE", "4"),
        ("MODELOPT_FSDP_SIZE", "world"),
        ("MODELOPT_FSDP_SIZE", "2"),
    ],
    ids=["no_knob", "ep_degree", "fsdp_world", "fsdp_part"],
)
@pytest.mark.parametrize("world_size", [None, 1], ids=["no_process_group", "one_rank"])
def test_a_single_process_holds_the_whole_model(
    run_as, serve, monkeypatch, tmp_path, world_size, knob
):
    """With no ranks to shard over, even an MoE loads whole, and no model is built to decide it. A
    shell still exporting a multi-GPU run's knobs must not reshape that: only the caller's own
    arguments ask for FSDP2 on one rank."""
    if world_size is not None:
        run_as(world_size)
    if knob is not None:
        monkeypatch.setenv(*knob)
    served = serve(_qwen3_moe())
    monkeypatch.setattr(
        model_load_utils, "build_meta_causal_lm", lambda *a, **k: pytest.fail("built a model")
    )

    assert resolve_distribution_plan(str(tmp_path)) == (DistributionPlan(), served)
    assert resolve_distribution_plan(str(tmp_path), dense="fsdp")[0] == DistributionPlan(
        dense="fsdp"
    )


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize("rank", [0, 1])
def test_an_ep_capable_moe_is_planned_expert_parallel(run_as, serve, tmp_path, capsys, rank):
    """Auto picks the largest EP degree dividing both the world and the expert count, hands back
    the config it decided from, and only rank 0 reports the decision."""
    run_as(4, rank)
    served = serve(_qwen3_moe(num_experts=8))

    plan, hf_config = resolve_distribution_plan(str(tmp_path))

    assert plan == DistributionPlan(4, ep_size=4, dense="replicate")
    assert hf_config is served
    assert ("Distribution:" in capsys.readouterr().out) == (rank == 0)


@needs_qwen3_moe_ep_plan
def test_no_ep_degree_dividing_the_experts_leaves_them_whole(run_as, serve, tmp_path):
    run_as(2)
    serve(_qwen3_moe(num_experts=3))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(2)


def test_an_moe_without_an_ep_plan_is_not_expert_parallel(run_as, serve, tmp_path):
    """The config counts 8 experts, but with no base_model_ep_plan nothing could shard them."""
    run_as(4)
    config = _qwen3_moe(num_experts=8)
    config.base_model_ep_plan = None
    serve(config)

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4)


@pytest.mark.parametrize("degree", [None, "1"], ids=["auto", "set-to-1"])
def test_an_ep_plan_naming_no_module_of_the_model_is_refused(
    run_as, serve, monkeypatch, tmp_path, degree
):
    """A plan whose expert containers match no module of the built model (a renamed submodule, a
    hand-written plan) leaves no expert to count or shard: the automatic degree would replicate
    every expert on every rank without a word. A degree set to 1 asks for exactly that."""
    run_as(4)
    config = _qwen3_moe(num_experts=8)
    config.base_model_ep_plan = {"layers.*.mlp.renamed_experts": "moe_tp_experts"}
    serve(config)

    if degree is None:
        with pytest.raises(ValueError, match=r"renamed_experts\) names .* MODELOPT_EP_DEGREE=1"):
            resolve_distribution_plan(str(tmp_path))
    else:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", degree)
        assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4)


@pytest.mark.parametrize("degree", [None, "1"], ids=["auto", "set-to-1"])
def test_routed_experts_the_config_does_not_count_are_refused(
    run_as, serve, monkeypatch, tmp_path, degree
):
    """The mirror case: the plan's containers are in the built model, but the config counts its
    experts under no field the decision reads, which would leave every expert replicated on every
    rank without a word. A degree set to 1 asks for exactly that."""
    plan = {"layers.*.mlp.experts": "moe_tp_experts"}
    serve(SimpleNamespace(model_type="moe", architectures=[], base_model_ep_plan=plan))
    built = model_with_decoder_at("model.")
    built.tie_weights = lambda: None  # a PreTrainedModel's; nothing here is tied
    monkeypatch.setattr(model_load_utils, "build_meta_causal_lm", lambda *a, **k: built)
    run_as(4)

    if degree is None:
        with pytest.raises(ValueError, match=r"no expert count .* MODELOPT_EP_DEGREE=1"):
            resolve_distribution_plan(str(tmp_path))
    else:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", degree)
        assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4)


@pytest.mark.parametrize("source", ["MODELOPT_EP_DEGREE", "ep_size"])
def test_an_ep_degree_that_does_not_divide_the_world_names_where_it_came_from(
    run_as, serve, monkeypatch, tmp_path, source
):
    """Every rank belongs to one EP group, so the degree must divide the world; the refusal names
    the knob or argument the degree came from, as the one for the experts does."""
    run_as(4)
    serve(LlamaConfig(**_TINY))
    arguments = {}
    if source == "ep_size":
        arguments["ep_size"] = 3
    else:
        monkeypatch.setenv(source, "3")

    with pytest.raises(ValueError, match=rf"{source}=3 must divide the world size \(4\)"):
        resolve_distribution_plan(str(tmp_path), **arguments)


@pytest.mark.parametrize("value", [[2], 2.9], ids=["a-list", "a-fraction"])
def test_an_ep_size_that_is_no_whole_number_is_refused(run_as, serve, tmp_path, value):
    """An argument int() fails on with a TypeError, or would truncate, is no degree either."""
    run_as(4)
    serve(LlamaConfig(**_TINY))

    with pytest.raises(ValueError, match=r"ep_size=.*: expected auto, 0 \(auto\) or an EP degree"):
        resolve_distribution_plan(str(tmp_path), ep_size=value)


@pytest.mark.parametrize("value", ["world", "-2"], ids=["word", "below-1"])
@pytest.mark.parametrize(
    ("world_size", "source"),
    [(4, "MODELOPT_EP_DEGREE"), (1, "ep_size")],
    ids=["knob", "argument-in-a-single-process"],
)
def test_an_ep_degree_that_is_not_a_number_names_where_it_came_from(
    run_as, serve, monkeypatch, tmp_path, world_size, source, value
):
    """A single process takes its own argument as given, but parses it as the decision does, and a
    degree below 1 is no degree either."""
    run_as(world_size)
    serve(LlamaConfig(**_TINY))
    arguments = {}
    if source == "ep_size":
        arguments["ep_size"] = value
    else:
        monkeypatch.setenv(source, value)

    with pytest.raises(ValueError, match=f"{source}='{value}'"):
        resolve_distribution_plan(str(tmp_path), **arguments)


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize(
    "value", ["0", "", "AUTO", " auto "], ids=["zero", "exported-empty", "capitals", "padded"]
)
def test_an_ep_degree_of_0_is_auto(run_as, serve, monkeypatch, tmp_path, value):
    """And so is one exported empty, and auto in any case or padded: the plan decides."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", value)
    run_as(4)
    serve(_qwen3_moe(num_experts=8))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, 4)


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied-embeddings"])
@pytest.mark.parametrize(
    ("share", "dense"), [(0.59, "replicate"), (0.61, "fsdp")], ids=["fits", "too-big"]
)
def test_a_dense_model_is_fsdp_sharded_only_when_too_big_for_one_gpu(
    run_as, serve, gpu_memory, tmp_path, share, dense, tied
):
    """Taking more than 60% of a GPU, the whole model would leave too little to calibration. It is
    sized as it loads, a tied embedding and LM head counting once, though its meta build unties
    them."""
    run_as(4)
    config = serve(LlamaConfig(**_TINY, tie_word_embeddings=tied))
    loaded = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    gpu_memory(math.ceil(_bytes(loaded) / share))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, dense=dense)


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize(
    ("experts_fitting", "dense"),
    [(1 / 2, "replicate"), (1 / 8, "fsdp")],
    ids=["fits", "the-experts-share-does-not"],
)
def test_what_a_rank_holds_with_the_dense_part_replicated_must_fit(
    run_as, serve, gpu_memory, tmp_path, experts_fitting, dense
):
    """At EP 4 a rank holds the rest of the model whole and a quarter of the routed experts, so the
    dense part is sharded when that does not fit one GPU, however small the rest; with no EP the
    whole MoE must fit, and here it does not."""
    run_as(4)
    config = serve(_qwen3_moe(num_experts=8))
    model = _meta(config)
    experts = sum(_bytes(layer.mlp.experts) for layer in model.model.layers)
    # A GPU 60% of which holds the rest of the model and that share of the experts.
    gpu_memory(int((_bytes(model) - experts + experts_fitting * experts) / 0.6))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, 4, dense)
    assert resolve_distribution_plan(str(tmp_path), ep_size=1)[0] == DistributionPlan(
        4, dense="fsdp"
    )


@needs_qwen3_moe_ep_plan
def test_the_env_knobs_override_the_automatic_plan(run_as, serve, monkeypatch, tmp_path):
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(4)
    serve(_qwen3_moe(num_experts=8))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, 2, "fsdp")


@needs_qwen3_moe_ep_plan
def test_arguments_override_the_env_knobs(run_as, serve, monkeypatch, tmp_path):
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "4")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(4)
    serve(_qwen3_moe(num_experts=8))

    plan, _ = resolve_distribution_plan(str(tmp_path), ep_size=2, dense="replicate")

    assert plan == DistributionPlan(4, 2, "replicate")


@pytest.mark.parametrize(
    ("value", "dense"),
    [
        ("1", "replicate"),
        ("4", "fsdp"),
        ("WORLD", "fsdp"),
        ("", "fsdp"),
        (" world ", "fsdp"),
        ("2", None),
    ],
    ids=["1", "4", "WORLD", "exported-empty", "padded", "part-of-the-world"],
)
def test_the_dense_knob_replicates_or_shards_over_the_whole_world(
    run_as, serve, gpu_memory, monkeypatch, tmp_path, value, dense
):
    """Exported empty, the knob is auto, which shards a model too big for one GPU. Sharding over
    part of the world is refused: no layout does that."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", value)
    run_as(4)
    serve(LlamaConfig(**_TINY))
    gpu_memory(1)  # too small for the whole model, so only auto tells "" from "1"

    if dense is None:
        with pytest.raises(ValueError, match=r"expected auto, 1 .* or world"):
            resolve_distribution_plan(str(tmp_path))
    else:
        assert resolve_distribution_plan(str(tmp_path))[0].dense == dense


@pytest.mark.parametrize(
    ("config", "ep_size", "experts"),
    [(LlamaConfig(**_TINY), 2, 0), (_qwen3_moe(num_experts=6), 4, 6)],
    ids=["dense-model", "experts-not-divisible"],
)
def test_an_ep_degree_that_cannot_shard_the_experts_is_refused(
    run_as, serve, tmp_path, config, ep_size, experts
):
    """Every rank of an EP group must hold whole experts, and a model without routed experts has
    none to shard."""
    if experts and Version(transformers.__version__) < Version("5.7"):
        pytest.skip("qwen3_moe's base_model_ep_plan ships in transformers 5.7")
    run_as(4)
    serve(config)

    with pytest.raises(ValueError, match=rf"ep_size={ep_size} must divide .* \({experts}\)"):
        resolve_distribution_plan(str(tmp_path), ep_size=ep_size)


@pytest.mark.parametrize(
    "config",
    [
        _tiny_qwen2_5_vl(),
        T5Config(d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2, vocab_size=32),
    ],
    ids=["vlm", "encoder-decoder-naming-no-architecture"],
)
def test_a_model_is_decided_from_its_whole_model(run_as, serve, gpu_memory, tmp_path, config):
    """AutoModelForCausalLM builds neither every VLM nor an encoder-decoder model, whose config need
    not name its architecture: the decision builds what the load builds."""
    run_as(2)
    serve(config)
    gpu_memory(1)

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(2, dense="fsdp")


def test_the_expert_count_is_read_from_the_config_holding_the_plan(
    run_as, serve, monkeypatch, tmp_path
):
    """A VLM whose nested config holds the MoE and its plan counts the experts there, not by an
    expert-ish field on its root config."""
    text = SimpleNamespace(
        base_model_ep_plan={"layers.*.mlp.experts": "moe_tp_experts"}, num_experts=8
    )
    serve(SimpleNamespace(model_type="vlm", architectures=[], text_config=text, num_experts=3))
    built = model_with_decoder_at("model.")
    built.tie_weights = lambda: None  # a PreTrainedModel's; nothing here is tied
    monkeypatch.setattr(model_load_utils, "build_meta_causal_lm", lambda *a, **k: built)
    run_as(4)

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, 4)


def test_a_model_that_cannot_be_built_on_meta_names_itself(run_as, serve, monkeypatch, tmp_path):
    def unbuildable(*args, **kwargs):
        raise NotImplementedError("custom init touches the weights")

    run_as(2)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(model_load_utils, "build_meta_causal_lm", unbuildable)

    with pytest.raises(RuntimeError, match=r"distribute llama .* custom init touches the weights"):
        resolve_distribution_plan(str(tmp_path))


def test_a_meta_build_leaves_the_auto_class_a_part_registers_in_as_it_found_it(monkeypatch):
    """A remote VLM's ``__init__`` can build its text model through ``AutoModelForCausalLM`` while
    the whole model builds through another auto class. That registers the text model's class in
    ``AutoModelForCausalLM`` too, which would leave rank 0's later load of it without the
    ``generate`` its peers' have."""

    class ProbeConfig(transformers.PretrainedConfig):
        model_type = "meta_build_part_probe"

    class ProbeModel(transformers.PreTrainedModel):
        config_class = ProbeConfig

    for auto in (AutoModel, AutoModelForCausalLM):
        mapping = auto._model_mapping
        monkeypatch.setattr(mapping, "_extra_content", dict(mapping._extra_content))
    from_config = AutoModel.from_config

    def remote_vlm_from_config(config, **kwargs):
        # What a remote VLM's __init__ does for its text model before the whole model is built.
        AutoModelForCausalLM.register(ProbeConfig, ProbeModel, exist_ok=True)
        return from_config(config, **kwargs)

    monkeypatch.setattr(AutoModel, "from_config", remote_vlm_from_config)

    model_load_utils.build_meta_causal_lm(
        None, False, None, LlamaConfig(**_TINY), auto_class=AutoModel
    )

    assert ProbeConfig not in AutoModelForCausalLM._model_mapping


def test_a_meta_build_leaves_the_auto_class_as_it_found_it(monkeypatch):
    """Building a remote-code model through an auto class registers its class there, as it is, and a
    later load through that auto class builds the registered class, without the ``generate`` a first
    load adds. Rank 0 alone builds one to decide a distribution: its load alone would lack it."""

    class ProbeConfig(transformers.PretrainedConfig):
        model_type = "meta_build_probe"

    class ProbeModel(transformers.PreTrainedModel):
        config_class = ProbeConfig

    mapping = AutoModelForCausalLM._model_mapping
    monkeypatch.setattr(mapping, "_extra_content", dict(mapping._extra_content))
    from_config = AutoModelForCausalLM.from_config

    def remote_from_config(config, **kwargs):
        # What transformers' from_config does for a remote-code config before it builds.
        AutoModelForCausalLM.register(ProbeConfig, ProbeModel, exist_ok=True)
        return from_config(config, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_config", remote_from_config)

    model_load_utils.build_meta_causal_lm(None, False, None, LlamaConfig(**_TINY))

    assert ProbeConfig not in mapping


def test_the_config_returned_with_the_plan_is_as_every_rank_read_it(run_as, serve, tmp_path):
    """Rank 0 alone decides, building the model, which writes to the config it builds from; a
    sharded load builds from the config returned, so rank 0's must be the one the others read."""
    run_as(2)
    serve(LlamaConfig(**_TINY))

    _, hf_config = resolve_distribution_plan(str(tmp_path))

    assert vars(hf_config) == vars(LlamaConfig(**_TINY))


def test_the_hub_options_locate_the_checkpoint_the_load_reads(tmp_path):
    """``revision``, ``cache_dir``, ``local_files_only`` and ``subfolder`` find the checkpoint as
    ``load_pretrained`` takes them, so the plan and its config are those of what the load reads."""
    repo = tmp_path / "models--org--model"
    (repo / "refs").mkdir(parents=True)
    for revision, commit, hidden_size in (("main", "a" * 40, 16), ("v2", "b" * 40, 32)):
        (repo / "refs" / revision).write_text(commit)
        config = LlamaConfig(**{**_TINY, "hidden_size": hidden_size})
        config.save_pretrained(repo / "snapshots" / commit / "nested")

    _, config = resolve_distribution_plan(
        "org/model",
        revision="v2",
        cache_dir=str(tmp_path),
        local_files_only=True,
        subfolder="nested",
    )

    assert config.hidden_size == 32


# ------------------------------------------------------------------------------ rank 0 decides


@contextmanager
def _knobs(**values):
    """This rank's distribution knobs, exactly ``values``, within the block.

    A pooled worker keeps the environment it was spawned with, so the knobs are cleared first. The
    environment is restored on the way out.
    """
    with mock.patch.dict(os.environ):
        for name in ("MODELOPT_EP_DEGREE", "MODELOPT_FSDP_SIZE"):
            os.environ.pop(name, None)
        os.environ.update(values)
        yield


def _every_rank_takes_rank0s_plan_job(rank, size, ckpt, knobs, expected):
    with bounded_collectives(), _knobs(**knobs[rank]):
        plan, _ = resolve_distribution_plan(ckpt)
    assert plan == expected, (rank, plan)


def test_every_rank_takes_the_plan_rank0_decided(cpu_dist_workers, tmp_path):
    """Over a real broadcast. Rank 1's own knobs, FSDP2 over the world, would decide another plan,
    and ranks loading different plans would build mismatched meshes and hang."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _every_rank_takes_rank0s_plan_job,
        ckpt=str(tmp_path),
        knobs=[{}, {"MODELOPT_FSDP_SIZE": "world"}],
        expected=DistributionPlan(2),
    )


def _one_rank_cannot_see_the_checkpoint_job(rank, size, ckpt, load):
    isdir = os.path.isdir
    with (
        bounded_collectives(),
        mock.patch("os.path.isdir", lambda path: (rank == 0 or path != ckpt) and isdir(path)),
    ):
        try:
            if load:
                load_pretrained(ckpt, distribution=DistributionPlan(size), device_map="cpu")
            else:
                resolve_distribution_plan(ckpt)
        except RuntimeError as error:
            raised = str(error)
        else:
            raised = "no error"
    # Rank 1's own error, carried to rank 0, rather than rank 0 timing out in the next collective.
    assert "which this rank cannot see" in raised, raised


@pytest.mark.parametrize("load", [False, True], ids=["plan", "load"])
def test_a_rank_that_cannot_see_the_checkpoint_fails_every_rank(cpu_dist_workers, tmp_path, load):
    """Deciding the plan, or loading by a handed one: rank 0 would otherwise wait for the rank that
    raised in the next collective, the plan's broadcast or the load's checks. The group is the
    caller's and the error is caught, so nothing else ends rank 0's wait."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(_one_rank_cannot_see_the_checkpoint_job, ckpt=str(tmp_path), load=load)


# ------------------------------------------------------------------------------ loading by the plan


class _Loaded(torch.nn.Module):
    """What a sharded loader returns: a model whose load recorded what it could not place."""

    def __init__(self):
        super().__init__()
        self._modelopt_unplaced_source_keys = []


@pytest.fixture
def loaders(monkeypatch):
    """Stand-ins for the sharded loaders, each recording its call by parameter name."""
    calls = {}

    def recording(name, loader):
        def stub(*args, **kwargs):
            calls[name] = inspect.signature(loader).bind(*args, **kwargs).arguments
            return _Loaded()

        return stub

    monkeypatch.setattr(
        model_load_utils,
        "parallel_load_and_prepare_fsdp2",
        recording("fsdp2", model_load_utils.parallel_load_and_prepare_fsdp2),
    )
    return calls


@pytest.fixture
def auto_class():
    """An ``auto_class`` whose from_pretrained records its arguments."""

    class Auto:
        calls: list = []

        @classmethod
        def from_pretrained(cls, ckpt_path, **kwargs):
            cls.calls.append(kwargs)
            return torch.nn.Linear(1, 1)

    return Auto


@pytest.fixture
def a_gpu(monkeypatch):
    """This rank has a GPU of its own, which the FSDP2 load shards onto; the stand-in loaders never
    use it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)


@pytest.fixture
def process_group(monkeypatch, a_gpu):
    """Rank 1 of a 2-rank process group, as the FSDP2 load reads it."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(dist, "local_rank", lambda: 1)


# ----------------------------------------------------------------------------------- the loads


@pytest.mark.parametrize(
    ("device_map", "gpus", "placed"),
    [
        (None, 4, {"device_map": {"": "cuda:1"}}),
        ("cpu", 4, {"device_map": "cpu"}),
        (None, 1, {}),
    ],
    ids=["defaults_to_the_rank_gpu", "caller_placement_wins", "no_gpu_of_its_own"],
)
def test_a_replicated_plan_loads_the_whole_model_on_every_rank(
    run_as, serve, loaders, auto_class, monkeypatch, tmp_path, device_map, gpus, placed
):
    """Each rank takes the whole model onto its own GPU; "auto" would let every rank's accelerate
    pick the same idle GPU. A rank with none of its own, on a host with fewer GPUs than ranks, is
    left to from_pretrained's placement rather than handed a device it does not have."""
    run_as(4, rank=1)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    kwargs = {} if device_map is None else {"device_map": device_map}

    model = load_pretrained(str(tmp_path), auto_class, **kwargs)

    assert auto_class.calls == [placed]
    assert distribution_plan(model) == DistributionPlan(4)
    assert loaders == {}


@pytest.mark.parametrize(
    "dtype",
    [{"dtype": "auto"}, {"dtype": None}, {"torch_dtype": "auto"}],
    ids=["auto", "none", "torch_dtype"],
)
def test_an_fsdp_plan_loads_through_the_parallel_fsdp2_loader(
    run_as, loaders, process_group, monkeypatch, tmp_path, dtype
):
    """FSDP2 shards over the process group it reads its rank and GPU from, by a config read once,
    with the Hub options the load was given (custom code an ``auto_map`` names in another repo).
    It keeps the checkpoint's dtype, as from_pretrained does for "auto" and for None, under either
    name of the argument."""
    run_as(2, rank=1)
    served, reads = LlamaConfig(**_TINY), []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(k) or served
    )
    plan = DistributionPlan(2, dense="fsdp")

    model = load_pretrained(
        str(tmp_path), distribution=plan, cpu_offload=True, local_files_only=True, **dtype
    )

    call = loaders.pop("fsdp2")
    assert (call["device"], call["rank"], call["world_size"]) == (torch.device("cuda:1"), 1, 2)
    assert call["cpu_offload"] is True
    assert call["hf_config"] is served
    assert [read.get("local_files_only") for read in reads] == [True]
    assert distribution_plan(model) == plan
    assert loaders == {}


@pytest.mark.parametrize(
    "unset",
    [{}, {"config": None}, {"variant": None}, {"hf_config": None}],
    ids=["none_passed", "config_none", "variant_none", "hf_config_none"],
)
def test_a_load_it_decides_reads_the_config_once(
    run_as, loaders, process_group, monkeypatch, tmp_path, unset
):
    """The config the plan was decided from is the one the FSDP2 load gets, not a second read. An
    option passed as None, its default, is as if not passed, not one the FSDP2 load refuses."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(2, rank=1)
    config, reads = LlamaConfig(**_TINY), []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(a) or config
    )

    model = load_pretrained(str(tmp_path), **unset)

    assert len(reads) == 1
    assert loaders["fsdp2"]["hf_config"] is config
    assert distribution_plan(model) == DistributionPlan(2, dense="fsdp")


@pytest.mark.parametrize("handed", [False, True], ids=["decided", "handed"])
def test_a_config_passed_for_the_load_is_the_one_it_decides_from(
    run_as, loaders, process_group, monkeypatch, tmp_path, handed
):
    """No config is read: the plan is decided from the one passed for the load, which the FSDP2
    load then builds from, as it does with a plan handed in. Building writes to that config, so the
    load works on a copy of it."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(2, rank=1)
    reads: list = []
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(a))
    config = LlamaConfig(**{**_TINY, "num_hidden_layers": 2})
    as_passed = config.to_dict()
    plan = {"distribution": DistributionPlan(2, dense="fsdp")} if handed else {}

    model = load_pretrained(str(tmp_path), config=config, **plan)

    assert reads == []
    assert loaders["fsdp2"]["hf_config"].num_hidden_layers == 2
    assert loaders["fsdp2"]["hf_config"] is not config  # building it writes to it
    assert config.to_dict() == as_passed
    assert distribution_plan(model) == DistributionPlan(2, dense="fsdp")


def test_hf_config_is_refused_without_the_plan_it_came_with(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path
):
    """A sharded load builds from hf_config as given, so the plan must be the one it came with: a
    plan decided from the checkpoint's own config would load a model it was not decided for."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(2, rank=1)
    serve(LlamaConfig(**_TINY))

    with pytest.raises(TypeError, match="returned with a plan that shards the model: pass that"):
        load_pretrained(str(tmp_path), hf_config=LlamaConfig(**{**_TINY, "num_hidden_layers": 2}))
    assert loaders == {}


class _OwnConfig(LlamaConfig):
    """A config class of the caller's own, which AutoConfig has no mapping for."""

    model_type = "modelopt_test_own_llama"


@pytest.mark.parametrize(
    ("fsdp_size", "own_config"),
    [(None, False), ("world", False), (None, True)],
    ids=["no-knob", "knob-shards", "own-config-class"],
)
def test_a_model_class_of_the_callers_own_loads_whole(
    run_as, serve, loaders, auto_class, monkeypatch, tmp_path, fsdp_size, own_config
):
    """The sharded loads build their own model class and refuse another, so with nothing asking to
    shard, a class of the caller's own loads whole on every rank, and with a knob asking to shard
    it is refused. Neither builds the model on the meta device, which the decision's own pick of
    class may not build (an encoder-decoder loaded through AutoModel). The decision reads the
    checkpoint's config as the class does, through its own config class, which AutoConfig need not
    know."""
    run_as(2)
    if own_config:
        _OwnConfig(**_TINY).save_pretrained(tmp_path)
        auto_class.config_class = _OwnConfig
    else:
        serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(
        model_load_utils, "build_meta_causal_lm", lambda *a, **k: pytest.fail("built a model")
    )
    if fsdp_size is None:
        model = load_pretrained(str(tmp_path), auto_class)
        assert distribution_plan(model) == DistributionPlan(2)
        assert len(auto_class.calls) == 1
    else:
        monkeypatch.setenv("MODELOPT_FSDP_SIZE", fsdp_size)
        with pytest.raises(NotImplementedError, match="builds its model class itself"):
            load_pretrained(str(tmp_path), auto_class)
    assert loaders == {}


@pytest.mark.parametrize("handed", [True, False], ids=["handed", "decided"])
def test_a_model_class_given_by_name_is_refused_by_name(
    run_as, serve, loaders, monkeypatch, tmp_path, handed
):
    """A sharded load refuses a model class it does not build, given by name too, rather than fail
    on the name having no __name__ of its own."""
    run_as(2)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    plan = {"distribution": DistributionPlan(2, dense="fsdp")} if handed else {}

    with pytest.raises(NotImplementedError, match="auto_class='AutoModelForCausalLM'"):
        load_pretrained(str(tmp_path), "AutoModelForCausalLM", **plan)
    assert loaders == {}


def test_an_expert_parallel_plan_is_refused(run_as, serve, loaders, auto_class, tmp_path):
    """No load here shards the routed experts: an EP plan fails before anything loads."""
    run_as(2)
    serve(Qwen3MoeConfig(num_experts=8, moe_intermediate_size=8, **_TINY))

    with pytest.raises(NotImplementedError, match="shards the routed experts"):
        load_pretrained(str(tmp_path), auto_class, distribution=DistributionPlan(2, ep_size=2))
    assert loaders == {}
    assert auto_class.calls == []


def _every_rank_loads_rank0s_plan_job(rank, size, ckpt, handed, placed):
    model, refused = None, None
    # Ranks without GPUs, whatever the host has: the pooled workers keep no CUDA context.
    with bounded_collectives(), mock.patch.object(torch.cuda, "is_available", return_value=False):
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            model = load_pretrained(ckpt, distribution=handed[rank], **placed)
        except Exception as error:
            refused = str(error)
    if isinstance(handed[0], DistributionPlan):
        assert refused is None and distribution_plan(model) == handed[0], (rank, refused)
    else:
        # Rank 0's refusal, on every rank, rather than a rank left waiting in the broadcast.
        assert "expected a DistributionPlan or None" in str(refused), (rank, refused)


@pytest.mark.parametrize(
    ("handed", "placed"),
    [
        ((DistributionPlan(2), DistributionPlan(2, dense="fsdp")), {"device_map": "cpu"}),
        ((DistributionPlan(2), None), {"device_map": "cpu"}),
        (("fsdp2", DistributionPlan(2)), {"device_map": "cpu"}),
        ((DistributionPlan(2), DistributionPlan(2)), {}),
    ],
    ids=["another-plan", "none", "not-a-plan", "placed-by-default"],
)
def test_every_rank_loads_the_plan_rank0_was_handed(cpu_dist_workers, tmp_path, handed, placed):
    """Over a real broadcast, whatever rank 1 was handed. Ranks loading different plans would build
    mismatched meshes, and one handed none would decide a plan through collectives the others never
    join: both would hang. A distribution rank 0 refuses fails every rank. Placed by default, each
    rank takes the model onto its GPU, and a run without GPUs, as these ranks are, loads it whole."""
    LlamaForCausalLM(LlamaConfig(**_TINY)).save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _every_rank_loads_rank0s_plan_job, ckpt=str(tmp_path), handed=handed, placed=placed
    )


def _fetched_in_step(ckpt_path, **hub_options):
    """The checkpoint's fetch, as the collective it is: a rank not in it leaves the others waiting."""
    torch.distributed.barrier()
    return None, ckpt_path


def _loaded_in_step(*args, **kwargs):
    """The FSDP2 load, as the collectives it runs, which a rank without a GPU fails before."""
    if dist.local_rank() >= torch.cuda.device_count():
        raise RuntimeError("no GPU to shard onto")
    torch.distributed.barrier()
    return _Loaded()


def _one_rank_refused_job(rank, size, ckpt, refused_on_1, gpus_on_1, match):
    refused = None
    gpus = size if rank == 0 else gpus_on_1  # what this rank's host shows it
    with (
        bounded_collectives(),
        mock.patch.object(torch.cuda, "is_available", return_value=gpus > 0),
        mock.patch.object(torch.cuda, "device_count", return_value=gpus),
        mock.patch.object(model_load_utils, "ensure_local_checkpoint", _fetched_in_step),
        mock.patch.object(model_load_utils, "parallel_load_and_prepare_fsdp2", _loaded_in_step),
    ):
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            load_pretrained(
                ckpt,
                distribution=DistributionPlan(2, dense="fsdp"),
                **(refused_on_1 if rank == 1 else {}),
            )
        except Exception as error:
            refused = str(error)
    # Rank 1's refusal on both ranks, rather than rank 0 failing at the group's timeout.
    assert refused is not None and match in refused, (rank, refused)
    assert rank == 1 or "rank 1 failed" in refused, refused


@pytest.mark.parametrize(
    ("refused_on_1", "gpus_on_1", "match"),
    [
        ({"config": LlamaConfig(**_TINY), "hf_config": LlamaConfig(**_TINY)}, 2, "both name"),
        ({"dtype": torch.float32}, 2, "keeps their dtype"),
        (
            {"hf_config": LlamaConfig(**_TINY, quantization_config={"quant_method": "fp8"})},
            2,
            "cannot dequantize",
        ),
        ({}, 0, "this rank has none"),
        ({}, 1, "this rank has none"),
    ],
    ids=["two-configs", "dtype", "quantized-config", "no-gpu", "fewer-gpus-than-ranks"],
)
def test_a_load_refused_on_one_rank_fails_every_rank(
    cpu_dist_workers, tmp_path, refused_on_1, gpus_on_1, match
):
    """Rank 1 alone is refused, by its arguments before the checkpoint is fetched, or by the config
    it was handed or its lack of a GPU before the FSDP2 load: every rank raises the refusal, rather
    than leave rank 0 waiting in the fetch's or the load's collectives, which stand-ins on a bounded
    group run here."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _one_rank_refused_job,
        ckpt=str(tmp_path),
        refused_on_1=refused_on_1,
        gpus_on_1=gpus_on_1,
        match=match,
    )


class _LoadsWholeExceptOnRank1:
    """A model class whose whole-model load fails on rank 1 alone, as an out-of-memory GPU would."""

    @classmethod
    def from_pretrained(cls, ckpt_path, **kwargs):
        if dist.rank() == 1:
            raise OSError("rank 1 could not load the model")
        return torch.nn.Linear(1, 1)


def _one_rank_fails_to_load_job(rank, size, ckpt):
    refused = None
    with (
        bounded_collectives(),
        mock.patch.object(model_load_utils, "ensure_local_checkpoint", _fetched_in_step),
    ):
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            load_pretrained(ckpt, _LoadsWholeExceptOnRank1, distribution=DistributionPlan(2))
        except Exception as error:
            refused = str(error)
    # Rank 1's failure on both ranks, rather than rank 0 going on to the next collective alone.
    assert refused is not None and "could not load the model" in refused, (rank, refused)


def test_a_whole_load_that_fails_on_one_rank_fails_every_rank(cpu_dist_workers, tmp_path):
    """A whole-model load runs no collective, so a rank failing in it alone would leave the others
    to wait for it in the next one, calibration's or the export's."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(_one_rank_fails_to_load_job, ckpt=str(tmp_path))


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"kwargs": {"device_map": "auto"}}, TypeError, r"\['device_map'\] do not apply"),
        ({"kwargs": {"dtype": torch.bfloat16}}, TypeError, "keeps their dtype"),
        (
            {"kwargs": {"torch_dtype": torch.bfloat16}},
            TypeError,
            r"keeps their dtype, so torch_dtype=torch\.bfloat16",
        ),
        ({"auto_class": AutoModel}, NotImplementedError, "builds its model class itself"),
        ({"mtp": True}, NotImplementedError, "cannot build the MTP head"),
        (
            {"kwargs": {"config": LlamaConfig(**_TINY), "hf_config": LlamaConfig(**_TINY)}},
            TypeError,
            "config and hf_config both name",
        ),
    ],
    ids=["placement", "dtype", "torch_dtype", "model_class", "mtp_head", "two_configs"],
)
def test_a_sharded_load_refuses_what_it_cannot_honor(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, change, error, match
):
    """Placement, a model class, a head only a whole model builds, two configs to load with: none
    reaches a sharded load, which fails before it starts rather than load something other than
    what was asked."""
    run_as(2)
    serve(LlamaConfig(**_TINY))
    if change.get("mtp"):
        monkeypatch.setattr(models_hf, "checkpoint_has_mtp", lambda *a: True)

    with pytest.raises(error, match=match):
        load_pretrained(
            str(tmp_path),
            change.get("auto_class"),
            distribution=DistributionPlan(2, dense="fsdp"),
            **change.get("kwargs", {}),
        )
    assert loaders == {}


@pytest.mark.parametrize(
    ("distribution", "checkpoint", "options", "error", "match"),
    [
        (
            DistributionPlan(4, dense="fsdp"),
            "causal-lm",
            None,
            ValueError,
            "is for 4 ranks, but this run has 2",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            "quantized",
            None,
            NotImplementedError,
            "cannot dequantize",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            "vlm",
            None,
            NotImplementedError,
            "builds a causal LM, which this llava checkpoint is not",
        ),
        ("fsdp2", "causal-lm", None, TypeError, "expected a DistributionPlan or None"),
        (
            DistributionPlan(2),
            "causal-lm",
            "hf_config",
            TypeError,
            "returned with a plan that shards the model",
        ),
        (
            DistributionPlan(2),
            "causal-lm",
            {"cpu_offload": True},
            TypeError,
            r"\['cpu_offload'\] only apply to the FSDP2 load",
        ),
    ],
    ids=[
        "another-world",
        "quantized-checkpoint",
        "vlm-checkpoint",
        "not-a-plan",
        "hf-config-on-a-whole-load",
        "fsdp2-option-on-a-whole-load",
    ],
)
def test_a_load_refuses_what_it_cannot_load(
    run_as,
    serve,
    loaders,
    auto_class,
    process_group,
    tmp_path,
    distribution,
    checkpoint,
    options,
    error,
    match,
):
    """Before anything loads: a plan for another world; a quantized checkpoint, which FSDP2 would
    read as stored; a VLM, which FSDP2 would build through AutoModelForCausalLM, refused or as its
    language model alone; anything but a plan; and the sharded loads' config, or an option only the
    FSDP2 load takes, which from_pretrained would take for a model argument and fail far from here,
    or drop."""
    run_as(2)
    config = serve(transformers.LlavaConfig() if checkpoint == "vlm" else LlamaConfig(**_TINY))
    if checkpoint == "quantized":
        config.quantization_config = {"quant_method": "fp8"}
    if options == "hf_config":
        options = {"auto_class": auto_class, "hf_config": config}

    with pytest.raises(error, match=match):
        load_pretrained(str(tmp_path), distribution=distribution, **(options or {}))
    assert loaders == {}
    assert auto_class.calls == []


def test_an_automatic_layout_says_so_when_it_refuses_a_placement(
    run_as, serve, loaders, monkeypatch, tmp_path
):
    """With neither a plan nor MODELOPT_FSDP_SIZE, rank 0's GPU memory decides to shard a model too
    big for it, and the sharded load refuses options that place a whole model. The refusal says the
    layout was decided rather than asked for, and how to keep the load whole."""
    run_as(2)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: 1)

    with pytest.raises(TypeError, match=r"\['device_map'\] place a whole model, but no plan"):
        load_pretrained(str(tmp_path), device_map="auto")
    assert loaders == {}


@pytest.mark.parametrize(
    ("distribution", "options", "error", "match"),
    [
        (DistributionPlan(4), {}, ValueError, "is for 4 ranks"),
        (DistributionPlan(2), {"hf_config": LlamaConfig(**_TINY)}, TypeError, "plan that shards"),
        (
            DistributionPlan(2, dense="fsdp"),
            {"device_map": "auto"},
            TypeError,
            r"\['device_map'\] do not apply to the FSDP2 load",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            {"auto_class": AutoModel},
            NotImplementedError,
            "auto_class=AutoModel does not apply",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            {"dtype": torch.float32},
            TypeError,
            "keeps their dtype",
        ),
    ],
    ids=[
        "another-world",
        "hf-config-on-a-whole-load",
        "whole-load-option-on-a-sharded-plan",
        "auto-class-on-a-sharded-plan",
        "dtype-on-a-sharded-plan",
    ],
)
def test_a_handed_plan_is_checked_before_the_checkpoint_is_fetched(
    run_as, monkeypatch, tmp_path, distribution, options, error, match
):
    """Making the checkpoint local can mean downloading it: a plan the load refuses fails first, as
    do hf_config with a plan that loads the model whole and an option a sharded plan's load
    refuses."""
    run_as(2)
    monkeypatch.setattr(
        model_load_utils, "ensure_local_checkpoint", lambda *a, **k: pytest.fail("fetched it")
    )

    with pytest.raises(error, match=match):
        load_pretrained(str(tmp_path), distribution=distribution, **options)


@pytest.mark.parametrize("quantized", [False, True])
def test_a_single_process_fsdp2_load_sets_up_its_group_of_one(
    monkeypatch, serve, loaders, a_gpu, tmp_path, quantized
):
    """FSDP2 on one rank still shards through a process group, of one, which the load sets up once
    nothing can refuse it: a refused load leaves no group behind."""
    config = serve(LlamaConfig(**_TINY))
    if quantized:
        config.quantization_config = {"quant_method": "fp8"}
    monkeypatch.setenv("RANK", "0")  # torchrun --nproc_per_node=1
    live = []
    monkeypatch.setattr(dist, "is_initialized", lambda: bool(live))
    monkeypatch.setattr(dist, "setup", lambda timeout=None: live.append(0))
    # What a group of one answers once it is up.
    monkeypatch.setattr(dist, "rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "size", lambda group=None: 1)
    monkeypatch.setattr(dist, "local_rank", lambda: 0)
    plan = DistributionPlan(dense="fsdp")

    if quantized:
        with pytest.raises(NotImplementedError, match="cannot dequantize"):
            load_pretrained(str(tmp_path), distribution=plan)
        assert live == []
        return
    load_pretrained(str(tmp_path), distribution=plan)

    assert live == [0]
    call = loaders.pop("fsdp2")
    assert (call["rank"], call["world_size"]) == (0, 1)


def test_a_single_process_fsdp2_load_needs_a_launcher(monkeypatch, serve, loaders, tmp_path):
    """Started without one, the process cannot set its group of one up; it is told to launch with
    torchrun rather than fail inside init_process_group."""
    serve(LlamaConfig(**_TINY))
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "setup", lambda timeout=None: pytest.fail("set a group up"))

    with pytest.raises(RuntimeError, match="torchrun --nproc_per_node=1"):
        load_pretrained(str(tmp_path), distribution=DistributionPlan(dense="fsdp"))
    assert loaders == {}


@pytest.fixture
def launch_without_its_group(monkeypatch):
    """Rank 0 of an 8-rank launch whose processes set no process group up."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "setup", lambda timeout=None: pytest.fail("set a group up"))


def test_a_launch_that_sets_no_group_up_loads_whole_on_each_process(
    launch_without_its_group, serve, loaders, auto_class, tmp_path
):
    """A launcher can start several processes that never set a group up (hf_ptq without
    --use_fsdp2): each loads the whole model as a single process does, with no collective to wait
    in and no plan to agree on."""
    serve(LlamaConfig(**_TINY))

    model = load_pretrained(str(tmp_path), auto_class)

    assert auto_class.calls == [{}]
    assert distribution_plan(model) == DistributionPlan()
    assert loaders == {}


@pytest.mark.parametrize(
    ("distribution", "knobs", "error", "match"),
    [
        (
            DistributionPlan(8),
            {},
            RuntimeError,
            "is for 8 ranks, whose process group this 8-rank launch has not",
        ),
        (
            DistributionPlan(dense="fsdp"),
            {},
            RuntimeError,
            r"shards through this 8-rank launch's .* DistributionPlan\(8, dense='fsdp'\)",
        ),
        (
            None,
            {"MODELOPT_FSDP_SIZE": "world"},
            RuntimeError,
            "MODELOPT_FSDP_SIZE=world ask to shard",
        ),
        (None, {"MODELOPT_EP_DEGREE": "4"}, RuntimeError, "MODELOPT_EP_DEGREE=4 ask to shard"),
        (None, {"MODELOPT_EP_DEGREE": "eigth"}, ValueError, "MODELOPT_EP_DEGREE='eigth'"),
    ],
    ids=["several-ranks", "sharded", "fsdp-knob", "ep-knob", "ep-knob-not-a-degree"],
)
def test_what_needs_the_group_is_refused_in_a_launch_that_sets_none_up(
    launch_without_its_group,
    serve,
    loaders,
    monkeypatch,
    tmp_path,
    distribution,
    knobs,
    error,
    match,
):
    """A plan for several ranks, a sharded load, and knobs asking to shard over the launch need its
    group, which one rank cannot set up for itself: refused before anything is fetched, rather than
    leave each process the whole model the knobs did not ask for. A knob that is not one is refused
    as the plan decision refuses it, not dropped."""
    serve(LlamaConfig(**_TINY))
    for name, value in knobs.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        model_load_utils, "ensure_local_checkpoint", lambda *a, **k: pytest.fail("fetched it")
    )

    with pytest.raises(error, match=match):
        load_pretrained(str(tmp_path), distribution=distribution)
    assert loaders == {}
