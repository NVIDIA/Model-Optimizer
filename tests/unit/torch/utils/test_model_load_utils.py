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

import copy
import dataclasses
import inspect
import json
import math
import os
import warnings
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from _test_utils.torch.moe_models import model_with_decoder_at
from safetensors.torch import save_file
from torch import nn

pytest.importorskip("accelerate")
transformers = pytest.importorskip("transformers")
import transformers.distributed as hf_distributed
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
    DeepseekV3Config,
    FineGrainedFP8Config,
    GptOssConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Phi4MultimodalConfig,
    PreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    T5Config,
    WhisperConfig,
)
from transformers.distributed import DistributedConfig

import modelopt.torch.models.hf as models_hf
import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as dist
import modelopt.torch.utils.moe as moe
from modelopt.torch.models import EPSpec, ModelSpec, get_spec
from modelopt.torch.quantization.plugins import accelerate as accelerate_plugin
from modelopt.torch.quantization.plugins.accelerate import init_quantized_weights
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.distributed import DistributionPlan, distribution_plan
from modelopt.torch.utils.moe import _EP_MIN_TRANSFORMERS
from modelopt.torch.utils.plugins import hf_checkpoint_utils, model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    _auto_class,
    _ckpt_quant_method,
    _conversion_plan,
    _convert_keys,
    _experts_sharded_over,
    _resolve_target,
    _seed_parallel_states,
    _transformers_mesh_builder,
    build_meta_causal_lm,
    load_and_prepare_ep,
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

# Applying a model's spec enforces the transformers floor the expert-parallel load needs.
needs_ep_transformers = pytest.mark.skipif(
    Version(transformers.__version__) < Version(_EP_MIN_TRANSFORMERS),
    reason=f"expert parallelism needs transformers >= {_EP_MIN_TRANSFORMERS}",
)

needs_qwen3_moe_ep_plan = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"),
    reason="qwen3_moe's base_model_ep_plan ships in transformers 5.7",
)


def _qwen3_moe(num_experts=8):
    """Qwen3-MoE, whose base_model_ep_plan ships with transformers itself."""
    return Qwen3MoeConfig(num_experts=num_experts, moe_intermediate_size=8, **_TINY)


def _deepseek_v3(n_routed_experts=8):
    """DeepSeek-V3 without a plan of its own, so only its model spec can make it EP-capable.

    Its one layer is an MoE layer: DeepSeek-V3's first layers are dense by default.
    """
    config = DeepseekV3Config(
        n_routed_experts=n_routed_experts,
        first_k_dense_replace=0,
        moe_intermediate_size=8,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        **_TINY,
    )
    config.base_model_ep_plan = None  # holds even if a later transformers ships a plan
    return config


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
    """With no ranks to shard over, even an MoE loads whole, and no model is built, nor its spec
    consulted, to decide it. A shell still exporting a multi-GPU run's knobs must not reshape that:
    only the caller's own arguments ask for FSDP2 on one rank."""
    if world_size is not None:
        run_as(world_size)
    if knob is not None:
        monkeypatch.setenv(*knob)
    served = serve(_qwen3_moe())
    monkeypatch.setattr(
        model_load_utils, "build_meta_causal_lm", lambda *a, **k: pytest.fail("built a model")
    )
    monkeypatch.setattr(
        model_load_utils, "_spec_refuses_ep", lambda *a: pytest.fail("consulted the spec")
    )

    assert resolve_distribution_plan(str(tmp_path)) == (DistributionPlan(), served)
    assert resolve_distribution_plan(str(tmp_path), ep_size="auto")[0] == DistributionPlan()
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
    assert hf_config.base_model_ep_plan == served.base_model_ep_plan
    assert ("Distribution:" in capsys.readouterr().out) == (rank == 0)


def test_an_moe_without_an_ep_plan_is_not_expert_parallel(run_as, serve, tmp_path):
    """The config counts 8 experts, but with no base_model_ep_plan nothing could shard them."""
    run_as(4)
    config = _qwen3_moe(num_experts=8)
    config.base_model_ep_plan = None
    serve(config)

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4)


@pytest.mark.parametrize(
    "degree",
    [pytest.param(None, id="auto", marks=needs_ep_transformers), pytest.param("1", id="set-to-1")],
)
def test_an_ep_plan_naming_no_module_of_the_model_is_refused(
    run_as, serve, monkeypatch, tmp_path, degree
):
    """A plan whose expert containers match no module of the built model (a renamed submodule, a
    hand-written plan) leaves no expert to count or shard: the automatic degree would replicate
    every expert on every rank without a word. A degree of 1 with the dense part replicated asks
    for exactly that."""
    run_as(4)
    config = _qwen3_moe(num_experts=8)
    config.base_model_ep_plan = {"layers.*.mlp.renamed_experts": "moe_tp_experts"}
    serve(config)

    if degree is None:
        with pytest.raises(
            ValueError,
            match=r"renamed_experts\) names .* MODELOPT_EP_DEGREE=1 and MODELOPT_FSDP_SIZE=1",
        ):
            resolve_distribution_plan(str(tmp_path))
    else:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", degree)
        monkeypatch.setenv("MODELOPT_FSDP_SIZE", "1")
        assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4)


@pytest.mark.parametrize(
    "degree",
    [pytest.param(None, id="auto", marks=needs_ep_transformers), pytest.param("1", id="set-to-1")],
)
def test_routed_experts_the_config_does_not_count_are_refused(
    run_as, serve, monkeypatch, tmp_path, degree
):
    """The mirror case: the plan's containers are in the built model, but the config counts its
    experts under no field the decision reads, which would leave every expert replicated on every
    rank without a word. A degree of 1 with the dense part replicated asks for exactly that."""
    plan = {"layers.*.mlp.experts": "moe_tp_experts"}
    serve(SimpleNamespace(model_type="moe", architectures=[], base_model_ep_plan=plan))
    built = model_with_decoder_at("model.")
    built.tie_weights = lambda: None  # a PreTrainedModel's; nothing here is tied
    monkeypatch.setattr(model_load_utils, "build_meta_causal_lm", lambda *a, **k: built)
    run_as(4)

    if degree is None:
        with pytest.raises(
            ValueError, match=r"no expert count .* MODELOPT_EP_DEGREE=1 and MODELOPT_FSDP_SIZE=1"
        ):
            resolve_distribution_plan(str(tmp_path))
    else:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", degree)
        monkeypatch.setenv("MODELOPT_FSDP_SIZE", "1")
        assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4)


@needs_ep_transformers
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

    with pytest.raises(ValueError, match=r"ep_size=.*EP degree"):
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


@needs_ep_transformers
def test_the_spec_is_applied_before_the_decision(run_as, serve, tmp_path):
    """DeepSeek-V3's plan comes only from its spec. Deciding before applying it would read the MoE
    as dense; applied first, it shards, and the config handed on carries the spec's plan."""
    run_as(4)
    serve(_deepseek_v3(n_routed_experts=8))

    plan, hf_config = resolve_distribution_plan(str(tmp_path))

    assert plan == DistributionPlan(4, 4)
    assert hf_config.base_model_ep_plan == get_spec("deepseek_v3").ep_spec.base_model_ep_plan


def test_only_an_expert_parallel_plan_carries_the_spec_plan_on(run_as, serve, tmp_path):
    """No EP degree divides 3 experts over 2 ranks, so the whole model is FSDP2-sharded: the config
    handed on keeps no expert-parallel plan, which the export would write into config.json."""
    run_as(2)
    serve(_deepseek_v3(n_routed_experts=3))

    plan, hf_config = resolve_distribution_plan(str(tmp_path), dense="fsdp")

    assert plan == DistributionPlan(2, dense="fsdp")
    assert hf_config.base_model_ep_plan is None


@pytest.mark.parametrize(
    ("config", "floor", "version", "error", "refusal"),
    [
        pytest.param(
            GptOssConfig(num_local_experts=4, **_TINY),
            None,
            None,
            NotImplementedError,
            "cannot be quantized expert-parallel yet",
            id="spec-refuses-ep",
        ),
        pytest.param(
            _qwen3_moe(num_experts=4),
            "99",
            None,
            RuntimeError,
            "requires transformers >= 99",
            id="below-the-specs-floor",
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=4),
            None,
            "5.6.2",
            RuntimeError,
            r"requires transformers >= 5\.7",
            id="below-the-expert-parallel-loads-floor",
        ),
    ],
)
def test_a_run_without_expert_parallelism_needs_no_spec(
    run_as, serve, monkeypatch, tmp_path, config, floor, version, error, refusal
):
    """GPT-OSS cannot be expert-parallel, which its spec says by refusing an EP degree above 1, nor
    can a model on a transformers older than its spec, or the expert-parallel load, asks for: a
    plan the spec supplies does not lift the load's floor. A run that rules expert parallelism out
    does not ask the spec, nor does one leaving the EP degree to auto, which picks none for
    either: asking for FSDP2 alone shards it whole."""
    if floor is not None:
        monkeypatch.setattr(get_spec(config.model_type).ep_spec, "requires_transformers", floor)
    if version is not None:
        monkeypatch.setattr(transformers, "__version__", version)
    run_as(2)
    serve(config)

    with pytest.raises(error, match=refusal):
        resolve_distribution_plan(str(tmp_path), ep_size=2)
    plan, _ = resolve_distribution_plan(str(tmp_path), ep_size=1, dense="fsdp")
    assert plan == DistributionPlan(2, dense="fsdp")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    assert resolve_distribution_plan(str(tmp_path))[0] == plan
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "1")
    assert resolve_distribution_plan(str(tmp_path))[0] == plan


# The refusals of an MoE left whole on every rank, by whether its spec allows expert parallelism.
_CAN_BE_EP = "can be expert-parallel, but the plan resolved .* divid(?:es|ing) both its {} experts"
_NO_EP_YET = (
    "cannot be quantized expert-parallel yet: .* The plan replicates .* FSDP2 .* if the "
    "FSDP2 load supports the checkpoint, which takes .*, load it in a single process, or "
    r"replicate .*DistributionPlan\(2, 1, 'replicate'\)"
)
_NEEDS_LATER = (
    "can be expert-parallel, but the plan resolved .* this transformers cannot shard its experts. "
    r"Upgrade transformers to 5\.7 or later to shard them, or shard the whole model with FSDP2"
)
# The same, for an EP degree the caller set to 1, which an upgrade alone leaves at 1.
_NEEDS_LATER_AND_A_DEGREE = (
    "can be expert-parallel, but the plan resolved .* this transformers cannot shard its experts. "
    r"Upgrade transformers to 5\.7 or later and set MODELOPT_EP_DEGREE above 1, dividing both its "
    "8 experts and the world size, to shard them, or shard the whole model with FSDP2"
)


@pytest.mark.parametrize(
    ("config", "ep_degree", "refusal", "version"),
    [
        pytest.param(
            _deepseek_v3(n_routed_experts=3),
            None,
            _CAN_BE_EP.format(3),
            None,
            id="auto_finds_no_degree",
        ),
        pytest.param(
            _qwen3_moe(num_experts=8), "1", _CAN_BE_EP.format(8), None, id="explicit_degree_one"
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=8),
            "1",
            _CAN_BE_EP.format(8),
            None,
            id="explicit_degree_one_spec_plan",
        ),
        pytest.param(
            GptOssConfig(num_local_experts=4, **_TINY),
            None,
            _NO_EP_YET,
            None,
            id="spec_refuses_ep",
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=8), None, _NEEDS_LATER, "5.6.2", id="below_the_ep_floor"
        ),
        pytest.param(
            _qwen3_moe(num_experts=8),
            1,
            _CAN_BE_EP.format(8),
            None,
            id="explicit_degree_one_as_an_argument",
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=3),
            None,
            _CAN_BE_EP.format(3),
            "5.6.2",
            id="below_the_ep_floor_no_degree",
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=8),
            "1",
            _NEEDS_LATER_AND_A_DEGREE,
            "5.6.2",
            id="below_the_ep_floor_explicit_degree_one",
        ),
    ],
)
def test_an_ep_capable_moe_left_whole_on_every_rank_is_an_error(
    run_as, serve, monkeypatch, tmp_path, config, ep_degree, refusal, version
):
    """No EP degree above 1 divides 3 experts over 2 ranks, and an explicit 1 asks for none,
    whether the plan comes with the config (Qwen3-MoE) or only from the model's spec, which an EP
    degree of 1 skips (DeepSeek-V3). With nothing sharded either, every rank would hold the whole
    MoE: that must fail, not run, and name the expert count an EP degree has to divide. GPT-OSS
    carries a plan too, but its spec refuses expert parallelism, so auto picks no EP degree for it
    and this check refuses it as an MoE that cannot be expert-parallel yet, offering every other
    way out (FSDP2 only for an unquantized causal LM); the spec's own refusal, a
    NotImplementedError and so a RuntimeError too, words it differently. Below the transformers
    release expert parallelism needs, no EP degree is offered, since none would load: upgrading
    is, where an EP degree divides the experts; where none does, upgrading would not shard them.
    The ways out are named as the caller set the plan, so arguments, which override the knobs, are
    named."""
    if version is not None:
        monkeypatch.setattr(transformers, "__version__", version)
    experts = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 0)
    below = Version(transformers.__version__) < Version(_EP_MIN_TRANSFORMERS)
    if version is None and refusal.startswith("can be") and below and experts % 2 == 0:
        # On a transformers below the floor, upgrading is the way out where an EP degree divides
        # the experts, with a degree set to 1 raised as well.
        refusal = _NEEDS_LATER_AND_A_DEGREE if ep_degree is not None else _NEEDS_LATER
    if ep_degree is not None:
        if config.model_type == "qwen3_moe" and Version(transformers.__version__) < Version("5.7"):
            pytest.skip("qwen3_moe's base_model_ep_plan ships in transformers 5.7")
        if isinstance(ep_degree, str):
            monkeypatch.setenv("MODELOPT_EP_DEGREE", ep_degree)
    arguments = {"ep_size": ep_degree, "dense": "auto"} if isinstance(ep_degree, int) else {}
    run_as(2)
    serve(config)

    with pytest.raises((RuntimeError, ValueError), match=refusal) as refused:
        resolve_distribution_plan(str(tmp_path), **arguments)
    if refusal is _NO_EP_YET:  # as the load refuses the same spec
        assert refused.type is NotImplementedError
    if refusal.startswith(_CAN_BE_EP[: _CAN_BE_EP.index("{")]):  # what its inputs resolve to
        assert refused.type is ValueError, refused.type
    if arguments:  # a knob the arguments override would lead straight back here
        assert "ep_size=1 with dense='replicate'" in str(refused.value), refused.value
        assert "MODELOPT_" not in str(refused.value), refused.value


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize("ep_degree", [None, "1"], ids=["auto", "set-to-1"])
@pytest.mark.parametrize(
    "refused_by",
    ["its-plan", "its-format", "its-spec", "its-transformers", "its-specs-transformers"],
)
def test_an_moe_the_expert_parallel_load_refuses_is_decided_without_it(
    run_as, serve, monkeypatch, tmp_path, refused_by, ep_degree
):
    """Left to the decision, an MoE the expert-parallel load would refuse -- by a plan that
    shards no expert weights through grouped_gemm, a checkpoint format it does not load, a model
    spec ruling it out (Llama-4's, GPT-OSS's), or a transformers below its floor or below the
    later one the spec states (MiniMax-M3-VL's) -- is planned without expert parallelism rather
    than committed to a load that refuses it, for some reasons only after the download. Nor is it
    refused for being left whole, whether the decision or the caller made its EP degree 1: it
    cannot be expert-parallel here. Below the floor its spec states, though, the decision's own
    refusal names the upgrade, as one would shard it, and an EP degree the caller set to 1 to
    raise."""
    if ep_degree is not None:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", ep_degree)
    run_as(2)
    config = _qwen3_moe(num_experts=8)
    if refused_by == "its-plan":
        config.base_model_ep_plan = {"layers.*.mlp.experts": "moe_tp_experts"}
    elif refused_by == "its-format":
        config.quantization_config = {"quant_method": "fp8"}
    elif refused_by == "its-spec":
        spec = ModelSpec(model_type="qwen3_moe", ep_spec=EPSpec(unsupported="Not yet."))
        monkeypatch.setattr(model_load_utils, "get_spec", lambda model_type: spec)
    elif refused_by == "its-specs-transformers":
        # Above the load's own 5.7 by version, as 5.10 is not, though not as strings compare.
        spec = ModelSpec(model_type="qwen3_moe", ep_spec=EPSpec(requires_transformers="5.13"))
        monkeypatch.setattr(model_load_utils, "get_spec", lambda model_type: spec)
        monkeypatch.setattr(transformers, "__version__", "5.10.0")
    else:
        monkeypatch.setattr(transformers, "__version__", "5.6.2")
    serve(config)
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: None)

    if refused_by == "its-transformers":
        refusal = _NEEDS_LATER if ep_degree is None else _NEEDS_LATER_AND_A_DEGREE
        with pytest.raises(RuntimeError, match=refusal):
            resolve_distribution_plan(str(tmp_path))
        return
    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(2)


def test_an_ep_capable_moe_may_be_fsdp2_sharded_instead(run_as, serve, monkeypatch, tmp_path):
    """Asked for no expert parallelism but for dense sharding, the MoE is sharded whole by FSDP2,
    which is not the waste of a replica on every rank."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "1")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(2)
    serve(_deepseek_v3(n_routed_experts=8))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(2, dense="fsdp")


@needs_qwen3_moe_ep_plan
def test_an_ep_capable_moe_is_replicated_when_both_knobs_ask_for_it(
    run_as, serve, monkeypatch, tmp_path
):
    """Both knobs asking to shard nothing is a replication the caller chose, not one the plan
    reached on its own: every rank holds the whole MoE."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "1")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "1")
    run_as(2)
    serve(_qwen3_moe(num_experts=8))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(2)


def test_an_ep_degree_of_1_auto_picks_does_not_ask_for_the_replication(
    run_as, serve, monkeypatch, tmp_path
):
    """Auto is an EP degree of 1 for GPT-OSS, whose spec refuses expert parallelism; with the dense
    part asked to be replicated, that is still a replication the plan reached, and is refused."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "1")
    run_as(2)
    serve(GptOssConfig(num_local_experts=4, **_TINY))

    with pytest.raises(RuntimeError, match="cannot be quantized expert-parallel yet"):
        resolve_distribution_plan(str(tmp_path))


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize(
    "value",
    ["0", "", " ", "AUTO", " auto "],
    ids=["zero", "exported-empty", "exported-blank", "capitals", "padded"],
)
def test_an_ep_degree_of_0_is_auto(run_as, serve, monkeypatch, tmp_path, value):
    """And so is one exported empty or blank, and auto in any case or padded: the plan decides."""
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
    whole MoE must fit, and here it does not. The expert-parallel load replicates the dense part,
    so an EP degree left to the decision gives way to FSDP2 for the whole model, where one asked
    for stands, for the load to refuse."""
    run_as(4)
    config = serve(_qwen3_moe(num_experts=8))
    model = _meta(config)
    experts = sum(_bytes(layer.mlp.experts) for layer in model.model.layers)
    # A GPU 60% of which holds the rest of the model and that share of the experts.
    gpu_memory(int((_bytes(model) - experts + experts_fitting * experts) / 0.6))

    decided = DistributionPlan(4, 4) if dense == "replicate" else DistributionPlan(4, dense="fsdp")
    assert resolve_distribution_plan(str(tmp_path))[0] == decided
    assert resolve_distribution_plan(str(tmp_path), ep_size=4)[0] == DistributionPlan(4, 4, dense)
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
        (" ", "fsdp"),
        (" world ", "fsdp"),
        ("2", None),
    ],
    ids=["1", "4", "WORLD", "exported-empty", "exported-blank", "padded", "part-of-the-world"],
)
def test_the_dense_knob_replicates_or_shards_over_the_whole_world(
    run_as, serve, gpu_memory, monkeypatch, tmp_path, value, dense
):
    """Exported empty or blank, the knob is auto, which shards a model too big for one GPU. Sharding
    over part of the world is refused: no layout does that."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", value)
    run_as(4)
    serve(LlamaConfig(**_TINY))
    gpu_memory(1)  # too small for the whole model, so only auto tells "" from "1"

    if dense is None:
        with pytest.raises(ValueError, match=r"expected auto, 1 .* or world"):
            resolve_distribution_plan(str(tmp_path))
    else:
        assert resolve_distribution_plan(str(tmp_path))[0].dense == dense


@needs_ep_transformers
@pytest.mark.parametrize(
    ("config", "ep_size", "experts"),
    [(LlamaConfig(**_TINY), 2, 0), (_qwen3_moe(num_experts=6), 4, 6)],
    ids=["dense-model", "experts-not-divisible"],
)
def test_an_ep_degree_that_cannot_shard_the_experts_is_refused(
    run_as, serve, tmp_path, config, ep_size, experts
):
    """Every rank of an EP group must hold whole experts, and a model without routed experts has
    none to shard. (Below the expert-parallel load's transformers, any EP degree is refused first.)"""
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


@needs_ep_transformers
def test_the_expert_count_is_read_from_the_config_holding_the_plan(
    run_as, serve, monkeypatch, tmp_path
):
    """A VLM whose nested config holds the MoE and its plan counts the experts there, not by an
    expert-ish field on its root config."""
    plan = {
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
    }
    text = SimpleNamespace(base_model_ep_plan=plan, num_experts=8)
    serve(SimpleNamespace(model_type="vlm", architectures=[], text_config=text, num_experts=3))
    built = model_with_decoder_at("model.")
    built.tie_weights = lambda: None  # a PreTrainedModel's; nothing here is tied
    monkeypatch.setattr(model_load_utils, "build_meta_causal_lm", lambda *a, **k: built)
    monkeypatch.setattr(transformers, "__version__", "5.7.0")  # one the expert-parallel load takes
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


def _every_rank_takes_rank0s_plan_job(rank, size, ckpt, knobs, expected, spec_applied):
    with bounded_collectives(), _knobs(**knobs[rank]):
        plan, hf_config = resolve_distribution_plan(ckpt)
    assert plan == expected, (rank, plan)
    if spec_applied is not None:
        spec_plan = get_spec(hf_config.model_type).ep_spec.base_model_ep_plan
        applied = getattr(hf_config, "base_model_ep_plan", None) == spec_plan
        assert applied is spec_applied, (rank, applied)


@pytest.mark.parametrize(
    ("config", "knobs", "expected", "spec_applied"),
    [
        pytest.param(
            LlamaConfig(**_TINY),
            [{}, {"MODELOPT_FSDP_SIZE": "world"}],
            DistributionPlan(2),
            None,
            id="rank1-would-shard",
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=8),
            [{}, {"MODELOPT_EP_DEGREE": "1"}],
            DistributionPlan(2, 2),
            True,
            marks=needs_ep_transformers,
            id="rank1-would-skip-the-spec",
        ),
        pytest.param(
            _deepseek_v3(n_routed_experts=8),
            [{"MODELOPT_EP_DEGREE": "1", "MODELOPT_FSDP_SIZE": "world"}, {}],
            DistributionPlan(2, dense="fsdp"),
            False,
            marks=needs_ep_transformers,
            id="rank1-would-apply-the-spec",
        ),
    ],
)
def test_every_rank_takes_the_plan_rank0_decided(
    cpu_dist_workers, tmp_path, config, knobs, expected, spec_applied
):
    """Over real broadcasts. Rank 1's own knobs would decide another plan, or apply the model's spec
    where rank 0 skipped it, or skip it where rank 0 applied it: ranks loading different plans, or
    one without the expert-parallel plan the others shard by, build mismatched meshes and hang."""
    config.save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _every_rank_takes_rank0s_plan_job,
        ckpt=str(tmp_path),
        knobs=knobs,
        expected=expected,
        spec_applied=spec_applied,
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
        "load_and_prepare_ep",
        recording("ep", model_load_utils.load_and_prepare_ep),
    )
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


def _pin_cuda(monkeypatch, gpus, bound=None):
    """Pin the CUDA state a rank's GPU is read from: ``gpus`` visible, and the process bound to
    cuda:``bound``, or to none yet, so its local rank picks one. Tests then read the case rather
    than the host, whose process may have set CUDA up already."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpus > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: bound is not None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: bound or 0)


@pytest.fixture
def a_gpu(monkeypatch):
    """This rank has a GPU of its own, which the FSDP2 load shards onto; the stand-in loaders never
    use it."""
    _pin_cuda(monkeypatch, 8)


@pytest.fixture
def process_group(monkeypatch, a_gpu):
    """Rank 1 of a 2-rank process group, as the FSDP2 load reads it."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(dist, "local_rank", lambda: 1)


# ----------------------------------------------------------------------------------- the loads


@pytest.mark.parametrize(
    ("passed", "gpus", "placed"),
    [
        ({}, 4, {"device_map": {"": "cuda:1"}}),
        ({"device_map": "cpu"}, 4, {"device_map": "cpu"}),
        ({"device_map": None}, 4, {"device_map": None}),
        ({}, 1, {}),
        ({"mp_policy": None, "cpu_offload": None}, 4, {"device_map": {"": "cuda:1"}}),
        # As hf_ptq passes --cpu_offload when it is not set: the FSDP2 load's own default.
        ({"cpu_offload": False}, 4, {"device_map": {"": "cuda:1"}}),
        ({"broadcast_chunk_size": 8}, 4, {"device_map": {"": "cuda:1"}}),
    ],
    ids=[
        "defaults_to_the_rank_gpu",
        "caller_placement_wins",
        "none_loads_on_the_cpu",
        "no_gpu_of_its_own",
        "fsdp2_options_at_none",
        "cpu_offload_at_its_default",
        "broadcast_chunk_size_at_its_default",
    ],
)
def test_a_replicated_plan_loads_the_whole_model_on_every_rank(
    run_as, serve, loaders, auto_class, monkeypatch, tmp_path, passed, gpus, placed
):
    """Each rank takes the whole model onto its own GPU; "auto" would let every rank's accelerate
    pick the same idle GPU. device_map=None is from_pretrained's own: a load on the CPU, which the
    caller places, as hf_ptq does. A rank with none of its own, on a host with fewer GPUs than
    ranks, is left to from_pretrained's placement rather than handed a device it does not have. The
    FSDP2 load's own options left unset or at its defaults ask nothing and do not reach
    from_pretrained."""
    run_as(4, rank=1)
    serve(LlamaConfig(**_TINY))
    _pin_cuda(monkeypatch, gpus)

    model = load_pretrained(str(tmp_path), auto_class, **passed)

    assert auto_class.calls == [placed]
    assert distribution_plan(model) == DistributionPlan(4)
    assert loaders == {}


def test_a_rank_bound_to_its_one_visible_gpu_loads_onto_it(
    run_as, serve, loaders, auto_class, monkeypatch, tmp_path
):
    """A launch that exposes one GPU to each process binds every rank to cuda:0, whatever its local
    rank: each takes the whole model onto the GPU it is bound to, rather than read as having
    none."""
    run_as(4, rank=1)
    serve(LlamaConfig(**_TINY))
    _pin_cuda(monkeypatch, 1, bound=0)

    load_pretrained(str(tmp_path), auto_class)

    assert auto_class.calls == [{"device_map": {"": "cuda:0"}}]


def test_a_rank_without_a_gpu_of_its_own_says_so(
    run_as, serve, loaders, auto_class, monkeypatch, tmp_path
):
    """On a host with fewer GPUs than ranks, a rank of a replicated load left without one loads the
    model where from_pretrained places it while its peers load onto theirs: it warns itself, as rank
    0 may have a GPU of its own and nothing to say."""
    run_as(4, rank=1)
    serve(LlamaConfig(**_TINY))
    _pin_cuda(monkeypatch, 1)

    with pytest.warns(UserWarning, match="local rank 1"):
        load_pretrained(str(tmp_path), auto_class)
    assert auto_class.calls == [{}]


@pytest.mark.parametrize(
    "defaults",
    [
        {"dtype": "auto"},
        {"dtype": None},
        {"torch_dtype": "auto"},
        # As hf_ptq passes them: placement options at from_pretrained's own default.
        {
            "device_map": None,
            "max_memory": None,
            "offload_folder": None,
            "quantization_config": None,
        },
    ],
    ids=["auto", "none", "torch_dtype", "placement-none"],
)
def test_an_fsdp_plan_loads_through_the_parallel_fsdp2_loader(
    run_as, loaders, process_group, monkeypatch, tmp_path, defaults
):
    """FSDP2 shards over the process group it reads its rank and GPU from, by a config read once,
    with the Hub options the load was given (custom code an ``auto_map`` names in another repo).
    It keeps the checkpoint's dtype, as from_pretrained does for "auto" and for None, under either
    name of the argument, and takes a placement option passed as None as not passed."""
    run_as(2, rank=1)
    served, reads = LlamaConfig(**_TINY), []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(k) or served
    )
    plan = DistributionPlan(2, dense="fsdp")

    model = load_pretrained(
        str(tmp_path), distribution=plan, cpu_offload=True, local_files_only=True, **defaults
    )

    call = loaders.pop("fsdp2")
    assert (call["device"], call["rank"], call["world_size"]) == (torch.device("cuda:1"), 1, 2)
    assert call["cpu_offload"] is True
    assert call["hf_config"] is served
    assert [read.get("local_files_only") for read in reads] == [True]
    assert distribution_plan(model) == plan
    assert loaders == {}


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_a_text_only_vlm_the_fsdp2_load_builds_reaches_it(
    run_as, serve, loaders, process_group, tmp_path, model_type
):
    """The FSDP2 load builds Qwen3.5's VLMs through their image-text class, as hf_ptq's --use_fsdp2
    does, so a plan that shards one reaches the load rather than being refused as no causal LM."""
    try:
        config = serve(AutoConfig.for_model(model_type))
    except ValueError:
        pytest.skip("Qwen3.5 requires a recent Transformers")
    run_as(2, rank=1)

    load_pretrained(str(tmp_path), distribution=DistributionPlan(2, dense="fsdp"))

    assert loaders.pop("fsdp2")["hf_config"] is config


@pytest.mark.parametrize("handed", [True, False], ids=["handed", "decided"])
def test_a_sharded_load_offers_the_knob_only_for_a_plan_it_decided(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, handed
):
    """MODELOPT_FSDP_SIZE counts only for a plan the load decides: to keep a model whole under a
    plan of its own, a caller passes a replicated one, which is all the refusal offers it."""
    run_as(2, rank=1)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    plan = {"distribution": DistributionPlan(2, dense="fsdp")} if handed else {}

    with pytest.raises(TypeError, match=r"\['device_map'\] do not apply") as refused:
        load_pretrained(str(tmp_path), device_map="auto", **plan)
    assert ("MODELOPT_FSDP_SIZE=1" in str(refused.value)) is not handed
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

    with pytest.raises(TypeError, match=r"hf_config.*resolve_distribution_plan"):
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
    class may not build (an encoder-decoder loaded through AutoModel), nor applies the model's spec,
    whose prepare could fail every rank, or patch the process, for a load that is never
    expert-parallel. The decision reads the checkpoint's config as the class does, through its own
    config class, which AutoConfig need not know."""
    run_as(2)
    if own_config:
        _OwnConfig(**_TINY).save_pretrained(tmp_path)
        auto_class.config_class = _OwnConfig
    else:
        serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(
        model_load_utils, "build_meta_causal_lm", lambda *a, **k: pytest.fail("built a model")
    )
    monkeypatch.setattr(
        model_load_utils, "apply_distributed_support", lambda *a, **k: pytest.fail("applied a spec")
    )
    # As on a transformers new enough for expert parallelism, whose floor would rule it out here.
    monkeypatch.setattr(model_load_utils, "_spec_refuses_ep", lambda model_type: False)
    if fsdp_size is None:
        model = load_pretrained(str(tmp_path), auto_class)
        assert distribution_plan(model) == DistributionPlan(2)
        assert len(auto_class.calls) == 1
    else:
        monkeypatch.setenv("MODELOPT_FSDP_SIZE", fsdp_size)
        with pytest.raises(NotImplementedError, match="auto_class=Auto "):
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


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize("knob", [None, "4"], ids=["left-to-decide", "asked-for"])
def test_a_load_it_decides_shards_the_experts(
    run_as, serve, loaders, a_gpu, monkeypatch, tmp_path, knob
):
    """An MoE whose EP degree is left to the decision is decided expert-parallel and loads through
    the expert-parallel load, as one MODELOPT_EP_DEGREE asks for does."""
    run_as(4)
    serve(_qwen3_moe())
    if knob is not None:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", knob)
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: None)

    model = load_pretrained(str(tmp_path))
    assert loaders["ep"]["plan"] == DistributionPlan(4, ep_size=4)
    assert distribution_plan(model) == DistributionPlan(4, ep_size=4)


@needs_ep_transformers
@pytest.mark.parametrize(
    ("dense", "one_gpu_per_process"),
    [("replicate", False), ("fsdp", False), ("replicate", True)],
    ids=["replicate", "fsdp", "one-gpu-per-process"],
)
def test_an_expert_parallel_plan_loads_through_the_ep_loader(
    run_as, serve, loaders, auto_class, a_gpu, monkeypatch, tmp_path, dense, one_gpu_per_process
):
    """Decided automatically, an EP plan loads through load_and_prepare_ep -- with the plan, the
    spec-prepared config and the Hub options the load was given -- and through no other load. It
    returns the model unfrozen, as the other loads do, and an option passed as None asks nothing
    of it. That load replicates the dense part, so a plan that asks for expert parallelism and
    FSDP2-shards the dense part too is refused before anything loads, naming the ways out. A launch
    that shows each process one GPU, bound to cuda:0, gives a rank of any local rank a GPU of its
    own."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", {"replicate": "1", "fsdp": "world"}[dense])
    if dense == "fsdp":  # left to the decision, the EP degree would give way to FSDP2
        monkeypatch.setenv("MODELOPT_EP_DEGREE", "4")
    run_as(4, rank=1 if one_gpu_per_process else 0)
    if one_gpu_per_process:
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    served = serve(_deepseek_v3(n_routed_experts=8))
    options = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "require_lm_head": None,
        "local_files_only": True,
    }

    if dense == "fsdp":
        with pytest.raises(
            NotImplementedError, match=r"load replicates it on every rank\..*\(MODELOPT_FSDP_SIZE=1"
        ):
            load_pretrained(str(tmp_path), **options)
    else:
        model = load_pretrained(str(tmp_path), **options)
        call = loaders.pop("ep")
        plan = DistributionPlan(4, ep_size=4, dense=dense)
        assert (call["ckpt_path"], call["plan"]) == (str(tmp_path), plan)
        assert (call["trust_remote_code"], call["attn_implementation"]) == (True, "eager")
        assert call["freeze"] is False
        assert "require_lm_head" not in call  # its default, not None read as False
        assert call["hub_options"] == {"local_files_only": True}
        assert (
            call["hf_config"].base_model_ep_plan
            == get_spec("deepseek_v3").ep_spec.base_model_ep_plan
        )
        assert served.base_model_ep_plan is None
        assert distribution_plan(model) == plan
    assert loaders == {}
    assert auto_class.calls == []


@needs_ep_transformers
@pytest.mark.parametrize(
    ("planned", "passed_as", "preparations"),
    [(False, None, 1), (True, None, 1), (True, "hf_config", 0), (True, "config", 1)],
    ids=[
        "decided_by_the_load",
        "planned",
        "planned_with_its_config",
        "planned_with_a_config_for_the_load",
    ],
)
def test_a_load_prepares_its_config_once(
    run_as, loaders, monkeypatch, tmp_path, planned, passed_as, preparations
):
    """A model's EP preparation can patch its modeling code, so a load runs it once, and not at all
    when handed the config resolve_distribution_plan prepared. A config passed for the load is the
    caller's own, and the one deciding reads may go on to a load that is not expert-parallel, so
    either is prepared as a copy. Every config it reads takes the Hub options the load was given
    (custom code an ``auto_map`` names in another repo)."""
    run_as(4)
    # A GPU of its own, which the expert-parallel load needs; the decision weighs no GPU memory.
    monkeypatch.setattr(model_load_utils, "_has_own_gpu", lambda: True)
    config, reads = _deepseek_v3(n_routed_experts=8), []
    as_passed = config.to_dict()
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(k) or config
    )
    prepared = []

    def apply_distributed_support(model_type, hf_config, **kwargs):
        prepared.append(model_type)
        hf_config.base_model_ep_plan = get_spec("deepseek_v3").ep_spec.base_model_ep_plan
        return hf_config

    monkeypatch.setattr(model_load_utils, "apply_distributed_support", apply_distributed_support)
    kwargs = {"distribution": DistributionPlan(4, ep_size=4)} if planned else {}
    if passed_as:
        kwargs[passed_as] = config

    load_pretrained(str(tmp_path), local_files_only=True, **kwargs)

    assert len(prepared) == preparations
    loaded = loaders["ep"]["hf_config"]
    if passed_as == "config" or not planned:
        assert loaded is not config and loaded.base_model_ep_plan
        assert config.to_dict() == as_passed
    else:
        assert loaded is config
    if planned and passed_as is None:  # read once, for the model type and for the load
        assert len(reads) == 1, reads
    assert all(read.get("local_files_only") is True for read in reads)


def test_a_model_its_spec_refuses_is_refused_with_its_config_handed_in(
    run_as, loaders, a_gpu, tmp_path
):
    """Handed the config prepared with its plan, the load does not prepare it again, but a model
    whose spec refuses expert parallelism is refused all the same, before anything loads."""
    run_as(4)
    config = GptOssConfig(**_TINY, num_local_experts=4)

    with pytest.raises(NotImplementedError, match="gpt_oss cannot be quantized expert-parallel"):
        load_pretrained(
            str(tmp_path), distribution=DistributionPlan(4, ep_size=4), hf_config=config
        )
    assert loaders == {}


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
        assert "expected a DistributionPlan" in str(refused), (rank, refused)


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


_FSDP2_PLAN = DistributionPlan(2, dense="fsdp")


def _one_rank_refused_job(
    rank, size, ckpt, refused_on_1, gpus_on_1, match, knobs_on_1=None, plan=_FSDP2_PLAN
):
    refused = None
    gpus = size if rank == 0 else gpus_on_1  # what this rank's host shows it
    with (
        bounded_collectives(),
        pytest.MonkeyPatch.context() as cuda,
        mock.patch.dict(os.environ, (knobs_on_1 or {}) if rank == 1 else {}),
        mock.patch.object(model_load_utils, "ensure_local_checkpoint", _fetched_in_step),
        mock.patch.object(model_load_utils, "parallel_load_and_prepare_fsdp2", _loaded_in_step),
    ):
        _pin_cuda(cuda, gpus)  # the pooled worker's own CUDA state is whatever an earlier job left
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            load_pretrained(ckpt, distribution=plan, **(refused_on_1 if rank == 1 else {}))
        except Exception as error:
            refused = str(error)
    # Rank 1's refusal on both ranks, rather than rank 0 failing at the group's timeout.
    assert refused is not None and match in refused, (rank, refused)
    assert rank == 1 or "rank 1 failed" in refused, refused


@pytest.mark.parametrize(
    ("refused_on_1", "gpus_on_1", "match"),
    [
        (
            {"config": LlamaConfig(**_TINY), "hf_config": LlamaConfig(**_TINY)},
            2,
            "config and hf_config",
        ),
        ({"dtype": torch.float32}, 2, "dtype=torch.float32"),
        (
            {"hf_config": LlamaConfig(**_TINY, quantization_config={"quant_method": "fp8"})},
            2,
            "dequantize",
        ),
        ({}, 0, "as local rank 1"),
        ({}, 1, "as local rank 1"),
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


def test_a_knob_one_rank_cannot_parse_fails_every_rank(cpu_dist_workers, tmp_path):
    """With no plan handed, rank 1 alone reads a malformed MODELOPT_EP_DEGREE: every rank raises it,
    rather than leave rank 0 waiting in the plan decision's collectives."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _one_rank_refused_job,
        ckpt=str(tmp_path),
        refused_on_1={},
        gpus_on_1=2,
        match="MODELOPT_EP_DEGREE='eigth'",
        knobs_on_1={"MODELOPT_EP_DEGREE": "eigth"},
        plan=None,
    )


def _one_visible_gpu_per_process_job(rank, size, ckpt):
    devices = []

    def load(ckpt_path, device, *args, **kwargs):
        devices.append(device)
        torch.distributed.barrier()
        return _Loaded()

    with (
        bounded_collectives(),
        pytest.MonkeyPatch.context() as cuda,
        mock.patch.object(model_load_utils, "ensure_local_checkpoint", _fetched_in_step),
        mock.patch.object(model_load_utils, "parallel_load_and_prepare_fsdp2", load),
    ):
        # Each rank sees only its own GPU, which its process bound as cuda:0, whatever its local
        # rank.
        _pin_cuda(cuda, 1, bound=0)
        load_pretrained(ckpt, distribution=DistributionPlan(2, dense="fsdp"))
    assert devices == [torch.device("cuda", 0)], (rank, devices)


def test_one_visible_gpu_per_process_shards_on_every_rank(cpu_dist_workers, tmp_path):
    """Two ranks, each shown only its own GPU and bound to it as cuda:0, both shard onto it: rank 1,
    whose local rank numbers no GPU it sees, is not refused as having none."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(_one_visible_gpu_per_process_job, ckpt=str(tmp_path))


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


def _one_rank_cannot_apply_the_spec_job(rank, size, ckpt, step):
    apply = model_load_utils.apply_distributed_support

    def apply_on_rank0_only(*args, **kwargs):
        if rank == 1:
            raise OSError("this rank cannot import the checkpoint's code")
        return apply(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(bounded_collectives())
        stack.enter_context(
            mock.patch.object(model_load_utils, "apply_distributed_support", apply_on_rank0_only)
        )
        if step == "load":
            # Every rank has a GPU of its own, without which the load refuses it first.
            stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock.patch.object(torch.cuda, "device_count", return_value=size))
            stack.enter_context(
                mock.patch.object(model_load_utils, "ensure_local_checkpoint", _fetched_in_step)
            )
            stack.enter_context(
                mock.patch.object(model_load_utils, "load_and_prepare_ep", _loaded_in_step)
            )
        try:
            if step == "decision":
                resolve_distribution_plan(ckpt)
            else:
                load_pretrained(ckpt, distribution=DistributionPlan(2, ep_size=2))
        except (OSError, RuntimeError) as error:
            raised = str(error)
        else:
            raised = "no error"
    assert "cannot import the checkpoint's code" in raised, raised


@needs_ep_transformers
@pytest.mark.parametrize("step", ["decision", "load"])
def test_a_rank_that_cannot_apply_the_spec_fails_every_rank(cpu_dist_workers, tmp_path, step):
    """A spec's preparation reads the checkpoint and imports its code on each rank, which can fail
    on one alone; the others must not wait for it, in the plan's broadcast when deciding, nor in the
    load a handed expert-parallel plan starts."""
    _deepseek_v3(n_routed_experts=8).save_pretrained(tmp_path)
    cpu_dist_workers.run(_one_rank_cannot_apply_the_spec_job, ckpt=str(tmp_path), step=step)


@pytest.mark.parametrize("sharding", ["expert-parallel", "fsdp2"])
@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"kwargs": {"device_map": "auto"}}, TypeError, r"\['device_map'\] do not apply"),
        ({"kwargs": {"dtype": torch.bfloat16}}, TypeError, r"dtype=torch\.bfloat16"),
        (
            {"kwargs": {"torch_dtype": torch.bfloat16}},
            TypeError,
            r"torch_dtype=torch\.bfloat16",
        ),
        (
            {"auto_class": AutoModel},
            NotImplementedError,
            r"auto_class=AutoModel: leave it out.*DistributionPlan\(2, (1, |dense=)'replicate'\)",
        ),
        ({"mtp": True}, NotImplementedError, "MTP head"),
        (
            {"kwargs": {"config": LlamaConfig(**_TINY), "hf_config": LlamaConfig(**_TINY)}},
            TypeError,
            "config and hf_config",
        ),
        ({"gpus": 0}, RuntimeError, "this rank has none"),
        ({"gpus": 0, "config": True}, RuntimeError, "this rank has none"),
    ],
    ids=[
        "placement",
        "dtype",
        "torch_dtype",
        "model_class",
        "mtp_head",
        "two_configs",
        "no_gpu",
        "no_gpu_config_passed",
    ],
)
def test_a_sharded_load_refuses_what_it_cannot_honor(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, sharding, change, error, match
):
    """Placement, a model class, a head only a whole model builds, two configs to load with, a rank
    without a GPU of its own: none reaches a sharded load, which fails before it starts rather than
    load something other than what was asked. A rank without a GPU is refused before the model's
    spec prepares its config, which can import the checkpoint's code, whichever config it loads
    with."""
    run_as(2)
    expert_parallel = sharding == "expert-parallel"
    config = serve(_deepseek_v3() if expert_parallel else LlamaConfig(**_TINY))
    if change.get("mtp"):
        monkeypatch.setattr(models_hf, "checkpoint_has_mtp", lambda *a: True)
    if "gpus" in change:  # a host that shows this rank no GPU
        monkeypatch.setattr(torch.cuda, "device_count", lambda: change["gpus"])
        monkeypatch.setattr(
            model_load_utils,
            "apply_distributed_support",
            lambda *a, **k: pytest.fail("prepared a rank that cannot load"),
        )
    plan = DistributionPlan(2, ep_size=2) if expert_parallel else DistributionPlan(2, dense="fsdp")
    kwargs = {**change.get("kwargs", {}), **({"config": config} if change.get("config") else {})}

    with pytest.raises(error, match=match) as refused:
        load_pretrained(str(tmp_path), change.get("auto_class"), distribution=plan, **kwargs)
    if "device_map" in change.get("kwargs", {}) or "gpus" in change:
        # The way to a whole model it names, for the plan handed in: an expert-parallel MoE's is a
        # plan with an EP degree of 1.
        whole_moe = "DistributionPlan(2, 1, 'replicate')"
        assert (whole_moe in str(refused.value)) == expert_parallel, refused.value
    if {"dtype", "torch_dtype"} & set(change.get("kwargs", {})):
        # Only the FSDP2 load takes an mp_policy to compute in another dtype.
        assert ("mp_policy" in str(refused.value)) != expert_parallel, refused.value
    assert loaders == {}


@pytest.mark.parametrize(
    ("distribution", "checkpoint", "options", "error", "match"),
    [
        (
            DistributionPlan(4, dense="fsdp"),
            "causal-lm",
            None,
            ValueError,
            "is for 4 ranks",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            "quantized",
            None,
            NotImplementedError,
            "dequantize",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            "quantized-language-model",
            {"trust_remote_code": True},
            NotImplementedError,
            "dequantize",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            "vlm",
            None,
            NotImplementedError,
            "llava",
        ),
        ("fsdp2", "causal-lm", None, TypeError, "expected a DistributionPlan"),
        (
            DistributionPlan(2),
            "causal-lm",
            "hf_config",
            TypeError,
            r"hf_config.*resolve_distribution_plan",
        ),
        (
            DistributionPlan(2),
            "causal-lm",
            {"cpu_offload": True},
            TypeError,
            r"\['cpu_offload'\] only apply",
        ),
        (
            DistributionPlan(2),
            "causal-lm",
            {"freeze": False},
            TypeError,
            r"\['freeze'\] only apply to the expert-parallel load",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            "causal-lm",
            {"require_lm_head": False},
            TypeError,
            (
                r"\['require_lm_head'\] only apply to the expert-parallel load, and the FSDP2 "
                r"load .*; drop them, or pass a plan with ep_size > 1"
            ),
        ),
        (
            DistributionPlan(2, ep_size=2),
            "causal-lm",
            {"broadcast_chunk_size": None},
            TypeError,
            (
                r"\['broadcast_chunk_size'\] only apply to the FSDP2 load, and the "
                r"expert-parallel load .*; drop them, or pass a plan with ep_size=1 and dense='fsdp'"
            ),
        ),
    ],
    ids=[
        "another-world",
        "quantized-checkpoint",
        "quantized-language-model",
        "vlm-checkpoint",
        "not-a-plan",
        "hf-config-on-a-whole-load",
        "fsdp2-option-on-a-whole-load",
        "ep-option-on-a-whole-load",
        "ep-option-on-the-fsdp2-load",
        "fsdp2-chunking-as-none-on-the-ep-load",
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
    read as stored, even when only a VLM's language model is quantized and its remote code builds it
    as a causal LM; a VLM but a text-only one it supports, which FSDP2 would build through
    AutoModelForCausalLM, refused or as its language model alone; anything but a plan; and the
    sharded loads' config, or an option only one sharded load takes, which from_pretrained would
    take for a model argument and fail far from here, or drop."""
    run_as(2)
    config = serve(transformers.LlavaConfig() if checkpoint == "vlm" else LlamaConfig(**_TINY))
    if checkpoint == "quantized":
        config.quantization_config = {"quant_method": "fp8"}
    if checkpoint == "quantized-language-model":  # Kimi-K2.5's layout
        config.auto_map = _REMOTE_VLM
        config.text_config = LlamaConfig(**_TINY, quantization_config={"quant_method": "fp8"})
    if options == "hf_config":
        options = {"auto_class": auto_class, "hf_config": config}

    with pytest.raises(error, match=match):
        load_pretrained(str(tmp_path), distribution=distribution, **(options or {}))
    assert loaders == {}
    assert auto_class.calls == []


@pytest.mark.parametrize("device_map", ["auto", None], ids=["placed", "none"])
def test_an_automatic_layout_says_so_when_it_refuses_a_placement(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, device_map
):
    """With neither a plan nor MODELOPT_FSDP_SIZE, rank 0's GPU memory decides to shard a model too
    big for it, and the sharded load refuses options it does not take. The refusal says the layout
    was decided rather than asked for, and how to keep the load whole. device_map=None, as hf_ptq
    passes it, places nothing, and the model is sharded."""
    run_as(2, rank=1)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: 1)

    if device_map is None:
        load_pretrained(str(tmp_path), AutoModelForCausalLM, device_map=None, dtype="auto")
        assert "fsdp2" in loaders
        return
    with pytest.raises(TypeError, match=r"\['device_map'\] do not apply.*MODELOPT_FSDP_SIZE=1"):
        load_pretrained(str(tmp_path), device_map="auto")
    assert loaders == {}


class _ContextLoadReachedError(Exception):
    """Raised where init_quantized_weights' from_pretrained builds the model, past its checks."""


@pytest.mark.parametrize("memory", [1 << 50, 1], ids=["fits", "too-big"])
def test_a_load_it_decides_inside_init_quantized_weights_is_whole_and_unplaced(
    run_as, loaders, process_group, monkeypatch, tmp_path, memory
):
    """Inside init_quantized_weights, with nothing asking to shard, every rank gets the whole model,
    whatever rank 0's GPU memory, and the load leaves the device map to the context, which computes
    it and refuses one passed: the context's from_pretrained gets past that refusal."""
    run_as(2, rank=1)
    LlamaConfig(**_TINY).save_pretrained(tmp_path)  # read from disk, as the context reads it too
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: memory)
    monkeypatch.setattr(
        accelerate_plugin, "init_empty_weights", mock.Mock(side_effect=_ContextLoadReachedError)
    )

    with init_quantized_weights(mtq.INT8_DEFAULT_CFG), pytest.raises(_ContextLoadReachedError):
        load_pretrained(str(tmp_path))
    assert loaders == {}


@pytest.mark.parametrize("handed", [False, True], ids=["knob", "handed"])
def test_a_sharded_load_is_refused_inside_init_quantized_weights(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, handed
):
    """init_quantized_weights quantizes and compresses in the from_pretrained it patches, which the
    FSDP2 load never calls: a knob or a plan asking to shard there is refused rather than return the
    model unquantized."""
    run_as(2, rank=1)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    plan = {"distribution": DistributionPlan(2, dense="fsdp")} if handed else {}

    with (
        init_quantized_weights(mtq.INT8_DEFAULT_CFG),
        pytest.raises(NotImplementedError, match=r"init_quantized_weights.*dense='replicate'"),
    ):
        load_pretrained(str(tmp_path), **plan)
    assert loaders == {}


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize("handed", [False, True], ids=["knob", "handed"])
def test_an_expert_parallel_load_is_refused_inside_init_quantized_weights(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, handed
):
    """The expert-parallel load never calls from_pretrained either, and its way to a whole model on
    every rank sets the EP degree to 1 as well, as the decision refuses to replicate an MoE unasked."""
    run_as(2, rank=1)
    serve(_qwen3_moe(num_experts=8))
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: None)
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    plan = {"distribution": DistributionPlan(2, 2)} if handed else {}

    with (
        init_quantized_weights(mtq.INT8_DEFAULT_CFG),
        pytest.raises(
            NotImplementedError,
            match=r"expert-parallel load .*init_quantized_weights.*DistributionPlan\(2, 1, 'replicate'\)",
        ),
    ):
        load_pretrained(str(tmp_path), **plan)
    assert loaders == {}


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize(
    "options",
    [{"device_map": "auto"}, {"freeze": True, "require_lm_head": False}],
    ids=["placement", "its-own-options"],
)
def test_an_automatic_expert_parallel_layout_says_so_when_it_refuses_a_placement(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, options
):
    """With neither a plan nor MODELOPT_EP_DEGREE, the decision shards an MoE's experts, and the
    expert-parallel load refuses options that place a whole model. The refusal says the layout was
    decided rather than asked for, and how to keep the load whole. The load's own options place
    nothing, and reach it."""
    run_as(2, rank=1)
    serve(_qwen3_moe(num_experts=8))
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: None)

    if "device_map" not in options:
        load_pretrained(str(tmp_path), **options)
        assert {name: loaders["ep"][name] for name in options} == options
        return
    with pytest.raises(
        TypeError, match=r"\['device_map'\] do not apply.*no plan or MODELOPT_EP_DEGREE"
    ):
        load_pretrained(str(tmp_path), **options)
    assert loaders == {}


@pytest.mark.parametrize(
    ("checkpoint", "match"),
    [("quantized", "dequantize"), ("mtp", "MTP head")],
    ids=["quantized", "mtp"],
)
def test_a_measured_layout_names_the_way_back_when_it_refuses_a_checkpoint(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, checkpoint, match
):
    """Nothing asked for the layout rank 0's GPU memory chose, so a checkpoint the FSDP2 load cannot
    take is refused with the way to load it whole, as every refusal of that load says."""
    run_as(2, rank=1)
    config = serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(model_load_utils, "_gpu_memory", lambda: 1)
    if checkpoint == "quantized":
        config.quantization_config = {"quant_method": "fp8"}
    else:
        monkeypatch.setattr(models_hf, "checkpoint_has_mtp", lambda *a: True)

    with pytest.raises(NotImplementedError, match=match) as refused:
        load_pretrained(str(tmp_path))
    assert "MODELOPT_FSDP_SIZE=1" in str(refused.value)
    assert loaders == {}


@pytest.mark.parametrize(
    ("distribution", "options", "error", "match"),
    [
        (DistributionPlan(4), {}, ValueError, "is for 4 ranks"),
        (
            DistributionPlan(2),
            {"hf_config": LlamaConfig(**_TINY)},
            TypeError,
            r"hf_config.*resolve_distribution_plan",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            {"device_map": "auto"},
            TypeError,
            r"\['device_map'\] do not apply",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            {"auto_class": AutoModel},
            NotImplementedError,
            "auto_class=AutoModel",
        ),
        (
            DistributionPlan(2, dense="fsdp"),
            {"dtype": torch.float32},
            TypeError,
            r"dtype=torch\.float32",
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
        with pytest.raises(NotImplementedError, match="dequantize"):
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
            r"is for 8 ranks.*distributed_run",
        ),
        (
            DistributionPlan(dense="fsdp"),
            {},
            RuntimeError,
            r"DistributionPlan\(8, dense='fsdp'\)",
        ),
        (
            None,
            {"MODELOPT_FSDP_SIZE": "world"},
            RuntimeError,
            "MODELOPT_FSDP_SIZE=world",
        ),
        (None, {"MODELOPT_EP_DEGREE": "4"}, RuntimeError, "MODELOPT_EP_DEGREE=4"),
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


# ======================================================================== the expert-parallel load
# What the expert-parallel load reads off the loaded model and its config: it takes an expert-parallel
# plan, ``_seed_parallel_states`` gives every module the parallel state its amax syncs over, and
# ``_ckpt_quant_method`` reads the checkpoint's native quantization. Everything runs in one CPU process
# but the bounded two-rank tests at the end.

# ------------------------------------------------------------------------------- the plan

# The model type of the configs that stand in for a checkpoint's here: no model spec names it.
_NO_SPEC = "toy_moe"

_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class _Mesh:
    """A device mesh, as far as the loader reads one; its groups run no collective (``-1``)."""

    device_type = "cpu"

    def __getitem__(self, name):
        return self

    def size(self):
        return 2

    def get_group(self):
        return -1


def _cuda(gpu, bound=True, count=1, uuid="GPU-0"):
    """Patches pinning the CUDA state the GPU check reads: ``count`` visible GPUs and this rank on
    cuda:0, the one named ``uuid``, bound to it or not yet; or, without ``gpu``, none. The check then
    reads the case, not the host the test runs on, whose visible GPU no test binds."""
    state = {
        "is_available": gpu,
        "is_initialized": gpu and bound,
        "current_device": 0,
        "device_count": count if gpu else 0,
        "get_device_properties": SimpleNamespace(uuid=uuid),
    }
    return [mock.patch(f"torch.cuda.{name}", return_value=value) for name, value in state.items()]


@pytest.fixture
def gloo_world_of_one(gloo_world_of_one):
    """The one-rank world, with no GPU whatever the host has (:func:`_cuda`)."""
    with ExitStack() as stack:
        for patch in _cuda(False):
            stack.enter_context(patch)
        yield gloo_world_of_one


@pytest.fixture
def ep_world_of_two(monkeypatch, gloo_world_of_one):
    """A two-rank expert-parallel world as far as the loader's checks see, and the checkpoint paths
    it resolves; nothing else is set up."""
    resolved = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(
        model_load_utils, "ensure_local_checkpoint", lambda p, **k: resolved.append(p) or (None, p)
    )
    return resolved


@pytest.mark.parametrize(
    ("plan", "error", "match"),
    [
        (DistributionPlan(2), ValueError, "shards no routed experts"),
        (DistributionPlan(2, ep_size=2, dense="fsdp"), NotImplementedError, "replicates the dense"),
    ],
    ids=["no-expert-parallelism", "sharded-dense"],
)
def test_the_loader_refuses_a_plan_it_cannot_load_by(plan, error, match):
    """Before anything else: no process group, transformers release or checkpoint is looked at."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("unused", plan)


def test_the_loader_refuses_a_plan_for_another_world(gloo_world_of_one):
    with pytest.raises(ValueError, match="is for 2 ranks, but this run has 1"):
        load_and_prepare_ep("unused", DistributionPlan(2, ep_size=2))


def test_the_loader_refuses_a_transformers_that_builds_no_model_with_torch_a2a(
    monkeypatch, gloo_world_of_one
):
    """transformers 5.6 checks a model's experts implementation against the ones it ships, so its
    from_pretrained would refuse torch_a2a once the load had set everything up: refused first."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr("transformers.__version__", "5.6.2")

    with pytest.raises(RuntimeError, match=r"requires transformers >= 5\.7"):
        load_and_prepare_ep("unused", DistributionPlan(2, ep_size=2), hf_config=SimpleNamespace())


def test_a_hub_id_is_made_local_before_the_weights_load(monkeypatch, gloo_world_of_one):
    """The export reads the tensors the load could not place back from the checkpoint it records,
    which must be a local directory: a Hub ID there leaves the export nothing to read. The load
    resolves it once the plan and the config pass, before anything reads a weight."""

    class _ResolvedError(Exception):
        """Stops the load where it resolves the checkpoint."""

    resolved = []

    def resolve(path, **hub_options):
        resolved.append(path)
        raise _ResolvedError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "ensure_local_checkpoint", resolve)

    with pytest.raises(_ResolvedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(2, ep_size=2),
            hf_config=SimpleNamespace(model_type=_NO_SPEC, base_model_ep_plan=_EP_PLAN),
        )

    assert resolved == ["org/model"]


def test_the_experts_are_counted_in_the_config_holding_the_plan(monkeypatch, gloo_world_of_one):
    """As the distribution plan's decision counts them: a VLM whose nested config holds the MoE and
    its plan has its experts counted there, not by an expert-ish field on its root config."""

    class _CheckedError(Exception):
        """Stops the load once its checks pass."""

    def resolve(path, **hub_options):
        raise _CheckedError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "ensure_local_checkpoint", resolve)
    text = SimpleNamespace(base_model_ep_plan=_EP_PLAN, num_experts=8)

    with pytest.raises(_CheckedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(4, ep_size=4),
            hf_config=SimpleNamespace(model_type=_NO_SPEC, text_config=text, num_experts=3),
        )


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        *(
            (
                SimpleNamespace(quantization_config={"quant_method": method}),
                NotImplementedError,
                f"cannot dequantize a '{method}'",
            )
            for method in ("gptq", "modelopt", "compressed-tensors", "mxfp4")
        ),
        (SimpleNamespace(), ValueError, "declare base_model_ep_plan"),
        (
            SimpleNamespace(
                model_type=_NO_SPEC, base_model_ep_plan={"layers.*.self_attn.q_proj": "colwise"}
            ),
            ValueError,
            "styles no routed experts",
        ),
        *(
            (
                SimpleNamespace(
                    model_type=_NO_SPEC,
                    base_model_ep_plan={"layers.*.mlp.experts": "moe_tp_experts", **params},
                ),
                ValueError,
                r"shards no weights of layers\.\*\.mlp\.experts through grouped_gemm",
            )
            for params in ({}, {"layers.*.mlp.experts.gate_up_proj": "packed_colwise"})
        ),
        (
            SimpleNamespace(
                model_type=_NO_SPEC,
                base_model_ep_plan={**_EP_PLAN, "layers.*.mlp.experts.down_proj": "packed_rowwise"},
            ),
            ValueError,
            r"styles layers\.\*\.mlp\.experts\.down_proj by a style this load drops",
        ),
        (
            SimpleNamespace(model_type="gpt_oss", base_model_ep_plan=_EP_PLAN),
            NotImplementedError,
            "the expert-parallel export cannot write that layout",
        ),
    ],
    ids=[
        "gptq",
        "modelopt",
        "compressed-tensors",
        "mxfp4",
        "no-ep-plan",
        "no-routed-experts",
        "experts-container-only",
        "experts-sharded-otherwise",
        "experts-sharded-partly-otherwise",
        "spec-refuses-ep",
    ],
)
def test_a_config_the_load_cannot_take_is_refused_before_anything_loads(
    ep_world_of_two, config, error, match
):
    """A format transformers' own quantizer would load as stored, for PTQ to treat as plain weights,
    or, MXFP4, dequantize into GPT-OSS's transposed expert layout whatever the model's, or a plan
    that leaves the experts whole on every rank: one naming no routed experts, or none of
    whose entries shards their weights by grouped_gemm, the only expert-weight style the load
    shards by, or one that styles some of their weights otherwise. So is a model whose spec rules
    expert parallelism out (GPT-OSS's), which the plan decision plans without it: an EP degree asked
    for, or a plan handed in, meets that refusal here. The load refuses each before it reads
    anything."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert ep_world_of_two == []


def test_an_ep_size_that_does_not_divide_the_experts_is_refused_before_anything_loads(
    ep_world_of_two,
):
    """The all-to-all transport finds an expert's rank by dividing its id by this rank's share of
    the experts: with 3 experts over 2 ranks the ranks would disagree on who holds which, and send
    tokens to the wrong experts."""
    config = SimpleNamespace(model_type=_NO_SPEC, base_model_ep_plan=_EP_PLAN, num_experts=3)

    with pytest.raises(ValueError, match="ep_size=2 must divide the 3 routed experts"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert ep_world_of_two == []


def test_a_plan_whose_expert_containers_match_no_module_is_refused(ep_world_of_two, monkeypatch):
    """A plan that styles the grouped-GEMM parameters a level below the experts names a container no
    module is. Loaded anyway, every module would get the dense state, and the experts' amax would
    reduce over ranks holding different experts."""

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return model_with_decoder_at("model.")

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: [])
    config = SimpleNamespace(
        model_type=_NO_SPEC,
        base_model_ep_plan={"layers.*.mlp.experts.*.gate_up_proj": "grouped_gemm"},
    )

    with pytest.raises(ValueError, match="that no module of"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)


@pytest.mark.parametrize(
    ("source", "on_the_config"),
    [
        ({"base_model_ep_plan": _EP_PLAN}, True),
        ({}, False),
        ({}, True),
        ('{"base_model_ep_plan": ', True),
    ],
    ids=["by-its-checkpoint", "a-class-default", "attached-by-its-spec", "in-an-unreadable-file"],
)
def test_the_model_keeps_the_plan_its_checkpoint_declares(
    ep_world_of_two, gloo_world_of_one, monkeypatch, tmp_path, source, on_the_config
):
    """The load shards by the torch_a2a styles, but the config the model keeps goes back to the
    plan its checkpoint's config.json declares: an export writes it out, and nothing without
    ModelOpt knows those styles. A plan the config class ships (transformers' own, e.g.
    Qwen3-MoE's) stays off the instance, and so does one the model's spec attached to the config,
    which names styles only ModelOpt registers (Nemotron-H's nemotron_experts_ep). A config.json
    the load cannot read declares none, with a warning, since an export then loses the plan."""
    declared = isinstance(source, dict) and "base_model_ep_plan" in source

    class _PastError(Exception):
        """Raised by the load's next check."""

    model = model_with_decoder_at("model.")

    class _Loader:
        @staticmethod
        def from_pretrained(path, config, **kwargs):
            model.config = copy.deepcopy(config)  # transformers keeps a copy of the config
            return model

    def next_check(*args, **kwargs):
        raise _PastError

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_seed_parallel_states", next_check)

    class _Config(transformers.PretrainedConfig):
        model_type = "_probe"
        base_model_ep_plan = _EP_PLAN

    unreadable = isinstance(source, str)
    (tmp_path / "config.json").write_text(source if unreadable else json.dumps(source))
    # 8 experts over 2 ranks: the stand-in model holds this rank's 4.
    plan = {"base_model_ep_plan": dict(_EP_PLAN)} if on_the_config else {}
    config = _Config(num_experts=8, **plan)

    warns = pytest.warns(UserWarning, match="config.json") if unreadable else nullcontext()
    with warns, pytest.raises(_PastError):
        load_and_prepare_ep(str(tmp_path), DistributionPlan(2, ep_size=2), hf_config=config)

    assert model.config.base_model_ep_plan == _EP_PLAN
    assert ("base_model_ep_plan" in vars(model.config)) is declared
    model.config.save_pretrained(tmp_path / "saved")
    written = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert ("base_model_ep_plan" in written) is declared


def test_a_loaded_model_keeps_its_sharding_styles_out_of_its_saved_config(
    ep_world_of_two, gloo_world_of_one, monkeypatch
):
    """What the loader leaves on a transformers model, as transformers treats it. Built with the
    torch_a2a styles, the model keeps them as ``model.tp_plan``, which calibration re-installs the
    all-to-all forwards by, and the expert-parallel mesh it re-installs them over, which this stand-in
    from_pretrained does not set; while its config goes back to the checkpoint's plan; and the experts
    implementation its forwards dispatch on, torch_a2a, stays out of the config it serializes,
    which an export writes."""
    if not Qwen3MoeConfig.base_model_ep_plan:
        pytest.skip("this transformers ships no expert-parallel plan for Qwen3-MoE")

    class _PastError(Exception):
        """Raised by the load's next check."""

    built = []

    class _Loader:
        @staticmethod
        def from_pretrained(path, config, **kwargs):
            # As transformers' expert-parallel load builds it: this rank's 2 of the 4 experts.
            config.distributed_config = DistributedConfig(enable_expert_parallel=True)
            share = copy.deepcopy(config)
            share.num_experts //= 2
            with torch.device("meta"):
                model = Qwen3MoeForCausalLM(share)
            model.config = config
            built.append(model)
            return model

    def next_check(*args, **kwargs):
        raise _PastError

    mesh = _Mesh()
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: mesh
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_seed_parallel_states", next_check)
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_experts=4,
        num_experts_per_tok=1,
    )

    with pytest.raises(_PastError):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)

    (model,) = built
    assert model.config.base_model_ep_plan == Qwen3MoeConfig.base_model_ep_plan
    assert "torch_a2a_experts" in set(model.tp_plan.values())
    assert model._device_mesh is mesh
    assert model.config._experts_implementation == "torch_a2a"
    assert not any("experts_implementation" in key for key in model.config.to_dict())


def test_the_weights_load_in_the_checkpoints_dtype_through_its_class(ep_world_of_two, monkeypatch):
    """The checkpoint's ``dtype`` reaches ``from_pretrained``, and the class that builds it is chosen
    with the trust the caller gave its code. The Hub options reach the checkpoint's download, the
    config read and ``from_pretrained``, which fetch the custom code an ``auto_map`` names in another
    repo."""

    class _LoadedError(Exception):
        """Stops the load at from_pretrained."""

    loaded, classes = {}, []

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            raise _LoadedError

    def auto_class(hf_config, require_lm_head=True, trust_remote_code=False):
        classes.append(trust_remote_code)
        return _Loader

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", auto_class)
    config = SimpleNamespace(model_type=_NO_SPEC, dtype=torch.float16, base_model_ep_plan=_EP_PLAN)
    reads, downloads = [], []
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", lambda *a, **k: reads.append(k) or config
    )
    monkeypatch.setattr(
        model_load_utils,
        "ensure_local_checkpoint",
        lambda p, **k: downloads.append(k) or (None, p),
    )

    with pytest.raises(_LoadedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(2, ep_size=2),
            trust_remote_code=True,
            hub_options={"local_files_only": True},
        )
    assert loaded["dtype"] is torch.float16
    assert [read["local_files_only"] for read in reads] == [True]
    assert downloads == [{"local_files_only": True}]
    assert loaded["local_files_only"] is True
    assert classes == [True]


# ----------------------------------------------------------------------- _seed_parallel_states


def test_expert_containers_and_their_children_get_the_expert_state():
    """Everything else, the root and the containers' own parents included, gets the dense state; so
    does a sibling whose name merely starts with a container's (Qwen2-MoE keeps
    ``shared_expert_gate`` beside ``shared_expert``)."""
    model = model_with_decoder_at("model.")
    for layer in model.model.layers:
        layer.mlp.experts.weight_quantizer = nn.Identity()  # a child, as quantizer insertion adds
    model.model.layers[0].mlp.experts_gate = nn.Linear(2, 1)
    experts = {f"model.layers.{i}.mlp.experts": model.model.layers[i].mlp.experts for i in (0, 1)}
    dense, expert = object(), object()

    _seed_parallel_states(model, experts, dense_state=dense, expert_state=expert)

    states = {name: module._parallel_state for name, module in model.named_modules()}
    assert {name for name, state in states.items() if state is expert} == {
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.weight_quantizer",
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.experts.weight_quantizer",
    }
    assert all(state is dense for state in states.values() if state is not expert)


# --------------------------------------------------------------------------- _ckpt_quant_method


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(quantization_config={"quant_method": "fp8"}), "fp8"),
        (SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4")), "mxfp4"),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
            ),
            "fp8",
        ),
        (
            SimpleNamespace(
                llm_config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
            ),
            "fp8",
        ),
        (
            SimpleNamespace(
                quantization_config={"quant_method": "mxfp4"},
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
            ),
            "mxfp4",
        ),
        (SimpleNamespace(quantization_config=FineGrainedFP8Config()), "fp8"),
        (SimpleNamespace(quantization_config={"weight_block_size": [128, 128]}), None),
        (SimpleNamespace(text_config=None), None),
    ],
    ids=[
        "dict",
        "object",
        "vlm-text-config",
        "vlm-llm-config",
        "top-level-wins",
        "transformers-config-object",
        "no-method",
        "unquantized",
    ],
)
def test_quant_method_comes_from_the_config_or_a_vlms_text_config(config, expected):
    """The top-level config first, else a VLM's text or LLM config. A loaded model's config holds
    transformers' config object, not the JSON dict, and callers compare against ``"fp8"`` /
    ``"mxfp4"``: it reports the plain method name, not its enum."""
    assert _ckpt_quant_method(config) == expected


def test_from_pretrained_builds_its_mesh_from_the_ep_mesh_it_is_given():
    """Newer transformers ignores ``from_pretrained``'s ``device_mesh`` and builds a world-sized one,
    which shards the experts over every rank when EP is below the world size. During the load it gets
    the EP mesh instead, and its own builder is back afterwards, even when the load raises."""
    mixin = _transformers_mesh_builder()
    if mixin is None:
        pytest.skip("this transformers honors from_pretrained's device_mesh")
    builder = mixin.initialize_distributed_mesh
    ep_mesh = SimpleNamespace(device_type="cpu")
    config = DistributedConfig(enable_expert_parallel=True, tp_plan="auto", tp_size=2)
    with _experts_sharded_over(ep_mesh):
        _, device_map, device_mesh = PreTrainedModel.prepare_distribute_model(config)

    assert device_mesh is ep_mesh
    assert device_map == torch.device("cpu")
    assert mixin.initialize_distributed_mesh is builder
    with pytest.raises(RuntimeError), _experts_sharded_over(ep_mesh):
        raise RuntimeError("the load failed")
    assert mixin.initialize_distributed_mesh is builder


@pytest.mark.parametrize(
    ("fields", "builds_its_mesh", "config", "tp_plan"),
    [
        # transformers 5.6-5.9, 5.11, 5.12
        (["enable_expert_parallel"], False, {"enable_expert_parallel": True}, "auto"),
        # 5.10, 5.13-5.17
        (
            ["tp_size", "tp_plan", "enable_expert_parallel"],
            False,
            {"enable_expert_parallel": True, "tp_plan": "auto"},
            None,
        ),
        # 5.18 on
        (
            ["tp_size", "tp_plan", "enable_expert_parallel"],
            True,
            {"enable_expert_parallel": True, "tp_plan": "auto", "tp_size": 2},
            None,
        ),
        # no release so far
        (
            ["tp_size", "enable_expert_parallel"],
            True,
            {"enable_expert_parallel": True, "tp_size": 2},
            "auto",
        ),
    ],
    ids=["no-tp-plan-field", "tp-plan-field", "own-mesh-builder", "own-mesh-builder-no-tp-plan"],
)
def test_each_load_argument_goes_where_this_transformers_takes_it(
    ep_world_of_two, monkeypatch, fields, builds_its_mesh, config, tp_plan
):
    """``tp_plan="auto"`` goes on ``DistributedConfig`` where it has the field, else to
    ``from_pretrained``, never both; a release that builds its own mesh gets it sized to the EP
    degree. Each is decided by its own field, so a release with one and not the other loads too."""

    class _LoadedError(Exception):
        """Stops the load at from_pretrained."""

    loaded = {}

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            raise _LoadedError

    distributed_config = dataclasses.make_dataclass(
        "DistributedConfig", [(name, object, dataclasses.field(default=None)) for name in fields]
    )
    monkeypatch.setattr(hf_distributed, "DistributedConfig", distributed_config)
    monkeypatch.setattr(
        model_load_utils,
        "_transformers_mesh_builder",
        lambda: SimpleNamespace(initialize_distributed_mesh=None) if builds_its_mesh else None,
    )
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)

    with pytest.raises(_LoadedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(2, ep_size=2),
            hf_config=SimpleNamespace(model_type=_NO_SPEC, base_model_ep_plan=_EP_PLAN),
        )

    given = {k: v for k, v in vars(loaded["distributed_config"]).items() if v is not None}
    assert given == config
    assert loaded.get("tp_plan") == tp_plan


# ----------------------------------------------------------------- one rank failing alone


class _PastTheMeshesError(Exception):
    """Raised by the meshes' stand-in: this rank got past the load's checks."""


def _meshes_in_step(*args, **kwargs):
    """The meshes' subgroups, as the collective they are: a rank in it waits for every peer."""
    torch.distributed.barrier()
    raise _PastTheMeshesError


class _InTheLoadError(Exception):
    """Raised by the weights' load stand-in: this rank got into the load."""


def _load_in_step(*args, **kwargs):
    """The weights' load, as the collective it is: a rank in it waits for every peer."""
    torch.distributed.barrier()
    raise _InTheLoadError


def _one_rank_fails_job(rank, size, ckpt, failure, failing_rank, match):
    path, hf_config, patches = ckpt, None, []
    if failure == "unreadable-plan-provenance":
        # A VLM's config, handed in: the plan to keep is read from where it sits in config.json.
        plan = {"base_model_ep_plan": dict(_EP_PLAN), "num_experts": 4}
        hf_config = SimpleNamespace(model_type=_NO_SPEC, text_config=SimpleNamespace(**plan))
    elif failure != "unreadable-config":
        hf_config = SimpleNamespace(
            model_type=_NO_SPEC, base_model_ep_plan=dict(_EP_PLAN), num_experts=4
        )
    if failure != "unseen-checkpoint":
        local = mock.patch.object(
            model_load_utils, "ensure_local_checkpoint", side_effect=lambda p, **k: (None, p)
        )
        patches.append(local)
    fails_here = rank == failing_rank
    # No rank has a GPU, but in the no-gpu row every rank other than the failing one has its own.
    patches += _cuda(failure == "no-gpu" and not fails_here, uuid=f"GPU-{rank}")

    def fail(*args, **kwargs):
        raise (MemoryError if failure == "load" else OSError)(match)

    if failure == "unreadable-config" and fails_here:
        path = ckpt + "-not-on-this-node"
    elif failure == "transformers" and fails_here:
        patches.append(mock.patch("transformers.__version__", "5.6.2"))
    elif failure == "quantized-config" and fails_here:
        hf_config.quantization_config = {"quant_method": "gptq"}
    elif failure == "unseen-checkpoint" and fails_here:
        # Rank 0 makes the checkpoint local; a rank on a node without that storage cannot see it.
        patches.append(mock.patch.object(hf_checkpoint_utils.os.path, "isdir", return_value=False))
    elif failure == "unreadable-plan-provenance" and fails_here:
        # This rank reads a config.json that is no JSON object: it declares no plan, not an error.
        reads = SimpleNamespace(load=lambda f: [])
        patches.append(mock.patch.object(model_load_utils, "json", reads))
    if failure in ("load", "record", "unreadable-plan-provenance"):
        loaded = model_with_decoder_at("model.")
        load = fail if failure == "load" and fails_here else (lambda path, **kwargs: loaded)
        if failure == "unreadable-plan-provenance":
            load = _load_in_step
        loader = SimpleNamespace(__name__="Loader", from_pretrained=load)
        record = fail if failure == "record" and fails_here else None
        patches += [
            mock.patch("torch.distributed.device_mesh.init_device_mesh", return_value=_Mesh()),
            mock.patch.object(model_load_utils, "_auto_class", return_value=loader),
            mock.patch.object(model_load_utils, "record_unplaced_source_keys", side_effect=record),
        ]
    else:
        patches.append(
            mock.patch("torch.distributed.device_mesh.init_device_mesh", _meshes_in_step)
        )
    refused = None
    with bounded_collectives(), ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        caught = stack.enter_context(warnings.catch_warnings(record=True))
        warnings.simplefilter("always")
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            load_and_prepare_ep(path, DistributionPlan(2, ep_size=2), hf_config=hf_config)
        except Exception as error:
            refused = f"{type(error).__name__}: {error}"
    # The one rank's failure on every rank, before any rank went on alone into a collective. The GPU
    # check decides on every rank at once, from one gather, so no rank relays another's failure.
    assert refused is not None and match in refused, (rank, refused)
    if match != _InTheLoadError.__name__:  # else no rank failed, and every rank got into the load
        assert fails_here or failure == "no-gpu" or f"rank {failing_rank} failed" in refused, (
            refused
        )
    else:  # the plan the one rank could not read is lost to its model alone, which says so
        warned = [
            str(w.message) for w in caught if "keeps no expert-parallel plan" in str(w.message)
        ]
        assert bool(warned) == fails_here, (rank, warned)
        assert not warned or f"rank {rank}'s model" in warned[0], warned


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"), reason="the EP load needs transformers 5.7"
)
@pytest.mark.parametrize(
    ("failure", "failing_rank", "match"),
    [
        ("unreadable-config", 1, "not-on-this-node"),
        ("transformers", 1, "requires transformers >= 5.7, but 5.6.2"),
        ("quantized-config", 1, "cannot dequantize a 'gptq'"),
        ("unseen-checkpoint", 1, "which this rank cannot see"),
        ("no-gpu", 1, "rank(s) [1] have none while the others have theirs"),
        ("load", 1, "the GPU ran out of memory"),
        ("record", 1, "a weight file this rank cannot read"),
        ("unreadable-plan-provenance", 1, _InTheLoadError.__name__),
    ],
)
def test_a_load_failing_on_one_rank_fails_every_rank(
    cpu_dist_workers, tmp_path, failure, failing_rank, match
):
    """One rank fails alone: reading the config, on its transformers, on its arguments, seeing the
    checkpoint rank 0 made local, without a GPU of its own where the others have theirs, or in the
    load. Every rank raises that failure, rather than leave the others waiting for it in the meshes
    (stood in by a bounded barrier) or the warmup's collectives. A config.json one rank cannot read
    as a JSON object declares no plan to keep there, so every rank goes on into the load (stood in
    by a bounded barrier too)."""
    Qwen3MoeConfig(num_experts=4, num_hidden_layers=1, hidden_size=8).save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _one_rank_fails_job,
        ckpt=str(tmp_path),
        failure=failure,
        failing_rank=failing_rank,
        match=match,
    )


def _gpu_layout_job(rank, size, visible, bound, shared, refused):
    hf_config = SimpleNamespace(
        model_type=_NO_SPEC, base_model_ep_plan=dict(_EP_PLAN), num_experts=4
    )
    reached = None
    with bounded_collectives(), ExitStack() as stack:
        # This rank is on cuda:0, bound to it or not yet; the local ranks run 0 and 1.
        for patch in _cuda(True, bound, visible, "GPU-0" if shared else f"GPU-{rank}"):
            stack.enter_context(patch)
        stack.enter_context(
            mock.patch.object(
                model_load_utils, "ensure_local_checkpoint", side_effect=lambda p, **k: (None, p)
            )
        )
        stack.enter_context(
            mock.patch("torch.distributed.device_mesh.init_device_mesh", _meshes_in_step)
        )
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=hf_config)
        except Exception as error:
            reached = error
    if refused is None:
        assert isinstance(reached, _PastTheMeshesError), (rank, reached)
    else:
        assert isinstance(reached, RuntimeError) and refused in str(reached), (rank, reached)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"), reason="the EP load needs transformers 5.7"
)
@pytest.mark.parametrize(
    ("visible", "bound", "shared", "refused"),
    [
        (1, True, False, None),
        (1, False, False, "rank(s) [1] are bound to none yet"),
        (2, False, False, None),
        (2, True, True, "rank(s) [0, 1] are bound to the same one"),
    ],
    ids=[
        "one-gpu-each-bound",
        "one-gpu-each-not-yet",
        "every-gpu-not-yet",
        "every-gpu-all-on-cuda-0",
    ],
)
def test_each_rank_builds_the_meshes_on_a_gpu_of_its_own(
    cpu_dist_workers, visible, bound, shared, refused
):
    """Each rank builds the meshes on the GPU it is bound to, and DeviceMesh binds one not bound yet
    by its local rank. Shown one GPU each, every rank bound to cuda:0 goes on; not yet bound, rank 1's
    local rank names no GPU it sees. Shown every GPU and not yet bound, DeviceMesh binds each to its
    own. Every rank left on cuda:0, as an import that initializes CUDA leaves a launch that binds
    none, shares one GPU. A refusal comes on every rank, from one gather."""
    cpu_dist_workers.run(
        _gpu_layout_job, visible=visible, bound=bound, shared=shared, refused=refused
    )
