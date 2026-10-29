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

Nothing here needs a GPU. The tests of ``resolve_distribution_plan`` answer the process group's rank
and size as a torchrun launch would, and serve tiny real transformers configs from memory, so the
decision builds their real models on the meta device; only the last ones use a process group, two
gloo ranks.
"""

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
    Phi4MultimodalConfig,
    Qwen2_5_VLConfig,
    Qwen3MoeConfig,
    T5Config,
    WhisperConfig,
)

import modelopt.torch.utils.distributed as dist
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.distributed import DistributionPlan
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    _auto_class,
    _conversion_plan,
    _convert_keys,
    _resolve_target,
    build_meta_causal_lm,
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

    Every rank reads the checkpoint as this one does. How the ranks agree on one plan is tested at
    the end of this file.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _gathered_alike)

    def enter(world_size, rank=0):
        monkeypatch.setattr(dist, "is_distributed_run", lambda: world_size > 1)
        monkeypatch.setattr(dist, "size", lambda group=None: world_size)
        monkeypatch.setattr(dist, "is_master", lambda group=None: rank == 0)
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
    ("config", "trust_remote_code", "expected"),
    [
        (SimpleNamespace(architectures=["LlamaForCausalLM"]), False, AutoModelForCausalLM),
        (_vlm_config(auto_map=_REMOTE_VLM), True, AutoModelForCausalLM),
        (_vlm_config(auto_map=_REMOTE_VLM), False, AutoModel),
        (_vlm_config(auto_map=_REMOTE_IMAGE_TEXT_VLM), True, AutoModelForImageTextToText),
        (_tiny_qwen2_5_vl(), False, AutoModelForImageTextToText),
        (Phi4MultimodalConfig(), False, AutoModelForCausalLM),
        (_vlm_config(), False, AutoModel),
        (T5Config(architectures=["T5ForConditionalGeneration"]), False, AutoModelForSeq2SeqLM),
        (WhisperConfig(), False, AutoModelForSpeechSeq2Seq),
    ],
    ids=[
        "causal-lm",
        "remote-vlm-with-head",
        "remote-vlm-untrusted",
        "remote-image-text-vlm",
        "native-vlm-with-head",
        "native-multimodal-causal-lm",
        "unknown-vlm",
        "encoder-decoder",
        "speech-encoder-decoder",
    ],
)
def test_the_class_that_builds_the_whole_model(config, trust_remote_code, expected):
    """A VLM's AutoModel leaves the LM head out: it builds through the class that carries the head,
    keeping the headless backbone only when no class recognizes the config. A remote class counts
    only when the checkpoint's code is trusted. An encoder-decoder model's architecture reads like a
    VLM's, but its seq2seq class carries its head, also for a config that names no architecture."""
    assert _auto_class(config, trust_remote_code) is expected


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


@pytest.mark.parametrize(
    ("world_size", "source"),
    [(4, "MODELOPT_EP_DEGREE"), (1, "ep_size")],
    ids=["knob", "argument-in-a-single-process"],
)
def test_an_ep_degree_that_is_not_a_number_names_where_it_came_from(
    run_as, serve, monkeypatch, tmp_path, world_size, source
):
    """A single process takes its own argument as given, but parses it as the decision does."""
    run_as(world_size)
    serve(LlamaConfig(**_TINY))
    arguments = {}
    if source == "ep_size":
        arguments["ep_size"] = "world"
    else:
        monkeypatch.setenv(source, "world")

    with pytest.raises(ValueError, match=f"{source}='world'"):
        resolve_distribution_plan(str(tmp_path), **arguments)


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize("value", ["0", "", "AUTO"], ids=["zero", "exported-empty", "capitals"])
def test_an_ep_degree_of_0_is_auto(run_as, serve, monkeypatch, tmp_path, value):
    """And so is one exported empty, and auto in any case: the plan decides."""
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
        ("2", None),
    ],
    ids=["1", "4", "WORLD", "exported-empty", "part-of-the-world"],
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


def _one_rank_cannot_see_the_checkpoint_job(rank, size, ckpt):
    isdir = os.path.isdir
    with (
        bounded_collectives(),
        mock.patch("os.path.isdir", lambda path: (rank == 0 or path != ckpt) and isdir(path)),
    ):
        try:
            resolve_distribution_plan(ckpt)
        except RuntimeError as error:
            raised = str(error)
        else:
            raised = "no error"
    # Rank 1's own error, carried to rank 0, rather than rank 0 timing out in the plan's broadcast.
    assert "which this rank cannot see" in raised, raised


def test_a_rank_that_cannot_see_the_checkpoint_fails_every_rank(cpu_dist_workers, tmp_path):
    """Before the plan's broadcast, which rank 0 would otherwise wait in for a rank that raised."""
    LlamaConfig(**_TINY).save_pretrained(tmp_path)
    cpu_dist_workers.run(_one_rank_cannot_see_the_checkpoint_job, ckpt=str(tmp_path))
