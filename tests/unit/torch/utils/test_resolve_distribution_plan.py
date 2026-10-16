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

"""How ``resolve_distribution_plan`` decides a model's distribution, and the MoE helpers it uses.

Nothing here needs a GPU, and only the tests at the end a process group, two gloo ranks: elsewhere
the process group's rank and size are answered the way a torchrun launch would, and the configs are
tiny real transformers configs served from memory, so the decision builds their real models on the
meta device.
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from torch import nn

transformers = pytest.importorskip("transformers")

from packaging.version import Version
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    LlamaConfig,
    Qwen2_5_VLConfig,
    Qwen3MoeConfig,
)

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.distributed import DistributionPlan
from modelopt.torch.utils.moe import (
    _routed_expert_modules,
    choose_ep_dp_degrees,
    num_experts_from_config,
)
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import _auto_class, resolve_distribution_plan

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


# ------------------------------------------------------------------------------- expert counting


@pytest.mark.parametrize(
    ("config", "experts"),
    [
        (SimpleNamespace(hidden_size=128), 0),
        (SimpleNamespace(num_experts=16), 16),
        (SimpleNamespace(num_local_experts=16), 16),
        (SimpleNamespace(n_routed_experts=16), 16),
        (SimpleNamespace(num_routed_experts=16), 16),
        (SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=32)), 32),
        (SimpleNamespace(llm_config=SimpleNamespace(n_routed_experts=32)), 32),
    ],
    ids=[
        "dense",
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_routed_experts",
        "vlm-text_config",
        "vlm-llm_config",
    ],
)
def test_the_expert_count_is_read_under_every_spelling(config, experts):
    """Architectures disagree on the field name, and a VLM's MoE is its nested LM, whose root
    config has no expert count of its own: missing either would silently disable EP."""
    assert num_experts_from_config(config) == experts


# ------------------------------------------------------------------------------- degree selection


@pytest.mark.parametrize(
    ("world", "experts", "degrees"),
    [
        (8, 64, (8, 1)),
        (8, 6, (2, 4)),
        (8, 5, (1, 8)),
        (8, 0, (1, 8)),
        (1, 64, (1, 1)),
    ],
    ids=["largest-valid", "divides-the-experts", "nothing-divides", "dense", "single-rank"],
)
def test_the_largest_ep_degree_dividing_the_world_and_the_experts_is_chosen(
    world, experts, degrees
):
    """EP must divide the expert count, so every rank holds whole experts; with nothing dividing, a
    dense model, or a single rank, there is no expert parallelism."""
    assert choose_ep_dp_degrees(world, experts) == degrees


# ---------------------------------------------------------------------------- routed experts


class _MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(2, 4, bias=False)  # the router
        self.mlp.experts = nn.Module()
        self.mlp.experts.gate_up_proj = nn.Parameter(torch.zeros(4, 4, 2))
        self.mlp.experts.down_proj = nn.Parameter(torch.zeros(4, 2, 2))
        self.mlp.shared_experts = nn.Linear(2, 4, bias=False)


def _model_with_decoder_at(prefix):
    """A two-layer MoE whose decoder (``layers.<i>``) sits under ``prefix`` (``""``: the decoder)."""
    decoder = nn.Module()
    decoder.embed_tokens = nn.Embedding(8, 2)
    decoder.layers = nn.ModuleList(_MoeLayer() for _ in range(2))
    if not prefix:
        return decoder
    root = parent = nn.Module()
    *path, leaf = prefix.rstrip(".").split(".")
    for name in path:
        parent.add_module(name, nn.Module())
        parent = getattr(parent, name)
    parent.add_module(leaf, decoder)
    return root


_STOCK_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


@pytest.mark.parametrize(
    "plan",
    [
        _STOCK_PLAN,
        {"layers.*.mlp.experts": "moe_tp_experts"},
        {"layers.*.mlp.experts": "nemotron_experts_ep"},
        {k: v for k, v in _STOCK_PLAN.items() if v == "grouped_gemm"},
    ],
    ids=["stock", "experts-style-only", "nemotron-style-only", "grouped-gemm-only"],
)
@pytest.mark.parametrize(
    "prefix",
    ["", "model.", "model.language_model.", "language_model.model."],
    ids=["base-model", "causal-lm", "vlm", "legacy-vlm"],
)
def test_routed_experts_are_the_plans_containers_under_any_prefix(plan, prefix):
    """The base-relative plan keys match the decoder wherever the model nests it, at every layer
    index, the container named by its style or as its grouped-GEMM parameters' parent. The router,
    the shared experts, and a module whose name merely ends in a key's text are not routed
    experts."""
    model = _model_with_decoder_at(prefix)
    model.add_module("sublayers", nn.ModuleList([_MoeLayer()]))  # "sublayers.0.mlp.experts"

    found = _routed_expert_modules(model, plan)

    modules = dict(model.named_modules())
    assert found == {
        name: modules[name] for name in [f"{prefix}layers.{i}.mlp.experts" for i in (0, 1)]
    }


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


@pytest.mark.parametrize(
    ("config", "require_lm_head", "expected"),
    [
        (SimpleNamespace(architectures=["LlamaForCausalLM"]), True, AutoModelForCausalLM),
        (_vlm_config(), False, AutoModel),
        (
            _vlm_config(auto_map={"AutoModelForCausalLM": "m.KimiK25ForConditionalGeneration"}),
            True,
            AutoModelForCausalLM,
        ),
        (_tiny_qwen2_5_vl(), True, AutoModelForImageTextToText),
        (_vlm_config(), True, AutoModel),
    ],
    ids=[
        "causal-lm",
        "vlm-backbone",
        "remote-vlm-with-head",
        "native-vlm-with-head",
        "unknown-vlm",
    ],
)
def test_the_class_that_builds_the_whole_model(config, require_lm_head, expected):
    """A VLM's AutoModel leaves the LM head out: with the head required it builds through the class
    that carries it, keeping the headless backbone only when no class recognizes the config."""
    assert _auto_class(config, require_lm_head) is expected


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


def test_the_hub_options_reach_the_plans_checkpoint_read(monkeypatch, tmp_path):
    """The plan's read resolves a Hub ID and loads its config with the options it was given: the
    config reaches the Hub for the custom code its ``auto_map`` names in another repo."""
    options = {"token": "secret", "cache_dir": str(tmp_path), "local_files_only": True}
    calls = []
    monkeypatch.setattr(
        model_load_utils,
        "ensure_local_checkpoint",
        lambda path, **kwargs: calls.append(("resolve", path, kwargs)) or (path, str(tmp_path)),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("config", path, kwargs)) or _qwen3_moe(),
    )

    resolve_distribution_plan("org/model", **options)

    assert calls == [
        ("resolve", "org/model", options),
        ("config", str(tmp_path), {"trust_remote_code": False, **options}),
    ]


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


@needs_qwen3_moe_ep_plan
@pytest.mark.parametrize("value", ["0", ""], ids=["zero", "exported-empty"])
def test_an_ep_degree_of_0_is_auto(run_as, serve, monkeypatch, tmp_path, value):
    """And so is one exported empty: unset, the plan decides too."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", value)
    run_as(4)
    serve(_qwen3_moe(num_experts=8))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, 4)


@pytest.mark.parametrize(
    ("gpus_worth", "dense"), [(2, "replicate"), (1, "fsdp")], ids=["fits", "too-big"]
)
def test_a_dense_model_is_fsdp_sharded_only_when_too_big_for_one_gpu(
    run_as, serve, gpu_memory, tmp_path, gpus_worth, dense
):
    """On a GPU only its own size, the whole model would leave less than 40% of it to calibration."""
    run_as(4)
    config = serve(LlamaConfig(**_TINY))
    gpu_memory(gpus_worth * _bytes(_meta(config)))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, dense=dense)


@needs_qwen3_moe_ep_plan
def test_only_what_expert_parallelism_leaves_whole_must_fit(run_as, serve, gpu_memory, tmp_path):
    """The routed experts are sharded at EP 4, so only the rest needs to fit one GPU; with no EP the
    whole MoE does, and here it does not."""
    run_as(4)
    config = serve(_qwen3_moe(num_experts=8))
    model = _meta(config)
    experts = sum(_bytes(layer.mlp.experts) for layer in model.model.layers)
    gpu_memory(int((_bytes(model) - experts / 2) / 0.6))

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(4, 4)
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
        ("full", "fsdp"),
        ("WORLD", "fsdp"),
        ("", "fsdp"),
        ("2", None),
    ],
    ids=["1", "4", "full", "WORLD", "exported-empty", "part-of-the-world"],
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


def test_a_vlm_is_decided_from_its_whole_model(run_as, serve, gpu_memory, tmp_path):
    """AutoModelForCausalLM does not build every VLM: the decision builds what the load builds."""
    run_as(2)
    serve(_tiny_qwen2_5_vl())
    gpu_memory(1)

    assert resolve_distribution_plan(str(tmp_path))[0] == DistributionPlan(2, dense="fsdp")


def test_a_model_that_cannot_be_built_on_meta_names_itself(run_as, serve, monkeypatch, tmp_path):
    def unbuildable(*args, **kwargs):
        raise NotImplementedError("custom init touches the weights")

    run_as(2)
    serve(LlamaConfig(**_TINY))
    monkeypatch.setattr(model_load_utils, "build_meta_causal_lm", unbuildable)

    with pytest.raises(RuntimeError, match=r"distribute llama .* custom init touches the weights"):
        resolve_distribution_plan(str(tmp_path))


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
