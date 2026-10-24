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

"""Which load ``load_pretrained`` routes a distribution plan to, and the plan it records.

Nothing here needs a GPU, and only one test a process group, two gloo ranks: elsewhere the process
group's rank and size are answered the way a torchrun launch would, the configs are tiny real
transformers configs served from memory, and the loaders are replaced by stand-ins that record how
they were called.
"""

import inspect

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives

transformers = pytest.importorskip("transformers")

from transformers import AutoModel, DeepseekV3Config, LlamaConfig, LlamaForCausalLM

import modelopt.torch.models.hf as models_hf
import modelopt.torch.utils.distributed as dist
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.distributed import DistributionPlan, distribution_plan
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

_TINY = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
}


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
    """Run as rank ``rank`` of a ``world_size``-rank launch.

    Every rank acts as its group's first, deciding for itself, and reads the checkpoint as this one
    does: these tests are about how a plan loads, and test_resolve_distribution_plan.py about how
    the ranks agree on it.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _gathered_alike)

    def enter(world_size, rank=0):
        monkeypatch.setattr(dist, "is_distributed_run", lambda: world_size > 1)
        monkeypatch.setattr(dist, "size", lambda group=None: world_size)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(dist, "is_master", lambda group=None: rank == 0)
        monkeypatch.setattr(dist, "local_rank", lambda: rank)

    return enter


@pytest.fixture
def serve(monkeypatch):
    """Answer every config read with ``config`` rather than a checkpoint on disk."""

    def serve(config):
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: config)
        return config

    return serve


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


@pytest.fixture
def process_group(monkeypatch):
    """Rank 1 of a 2-rank process group, as the FSDP2 load reads it."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(dist, "local_rank", lambda: 1)


# ----------------------------------------------------------------------------------- the loads


@pytest.mark.parametrize(
    ("device_map", "placed"),
    [(None, {"": "cuda:1"}), ("cpu", "cpu")],
    ids=["defaults_to_the_rank_gpu", "caller_placement_wins"],
)
def test_a_replicated_plan_loads_the_whole_model_on_every_rank(
    run_as, serve, loaders, auto_class, tmp_path, device_map, placed
):
    """Each rank takes the whole model onto its own GPU; "auto" would let every rank's accelerate
    pick the same idle GPU."""
    run_as(4, rank=1)
    serve(LlamaConfig(**_TINY))
    kwargs = {} if device_map is None else {"device_map": device_map}

    model = load_pretrained(str(tmp_path), auto_class, **kwargs)

    assert auto_class.calls == [{"device_map": placed}]
    assert distribution_plan(model) == DistributionPlan(4)
    assert loaders == {}


def test_an_fsdp_plan_loads_through_the_parallel_fsdp2_loader(
    run_as, loaders, process_group, monkeypatch, tmp_path
):
    """FSDP2 shards over the process group it reads its rank and GPU from, by a config read with
    the Hub options the load was given (custom code an ``auto_map`` names in another repo)."""
    run_as(2, rank=1)
    served, reads = LlamaConfig(**_TINY), []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(k) or served
    )
    plan = DistributionPlan(2, dense="fsdp")

    model = load_pretrained(
        str(tmp_path), distribution=plan, cpu_offload=True, dtype="auto", local_files_only=True
    )

    call = loaders.pop("fsdp2")
    assert (call["device"], call["rank"], call["world_size"]) == (torch.device("cuda:1"), 1, 2)
    assert call["cpu_offload"] is True
    assert call["hf_config"] is served
    assert reads and all(read.get("local_files_only") is True for read in reads)
    assert distribution_plan(model) == plan
    assert loaders == {}


def test_a_load_it_decides_reads_the_config_once(
    run_as, loaders, process_group, monkeypatch, tmp_path
):
    """The config the plan was decided from is the one the FSDP2 load gets, not a second read."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    run_as(2, rank=1)
    config, reads = LlamaConfig(**_TINY), []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(a) or config
    )

    model = load_pretrained(str(tmp_path))

    assert len(reads) == 1
    assert loaders["fsdp2"]["hf_config"] is config
    assert distribution_plan(model) == DistributionPlan(2, dense="fsdp")


@pytest.mark.parametrize("handed", [False, True], ids=["decided", "handed"])
def test_a_config_passed_for_the_load_is_the_one_it_decides_from(
    run_as, loaders, process_group, monkeypatch, tmp_path, handed
):
    """No config is read: the plan is decided from the one passed for the load, which the FSDP2
    load then builds from, as it does with a plan handed in. Deciding adds to that config, so it
    works on a copy of it."""
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

    with pytest.raises(TypeError, match="returned with a plan: pass that plan"):
        load_pretrained(str(tmp_path), hf_config=LlamaConfig(**{**_TINY, "num_hidden_layers": 2}))
    assert loaders == {}


def test_an_expert_parallel_plan_loads_through_the_ep_loader(
    run_as, serve, loaders, auto_class, tmp_path
):
    """An EP plan loads through load_and_prepare_ep, with the plan, and through no other load."""
    run_as(4)
    serve(_deepseek_v3(n_routed_experts=8))
    plan = DistributionPlan(4, ep_size=4, dense="fsdp")

    model = load_pretrained(
        str(tmp_path),
        distribution=plan,
        trust_remote_code=True,
        attn_implementation="eager",
        freeze=False,
    )

    call = loaders.pop("ep")
    assert (call["ckpt_path"], call["plan"]) == (str(tmp_path), plan)
    assert (call["trust_remote_code"], call["attn_implementation"]) == (True, "eager")
    assert call["freeze"] is False
    assert distribution_plan(model) == plan
    assert loaders == {}
    assert auto_class.calls == []


def _every_rank_loads_rank0s_plan_job(rank, size, ckpt):
    handed = DistributionPlan(size) if rank == 0 else DistributionPlan(size, dense="fsdp")
    with bounded_collectives():
        model = load_pretrained(ckpt, distribution=handed, device_map="cpu")
    assert distribution_plan(model) == DistributionPlan(size), rank


def test_every_rank_loads_the_plan_rank0_was_handed(cpu_dist_workers, tmp_path):
    """Over a real broadcast. Ranks loading different plans would build mismatched meshes and hang:
    rank 1, handed an FSDP2 plan of its own, loads the whole model by the plan rank 0 was handed."""
    LlamaForCausalLM(LlamaConfig(**_TINY)).save_pretrained(tmp_path)
    cpu_dist_workers.run(_every_rank_loads_rank0s_plan_job, ckpt=str(tmp_path))


@pytest.mark.parametrize("sharding", ["expert-parallel", "fsdp2"])
@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"kwargs": {"device_map": "auto"}}, TypeError, r"\['device_map'\] do not apply"),
        ({"auto_class": AutoModel}, NotImplementedError, "builds its model class itself"),
        ({"mtp": True}, NotImplementedError, "cannot build the MTP head"),
        (
            {"kwargs": {"config": LlamaConfig(**_TINY), "hf_config": LlamaConfig(**_TINY)}},
            TypeError,
            "config and hf_config both name",
        ),
    ],
    ids=["placement", "model_class", "mtp_head", "two_configs"],
)
def test_a_sharded_load_refuses_what_it_cannot_honor(
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, sharding, change, error, match
):
    """Placement, a model class, a head only a whole model builds, two configs to load with: none
    reaches a sharded load, which fails before it starts rather than load something other than
    what was asked."""
    run_as(2)
    expert_parallel = sharding == "expert-parallel"
    serve(_deepseek_v3() if expert_parallel else LlamaConfig(**_TINY))
    if change.get("mtp"):
        monkeypatch.setattr(models_hf, "checkpoint_has_mtp", lambda *a: True)
    plan = DistributionPlan(2, ep_size=2) if expert_parallel else DistributionPlan(2, dense="fsdp")

    with pytest.raises(error, match=match):
        load_pretrained(
            str(tmp_path), change.get("auto_class"), distribution=plan, **change.get("kwargs", {})
        )
    assert loaders == {}


@pytest.mark.parametrize(
    ("distribution", "quantized", "whole_model_kwargs", "error", "match"),
    [
        (
            DistributionPlan(4, dense="fsdp"),
            False,
            False,
            ValueError,
            "is for 4 ranks, but this run has 2",
        ),
        (DistributionPlan(2, dense="fsdp"), True, False, NotImplementedError, "cannot dequantize"),
        ("fsdp2", False, False, TypeError, "expected a DistributionPlan or None"),
        (
            DistributionPlan(2),
            False,
            True,
            TypeError,
            "hf_config applies to a sharded load only",
        ),
    ],
    ids=["another-world", "quantized-checkpoint", "not-a-plan", "hf-config-on-a-whole-load"],
)
def test_a_load_refuses_what_it_cannot_load(
    run_as,
    serve,
    loaders,
    auto_class,
    process_group,
    tmp_path,
    distribution,
    quantized,
    whole_model_kwargs,
    error,
    match,
):
    """Before anything loads: a plan for another world; a quantized checkpoint, which FSDP2 would
    read as stored; anything but a plan; and the sharded loads' config, which from_pretrained would
    take for a model argument and fail far from here."""
    run_as(2)
    config = serve(LlamaConfig(**_TINY))
    if quantized:
        config.quantization_config = {"quant_method": "fp8"}
    whole = {"auto_class": auto_class, "hf_config": config} if whole_model_kwargs else {}

    with pytest.raises(error, match=match):
        load_pretrained(str(tmp_path), distribution=distribution, **whole)
    assert loaders == {}
    assert auto_class.calls == []


@pytest.mark.parametrize("quantized", [False, True])
def test_a_single_process_fsdp2_load_sets_up_its_group_of_one(
    monkeypatch, serve, loaders, tmp_path, quantized
):
    """FSDP2 on one rank still shards through a process group, of one, which the load sets up once
    nothing can refuse it: a refused load leaves no group behind."""
    config = serve(LlamaConfig(**_TINY))
    if quantized:
        config.quantization_config = {"quant_method": "fp8"}
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


# --------------------------------------------------------------------------------- fsdp2_wrap


def test_fsdp2_wrap_leaves_the_plan_to_the_load(monkeypatch):
    """load_pretrained records the plan once, at the end of the load; the wrap it calls records none,
    so a caller that wraps a model itself must record one before anything reads it."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    monkeypatch.setattr(
        LayerActivationCollector, "get_decoder_layers", staticmethod(lambda model: list(model))
    )
    monkeypatch.setattr(model_load_utils, "fully_shard", lambda module, **kwargs: None)
    monkeypatch.setattr(dist, "size", lambda group=None: 4)

    model_load_utils.fsdp2_wrap(model)

    with pytest.raises(RuntimeError, match="no distribution plan recorded"):
        distribution_plan(model)
