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

"""How ``load_pretrained`` decides a model's distribution, and which load that routes it to.

Nothing here needs a process group or a GPU: the active ``ModelOptSession`` is initialized the way a
torchrun launch leaves it, the configs are tiny real transformers configs served from memory, and the
loaders are replaced by stand-ins that record how they were called.
"""

import inspect

import pytest
import torch

transformers = pytest.importorskip("transformers")

from packaging.version import Version
from transformers import AutoModel, LlamaConfig, Qwen3MoeConfig

import modelopt.torch.models.hf as models_hf
import modelopt.torch.utils.distributed as dist
import modelopt.torch.utils.moe as moe
from modelopt.torch.utils import env
from modelopt.torch.utils.distributed import DistributionPlan, get_active_mo_session
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained, resolve_distribution_plan

# Small enough that the dense part fits any GPU, so "auto" dense sharding always resolves to
# replication whether or not the machine running the tests has one.
_TINY = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
}


def _qwen3_moe(num_experts):
    """Qwen3-MoE, whose base_model_ep_plan ships with transformers itself."""
    return Qwen3MoeConfig(num_experts=num_experts, moe_intermediate_size=8, **_TINY)


@pytest.fixture(autouse=True)
def _no_distribution_knobs(monkeypatch):
    """A shell that exports the knobs would otherwise steer every decision below."""
    monkeypatch.delenv("MODELOPT_EP_DEGREE", raising=False)
    monkeypatch.delenv("MODELOPT_FSDP_SIZE", raising=False)


@pytest.fixture
def session(monkeypatch):
    """Enter the active session as rank ``rank`` of a ``world_size``-rank launch would.

    Every rank acts as its group's first, deciding for itself: these tests are about what is
    decided, and test_distributed_session.py about how the ranks agree on it.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)
    active = get_active_mo_session(required=False)

    def enter(world_size, rank=0):
        active.initialize(
            distributed=world_size > 1,
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            device=torch.device(f"cuda:{rank}"),
            is_main=rank == 0,
        )
        return active

    yield enter
    active._reset()


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


# ----------------------------------------------------------------------------------- env knobs


@pytest.mark.parametrize(
    ("knob", "name"),
    [(env.ep_degree, "MODELOPT_EP_DEGREE"), (env.fsdp_size, "MODELOPT_FSDP_SIZE")],
)
def test_distribution_knobs_default_to_auto(monkeypatch, knob, name):
    """Unset and exported-but-empty both mean "let the plan decide"."""
    assert knob() == "auto"
    monkeypatch.setenv(name, "")
    assert knob() == "auto"


@pytest.mark.parametrize(
    ("knob", "name", "other"),
    [
        (env.ep_degree, "MODELOPT_EP_DEGREE", env.fsdp_size),
        (env.fsdp_size, "MODELOPT_FSDP_SIZE", env.ep_degree),
    ],
)
def test_distribution_knobs_are_read_when_called(monkeypatch, knob, name, other):
    """A value snapshotted at import would miss what the launcher sets later; and each knob reads
    only its own variable."""
    monkeypatch.setenv(name, "2")
    assert knob() == "2"
    monkeypatch.setenv(name, "8")
    assert knob() == "8"
    assert other() == "auto"


# -------------------------------------------------------------------------------- the decision


@pytest.mark.parametrize("world_size", [None, 1], ids=["no_session", "single_process_session"])
def test_outside_a_distributed_session_nothing_is_distributed(session, serve, tmp_path, world_size):
    """With no ranks to shard over, even a MoE loads whole."""
    if world_size is not None:
        session(world_size)
    serve(_qwen3_moe(num_experts=8))

    assert resolve_distribution_plan(str(tmp_path)) == (None, None)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"),
    reason="qwen3_moe's base_model_ep_plan ships in transformers 5.7",
)
@pytest.mark.parametrize("rank", [0, 1])
def test_an_ep_capable_moe_is_planned_expert_parallel(session, serve, tmp_path, capsys, rank):
    """Auto picks the largest EP degree dividing both the world and the expert count, hands back
    the config it decided from, and only rank 0 reports the decision."""
    session(4, rank)
    serve(_qwen3_moe(num_experts=8))

    plan, hf_config = resolve_distribution_plan(str(tmp_path))

    assert (plan.world_size, plan.ep_size, plan.edp) == (4, 4, 1)
    assert not plan.is_fsdp_dense
    assert moe.is_expert_parallel_capable(hf_config)
    assert ("Distribution:" in capsys.readouterr().out) == (rank == 0)


def test_a_dense_model_is_replicated(session, serve, tmp_path):
    """A dense model that fits one GPU has nothing to shard: every rank holds it whole."""
    session(4)
    serve(LlamaConfig(**_TINY))

    plan, _ = resolve_distribution_plan(str(tmp_path))

    assert (plan.ep_size, plan.fsdp_size) == (1, 1)


def test_the_env_knobs_override_the_automatic_plan(session, serve, monkeypatch, tmp_path):
    """The decision honours both knobs over its automatic choice."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    session(4)
    serve(_qwen3_moe(num_experts=8))

    plan, _ = resolve_distribution_plan(str(tmp_path))

    assert (plan.ep_size, plan.edp, plan.fsdp_size) == (2, 2, 4)


def test_arguments_override_the_env_knobs(session, serve, monkeypatch, tmp_path):
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "4")
    session(4)
    serve(_qwen3_moe(num_experts=8))

    plan, _ = resolve_distribution_plan(str(tmp_path), ep_size=2, fsdp_size=1)

    assert (plan.ep_size, plan.fsdp_size) == (2, 1)


# ----------------------------------------------------------------------------------- the loads


def test_without_a_session_the_model_loads_whole_as_asked(serve, loaders, auto_class, tmp_path):
    """No session, no plan: from_pretrained places the model exactly as the caller says."""
    serve(LlamaConfig(**_TINY))

    model = load_pretrained(str(tmp_path), auto_class, device_map="auto", dtype="auto")

    assert auto_class.calls == [{"device_map": "auto", "dtype": "auto"}]
    assert model._modelopt_unplaced_source_keys == []
    assert loaders == {}


@pytest.mark.parametrize(
    ("device_map", "placed"),
    [(None, {"": "cuda:1"}), ("cpu", "cpu")],
    ids=["defaults_to_the_rank_gpu", "caller_placement_wins"],
)
def test_a_replicated_plan_loads_the_whole_model_on_every_rank(
    session, serve, loaders, auto_class, tmp_path, device_map, placed
):
    """Each rank takes the whole model onto its own GPU; "auto" would let every rank's accelerate
    pick the same idle GPU. The plan the load decided is kept on the session."""
    active = session(4, rank=1)
    serve(LlamaConfig(**_TINY))
    kwargs = {} if device_map is None else {"device_map": device_map}

    load_pretrained(str(tmp_path), auto_class, **kwargs)

    assert auto_class.calls == [{"device_map": placed}]
    assert loaders == {}
    assert (active.plan.ep_size, active.plan.fsdp_size) == (1, 1)


@pytest.mark.parametrize("distribution", ["fsdp2", DistributionPlan(2, fsdp_size=2)])
def test_fsdp2_loads_through_the_parallel_fsdp2_loader(
    session, serve, loaders, process_group, tmp_path, distribution
):
    """Named or planned, FSDP2 shards over the process group it reads its rank and GPU from."""
    session(2, rank=1)
    served = serve(LlamaConfig(**_TINY))

    load_pretrained(str(tmp_path), distribution=distribution, cpu_offload=True, dtype="auto")

    call = loaders.pop("fsdp2")
    assert (call["device"], call["rank"], call["world_size"]) == (torch.device("cuda:1"), 1, 2)
    assert call["cpu_offload"] is True
    assert call["hf_config"] is served
    assert loaders == {}


def test_a_load_it_decides_reads_the_config_once(
    session, loaders, process_group, monkeypatch, tmp_path
):
    """The config the plan was decided from is the one the FSDP2 load gets, not a second read."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    session(2, rank=1)
    config, reads = LlamaConfig(**_TINY), []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: reads.append(a) or config
    )

    load_pretrained(str(tmp_path))

    assert len(reads) == 1
    assert loaders["fsdp2"]["hf_config"] is config


def test_partial_dense_sharding_without_expert_parallelism_is_refused(session, serve, tmp_path):
    """The FSDP2 load shards over the whole world; it cannot honour a smaller dense degree."""
    session(4)
    serve(LlamaConfig(**_TINY))

    with pytest.raises(NotImplementedError, match="fsdp_size must be 4"):
        load_pretrained(str(tmp_path), distribution=DistributionPlan(4, fsdp_size=2))


def test_an_expert_parallel_plan_is_refused(session, serve, loaders, auto_class, tmp_path):
    """No load here shards the routed experts: an EP plan fails before anything loads, and is not
    kept on the session."""
    active = session(2)
    serve(_qwen3_moe(num_experts=8))

    with pytest.raises(NotImplementedError, match="shards the routed experts"):
        load_pretrained(str(tmp_path), auto_class, distribution=DistributionPlan(2, ep_size=2))
    assert active.plan is None
    assert loaders == {}
    assert auto_class.calls == []


def test_the_plan_a_load_runs_is_kept_on_the_session(
    session, serve, loaders, process_group, tmp_path
):
    """Handed a plan, the load records it, as it does one it decides itself."""
    active = session(2)
    serve(LlamaConfig(**_TINY))

    load_pretrained(str(tmp_path), distribution=DistributionPlan(2, fsdp_size=2))

    assert active.plan.fsdp_size == 2


def test_a_plan_for_another_world_is_refused(session, serve, tmp_path):
    session(2)
    serve(LlamaConfig(**_TINY))

    with pytest.raises(ValueError, match="is for 4 ranks, but this run has 2"):
        load_pretrained(str(tmp_path), distribution=DistributionPlan(4, fsdp_size=4))


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"kwargs": {"device_map": "auto"}}, TypeError, r"\['device_map'\] do not apply"),
        ({"auto_class": AutoModel}, NotImplementedError, "builds its model class itself"),
        ({"mtp": True}, NotImplementedError, "cannot build the MTP head"),
    ],
    ids=["placement", "model_class", "mtp_head"],
)
def test_a_sharded_load_refuses_what_it_cannot_honor(
    session, serve, loaders, process_group, monkeypatch, tmp_path, change, error, match
):
    """Placement, a model class, a head only a whole model builds: none reaches a sharded load,
    which fails before it starts rather than load something other than what was asked."""
    session(2)
    serve(LlamaConfig(**_TINY))
    if change.get("mtp"):
        monkeypatch.setattr(models_hf, "checkpoint_has_mtp", lambda *a: True)

    with pytest.raises(error, match=match):
        load_pretrained(
            str(tmp_path),
            change.get("auto_class"),
            distribution="fsdp2",
            **change.get("kwargs", {}),
        )
    assert loaders == {}


def test_the_fsdp2_load_refuses_a_quantized_checkpoint(serve, loaders, process_group, tmp_path):
    """It reads the tensors as stored, so quantized weights would load as if they were not."""
    config = LlamaConfig(**_TINY)
    config.quantization_config = {"quant_method": "fp8"}
    serve(config)

    with pytest.raises(NotImplementedError, match="cannot dequantize"):
        load_pretrained(str(tmp_path), distribution="fsdp2")
    assert loaders == {}


def test_the_fsdp2_load_needs_a_process_group(serve, loaders, tmp_path):
    serve(LlamaConfig(**_TINY))

    with pytest.raises(RuntimeError, match=r"needs a torch\.distributed process group"):
        load_pretrained(str(tmp_path), distribution="fsdp2")


def test_an_unknown_distribution_is_refused(serve, tmp_path):
    serve(LlamaConfig(**_TINY))

    with pytest.raises(ValueError, match="expected None, 'sequential', 'fsdp2'"):
        load_pretrained(str(tmp_path), distribution="ep")


def test_a_whole_model_load_refuses_a_sharded_loads_config(serve, auto_class, tmp_path):
    """from_pretrained would take hf_config for a model argument and fail far from here."""
    config = serve(LlamaConfig(**_TINY))

    with pytest.raises(TypeError, match="hf_config applies to a sharded load only"):
        load_pretrained(str(tmp_path), auto_class, hf_config=config)
