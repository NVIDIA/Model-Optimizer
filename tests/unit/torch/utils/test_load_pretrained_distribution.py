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

Nothing here needs a process group or a GPU: the process group's rank and size are answered the way
a torchrun launch would, the configs are tiny real transformers configs served from memory, and the
loaders are replaced by stand-ins that record how they were called.
"""

import inspect

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import AutoModel, LlamaConfig, Qwen3MoeConfig

import modelopt.torch.models.hf as models_hf
import modelopt.torch.utils.distributed as dist
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


@pytest.fixture(autouse=True)
def _no_distribution_knobs(monkeypatch):
    """A shell that exports the knobs would otherwise steer every decision below."""
    monkeypatch.delenv("MODELOPT_EP_DEGREE", raising=False)
    monkeypatch.delenv("MODELOPT_FSDP_SIZE", raising=False)


@pytest.fixture
def run_as(monkeypatch):
    """Run as rank ``rank`` of a ``world_size``-rank launch.

    Every rank acts as its group's first, deciding for itself: these tests are about how a plan
    loads, and test_resolve_distribution_plan.py about how the ranks agree on it.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)

    def enter(world_size, rank=0):
        monkeypatch.setattr(dist, "is_distributed_run", lambda: world_size > 1)
        monkeypatch.setattr(dist, "size", lambda group=None: world_size)
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


@pytest.mark.parametrize("kwargs", [{"device_map": "auto", "dtype": "auto"}, {}])
def test_a_single_process_loads_the_model_whole_as_asked(
    serve, loaders, auto_class, tmp_path, kwargs
):
    """No process group, a single process's plan: from_pretrained places the model exactly as the
    caller says, and with nothing said, as it does by default."""
    serve(LlamaConfig(**_TINY))

    model = load_pretrained(str(tmp_path), auto_class, **kwargs)

    assert auto_class.calls == [kwargs]
    assert model._modelopt_unplaced_source_keys == []
    assert distribution_plan(model) == DistributionPlan()
    assert loaders == {}


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
    run_as, serve, loaders, process_group, tmp_path
):
    """FSDP2 shards over the process group it reads its rank and GPU from."""
    run_as(2, rank=1)
    served = serve(LlamaConfig(**_TINY))
    plan = DistributionPlan(2, dense="fsdp")

    model = load_pretrained(str(tmp_path), distribution=plan, cpu_offload=True, dtype="auto")

    call = loaders.pop("fsdp2")
    assert (call["device"], call["rank"], call["world_size"]) == (torch.device("cuda:1"), 1, 2)
    assert call["cpu_offload"] is True
    assert call["hf_config"] is served
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


def test_an_expert_parallel_plan_is_refused(run_as, serve, loaders, auto_class, tmp_path):
    """No load here shards the routed experts: an EP plan fails before anything loads."""
    run_as(2)
    serve(Qwen3MoeConfig(num_experts=8, moe_intermediate_size=8, **_TINY))

    with pytest.raises(NotImplementedError, match="shards the routed experts"):
        load_pretrained(str(tmp_path), auto_class, distribution=DistributionPlan(2, ep_size=2))
    assert loaders == {}
    assert auto_class.calls == []


def test_every_rank_loads_the_plan_rank0_was_handed(
    run_as, serve, loaders, process_group, auto_class, monkeypatch, tmp_path
):
    """Ranks loading different plans would build mismatched meshes and hang: rank 1, handed a plan
    of its own, loads the one rank 0 was handed."""
    run_as(2, rank=1)
    serve(LlamaConfig(**_TINY))
    rank0_plan = DistributionPlan(2, dense="fsdp")

    def broadcast_object_list(message, group=None, group_src=None):
        label = message[0][0]
        message[0] = (label, rank0_plan if label == "plan" else [], None)

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(model_load_utils, "AutoModelForCausalLM", auto_class)

    model = load_pretrained(str(tmp_path), distribution=DistributionPlan(2))

    assert auto_class.calls == [], "rank 1 loaded the whole model, as its own plan says"
    assert "fsdp2" in loaders, "rank 0's plan shards the model"
    assert distribution_plan(model) == rank0_plan


def test_a_plan_for_another_world_is_refused(run_as, serve, tmp_path):
    run_as(2)
    serve(LlamaConfig(**_TINY))

    with pytest.raises(ValueError, match="is for 4 ranks, but this run has 2"):
        load_pretrained(str(tmp_path), distribution=DistributionPlan(4, dense="fsdp"))


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
    run_as, serve, loaders, process_group, monkeypatch, tmp_path, change, error, match
):
    """Placement, a model class, a head only a whole model builds: none reaches a sharded load,
    which fails before it starts rather than load something other than what was asked."""
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


def test_the_fsdp2_load_refuses_a_quantized_checkpoint(
    run_as, serve, loaders, process_group, tmp_path
):
    """It reads the tensors as stored, so quantized weights would load as if they were not."""
    run_as(2)
    config = LlamaConfig(**_TINY)
    config.quantization_config = {"quant_method": "fp8"}
    serve(config)

    with pytest.raises(NotImplementedError, match="cannot dequantize"):
        load_pretrained(str(tmp_path), distribution=DistributionPlan(2, dense="fsdp"))
    assert loaders == {}


def test_the_fsdp2_load_needs_a_process_group(serve, loaders, tmp_path):
    """FSDP2 on one rank still shards through a process group, of one."""
    serve(LlamaConfig(**_TINY))

    with pytest.raises(RuntimeError, match=r"needs a torch\.distributed process group"):
        load_pretrained(str(tmp_path), distribution=DistributionPlan(dense="fsdp"))


def test_anything_but_a_plan_is_refused(serve, tmp_path):
    serve(LlamaConfig(**_TINY))

    with pytest.raises(TypeError, match="expected a DistributionPlan or None"):
        load_pretrained(str(tmp_path), distribution="fsdp2")


def test_a_whole_model_load_refuses_a_sharded_loads_config(serve, auto_class, tmp_path):
    """from_pretrained would take hf_config for a model argument and fail far from here."""
    config = serve(LlamaConfig(**_TINY))

    with pytest.raises(TypeError, match="hf_config applies to a sharded load only"):
        load_pretrained(str(tmp_path), auto_class, hf_config=config)
