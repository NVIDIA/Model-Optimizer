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

"""Unit tests for distributed calibration: per-rank data shards and the distributed forward loop.

Each rank of a distributed (e.g. expert-parallel) PTQ run calibrates on its own shard of the data,
and the EP collectives inside every forward hang unless all ranks run the same number of batches.
The torch_a2a TP forward wrappers come from transformers >= 5.16 and need a process group, so they
are faked here; everything runs in one CPU process.
"""

import re
import sys
import types
from collections import Counter, OrderedDict

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as mto_dist
from modelopt.torch.quantization.model_quant import _refuse_unsupported_distributed_calibration
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    ParallelState,
    _reinstall_a2a_forwards,
    distributed_forward_loop,
    set_distribution_plan,
    shard_dataloader,
)

_EXPERT_PARALLEL = DistributionPlan(world_size=2, ep_size=2)

# ------------------------------------------------------------------------------------- data helpers


def _collate(ids):
    """Calibration-style collate: a dict of batched tensors, as ``model(**batch)`` expects."""
    return {"input_ids": torch.tensor(ids).unsqueeze(-1)}


class _Stream(IterableDataset):
    """A deterministic stream with no ``len()``, like the streaming image-text calibration sets."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))


def _loader(dataset, batch_size=2):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)


def _batches(loader):
    """The sample ids in each batch the loader yields."""
    return [batch["input_ids"].flatten().tolist() for batch in loader]


def _ids(loader):
    return [i for batch in _batches(loader) for i in batch]


def _seed_worker(worker_id):
    torch.manual_seed(worker_id)


def _dataset(kind, n):
    """Samples ``0 .. n - 1`` as a list (map-style) or as a stream with no ``len()``."""
    return list(range(n)) if kind == "map-style" else _Stream(n)


# --------------------------------------------------------------------------------- shard_dataloader


@pytest.mark.parametrize("kind", ["map-style", "iterable"])
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_shards_are_strided_cover_every_sample_once_and_replay(kind, world_size):
    """Rank ``r`` gets samples ``r, r + W, ...``, batched with the original size and collate_fn, and
    a second pass replays the same shard: multi-pass calibration algorithms (e.g. AWQ) iterate the
    loader again. A lone rank's shard is the whole (unshuffled) loader, batch for batch.
    """
    n = 12
    shards = [
        shard_dataloader(_loader(_dataset(kind, n)), r, world_size) for r in range(world_size)
    ]

    assert sorted(i for shard in shards for i in _ids(shard)) == list(range(n))
    for r, shard in enumerate(shards):
        ids = list(range(r, n, world_size))
        assert _batches(shard) == _batches(shard) == [ids[i : i + 2] for i in range(0, len(ids), 2)]


@pytest.mark.parametrize(
    ("kind", "padded", "dropped"),
    [("map-style", 2, set()), ("iterable", 0, {8, 9})],
    ids=["map-style", "iterable"],
)
def test_an_uneven_split_gives_every_rank_the_same_batch_count(kind, padded, dropped):
    """No rank runs short: it would hang the others' EP collectives. A map-style loader repeats a
    few samples to even the ranks out (``drop_last=False``); a stream, with no ``len()`` to pad to,
    drops its trailing partial group instead.
    """
    n, world_size = 10, 4
    shards = [
        _batches(shard_dataloader(_loader(_dataset(kind, n)), r, world_size))
        for r in range(world_size)
    ]

    assert len({len(batches) for batches in shards}) == 1
    seen = Counter(i for batches in shards for batch in batches for i in batch)
    assert set(seen) == set(range(n)) - dropped
    assert seen.total() - (n - len(dropped)) == padded


def test_a_stream_shorter_than_the_world_is_refused_on_every_rank():
    """With fewer samples than ranks there is no complete group, so every rank would calibrate on
    nothing; each rank refuses the stream instead, before any forward."""
    for rank in range(4):
        with pytest.raises(ValueError, match="fewer than the 4 ranks"):
            _ids(shard_dataloader(_loader(_Stream(3)), rank, 4))


def test_a_stream_loaded_with_workers_is_refused():
    """A stream that splits itself across the loader's workers, as a Hugging Face streaming dataset
    does, would be split again by rank and silently lose samples. A lone rank keeps its loader."""
    loader = DataLoader(_Stream(16), batch_size=2, collate_fn=_collate, num_workers=2)

    with pytest.raises(ValueError, match="num_workers=2 is refused"):
        shard_dataloader(loader, 0, 2)
    assert shard_dataloader(loader, 0, 1) is loader


@pytest.mark.parametrize(
    "picker",
    [
        {"sampler": SubsetRandomSampler(range(4)), "batch_size": 2},
        {"sampler": SubsetRandomSampler(range(4)), "batch_size": None},
        {"batch_sampler": [[0, 1], [2, 3]]},
        {"batch_sampler": BatchSampler(SubsetRandomSampler(range(4)), 2, drop_last=False)},
        {"sampler": RandomSampler(range(8), num_samples=4), "batch_size": 2},
        {"sampler": RandomSampler(range(8), replacement=True), "batch_size": 2},
        {"sampler": SequentialSampler(range(4)), "batch_size": 2},
        {"sampler": RandomSampler(range(4)), "batch_size": 2},
        {"sampler": DistributedSampler(range(4), num_replicas=2, rank=0), "batch_size": 2},
    ],
    ids=[
        "subset-sampler",
        "subset-sampler-unbatched",
        "batch-sampler",
        "subset-in-a-plain-batch-sampler",
        "random-sampler-drawing-fewer",
        "random-sampler-with-replacement",
        "sequential-sampler-over-fewer",
        "random-sampler-over-fewer",
        "distributed-sampler-over-fewer",
    ],
)
def test_a_loader_that_picks_its_own_samples_is_refused(picker):
    """The rank's DistributedSampler would replace what it picks, silently calibrating on the whole
    dataset instead, however the loader batches the picks. A lone rank has nothing to shard and
    keeps the loader as it is."""
    loader = DataLoader(list(range(8)), collate_fn=_collate, **picker)

    with pytest.raises(ValueError, match="picks which samples it reads"):
        shard_dataloader(loader, 0, 2)
    assert shard_dataloader(loader, 0, 1) is loader


@pytest.mark.parametrize(
    ("kind", "batching", "batches"),
    [
        (
            "map-style",
            {"batch_sampler": BatchSampler(SequentialSampler(range(10)), 2, drop_last=True)},
            [[1, 3], [5, 7]],
        ),
        ("map-style", {"batch_size": None}, [[1], [3], [5], [7], [9]]),
        ("iterable", {"batch_size": None}, [[1], [3], [5], [7], [9]]),
        ("map-style", {"batch_size": 2, "shuffle": True}, [[1, 3], [5, 7], [9]]),
    ],
    ids=["plain-batch-sampler", "auto-batching-off", "auto-batching-off-stream", "shuffled"],
)
def test_the_shard_batches_as_the_loader_does(kind, batching, batches):
    """A loader given a plain BatchSampler reports ``batch_size=None``; its shard still batches by the
    sampler's size and drop_last. A loader with auto-batching off reads one item per step, and so
    does its shard. A loader shuffling the whole dataset gives the rank its unshuffled share."""
    loader = DataLoader(_dataset(kind, 10), collate_fn=_collate, **batching)

    assert _batches(shard_dataloader(loader, 1, 2)) == batches


@pytest.mark.parametrize("dataset", [list(range(8)), _Stream(8)], ids=["map-style", "iterable"])
def test_sharded_loader_keeps_the_original_loader_settings(dataset):
    """Batching, pinning, timeout, worker init and RNG carry over to the shard, and for a map-style
    loader its workers and prefetching too (a stream is refused workers)."""
    workers = (
        {}
        if isinstance(dataset, _Stream)
        else {
            "num_workers": 2,
            "multiprocessing_context": "spawn",
            "prefetch_factor": 3,
            "persistent_workers": True,
        }
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        drop_last=True,
        collate_fn=_collate,
        pin_memory=True,
        timeout=7.0,
        worker_init_fn=_seed_worker,
        generator=torch.Generator(),
        **workers,
    )
    shard = shard_dataloader(loader, 1, 2)

    for setting in (
        "batch_size",
        "drop_last",
        "collate_fn",
        "num_workers",
        "pin_memory",
        "timeout",
        "worker_init_fn",
        "multiprocessing_context",
        "generator",
        "prefetch_factor",
        "persistent_workers",
    ):
        assert getattr(shard, setting) == getattr(loader, setting), setting


# -------------------------------------------------------------------------- _reinstall_a2a_forwards

_TP_MODULE = "transformers.distributed.tensor_parallel"
_LEGACY_TP_MODULE = "transformers.integrations.tensor_parallel"


class _FakeMesh:
    """The EP device mesh; the TP wrappers only need its process group, the loop its size."""

    def __init__(self, ranks=2):
        self.group = object()
        self.ranks = ranks

    def get_group(self):
        return self.group

    def size(self):
        return self.ranks


class _WrapperStyle:
    """A transformers >= 5.16 TP style: ``install_forward`` wraps the module's current forward."""

    def __init__(self, inject_group=False):
        self.inject_group = inject_group
        self.installed = []

    def install_forward(self, module, mesh):
        self.installed.append(module)
        inner = module.forward
        extra = (mesh.get_group(),) if self.inject_group else ()

        def forward(*args, **kwargs):
            return inner(*args, *extra, **kwargs)

        module.forward = forward


class _PreHookStyle:
    """A transformers < 5.16 TP style: its transforms are forward pre-hooks, no ``install_forward``."""


def _get_parameter_tp_plan(parameter_name, tp_plan, is_weight=True):
    """transformers' plan lookup: layer indices match ``*``; a weight also takes its module's entry."""
    name = re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), parameter_name)
    if name in tp_plan:
        return tp_plan[name]
    if is_weight and "." in name and (module_name := name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    return None


def _install_fake_tp(monkeypatch, styles=None, where=_TP_MODULE):
    """Serve ``styles`` as transformers' ``ALL_PARALLEL_STYLES`` from the module ``where``."""
    if styles is None:
        styles = {
            "torch_a2a_experts": _WrapperStyle(inject_group=True),
            "torch_a2a_router": _WrapperStyle(),
            "colwise": _WrapperStyle(),
        }
    module = types.ModuleType(where)
    module.ALL_PARALLEL_STYLES = styles
    module._get_parameter_tp_plan = _get_parameter_tp_plan
    if where != _TP_MODULE:
        monkeypatch.setitem(sys.modules, _TP_MODULE, None)  # importing it now raises ImportError
    monkeypatch.setitem(sys.modules, where, module)
    return styles


class _Experts(nn.Module):
    """A torch_a2a experts module: its forward needs the EP group its TP wrapper injects."""

    def __init__(self):
        super().__init__()
        self.weight_quantizer = nn.Identity()  # quantizer insertion adds children like this
        self.groups = []

    def forward(self, hidden_states, top_k_index, top_k_weights, process_group=None):
        if process_group is None:
            raise ValueError("torch_a2a experts forward requires the EP `process_group`")
        self.groups.append(process_group)
        return hidden_states


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Identity()  # the router
        self.proj = nn.Identity()  # a dense layer under a non-a2a TP style
        self.experts = _Experts()

    def forward(self, hidden_states):
        return self.experts(self.proj(self.gate(hidden_states)), None, None)


class _MoEModel(nn.Module):
    """Two MoE layers whose plan sends the experts and routers through torch_a2a."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.config = types.SimpleNamespace(use_cache=True)
        self.tp_plan = {
            "layers.*.gate": "torch_a2a_router",
            "layers.*.proj": "colwise",
            "layers.*.experts": "torch_a2a_experts",
        }

    def forward(self, input_ids):
        hidden_states = input_ids.float()
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _load_tp_wrappers(model, styles, mesh):
    """Do what transformers' expert-parallel load does: wrap every planned module's forward."""
    for name, module in model.named_modules():
        if (style := _get_parameter_tp_plan(name, model.tp_plan, is_weight=False)) is not None:
            styles[style].install_forward(module, mesh)
    for style in styles.values():
        style.installed.clear()


def _drop_tp_wrapper(module):
    """Do what quantizer insertion does to the instance: rebind the class forward over the wrapper.

    (``DynamicModule.convert`` -> ``bind_forward_method``.)
    """
    module.forward = types.MethodType(type(module).forward, module)


def test_reinstall_is_a_noop_without_an_ep_mesh(monkeypatch):
    """A single-process or non-EP model is left untouched, even with the torch_a2a styles around."""
    styles = _install_fake_tp(monkeypatch)

    assert _reinstall_a2a_forwards(_MoEModel(), None) == 0
    assert not any(style.installed for style in styles.values())


@pytest.mark.parametrize(
    ("experts_style", "planted"),
    [(_PreHookStyle(), 0), (_WrapperStyle(inject_group=True), 2)],
    ids=["all-pre-hooks", "only-the-experts-wrap"],
)
def test_reinstall_skips_styles_whose_transforms_are_pre_hooks(monkeypatch, experts_style, planted):
    """transformers < 5.16 has no ``install_forward``; its pre-hooks survive quantizer insertion.

    Each style is checked for it, the one about to be called: a style without one is skipped even
    where the other has one. Those releases keep the styles in ``transformers.integrations``: the
    fallback import.
    """
    _install_fake_tp(
        monkeypatch,
        styles={"torch_a2a_experts": experts_style, "torch_a2a_router": _PreHookStyle()},
        where=_LEGACY_TP_MODULE,
    )
    assert _reinstall_a2a_forwards(_MoEModel(), _FakeMesh()) == planted


# ------------------------------------------------------------------------- distributed_forward_loop


class _Recorder(nn.Module):
    """Records each forward's kwargs and the ``use_cache`` flag it ran under."""

    def __init__(self, on_forward=None):
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=True)
        self.calls = []
        self.use_cache_seen = []
        self.on_forward = on_forward

    def forward(self, **batch):
        self.calls.append(batch)
        self.use_cache_seen.append(self.config.use_cache)
        if self.on_forward is not None:
            self.on_forward()


def test_forward_loop_runs_this_ranks_shard_through_the_captured_model():
    """Every batch of the rank's shard reaches the top-level model, not the module ``mtq`` passes.

    Only the full model's forward runs the EP collectives; without a device the batches stay put.
    """
    model, inner = _Recorder(), _Recorder()
    shard = shard_dataloader(_loader(list(range(8))), 0, world_size=2)

    distributed_forward_loop(model, shard)(inner)

    calls = [call["input_ids"].flatten().tolist() for call in model.calls]
    assert calls == [[0, 2], [4, 6]]
    assert inner.calls == []


def test_forward_loop_moves_tensors_to_the_device_it_is_given():
    """Batch tensors go to ``device`` when one is given; other values pass through.

    The meta device makes the move observable on a CPU-only host. (Without a device the batches stay
    put: the shard-routing test reads their values back.)
    """
    batches = [{"input_ids": torch.zeros(2, 3), "logits_to_keep": 1}] * 2
    model = _Recorder()
    distributed_forward_loop(model, batches, device="meta")(model)

    assert [call["input_ids"].device.type for call in model.calls] == ["meta"] * 2
    assert [call["logits_to_keep"] for call in model.calls] == [1, 1]


def test_forward_loop_disables_use_cache_only_while_calibrating():
    """KV/state caching is off for the calibration forwards (it breaks hybrid Mamba models)."""
    model = _Recorder()
    distributed_forward_loop(model, [{"input_ids": torch.zeros(1, 1)}] * 2)(model)

    assert model.use_cache_seen == [False, False]
    assert model.config.use_cache is True


@pytest.mark.parametrize("cuda_started", [True, False], ids=["cuda", "cpu-only"])
def test_forward_loop_frees_the_allocator_cache_around_the_forwards(monkeypatch, cuda_started):
    """Re-plant the a2a wrappers and free the cached pool before the first forward, and again after.

    Building the loop does neither: quantizers are inserted between building and calling it. NCCL
    allocates its buffers outside torch's caching allocator, so a pool hoarded by the model load or
    by the calibration activations starves the next collective. A process that never started CUDA
    has no pool to free and skips the garbage collection, which alone takes about a second.
    """
    events = []
    monkeypatch.setattr(
        mto_dist, "_reinstall_a2a_forwards", lambda *args: events.append(("reinstall", *args))
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_started)
    monkeypatch.setattr(mto_dist.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    model, mesh = _Recorder(on_forward=lambda: events.append("forward")), _FakeMesh()
    set_distribution_plan(model, _EXPERT_PARALLEL)
    forward_loop = distributed_forward_loop(
        model, [{"input_ids": torch.zeros(1, 1)}] * 2, ep_mesh=mesh
    )
    assert events == []

    forward_loop(model)

    release = ["collect", "empty_cache"] if cuda_started else []
    assert events == [("reinstall", model, mesh), *release, "forward", "forward", *release]


def test_forward_loop_replants_a2a_wrappers_dropped_after_it_was_built(monkeypatch):
    """hf_ptq's order -- load, build the loop, insert quantizers, calibrate -- hands the experts the
    EP group.

    It holds on every pass (multi-pass algorithms call the loop again), with one wrapper per module,
    and again once another conversion drops the wrappers anew. Only the wrappers conversion dropped
    are planted again: a router quantization left alone keeps the one from the load, since wrapping
    it again would run its transforms twice. Other TP styles keep their forward, and so do the
    quantizers under an experts module, since the plan is looked up per module (``is_weight=False``)
    and children do not inherit their parent's style.
    """
    styles = _install_fake_tp(monkeypatch)
    model, mesh = _MoEModel(), _FakeMesh()
    _load_tp_wrappers(model, styles, mesh)
    set_distribution_plan(model, _EXPERT_PARALLEL)
    forward_loop = distributed_forward_loop(
        model, [{"input_ids": torch.ones(2, 4)}] * 3, ep_mesh=mesh
    )
    for layer in model.layers:
        _drop_tp_wrapper(layer.experts)

    forward_loop(model)
    forward_loop(model)
    for layer in model.layers:
        _drop_tp_wrapper(layer.experts)
    forward_loop(model)

    assert [layer.experts.groups for layer in model.layers] == [[mesh.group] * 9] * 2
    assert styles["torch_a2a_experts"].installed == [layer.experts for layer in model.layers] * 2
    assert styles["torch_a2a_router"].installed == []
    assert styles["colwise"].installed == []


@pytest.mark.parametrize(
    ("plan", "mesh", "match"),
    [
        (_EXPERT_PARALLEL, None, "calibrates with the ep_mesh"),
        (DistributionPlan(world_size=2), _FakeMesh(), "calibrates with the ep_mesh"),
        (DistributionPlan(world_size=4, ep_size=2), _FakeMesh(ranks=4), "spans 4 ranks"),
    ],
    ids=["expert-parallel-without-mesh", "mesh-for-a-replicated-model", "mesh-of-the-whole-world"],
)
def test_the_ep_mesh_is_the_plans_expert_parallel_one(plan, mesh, match):
    """Without its mesh, the experts' forward would run with no EP group at the first step; a mesh
    for a model whose plan shards no experts would be ignored; and a mesh over other ranks than an
    expert-parallel group would hand the all-to-all the wrong group."""
    model = _Recorder()
    set_distribution_plan(model, plan)

    with pytest.raises(ValueError, match=match):
        distributed_forward_loop(model, [{"input_ids": torch.zeros(1, 1)}], ep_mesh=mesh)


def test_forward_loop_refuses_a_batch_that_is_not_a_mapping():
    """The loop runs ``model(**batch)``; a loader of bare tensors is refused as such at its first
    batch, rather than failing on ``batch.items()`` inside the loop."""
    model = _Recorder()

    with pytest.raises(TypeError, match="yields Tensor"):
        distributed_forward_loop(model, [torch.zeros(1, 1)], device="meta")(model)


# ------------------------------------------------------------------------ rank-local calibration


@pytest.mark.parametrize(
    ("entry", "algorithm", "refused"),
    [
        ("quantize", {"method": "awq_lite"}, "awq_lite"),
        ("calibrate", "awq_lite", "awq_lite"),
        (
            "calibrate",
            {"method": "nvfp4_act_headroom", "weight_scale_algorithm": {"method": "local_hessian"}},
            "local_hessian",
        ),
        (
            "calibrate",
            {"method": "max", "distributed_sync": "false"},
            "max with distributed_sync=False",
        ),
        (
            "calibrate",
            {
                "method": "nvfp4_act_headroom",
                "weight_scale_algorithm": {"method": "mse", "distributed_sync": None},
            },
            "mse with distributed_sync=None",
        ),
        ("auto_quantize", {"method": "awq_lite"}, "awq_lite"),
    ],
    ids=[
        "quantize",
        "calibrate",
        "for-the-weight-scales",
        "max-without-its-sync",
        "for-the-weight-scales-without-their-sync",
        "auto_quantize",
    ],
)
def test_rank_local_calibration_is_refused_before_it_changes_a_distributed_model(
    entry, algorithm, refused
):
    """Each rank calibrates on its own data, and AWQ keeps what its search gathers on the rank: the
    ranks' copies of a weight would come out different. So would local_hessian, run by another
    algorithm for its weight scales, and max or MSE told not to sync, by False in any spelling its
    schema reads as False, or by None, which skips the sync too. The refusal comes before any
    change."""
    model = nn.Sequential(nn.Linear(16, 16))
    set_distribution_plan(model, DistributionPlan(world_size=2))
    if entry == "calibrate":
        mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 16)))
    modules = [type(module) for module in model.modules()]
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    with pytest.raises(NotImplementedError, match=rf"{refused} is not supported .* over 2 ranks"):
        if entry == "quantize":
            config = {**mtq.INT4_AWQ_CFG, "algorithm": algorithm}
            mtq.quantize(model, config, lambda m: m(torch.randn(2, 16)))
        elif entry == "calibrate":
            mtq.calibrate(model, algorithm, lambda m: m(torch.randn(2, 16)))
        else:
            mtq.auto_quantize(
                model,
                constraints={"effective_bits": 8.0},
                quantization_formats=[({**mtq.INT4_AWQ_CFG, "algorithm": algorithm}, "awq")],
                data_loader=[torch.randn(2, 16)],
                forward_step=lambda m, batch: m(batch),
                loss_func=lambda output, batch: output.sum(),
            )

    assert [type(module) for module in model.modules()] == modules
    after = model.state_dict()
    assert after.keys() == state.keys()
    assert all(torch.equal(after[name], tensor) for name, tensor in state.items())


# {module: (data_parallel_group, expert_model_parallel_group)} each layout seeds. With the experts
# sharded over every rank, their expert-parallel group is every rank (None), and no other rank holds
# the same experts (-1: no data-parallel group).
_EXPERT_LAYOUTS = {
    "seeded": {"experts": (-1, None)},
    "without-the-group": {},
    "on-the-router-only": {"gate": (-1, None)},
    "data-parallel-over-every-rank": {"experts": (None, None)},
}


def _expert_parallel_calibration_job(rank, size, layout):
    # A router and a routed expert, named as expert-parallel plans name the experts' container.
    model = nn.Sequential(OrderedDict(gate=nn.Linear(16, 16), experts=nn.Linear(16, 16)))
    for name, (dp, ep) in _EXPERT_LAYOUTS[layout].items():
        model.get_submodule(name)._parallel_state = ParallelState(
            data_parallel_group=dp, expert_model_parallel_group=ep
        )
    set_distribution_plan(model, DistributionPlan(world_size=size, ep_size=size))
    raised = None
    with bounded_collectives():
        # Not pytest.raises: when nothing is raised it fails with a BaseException, which would take
        # the shared worker down instead of failing this test alone.
        try:
            mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 16)))
        except NotImplementedError as e:
            raised = str(e)
    if layout == "seeded":
        assert raised is None, raised
        return
    assert raised is not None and "expert-parallel group" in raised, raised
    # Refused before converting anything, but for unseeded experts beside a seeded router: those
    # are told apart once quantization has converted them, and refused before they calibrate.
    assert (type(model.experts) is nn.Linear) == (layout != "on-the-router-only")


@pytest.mark.parametrize("layout", list(_EXPERT_LAYOUTS))
def test_expert_parallel_calibration_needs_the_experts_expert_parallel_group(
    cpu_dist_workers, layout
):
    """Each rank's per-expert quantizers hold its own experts, and max syncs them over the groups
    their module's ParallelState names, every rank unless set, which would merge different experts'
    scales. Calibration is refused until the experts carry their groups: with no expert-parallel
    group at all, with one on the router alone, and with a data-parallel group spanning ranks that
    hold other experts."""
    cpu_dist_workers.run(_expert_parallel_calibration_job, layout=layout)


def _expert_parallel_auto_quantize_job(rank, size):
    model = nn.Sequential(OrderedDict(gate=nn.Linear(16, 16), experts=nn.Linear(16, 16)))
    dp, ep = _EXPERT_LAYOUTS["seeded"]["experts"]
    model.experts._parallel_state = ParallelState(
        data_parallel_group=dp, expert_model_parallel_group=ep
    )
    set_distribution_plan(model, DistributionPlan(world_size=size, ep_size=size))
    modules = [type(module) for module in model.modules()]
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    raised = None
    with bounded_collectives():
        try:  # not pytest.raises, as in _expert_parallel_calibration_job
            mtq.auto_quantize(
                model,
                constraints={"effective_bits": 8.0},
                quantization_formats=[(mtq.INT8_DEFAULT_CFG, "int8")],
                data_loader=[torch.randn(2, 16)],
                forward_step=lambda m, batch: m(batch),
                loss_func=lambda output, batch: output.sum(),
            )
        except NotImplementedError as e:
            raised = str(e)
    assert raised is not None and f"experts are sharded over {size} ranks" in raised, raised
    assert [type(module) for module in model.modules()] == modules
    after = model.state_dict()
    assert after.keys() == state.keys()
    assert all(torch.equal(after[name], tensor) for name, tensor in state.items())


def test_auto_quantize_refuses_an_expert_parallel_model_before_changing_it(cpu_dist_workers):
    """auto_quantize's forwards would run the experts without the expert-parallel forwards that
    quantization replaces, which distributed_forward_loop restores. So it refuses such a model, with
    its experts seeded and a candidate every rank syncs, before it converts anything."""
    cpu_dist_workers.run(_expert_parallel_auto_quantize_job)


# Rank 0 of four at EP 2, in the (edp, ep) = (2, 2) mesh: ranks 0 and 1 shard the experts between
# them, ranks 2 and 3 do the same, and rank 2 holds the same shard as rank 0. The two-rank pool cannot
# build these groups, so the process group answers as a four-rank launch's would.
_FOUR_RANK_GROUPS = {
    "0,1": [0, 1],
    "1,0": [1, 0],
    "0,2": [0, 2],
    "0,3": [0, 3],
    "all": [0, 1, 2, 3],
}


@pytest.mark.parametrize(
    ("ep", "dp", "seeded"),
    [
        ("0,1", "0,2", True),
        ("0,2", "0,1", False),
        ("0,1", "0,1", False),
        ("0,1", "0,3", False),
        ("1,0", "0,2", False),
        ("all", "0,2", False),
    ],
    ids=[
        "seeded",
        "groups-transposed",
        "data-parallel-over-the-expert-parallel-ranks",
        "data-parallel-over-another-shard",
        "expert-parallel-in-another-order",
        "expert-parallel-over-every-rank",
    ],
)
def test_the_experts_groups_are_checked_by_their_ranks(monkeypatch, ep, dp, seeded):
    """A data-parallel group of the plan's edp ranks is not enough: over the experts'
    expert-parallel ranks, or ranks that meet them only at this one but hold another shard (rank 3
    holds rank 1's), max's and GPTQ's syncs would merge different experts' scales. Nor is an
    expert-parallel group over more ranks than the plan shards over, or in another order, which
    gives this rank another shard, nor the two groups transposed, the mesh's column as the
    expert-parallel group and its row as the data-parallel one, which match in size when edp is
    ep_size."""
    monkeypatch.setattr(mto_dist, "backend", lambda: "torch")
    monkeypatch.setattr(mto_dist, "rank", lambda group=None: 0)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 4 if group is None else len(_FOUR_RANK_GROUPS[group]),
    )
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: _FOUR_RANK_GROUPS[group]
    )
    model = nn.Sequential(OrderedDict(experts=nn.Linear(16, 16)))
    model.experts._parallel_state = ParallelState(
        data_parallel_group=dp, expert_model_parallel_group=ep
    )
    set_distribution_plan(model, DistributionPlan(world_size=4, ep_size=2))

    if seeded:
        _refuse_unsupported_distributed_calibration(model, "max")
    else:
        with pytest.raises(NotImplementedError, match=r"row of the \(2, 2\) expert-parallel mesh"):
            _refuse_unsupported_distributed_calibration(model, "max")


@pytest.mark.parametrize(
    "config",
    [
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": "max"},
        mtq.INT8_DEFAULT_CFG,
    ],
    ids=["all-disabled", "quantized"],
)
def test_a_module_without_routed_experts_calibrates_under_an_expert_parallel_plan(config):
    """hf_ptq quantizes a VLM's vision tower apart from its language model, every quantizer
    disabled. It holds no routed experts, so it has no experts' scales to merge, quantized or not."""
    model = nn.Sequential(nn.Linear(16, 16))
    set_distribution_plan(model, _EXPERT_PARALLEL)

    mtq.quantize(model, config, lambda m: m(torch.randn(2, 16)))
