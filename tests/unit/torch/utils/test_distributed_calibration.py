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
from collections import Counter

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

import modelopt.torch.utils.distributed as mto_dist
from modelopt.torch.utils.distributed import (
    ModelOptSession,
    _reinstall_a2a_forwards,
    distributed_forward_loop,
    modelopt_session,
    shard_dataloader,
)

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


# --------------------------------------------------------------------------------- shard_dataloader


@pytest.mark.parametrize("world_size", [2, 3])
def test_map_style_shards_are_strided_and_cover_every_sample_once(world_size):
    """Rank ``r`` gets samples ``r, r + W, ...``, batched with the original size and collate_fn."""
    n = 12
    shards = [shard_dataloader(_loader(list(range(n))), r, world_size) for r in range(world_size)]

    assert sorted(i for shard in shards for i in _ids(shard)) == list(range(n))
    for r, shard in enumerate(shards):
        ids = list(range(r, n, world_size))
        assert _batches(shard) == [ids[i : i + 2] for i in range(0, len(ids), 2)]


def test_map_style_uneven_split_pads_every_rank_to_the_same_batch_count():
    """No sample is dropped and no rank runs short (it would hang the others' EP collectives).

    The price of ``drop_last=False`` is that the sampler repeats fewer than ``world_size`` samples.
    """
    n, world_size = 10, 4
    shards = [
        _batches(shard_dataloader(_loader(list(range(n))), r, world_size))
        for r in range(world_size)
    ]

    assert [len(batches) for batches in shards] == [2] * world_size  # 3 samples per rank
    seen = Counter(i for batches in shards for batch in batches for i in batch)
    assert set(seen) == set(range(n))
    assert seen.total() - n == 2  # padding up to 3 * world_size samples


def test_iterable_shards_are_strided_cover_every_sample_once_and_replay():
    """Each rank strides its own copy of a stream; a second pass replays the same shard.

    Multi-pass calibration algorithms (e.g. AWQ) iterate the loader again.
    """
    n, world_size = 12, 3
    shards = [shard_dataloader(_loader(_Stream(n)), r, world_size) for r in range(world_size)]

    assert sorted(i for shard in shards for i in _ids(shard)) == list(range(n))
    for r, shard in enumerate(shards):
        assert _batches(shard) == _batches(shard) == [[r, r + 3], [r + 6, r + 9]]


def test_iterable_shards_split_their_groups_across_worker_processes():
    """Each worker process iterates its own copy of the stream. Unless the workers split the groups
    between them, every one yields the whole shard and each sample comes out once per worker."""
    n, world_size = 16, 2
    shards = [
        _ids(
            shard_dataloader(
                DataLoader(_Stream(n), batch_size=2, collate_fn=_collate, num_workers=2),
                r,
                world_size,
            )
        )
        for r in range(world_size)
    ]

    assert [sorted(ids) for ids in shards] == [list(range(r, n, world_size)) for r in range(2)]


def test_iterable_uneven_split_drops_the_trailing_partial_group():
    """Equal counts without ``len()``: the last ``n % world_size`` samples go to no rank."""
    n, world_size = 10, 4
    shards = [_ids(shard_dataloader(_loader(_Stream(n)), r, world_size)) for r in range(world_size)]

    assert shards == [[0, 4], [1, 5], [2, 6], [3, 7]]


@pytest.mark.parametrize("dataset", [list(range(5)), _Stream(5)], ids=["map-style", "iterable"])
def test_world_size_one_yields_the_original_batches(dataset):
    """A lone rank's shard is the whole (unshuffled) loader, batch for batch."""
    loader = _loader(dataset)
    assert _batches(shard_dataloader(loader, 0, 1)) == _batches(loader) == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize("dataset", [list(range(8)), _Stream(8)], ids=["map-style", "iterable"])
def test_sharded_loader_keeps_the_original_loader_settings(dataset):
    """Workers, pinning, timeout, worker init, RNG and prefetching carry over to the shard."""
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
        timeout=7.0,
        worker_init_fn=_seed_worker,
        multiprocessing_context="spawn",
        generator=torch.Generator(),
        prefetch_factor=3,
        persistent_workers=True,
    )
    shard = shard_dataloader(loader, 1, 2)

    for setting in (
        "batch_size",
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
    """The EP device mesh; the TP wrappers only need its process group."""

    def __init__(self):
        self.group = object()

    def get_group(self):
        return self.group


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

    def __init__(self, with_tp_plan=True):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.config = types.SimpleNamespace(use_cache=True)
        self.tp_plan = (
            {
                "layers.*.gate": "torch_a2a_router",
                "layers.*.proj": "colwise",
                "layers.*.experts": "torch_a2a_experts",
            }
            if with_tp_plan
            else None
        )

    def forward(self, input_ids):
        hidden_states = input_ids.float()
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _ep_session(rank=0, *, with_mesh=True):
    """An EP rank's session: its identity plus the EP mesh the model loader recorded."""
    session = ModelOptSession()
    session.initialize(
        distributed=True, rank=rank, world_size=2, local_rank=rank, device=None, is_main=rank == 0
    )
    session.ep_mesh = _FakeMesh() if with_mesh else None
    return session


def _drop_tp_wrapper(module):
    """Do what quantizer insertion does to the instance: rebind the class forward over the wrapper.

    (``DynamicModule.convert`` -> ``bind_forward_method``.)
    """
    module.forward = types.MethodType(type(module).forward, module)


@pytest.mark.parametrize("rank", [0, 1])
def test_reinstall_replants_the_ep_wrapper_on_the_a2a_modules_only(monkeypatch, capsys, rank):
    """Every rank re-wraps exactly the planned torch_a2a experts and routers; rank 0 logs the count.

    Other TP styles keep their forward, and so do the quantizers under an experts module: the plan
    is looked up per module (``is_weight=False``), so children do not inherit their parent's style.
    """
    styles = _install_fake_tp(monkeypatch)
    model, session = _MoEModel(), _ep_session(rank)
    hidden_states = torch.ones(2, 4)
    with pytest.raises(ValueError, match="process_group"):
        model(hidden_states)  # as quantizer insertion leaves it: no wrapper, no EP group

    assert _reinstall_a2a_forwards(model, session) == 4

    assert styles["torch_a2a_experts"].installed == [layer.experts for layer in model.layers]
    assert styles["torch_a2a_router"].installed == [layer.gate for layer in model.layers]
    assert styles["colwise"].installed == []
    model(hidden_states)
    assert [layer.experts.groups for layer in model.layers] == [[session.ep_mesh.group]] * 2
    assert ("re-installed 4" in capsys.readouterr().out) == (rank == 0)


def test_reinstall_does_not_stack_wrappers_across_calibration_passes(monkeypatch):
    """A repeat call is a no-op; a second wrapper would pass the experts the EP group twice."""
    styles = _install_fake_tp(monkeypatch)
    model, session = _MoEModel(), _ep_session()
    assert _reinstall_a2a_forwards(model, session) == 4

    assert _reinstall_a2a_forwards(model, session) == 0
    assert len(styles["torch_a2a_experts"].installed) == 2
    model(torch.ones(2, 4))
    assert [layer.experts.groups for layer in model.layers] == [[session.ep_mesh.group]] * 2


@pytest.mark.parametrize(
    ("has_session", "has_ep_mesh", "has_tp_plan"),
    [(False, True, True), (True, False, True), (True, True, False)],
    ids=["no-session", "no-ep-mesh", "no-tp-plan"],
)
def test_reinstall_is_a_noop_off_the_expert_parallel_path(
    monkeypatch, has_session, has_ep_mesh, has_tp_plan
):
    """A single-process or non-EP model is left untouched, even with the torch_a2a styles around."""
    styles = _install_fake_tp(monkeypatch)
    model = _MoEModel(with_tp_plan=has_tp_plan)
    session = _ep_session(with_mesh=has_ep_mesh) if has_session else None

    assert _reinstall_a2a_forwards(model, session) == 0
    assert not any(style.installed for style in styles.values())


def test_reinstall_is_a_noop_where_tp_styles_are_pre_hooks(monkeypatch):
    """transformers < 5.16 has no ``install_forward``; its pre-hooks survive quantizer insertion.

    Those releases keep the styles in ``transformers.integrations``: the fallback import.
    """
    _install_fake_tp(
        monkeypatch,
        styles={"torch_a2a_experts": _PreHookStyle(), "torch_a2a_router": _PreHookStyle()},
        where=_LEGACY_TP_MODULE,
    )
    assert _reinstall_a2a_forwards(_MoEModel(), _ep_session()) == 0


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


@pytest.mark.parametrize("dataset", [list(range(8)), _Stream(8)], ids=["map-style", "iterable"])
@pytest.mark.parametrize("rank", [0, 1])
def test_forward_loop_runs_this_ranks_shard_through_the_captured_model(dataset, rank):
    """Every batch of the rank's shard reaches the top-level model, not the module ``mtq`` passes.

    Only the full model's forward runs the EP collectives; no session means no device move.
    """
    model, inner = _Recorder(), _Recorder()
    shard = shard_dataloader(_loader(dataset), rank, world_size=2)

    distributed_forward_loop(model, shard)(inner)

    calls = [call["input_ids"].flatten().tolist() for call in model.calls]
    assert calls == [[rank, rank + 2], [rank + 4, rank + 6]]
    assert inner.calls == []


@pytest.mark.parametrize(("device", "expected"), [(None, "cpu"), ("meta", "meta")])
def test_forward_loop_moves_tensors_to_the_single_process_session_device(
    monkeypatch, device, expected
):
    """Batch tensors go to ``mo_session.device`` when it names one; other values pass through.

    The meta device makes the move observable on a CPU-only host.
    """
    monkeypatch.delenv("WORLD_SIZE", raising=False)  # single process: no process group
    batches = [{"input_ids": torch.zeros(2, 3), "logits_to_keep": 1}] * 2
    model = _Recorder()
    with modelopt_session(device=device) as session:
        distributed_forward_loop(model, batches, session)(model)

    assert [call["input_ids"].device.type for call in model.calls] == [expected] * 2
    assert [call["logits_to_keep"] for call in model.calls] == [1, 1]


def test_forward_loop_disables_use_cache_only_while_calibrating():
    """KV/state caching is off for the calibration forwards (it breaks hybrid Mamba models)."""
    model = _Recorder()
    distributed_forward_loop(model, [{"input_ids": torch.zeros(1, 1)}] * 2)(model)

    assert model.use_cache_seen == [False, False]
    assert model.config.use_cache is True


def test_forward_loop_frees_the_allocator_cache_around_the_forwards(monkeypatch):
    """Re-plant the a2a wrappers and free the cached pool before the first forward, and again after.

    Building the loop does neither: quantizers are inserted between building and calling it. NCCL
    allocates its buffers outside torch's caching allocator, so a pool hoarded by the model load or
    by the calibration activations starves the next collective.
    """
    events = []
    monkeypatch.setattr(
        mto_dist, "_reinstall_a2a_forwards", lambda *args: events.append(("reinstall", *args))
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    model, session = _Recorder(on_forward=lambda: events.append("forward")), _ep_session()
    forward_loop = distributed_forward_loop(model, [{"input_ids": torch.zeros(1, 1)}] * 2, session)
    assert events == []

    forward_loop(model)

    assert events == [
        ("reinstall", model, session),
        "empty_cache",
        "forward",
        "forward",
        "empty_cache",
    ]


def test_forward_loop_replants_a2a_wrappers_dropped_after_it_was_built(monkeypatch):
    """hf_ptq's order -- build the loop, insert quantizers, calibrate -- hands the experts the EP group.

    It holds on every pass (multi-pass algorithms call the loop again), with one wrapper per module.
    """
    styles = _install_fake_tp(monkeypatch)
    model, session = _MoEModel(), _ep_session()
    forward_loop = distributed_forward_loop(model, [{"input_ids": torch.ones(2, 4)}] * 3, session)
    for layer in model.layers:
        _drop_tp_wrapper(layer.experts)

    forward_loop(model)
    forward_loop(model)

    assert [layer.experts.groups for layer in model.layers] == [[session.ep_mesh.group] * 6] * 2
    assert len(styles["torch_a2a_experts"].installed) == 2
