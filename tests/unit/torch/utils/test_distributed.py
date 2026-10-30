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

"""Tests for ``modelopt.torch.utils.distributed``: its pure functions, the distribution plan and a
model's record of it, the collectives that fail together, and the process group's lifetime."""

import os
import pickle
import sys
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from unittest import mock

import pytest
import torch
import torch.distributed
import torch.nn as nn
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

import modelopt.torch.utils.distributed as dist
from modelopt.torch.nas.hparams.container import DepthHparam
from modelopt.torch.utils.distributed import (
    DistributedState,
    DistributionPlan,
    _deserialize,
    _serialize,
    all_reduce,
    broadcast_from,
    distributed_run,
    distribution_plan,
    set_distribution_plan,
)


class _Gadget:
    def __reduce__(self):
        return (os.system, ("exit 1",))


@pytest.mark.parametrize("force_no_weights_only", [False, True])
def test_deserialize_rejects_pickle_gadget(monkeypatch, force_no_weights_only):
    # The override that trusts local checkpoints must not also trust peer payloads.
    if force_no_weights_only:
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(pickle.UnpicklingError):
        _deserialize(_serialize(_Gadget()))


def test_deserialize_traced_hp_subclass():
    # The allowlist matches exact classes, so each TracedHp subclass must be registered too.
    hp = _deserialize(_serialize(DepthHparam([1, 2])))
    assert type(hp) is DepthHparam
    assert hp.choices == [1, 2]


# --------------------------------------------------------------------------------- DistributionPlan


@pytest.mark.parametrize(
    ("degrees", "match"),
    [
        ({"world_size": 8, "ep_size": 3}, "must divide world_size"),
        ({"world_size": 0}, "must be at least 1"),
        ({"world_size": 4, "ep_size": -2}, "must be at least 1"),
        ({"world_size": 8, "dense": "sharded"}, "expected 'replicate' or 'fsdp'"),
    ],
    ids=["ep-not-dividing", "world-0", "ep-negative", "unknown-dense"],
)
def test_an_invalid_plan_is_rejected(degrees, match):
    """A plan no load could carry out fails at construction."""
    with pytest.raises(ValueError, match=match):
        DistributionPlan(**degrees)


# ------------------------------------------------------------------- a model's recorded plan


def _plan_model():
    return nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))


@pytest.mark.parametrize(
    ("recorded_on", "again"),
    [
        ("model", DistributionPlan(world_size=8, ep_size=8)),
        ("model", DistributionPlan(world_size=8, dense="fsdp")),
        ("submodule", DistributionPlan(world_size=8)),
    ],
    ids=["reload", "redistribute", "submodule"],
)
def test_a_model_records_its_plan_once(recorded_on, again):
    """Every submodule reads the recorded plan, and a second record is refused, recording nothing."""
    model, plan = _plan_model(), DistributionPlan(world_size=8, ep_size=8)
    assert distribution_plan(model) == DistributionPlan()  # no plan: a single process
    holder = model if recorded_on == "model" else model[1]
    set_distribution_plan(holder, plan)
    holders = list(holder.modules())
    assert all(distribution_plan(module) is plan for module in holders)

    with pytest.raises(ValueError, match="its plan recorded once"):
        set_distribution_plan(model, again)
    for module in model.modules():
        assert distribution_plan(module) == (plan if module in holders else DistributionPlan())


def _outcome(call):
    """``(value, None)``, or ``(None, error)`` if ``call()`` raises, a ``SystemExit`` included."""
    try:
        return call(), None
    except (Exception, SystemExit) as error:
        return None, error


def _no_plan_on_several_ranks_job(rank, size):
    model = _plan_model()
    _, error = _outcome(lambda: set_distribution_plan(model, DistributionPlan()))
    assert isinstance(error, ValueError), error
    _, error = _outcome(lambda: distribution_plan(model))
    assert isinstance(error, RuntimeError), error
    assert f"is held in a run of {size} ranks" in str(error), error
    model = _plan_model()
    fully_shard(model, mesh=init_device_mesh("cpu", (size,)))
    _, error = _outcome(lambda: distribution_plan(model))
    assert isinstance(error, RuntimeError), error
    assert "is FSDP2-sharded" in str(error), error
    plan = DistributionPlan(world_size=size, dense="fsdp")
    set_distribution_plan(model, plan)
    assert distribution_plan(model) is plan
    model = _plan_model()
    set_distribution_plan(model, DistributionPlan(world_size=size))
    fully_shard(model, mesh=init_device_mesh("cpu", (size,)))
    _, error = _outcome(lambda: distribution_plan(model))
    assert isinstance(error, RuntimeError), error
    assert "sharded after its plan was recorded" in str(error), error
    model = _plan_model()
    set_distribution_plan(model, DistributionPlan(world_size=size))
    model.add_module("added", nn.Linear(2, 2))
    _, error = _outcome(lambda: distribution_plan(model.added))
    assert isinstance(error, RuntimeError), error


def test_a_model_with_no_plan_raises_in_a_run_of_several_ranks(cpu_dist_workers):
    """On two ranks a model needs a recorded plan, whole, FSDP2-sharded or added to later, and one
    sharded after its plan was recorded is refused rather than read as replicated."""
    cpu_dist_workers.run(_no_plan_on_several_ranks_job)


class _Decoder(nn.Module):
    """Decoder layers of plain PyTorch, which no Hugging Face plugin knows."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


# As an install without the hf extra has them: importing any of them fails.
_HUGGING_FACE = ("transformers", "huggingface_hub", "accelerate")


def _wraps_without_hugging_face_job(rank, size):
    # Imported while the Hugging Face packages are there, as the pooled worker keeps them.
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
    from modelopt.torch.utils import fsdp2_utils

    model, mesh = _Decoder(), init_device_mesh("cpu", (size,))
    hidden = (*_HUGGING_FACE, "modelopt.torch.utils.plugins.model_load_utils")
    saved = {name: sys.modules.pop(name) for name in hidden if name in sys.modules}
    sys.modules.update(dict.fromkeys(_HUGGING_FACE))
    try:
        with (
            mock.patch.object(LayerActivationCollector, "_decoder_layer_support", []),
            # A CPU mesh, whatever accelerator the host has.
            mock.patch.object(fsdp2_utils, "fully_shard", partial(fully_shard, mesh=mesh)),
        ):
            LayerActivationCollector.register_decoder_layer_support(
                lambda model: isinstance(model, _Decoder), lambda model: model.layers
            )
            layers = dist.fsdp2_wrap(model, shard_root=False)
    finally:
        for name in hidden:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    assert layers is model.layers
    assert all(isinstance(layer, FSDPModule) for layer in layers)
    assert not isinstance(model, FSDPModule)


def test_fsdp2_wrap_needs_no_hugging_face(cpu_dist_workers):
    """The old path wraps a model whose decoder layers are registered, with no Hugging Face package
    to import and the loaders unimportable, as it did before fsdp2_wrap moved."""
    cpu_dist_workers.run(_wraps_without_hugging_face_job)


# ------------------------------------------------------------------------------------ collectives


def test_a_single_process_computes_alone(monkeypatch):
    """Nothing to sync with: no collective."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    for collective in ("broadcast_object_list", "all_gather_object"):
        monkeypatch.setattr(
            torch.distributed, collective, lambda *a, **k: pytest.fail("a collective ran")
        )
    assert broadcast_from("x", lambda: 3) == 3
    assert all_reduce("smallest", lambda: 4, min) == 4
    with pytest.raises(ValueError, match="src=1"):
        broadcast_from("x", lambda: 3, src=1)


def test_a_launch_without_its_group_refuses_the_collectives(monkeypatch):
    """A rank of a launch whose group is not set up raises instead of computing its own value."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    ran = []
    for collective in (
        lambda: broadcast_from("x", lambda: ran.append("broadcast_from")),
        lambda: all_reduce("x", lambda: ran.append("all_reduce"), min),
    ):
        with pytest.raises(RuntimeError, match="'x' needs this 2-rank launch's process group"):
            collective()
    assert ran == []


class _Protocol2Only:
    """Pickles at protocol 2, as torch.save does, and refuses plain pickle's higher default."""

    def __init__(self, n):
        self.n = n

    def __reduce_ex__(self, protocol):
        if protocol > 2:
            raise TypeError("protocol 2 only")
        return _Protocol2Only, (self.n,)


# The collectives load what the ranks send with weights_only.
torch.serialization.add_safe_globals([_Protocol2Only])


def _refuse_to_load():
    raise RuntimeError("this value cannot be loaded")


class _Unloadable:
    """Pickles, and is refused when loaded back: weights_only runs no callable it was not given."""

    def __reduce__(self):
        return _refuse_to_load, ()


class _UnprintableError(Exception):
    """An error whose text cannot be formatted."""

    def __str__(self):
        raise ValueError("no text")


def _raise_unprintable():
    raise _UnprintableError


@pytest.mark.parametrize(
    ("value", "error"),
    [((n for n in range(1)), TypeError), (_Unloadable(), pickle.UnpicklingError)],
    ids=["does-not-pickle", "does-not-load"],
)
def test_the_collectives_encoding_refuses_a_value_that_does_not_round_trip(value, error):
    """Both collectives encode through it, inside the guard that tells every rank."""
    with pytest.raises(error):
        dist._encoded(value)


def test_the_collectives_carry_a_plan():
    """The one class they carry is registered, so it loads back under weights_only."""
    plan = DistributionPlan(world_size=8, ep_size=4, dense="fsdp")
    assert dist._decoded(dist._encoded(plan)) == plan


def _broadcast_from_job(rank, size):
    ran = []

    def compute():
        ran.append(rank)
        return f"rank {rank}'s"

    def fail():
        raise ValueError("bad degrees")

    def exit_2():
        raise SystemExit(2)

    with bounded_collectives():
        for sender in range(size):
            ran.clear()
            assert broadcast_from("x", compute, src=sender) == f"rank {sender}'s"
            assert ran == ([rank] if rank == sender else []), "a rank that does not send ran fn"
        _, error = _outcome(lambda: broadcast_from("x", compute, src=size))
        assert isinstance(error, ValueError), error
        assert f"src={size} is not a rank of this {size}-rank run" in str(error), error
        _, error = _outcome(lambda: broadcast_from("x", fail))
        if rank == 0:
            assert isinstance(error, ValueError), error
        else:
            assert isinstance(error, RuntimeError), error
            assert "rank 0 failed to compute x" in str(error), error
        _, error = _outcome(lambda: broadcast_from("x", exit_2))
        assert isinstance(error, SystemExit) and error.code == 2, error
        # Serialized once, where the sending rank catches a failure, not again by the broadcast.
        assert broadcast_from("x", lambda: _Protocol2Only(7)).n == 7
        # Every rank gets the value loaded from the bytes sent, the sending rank too.
        built: list = []
        got = broadcast_from("x", lambda: built)
        assert got == [] and got is not built
        # A value that does not round-trip fails on the sending rank, and reaches every rank.
        _, error = _outcome(lambda: broadcast_from("x", _Unloadable))
        if rank == 0:
            assert isinstance(error, pickle.UnpicklingError), error
        else:
            assert "rank 0 failed to compute x" in str(error), error
        sent, error = _outcome(lambda: broadcast_from("plan" if rank == 0 else "other", lambda: 7))
        if rank == 0:
            assert (sent, error) == (7, None)
        else:
            assert isinstance(error, RuntimeError), error
            assert "out of step" in str(error), error


def test_broadcast_from_gives_every_rank_the_sending_ranks_value(cpu_dist_workers):
    """Over a real broadcast, on two ranks: failures, exits and ranks out of step reach every rank."""
    cpu_dist_workers.run(_broadcast_from_job)


def _all_reduce_job(rank, size):
    def fail_on_rank_1():
        if rank == 1:
            raise ValueError("no memory reading")
        return 4

    def exit_on_rank_1():
        if rank == 1:
            raise SystemExit(3)
        return 4

    def exit_on_rank_0_fail_on_rank_1():
        raise SystemExit(3) if rank == 0 else ValueError("no memory reading")

    with bounded_collectives():
        assert all_reduce("smallest", lambda: [8, 2][rank], min) == 2
        _, error = _outcome(
            lambda: all_reduce("smallest", _raise_unprintable if rank == 1 else lambda: 4, min)
        )
        if rank == 1:
            assert isinstance(error, _UnprintableError), type(error)
        else:
            assert "rank 1 failed to compute smallest: _UnprintableError" in str(error), error
        for fail in (fail_on_rank_1, exit_on_rank_0_fail_on_rank_1):
            _, error = _outcome(lambda: all_reduce("smallest", fail, min))
            if rank == 1:
                assert isinstance(error, ValueError), error
            else:
                assert isinstance(error, RuntimeError), error
                assert "rank 1 failed to compute smallest: ValueError" in str(error), error
        _, error = _outcome(lambda: all_reduce("smallest", exit_on_rank_1, min))
        assert isinstance(error, SystemExit) and error.code == 3, error
        _, error = _outcome(
            lambda: all_reduce("smallest", lambda: (n for n in range(1)) if rank == 1 else 4, min)
        )
        if rank == 1:
            assert isinstance(error, TypeError), error
        else:
            assert isinstance(error, RuntimeError), error
            assert "rank 1 failed to compute smallest: TypeError" in str(error), error
        _, error = _outcome(
            lambda: all_reduce("smallest" if rank == 0 else "other", lambda: 4, min)
        )
        assert isinstance(error, RuntimeError), error
        assert "out of step" in str(error), error


def test_all_reduce_gives_every_rank_the_combined_value(cpu_dist_workers):
    """Over a real gather, on two ranks: one rank's failure or exit, and ranks out of step, reach all."""
    cpu_dist_workers.run(_all_reduce_job)


# ---------------------------------------------------------------------------------- process group


@pytest.mark.parametrize("callers_group", [False, True], ids=["no-group", "callers-one-rank-group"])
def test_a_single_process_is_rank_0_of_one_on_the_given_device(monkeypatch, recwarn, callers_group):
    """No ranks to sync with: no group set up or torn down, and a timeout taken quietly."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if callers_group:
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "rank", lambda group=None: 0)
        monkeypatch.setattr(dist, "size", lambda group=None: 1)
    monkeypatch.setattr(dist, "setup", lambda **kwargs: pytest.fail("set a group up"))
    monkeypatch.setattr(dist, "_destroy_group", lambda exc: pytest.fail("tore a group down"))
    with distributed_run(timeout=timedelta(hours=3), device="cpu") as state:
        assert state == DistributedState(rank=0, world_size=1, device=torch.device("cpu"))
        assert state.is_main
    assert not [w for w in recwarn if "does not apply" in str(w.message)]


@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_a_distributed_launch_refuses_another_device(monkeypatch, device):
    """A launch runs each rank on the CUDA device LOCAL_RANK names, and refuses any other."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "setup", lambda **kwargs: pytest.fail("set a group up"))
    with pytest.raises(ValueError, match=f"device='{device}'"), distributed_run(device=device):
        pass


class _ExitedError(Exception):
    """Raised by a stand-in for abort(), which exits the process instead of returning."""


@pytest.mark.parametrize(
    ("launch", "body_raises", "ends_by", "in_except"),
    [
        (True, None, ["barrier", "destroy"], False),
        (True, ValueError, ["abort"], False),
        (True, _UnprintableError, ["abort"], False),
        (True, SystemExit, ["barrier", "destroy"], False),
        (False, None, ["barrier", "destroy"], False),
        (False, ValueError, ["destroy"], False),
        (True, None, ["barrier", "destroy"], True),
        (False, ValueError, ["destroy"], True),
    ],
    ids=[
        "launch",
        "launch-raises",
        "launch-raises-unprintable",
        "launch-exits",
        "single-process",
        "single-process-raises",
        "launch-entered-in-except",
        "single-process-raises-entered-in-except",
    ],
)
def test_the_run_ends_the_group_it_set_up(
    monkeypatch, capsys, launch, body_raises, ends_by, in_except
):
    """The group the run set up ends with a barrier and destroy, or with abort() if a rank raised."""
    # Under a launch, rank 2 of 4 on local rank 1: the device follows the local rank, not the global.
    this_rank, world = (2, 4) if launch else (0, 1)
    if launch:
        monkeypatch.setenv("WORLD_SIZE", str(world))
    else:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    live, ended, timeouts = [], [], []

    def setup(timeout=None):
        timeouts.append(timeout)
        live.append("the launch's group")

    monkeypatch.setattr(dist, "setup", setup)
    monkeypatch.setattr(dist, "is_initialized", lambda: bool(live))
    monkeypatch.setattr(dist, "rank", lambda group=None: this_rank)
    monkeypatch.setattr(dist, "size", lambda group=None: world)
    monkeypatch.setattr(dist, "local_rank", lambda: 1)
    monkeypatch.setattr(dist, "barrier", lambda group=None: ended.append("barrier"))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: ended.append("destroy"))

    def abort():
        ended.append("abort")
        raise _ExitedError

    monkeypatch.setattr(dist, "abort", abort)
    escapes = _ExitedError if ends_by == ["abort"] else body_raises
    expected = pytest.raises(escapes) if escapes else nullcontext()

    states = []

    def run():
        with (
            expected,
            distributed_run(
                timeout=timedelta(hours=1), device="cuda" if launch else "cpu"
            ) as state,
        ):
            states.append(state)
            if not launch:
                live.append("the body's group")
            if body_raises:
                raise body_raises

    if in_except:
        try:
            raise KeyError("the caller's")
        except KeyError:
            run()
    else:
        run()

    assert ended == ends_by
    assert timeouts == ([timedelta(hours=1)] if launch else [])
    device = torch.device("cuda:1" if launch else "cpu")
    assert states == [DistributedState(rank=this_rank, world_size=world, device=device)]
    assert states[0].is_main is not launch
    # An intentional exit is not reported as an error. An error is reported once, under this rank's
    # place in the group; its traceback is abort()'s to print, or goes on with it to the caller.
    out, err = capsys.readouterr()
    assert (f"[distributed_run][rank {this_rank}] body raised" in out) == (
        body_raises not in (None, SystemExit)
    )
    assert "Traceback" not in err


@pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
@pytest.mark.parametrize("world_size", [None, "2"], ids=["single-process", "launch"])
def test_the_run_leaves_a_group_it_found_alone(monkeypatch, recwarn, world_size, raises):
    """A group the caller set up stays, the state is the rank's place in it, and a timeout warns."""
    bound: list[int] = []
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
        monkeypatch.setattr(
            torch.distributed, "init_process_group", lambda *a, **k: pytest.fail("set it up again")
        )
        monkeypatch.setattr(torch.cuda, "set_device", bound.append)
        monkeypatch.setattr(dist, "local_rank", lambda: 1)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(
        dist, "_destroy_group", lambda exc: pytest.fail("destroyed the caller's group")
    )
    monkeypatch.setattr(dist, "abort", lambda: pytest.fail("exited the caller's process"))
    expected = pytest.raises(ValueError, match="calibration failed") if raises else nullcontext()
    with (
        expected,
        distributed_run(
            timeout=timedelta(hours=1), device="cpu" if world_size is None else "cuda"
        ) as state,
    ):
        assert (state.rank, state.world_size) == (1, 2)
        assert state.device == torch.device("cpu" if world_size is None else "cuda:1")
        assert bound == ([] if world_size is None else [1])
        if raises:
            raise ValueError("calibration failed")
    assert "timeout=1:00:00) does not apply" in str(recwarn.pop(UserWarning).message)
