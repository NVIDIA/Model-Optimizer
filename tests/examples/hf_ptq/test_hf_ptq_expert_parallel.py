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

"""Expert-parallel PTQ wiring in ``examples/hf_ptq``.

Under torchrun, hf_ptq.py applies the model's spec to its config, decides a distribution plan from
the ``MODELOPT_EP_DEGREE`` / ``MODELOPT_FSDP_SIZE`` knobs, and loads the model through
``load_and_prepare_ep`` when that plan shards the experts. Every rank then calibrates on its own
shard of the data, and a dense model is pinned to the rank's own GPU instead of ``"auto"``.

Nothing here needs a process group or a checkpoint: an initialized ``ModelOptSession`` stands in
for the torchrun session, the configs are tiny real transformers configs served from memory, and
the loaders are replaced by stand-ins that record how they were called.
"""

import contextlib
import importlib
import inspect
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from _test_utils.examples.hf_ptq_example_utils import example_utils
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from packaging.version import Version
from torch.utils.data import DataLoader
from transformers import DeepseekV3Config, LlamaConfig, Qwen3MoeConfig

from modelopt.torch.models import get_spec
from modelopt.torch.utils import env
from modelopt.torch.utils.distributed import ModelOptSession
from modelopt.torch.utils.moe import is_expert_parallel_capable, load_and_prepare_ep

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"

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

_CKPT = "/ckpt/model"


class _LoaderReachedError(Exception):
    """Raised by a loader stand-in once the call under test has been recorded."""


class _RunFailedError(Exception):
    """Raised by a stand-in for the body of a run that fails."""


@pytest.fixture
def hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


@pytest.fixture(autouse=True)
def _no_distribution_knobs(monkeypatch):
    """A shell that exports the knobs would otherwise steer every decision below."""
    monkeypatch.delenv("MODELOPT_EP_DEGREE", raising=False)
    monkeypatch.delenv("MODELOPT_FSDP_SIZE", raising=False)


def _session(world_size, rank=0, distributed=True):
    """The session modelopt_session() yields on ``rank`` of a ``world_size``-rank launch."""
    session = ModelOptSession()
    session.initialize(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        device=torch.device(f"cuda:{rank}"),
        is_main=rank == 0,
    )
    return session


def _args(session, **overrides):
    """What load_model reads before it picks a loader, as main() leaves it."""
    args = SimpleNamespace(
        mo_session=session,
        use_fsdp2=False,
        recipe=None,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
        attn_implementation=None,
    )
    vars(args).update(overrides)
    return args


def _serve(monkeypatch, hf_ptq, config):
    """Answer hf_ptq's config read with ``config`` rather than a checkpoint on disk."""
    monkeypatch.setattr(hf_ptq.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    return config


def _qwen3_moe(num_experts):
    """Qwen3-MoE, whose base_model_ep_plan ships with transformers itself."""
    return Qwen3MoeConfig(num_experts=num_experts, moe_intermediate_size=8, **_TINY)


def _deepseek_v3(n_routed_experts):
    """DeepSeek-V3 without a plan of its own, so only its model spec can make it EP-capable."""
    config = DeepseekV3Config(n_routed_experts=n_routed_experts, **_TINY)
    config.base_model_ep_plan = None  # holds even if a later transformers ships a plan
    return config


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


@pytest.mark.parametrize("single_process_session", [False, True])
def test_a_single_process_run_is_never_expert_parallel(hf_ptq, monkeypatch, single_process_session):
    """With no distributed session there is nothing to shard over, even for an EP-capable MoE
    (which, decided at world_size=1, would otherwise be rejected as an unsharded MoE)."""
    session = _session(world_size=1, distributed=False) if single_process_session else None
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))

    assert hf_ptq._decide_expert_parallel(_args(session)) == (None, None)


def test_use_fsdp2_keeps_the_standalone_fsdp2_loader(hf_ptq, monkeypatch):
    """--use_fsdp2 names its loader, so even an explicit EP degree for an EP-capable MoE must not
    divert the run to the expert-parallel one."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    args = _args(_session(world_size=2), use_fsdp2=True)

    assert hf_ptq._decide_expert_parallel(args) == (None, None)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"),
    reason="qwen3_moe's base_model_ep_plan ships in transformers 5.7",
)
@pytest.mark.parametrize("rank", [0, 1])
def test_an_ep_capable_moe_is_planned_expert_parallel(hf_ptq, monkeypatch, capsys, rank):
    """Auto picks the largest EP degree dividing both the world and the expert count, hands back a
    config the EP loader can shard, and only rank 0 reports the decision."""
    _serve(monkeypatch, hf_ptq, _qwen3_moe(num_experts=8))

    plan, hf_config = hf_ptq._decide_expert_parallel(_args(_session(world_size=4, rank=rank)))

    assert (plan.world_size, plan.ep_size, plan.edp) == (4, 4, 1)
    assert not plan.is_fsdp_dense
    assert is_expert_parallel_capable(hf_config)
    assert ("[ep] decided" in capsys.readouterr().out) == (rank == 0)


def test_the_spec_is_applied_before_the_decision(hf_ptq, monkeypatch):
    """DeepSeek-V3's plan comes only from its spec. Deciding before applying it would read the MoE
    as dense; applied first, it shards, and the config handed on carries the spec's plan."""
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))

    plan, hf_config = hf_ptq._decide_expert_parallel(_args(_session(world_size=4)))

    assert plan.ep_size == 4
    assert hf_config.base_model_ep_plan == get_spec("deepseek_v3").ep_spec.base_model_ep_plan


@pytest.mark.parametrize(
    ("ep_degree", "experts"),
    [(None, 3), ("1", 8)],
    ids=["auto_finds_no_degree", "explicit_degree_one"],
)
def test_an_ep_capable_moe_left_unsharded_is_an_error(hf_ptq, monkeypatch, ep_degree, experts):
    """No EP degree above 1 divides 3 experts over 2 ranks, and an explicit 1 asks for none. With
    the dense part replicated, every rank would hold the whole MoE: that must fail, not run."""
    if ep_degree is not None:
        monkeypatch.setenv("MODELOPT_EP_DEGREE", ep_degree)
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=experts))

    with pytest.raises(RuntimeError, match="can be expert-parallel, but the plan resolved"):
        hf_ptq._decide_expert_parallel(_args(_session(world_size=2)))


@pytest.mark.parametrize(
    ("config", "ep_capable"),
    [(_deepseek_v3(n_routed_experts=3), True), (LlamaConfig(**_TINY), False)],
    ids=["moe_no_ep_degree", "dense"],
)
def test_dense_sharding_without_expert_parallelism_is_an_error(
    hf_ptq, monkeypatch, config, ep_capable
):
    """Only the EP loader shards a dense part by itself. Without EP the run would load the whole
    model on every rank and drop the sharding the plan asked for, so it must fail and point at the
    loader that can shard it."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    _serve(monkeypatch, hf_ptq, config)

    with pytest.raises(RuntimeError, match="--use_fsdp2") as raised:
        hf_ptq._decide_expert_parallel(_args(_session(world_size=2)))
    assert ("MODELOPT_EP_DEGREE" in str(raised.value)) == ep_capable


def test_a_dense_model_under_torchrun_runs_data_parallel(hf_ptq, monkeypatch):
    """A dense model has no experts to shard, so replicating it per rank is not an error."""
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))

    assert hf_ptq._decide_expert_parallel(_args(_session(world_size=4))) == (None, None)


def test_the_env_knobs_override_the_automatic_plan(hf_ptq, monkeypatch):
    """The decision honours both knobs; left on auto it would pick ep_size=4, dense replicated."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))

    plan, _ = hf_ptq._decide_expert_parallel(_args(_session(world_size=4)))

    assert (plan.ep_size, plan.edp, plan.fsdp_size) == (2, 2, 4)


# ---------------------------------------------------------------------------------- load_model


def test_load_model_hands_an_expert_parallel_plan_to_the_ep_loader(hf_ptq, monkeypatch):
    """With an EP plan, load_model loads through load_and_prepare_ep -- with the session, the
    decided degree and dense sharding, and the spec-prepared config -- and through no other
    loader."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    served = _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    session = _session(world_size=4)
    calls = []

    def load_and_prepare_ep_stub(*args, **kwargs):
        calls.append(inspect.signature(load_and_prepare_ep).bind(*args, **kwargs).arguments)
        raise _LoaderReachedError

    monkeypatch.setattr(hf_ptq, "load_and_prepare_ep", load_and_prepare_ep_stub)
    monkeypatch.setattr(hf_ptq, "get_model", lambda *a, **k: pytest.fail("dense loader ran"))
    monkeypatch.setattr(
        hf_ptq, "parallel_load_and_prepare_fsdp2", lambda *a, **k: pytest.fail("FSDP2 loader ran")
    )

    with pytest.raises(_LoaderReachedError):
        hf_ptq.load_model(_args(session, trust_remote_code=True, attn_implementation="eager"))

    (call,) = calls
    assert call["ckpt_path"] == _CKPT
    assert call["mo_session"] is session
    assert call["ep_degree"] == 4
    assert call["fsdp_dense"] is True
    assert call["trust_remote_code"] is True
    assert call["attn_implementation"] == "eager"
    assert call["hf_config"] is served
    assert served.base_model_ep_plan == get_spec("deepseek_v3").ep_spec.base_model_ep_plan


# ------------------------------------------------------------------------- per-rank placement


@pytest.mark.parametrize(
    ("device", "world_size", "device_map"),
    [
        ("cuda:1", 4, {"": "cuda:1"}),
        (torch.device("cuda:1"), 4, {"": "cuda:1"}),
        ("cuda", 4, "auto"),
        ("cuda:1", 1, "auto"),
    ],
    ids=["str", "torch_device", "unindexed", "single_process"],
)
def test_a_rank_device_pins_the_whole_model_to_it(
    monkeypatch, tmp_path, device, world_size, device_map
):
    """Under torchrun each rank asks for its own cuda:<local_rank>; "auto" would let every rank's
    accelerate pick the same idle GPU. An unindexed "cuda" still spreads the model over the GPUs,
    and so does a single process, whatever device it names."""
    LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY).save_pretrained(tmp_path)
    placed = []

    def from_pretrained(auto_class, ckpt_path, **kwargs):
        placed.append(kwargs["device_map"])
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(example_utils, "_from_pretrained_recording", from_pretrained)
    monkeypatch.setattr(example_utils.dist_utils, "size", lambda group=None: world_size)

    example_utils.get_model(str(tmp_path), device=device)

    assert placed == [device_map]


# -------------------------------------------------------------------- session and dist_state


@pytest.mark.parametrize("use_fsdp2", [False, True])
def test_dist_state_mirrors_the_distributed_session(monkeypatch, use_fsdp2):
    """Under torchrun dist_state is read from the session that owns the process group, even with
    --use_fsdp2, whose own branch would initialize a second group."""
    monkeypatch.setattr(
        example_utils.dist_utils, "setup", lambda *a, **k: pytest.fail("second process group")
    )
    session = _session(world_size=4, rank=1)
    args = SimpleNamespace(mo_session=session, use_fsdp2=use_fsdp2, device="cuda")

    example_utils.setup_distributed_args(args)

    assert args.dist_state == example_utils.DistributedState(
        rank=1, world_size=4, device=torch.device("cuda:1"), is_main=False
    )


def test_a_single_process_session_leaves_fsdp2_to_set_up_its_group(monkeypatch):
    """A one-rank --use_fsdp2 launch enters a session that owns no process group, so the FSDP2
    branch must still create one."""
    setups = []
    monkeypatch.setattr(example_utils.dist_utils, "setup", lambda timeout=None: setups.append(0))
    monkeypatch.setattr(example_utils.dist_utils, "rank", lambda group=None: 0)
    monkeypatch.setattr(example_utils.dist_utils, "size", lambda group=None: 1)
    monkeypatch.setattr(example_utils.dist_utils, "local_rank", lambda: 0)
    args = SimpleNamespace(mo_session=_session(world_size=1, distributed=False), use_fsdp2=True)

    example_utils.setup_distributed_args(args)

    assert len(setups) == 1
    assert args.dist_state.world_size == 1


@pytest.mark.parametrize(
    ("world_size", "use_fsdp2", "enters_session"),
    [("2", False, True), ("1", True, True), (None, False, False)],
    ids=["torchrun", "fsdp2", "single_process"],
)
def test_main_runs_a_distributed_launch_inside_a_session(
    hf_ptq, monkeypatch, world_size, use_fsdp2, enters_session
):
    """main() opens the session around the whole run, with a collective timeout above NCCL's
    30-minute default, and closes it even when the run raises -- or the other ranks would wait in
    their next collective until that timeout."""
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    session = _session(world_size=2)
    events, timeouts = [], []

    @contextlib.contextmanager
    def modelopt_session(timeout=None, device=None):
        timeouts.append(timeout)
        events.append(("enter", device))
        try:
            yield session
        finally:
            events.append(("exit",))

    def main_inner(args):
        events.append(("run", args.mo_session))
        raise _RunFailedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hf_ptq, "modelopt_session", modelopt_session)
    monkeypatch.setattr(hf_ptq, "_main_inner", main_inner)

    with pytest.raises(_RunFailedError):
        hf_ptq.main(SimpleNamespace(use_fsdp2=use_fsdp2, device="cuda"))

    if enters_session:
        assert events == [("enter", "cuda"), ("run", session), ("exit",)]
        assert timeouts[0] > timedelta(minutes=30)
    else:
        assert events == [("run", None)]


# --------------------------------------------------------------------------------- calibration


def test_distributed_calibration_gives_each_rank_its_own_shard(hf_ptq, monkeypatch):
    """Every distributed path calibrates data-parallel: the ranks get equal, disjoint shards that
    together cover the calibration set. A single process keeps the loader as built."""
    loader = DataLoader(list(range(8)), batch_size=1)
    monkeypatch.setattr(hf_ptq, "get_dataset_dataloader", lambda **kwargs: loader)
    args = SimpleNamespace(
        specdec_offline_dataset=None,
        calib_with_images=False,
        dataset=["cnn_dailymail"],
        batch_size=1,
        calib_size=[8],
        calib_seq=16,
    )
    tokenizer = get_tiny_tokenizer()

    def calib_loader(session):
        built, _ = hf_ptq.make_calib_dataloader(
            args, None, None, tokenizer, torch.device("cpu"), "llama", session
        )
        return built

    shards = [[int(b) for b in calib_loader(_session(world_size=2, rank=r))] for r in range(2)]

    assert len(shards[0]) == len(shards[1]) == 4
    assert sorted(shards[0] + shards[1]) == list(range(8))
    assert calib_loader(None) is loader


def _calibration_loop_choice(hf_ptq, monkeypatch, session):
    """Run mono_quantize under ``session`` and return which calibration loop it built."""
    model = torch.nn.Linear(2, 2)
    chosen, quantized = [], []
    monkeypatch.setattr(hf_ptq, "is_quantized", lambda m: False)
    monkeypatch.setattr(hf_ptq, "need_calibration", lambda cfg: True)
    monkeypatch.setattr(
        hf_ptq, "distributed_forward_loop", lambda *a: chosen.append("distributed") or object()
    )
    monkeypatch.setattr(
        hf_ptq, "create_forward_loop", lambda **k: chosen.append("plain") or object()
    )
    monkeypatch.setattr(
        hf_ptq.mtq, "quantize", lambda m, cfg, forward_loop: quantized.append(forward_loop) or m
    )
    args = SimpleNamespace(
        qformat="fp8",
        calib_with_images=False,
        mo_session=session,
        layerwise_export=False,
        specdec_offline_dataset=None,
    )

    hf_ptq.mono_quantize(
        args,
        {"algorithm": "max"},
        model,
        model,
        model_type="qwen3_moe",
        calibration_only=False,
        calib_dataloader=object(),
        is_nemotron_vl_model=False,
    )

    assert len(quantized) == 1
    return chosen


def test_expert_parallel_calibration_runs_through_the_distributed_forward_loop(hf_ptq, monkeypatch):
    """Only distributed_forward_loop re-plants the all-to-all forwards that quantizer insertion
    clobbers, so an expert-parallel run must calibrate through it rather than the plain loop."""
    session = _session(world_size=2)
    session.ep_mesh = object()  # what load_and_prepare_ep records

    assert _calibration_loop_choice(hf_ptq, monkeypatch, session) == ["distributed"]


def test_fsdp2_calibration_keeps_the_plain_forward_loop(hf_ptq, monkeypatch):
    """Without expert parallelism there is nothing to re-plant: FSDP2 keeps the plain loop and its
    split-on-OOM retry, as on main."""
    assert _calibration_loop_choice(hf_ptq, monkeypatch, _session(world_size=2)) == ["plain"]
