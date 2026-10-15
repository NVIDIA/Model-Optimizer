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

hf_ptq.py decides a distribution plan once, with ``resolve_distribution_plan``, and loads the model
by it through ``load_pretrained``: expert-parallel, FSDP2-sharded under ``--use_fsdp2``, or whole
through ``get_model``. Everything after reads the plan back from the model: every rank of a
distributed plan calibrates on its own shard of the data, an expert-parallel one through the
distributed forward loop, and a whole model is pinned to the rank's own GPU instead of ``"auto"``.

Nothing here needs a process group or a checkpoint: the process group's rank and size are answered
the way a torchrun launch would, the configs are tiny real transformers configs served from memory,
and the loaders are replaced by stand-ins that record how they were called.
"""

import contextlib
import importlib
import inspect
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoConfig, DeepseekV3Config, LlamaConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.models import get_spec
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

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


@pytest.fixture(autouse=True)
def _checkpoint_is_local(monkeypatch):
    """The checkpoint path names no files; resolving it must not reach the Hub."""
    monkeypatch.setattr(
        model_load_utils, "ensure_local_checkpoint", lambda path, group=None: (None, path)
    )


@pytest.fixture
def torchrun(monkeypatch):
    """Run as rank ``rank`` of a ``world_size``-rank torchrun launch; returns its ``dist_state``.

    Every rank acts as its group's first, deciding for itself; how the ranks agree on what they
    decide is tested with the collectives.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)

    def enter(world_size, rank=0):
        monkeypatch.setattr(dist, "is_initialized", lambda: world_size > 1)
        monkeypatch.setattr(dist, "is_distributed_run", lambda: world_size > 1)
        monkeypatch.setattr(dist, "size", lambda group=None: world_size)
        monkeypatch.setattr(dist, "is_master", lambda group=None: rank == 0)
        monkeypatch.setattr(dist, "local_rank", lambda: rank)
        return _dist_state(world_size, rank)

    return enter


def _dist_state(world_size, rank=0):
    """What distributed_run yields to ``rank`` of a ``world_size``-rank launch."""
    return dist.DistributedState(
        rank=rank, world_size=world_size, device=torch.device(f"cuda:{rank}")
    )


def _args(dist_state, **overrides):
    """What load_model reads before it picks a loader, as main() leaves it."""
    args = SimpleNamespace(
        dist_state=dist_state,
        use_fsdp2=False,
        recipe=None,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
        attn_implementation=None,
    )
    vars(args).update(overrides)
    return args


def _serve(monkeypatch, hf_ptq, config):
    """Answer every config read with ``config`` rather than a checkpoint on disk."""
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    return config


def _deepseek_v3(n_routed_experts):
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


# -------------------------------------------------------------------------------- the decision


@pytest.mark.parametrize("one_rank_group", [False, True], ids=["no_process_group", "one_rank"])
def test_a_single_process_run_is_never_expert_parallel(
    hf_ptq, monkeypatch, torchrun, one_rank_group
):
    """With no distributed run there is nothing to shard over, even for an EP-capable MoE."""
    dist_state = torchrun(world_size=1) if one_rank_group else _dist_state(world_size=1)
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))

    assert hf_ptq._resolve_distribution(_args(dist_state))[0] == DistributionPlan()


def test_use_fsdp2_plans_fsdp2_without_expert_parallelism(hf_ptq, monkeypatch, torchrun):
    """--use_fsdp2 names its load, so even an explicit EP degree for an EP-capable MoE must not
    divert the run to the expert-parallel one."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    args = _args(torchrun(world_size=2), use_fsdp2=True)

    assert hf_ptq._resolve_distribution(args)[0] == DistributionPlan(2, dense="fsdp")


def test_dense_sharding_without_expert_parallelism_needs_use_fsdp2(hf_ptq, monkeypatch, torchrun):
    """Without expert parallelism a plan that shards the dense part is an FSDP2 run, which this
    script runs only under --use_fsdp2, the flag its FSDP2 checks hang off; so it must fail and point
    there rather than load the model whole on every rank."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))

    with pytest.raises(RuntimeError, match="--use_fsdp2"):
        hf_ptq._resolve_distribution(_args(torchrun(world_size=2)))


def test_a_dense_model_under_torchrun_runs_data_parallel(hf_ptq, monkeypatch, torchrun):
    """A dense model has no experts to shard, so it loads whole on every rank, through get_model."""
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))

    assert hf_ptq._resolve_distribution(_args(torchrun(world_size=4)))[0] == DistributionPlan(4)


# ---------------------------------------------------------------------------------- load_model


def _record_load_pretrained(hf_ptq, monkeypatch):
    """Replace hf_ptq's load_pretrained and every other loader; return the recorded call."""
    calls = []

    def load_pretrained_stub(*args, **kwargs):
        calls.append(inspect.signature(load_pretrained).bind(*args, **kwargs).arguments)
        raise _LoaderReachedError

    monkeypatch.setattr(hf_ptq, "load_pretrained", load_pretrained_stub)
    monkeypatch.setattr(hf_ptq, "get_model", lambda *a, **k: pytest.fail("whole-model loader ran"))
    return calls


def test_load_model_loads_an_expert_parallel_plan_through_load_pretrained(
    hf_ptq, monkeypatch, torchrun
):
    """With an EP plan, load_model hands load_pretrained the plan -- the decided degree and dense
    sharding -- and the spec-prepared config it was decided on."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    served = _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    calls = _record_load_pretrained(hf_ptq, monkeypatch)
    args = _args(torchrun(world_size=4), trust_remote_code=True, attn_implementation="eager")

    with pytest.raises(_LoaderReachedError):
        hf_ptq.load_model(args)

    (call,) = calls
    assert call["ckpt_path"] == _CKPT
    assert call["distribution"] == DistributionPlan(4, ep_size=4, dense="fsdp")
    assert call["kwargs"] == {
        "hf_config": served,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    assert served.base_model_ep_plan == get_spec("deepseek_v3").ep_spec.base_model_ep_plan


def test_use_fsdp2_loads_through_load_pretrained(hf_ptq, monkeypatch, torchrun):
    """--use_fsdp2 asks load_pretrained for its FSDP2 load, with the CPU-offload choice and the
    config its checks already read."""
    served = _serve(monkeypatch, hf_ptq, LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY))
    calls = _record_load_pretrained(hf_ptq, monkeypatch)
    args = _args(
        torchrun(world_size=2),
        use_fsdp2=True,
        cpu_offload=True,
        specdec_offline_dataset=None,
        low_memory_mode=False,
    )

    with pytest.raises(_LoaderReachedError):
        hf_ptq.load_model(args)

    (call,) = calls
    assert call["distribution"] == DistributionPlan(2, dense="fsdp")
    assert call["kwargs"] == {
        "hf_config": served,
        "trust_remote_code": False,
        "cpu_offload": True,
        "attn_implementation": None,
    }


def test_a_whole_model_loads_by_the_plan_through_get_model(hf_ptq, monkeypatch, torchrun):
    """Data-parallel, every rank loads the whole model, by the plan, onto its own GPU."""
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))
    loads = []

    def get_model(ckpt_path, device, **kwargs):
        loads.append((device, kwargs["distribution"]))
        raise _LoaderReachedError

    monkeypatch.setattr(hf_ptq, "get_model", get_model)
    args = _args(
        torchrun(world_size=4, rank=1),
        specdec_offline_dataset=None,
        low_memory_mode=False,
        gpu_max_mem_percentage=0.8,
        use_seq_device_map=False,
        offload_folder=None,
        max_cpu_memory_gb=None,
        max_gpu_memory_gb=None,
    )

    with pytest.raises(_LoaderReachedError):
        hf_ptq.load_model(args)

    assert loads == [(torch.device("cuda:1"), DistributionPlan(4))]


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

    plan = DistributionPlan(world_size)

    def load_pretrained(ckpt_path, auto_class, **kwargs):
        assert kwargs["distribution"] == plan
        placed.append(kwargs["device_map"])
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(example_utils, "load_pretrained", load_pretrained)

    example_utils.get_model(str(tmp_path), device=device, distribution=plan)

    assert placed == [device_map]


# ------------------------------------------------------------------ process group and dist_state


def test_main_owns_the_process_group_around_the_whole_run(hf_ptq, monkeypatch):
    """main() runs inside distributed_run, with a collective timeout above NCCL's 30-minute
    default, and leaves it even when the run raises -- or the other ranks would wait in their next
    collective until that timeout. The run reads its rank, size and device from the state
    distributed_run yields, which puts a single process on --device."""
    events, calls, seen = [], [], []
    state = dist.DistributedState(rank=0, world_size=1, device="cpu")

    @contextlib.contextmanager
    def distributed_run(timeout=None, device="cuda"):
        calls.append((timeout, device))
        events.append("enter")
        try:
            yield state
        finally:
            events.append("exit")

    def main_inner(args):
        events.append("run")
        seen.append(args.dist_state)
        raise _RunFailedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hf_ptq, "distributed_run", distributed_run)
    monkeypatch.setattr(hf_ptq, "_main_inner", main_inner)

    with pytest.raises(_RunFailedError):
        hf_ptq.main(SimpleNamespace(use_fsdp2=False, device="cpu"))

    assert events == ["enter", "run", "exit"]
    timeout, device = calls[0]
    assert timeout > timedelta(minutes=30)
    assert device == "cpu"
    assert len(seen) == 1
    assert seen[0] is state


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

    def calib_loader(world_size, rank=0):
        args.dist_state = _dist_state(world_size, rank)
        model = torch.nn.Linear(2, 2)
        set_distribution_plan(model, DistributionPlan(world_size))
        built, _ = hf_ptq.make_calib_dataloader(
            args, model, None, tokenizer, torch.device("cpu"), "llama"
        )
        return built

    shards = [[int(b) for b in calib_loader(world_size=2, rank=r)] for r in range(2)]

    assert len(shards[0]) == len(shards[1]) == 4
    assert sorted(shards[0] + shards[1]) == list(range(8))
    assert calib_loader(world_size=1) is loader


def _calibration_loop_choice(hf_ptq, monkeypatch, plan, ep_mesh=None):
    """Run mono_quantize on a model distributed as ``plan``, its experts loaded over ``ep_mesh``, and
    return which calibration loop it built."""
    model = torch.nn.Linear(2, 2)
    set_distribution_plan(model, plan)
    chosen, quantized = [], []
    monkeypatch.setattr(hf_ptq, "is_quantized", lambda m: False)
    monkeypatch.setattr(hf_ptq, "need_calibration", lambda cfg: True)
    monkeypatch.setattr(hf_ptq, "expert_parallel_mesh", lambda full_model: ep_mesh)
    monkeypatch.setattr(
        hf_ptq,
        "distributed_forward_loop",
        lambda *a, **k: chosen.append(("distributed", k["ep_mesh"])) or object(),
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
        dist_state=_dist_state(world_size=2),
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
    ep_mesh = object()  # the mesh load_and_prepare_ep loaded the experts over
    plan = DistributionPlan(2, ep_size=2)

    assert _calibration_loop_choice(hf_ptq, monkeypatch, plan, ep_mesh) == [
        ("distributed", ep_mesh)
    ]


def test_fsdp2_calibration_keeps_the_plain_forward_loop(hf_ptq, monkeypatch):
    """Without expert parallelism there is nothing to re-plant: FSDP2 keeps the plain loop and its
    split-on-OOM retry, as on main."""
    plan = DistributionPlan(2, dense="fsdp")

    assert _calibration_loop_choice(hf_ptq, monkeypatch, plan, ep_mesh=object()) == ["plain"]


class _PreviewModel(torch.nn.Module):
    """A model whose ``generate`` records the kwargs each preview passes it."""

    def __init__(self, plan=None):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False, model_type="qwen3_moe")
        self.calls = []
        if plan is not None:
            set_distribution_plan(self, plan)

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        return torch.cat([input_ids, input_ids[:, :1]], dim=1)


_PREVIEW_PLANS = pytest.mark.parametrize(
    ("plan", "lockstep"),
    [
        (None, False),
        (DistributionPlan(4), False),
        (DistributionPlan(4, ep_size=4), True),
        (DistributionPlan(4, dense="fsdp"), True),
        (DistributionPlan(1, dense="fsdp"), False),
    ],
    ids=["single_process", "replicated", "expert_parallel", "fsdp2", "one_rank_fsdp2"],
)


@_PREVIEW_PLANS
def test_a_sharded_model_previews_in_lockstep(hf_ptq, monkeypatch, plan, lockstep):
    """Each rank previews its own calibration shard's first sample, so the ranks finish decoding at
    different steps. A sharded model runs a collective in every forward, and a rank that stopped
    early left the others blocked in it: Qwen3-30B-A3B NVFP4 at EP4 hung in the post-PTQ preview.
    Both previews keep every rank of a sharded model decoding until all have finished."""
    model = _PreviewModel(plan)
    loader = [{"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}]
    args = SimpleNamespace(
        specdec_offline_dataset=None,
        skip_generate=False,
        verbose=False,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
    )
    monkeypatch.setattr(hf_ptq, "export_quantized", lambda *args, **kwargs: None)

    ids, mask, before = hf_ptq.pre_quantize(args, model, "qwen3_moe", None, loader, False)
    hf_ptq.post_quantize(
        args,
        model,
        model,
        "qwen3_moe",
        get_tiny_tokenizer(),
        None,
        ids,
        mask,
        before,
        False,
        None,
        loader,
    )

    synced = [call.get("synced_gpus", False) for call in model.calls]
    assert synced == [lockstep, lockstep]  # before and after quantization


@_PREVIEW_PLANS
def test_a_sharded_nemotron_vl_model_previews_in_lockstep(monkeypatch, plan, lockstep):
    """The Nemotron-VL preview's text, fallback and image generations keep a sharded model's ranks
    decoding together too."""
    recorded = {}
    vlm_utils = SimpleNamespace(
        run_text_only_generation=lambda model, tokenizer, question, config, *args: recorded.update(
            text=config
        ),
        run_vl_preview_generation=lambda model, tokenizer, path, stage, trust, kwargs: (
            recorded.update(image=kwargs)
        ),
    )
    monkeypatch.setitem(sys.modules, "vlm_utils", vlm_utils)
    model = _PreviewModel(plan)

    example_utils.run_nemotron_vl_preview(
        model, get_tiny_tokenizer(), torch.tensor([[1, 2, 3]]), _CKPT, "stage", allow_fallback=True
    )

    expected = {"synced_gpus": True} if lockstep else {}
    assert recorded["text"].get("synced_gpus", False) == lockstep
    assert recorded["image"] == expected
    assert model.calls[0].get("synced_gpus", False) == lockstep  # the fallback generate


@pytest.mark.parametrize("generation_kwargs", [None, {"synced_gpus": True}])
def test_the_vl_image_preview_generates_with_the_kwargs_it_is_handed(
    monkeypatch, tmp_path, generation_kwargs
):
    """run_vl_preview_generation hands the kwargs it is given to the model's chat(), in its config."""
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")
    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "example1a.jpeg")
    monkeypatch.setattr(vlm_utils, "__file__", str(tmp_path / "vlm_utils.py"))
    monkeypatch.setattr(
        vlm_utils.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: lambda images: {}
    )
    received = []

    class _ChatModel:
        device = torch.device("cpu")

        def chat(self, tokenizer, question, generation_config, **image_features):
            received.append(generation_config)
            return "a picture"

    vlm_utils.run_vl_preview_generation(
        _ChatModel(), SimpleNamespace(eos_token_id=0), _CKPT, "stage", False, generation_kwargs
    )

    assert len(received) == 1
    assert received[0].get("synced_gpus", False) == (generation_kwargs is not None)
