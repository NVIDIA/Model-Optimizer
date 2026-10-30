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

"""What the expert-parallel loader reads off the loaded model and its config.

The loader takes an expert-parallel plan, ``_seed_parallel_states`` gives every module the
parallel state its amax syncs over,
``expert_parallel_mesh`` hands calibration the mesh an expert-parallel plan's experts are sharded
over, and ``_ckpt_quant_method`` reads the checkpoint's native quantization. Everything runs in one
CPU process.
"""

import copy
import dataclasses
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from _test_utils.torch.moe_models import model_with_decoder_at
from packaging.version import Version
from torch import nn

pytest.importorskip("transformers")

import transformers
import transformers.distributed as hf_distributed
from transformers import FineGrainedFP8Config, PreTrainedModel, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.distributed import DistributedConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import moe
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan
from modelopt.torch.utils.moe import expert_parallel_mesh
from modelopt.torch.utils.plugins import hf_checkpoint_utils, model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    _ckpt_quant_method,
    _experts_sharded_over,
    _seed_parallel_states,
    _transformers_mesh_builder,
    load_and_prepare_ep,
)

# ------------------------------------------------------------------------------- the plan

_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class _Mesh:
    """A device mesh, as far as the loader reads one; its groups run no collective (``-1``)."""

    device_type = "cpu"

    def __getitem__(self, name):
        return self

    def size(self):
        return 2

    def get_group(self):
        return -1


@pytest.fixture
def ep_world_of_two(monkeypatch, gloo_world_of_one):
    """A two-rank expert-parallel world as far as the loader's checks see, and the checkpoint paths
    it resolves; nothing else is set up."""
    resolved = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(
        model_load_utils, "ensure_local_checkpoint", lambda p, **k: resolved.append(p) or (None, p)
    )
    return resolved


@pytest.mark.parametrize(
    ("plan", "error", "match"),
    [
        (DistributionPlan(2), ValueError, "shards no routed experts"),
        (DistributionPlan(2, ep_size=2, dense="fsdp"), NotImplementedError, "replicates the dense"),
    ],
    ids=["no-expert-parallelism", "sharded-dense"],
)
def test_the_loader_refuses_a_plan_it_cannot_load_by(plan, error, match):
    """Before anything else: no process group, transformers release or checkpoint is looked at."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("unused", plan)


def test_the_loader_refuses_a_plan_for_another_world(gloo_world_of_one):
    with pytest.raises(ValueError, match="is for 2 ranks, but this run has 1"):
        load_and_prepare_ep("unused", DistributionPlan(2, ep_size=2))


def test_the_loader_refuses_a_transformers_that_builds_no_model_with_torch_a2a(
    monkeypatch, gloo_world_of_one
):
    """transformers 5.6 checks a model's experts implementation against the ones it ships, so its
    from_pretrained would refuse torch_a2a once the load had set everything up: refused first."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr("transformers.__version__", "5.6.2")

    with pytest.raises(RuntimeError, match=r"requires transformers >= 5\.7"):
        load_and_prepare_ep("unused", DistributionPlan(2, ep_size=2), hf_config=SimpleNamespace())


def test_a_hub_id_is_made_local_before_the_weights_load(monkeypatch, gloo_world_of_one):
    """The export reads the tensors the load could not place back from the checkpoint it records,
    which must be a local directory: a Hub ID there leaves the export nothing to read. The load
    resolves it once the plan and the config pass, before anything reads a weight."""

    class _ResolvedError(Exception):
        """Stops the load where it resolves the checkpoint."""

    resolved = []

    def resolve(path, **hub_options):
        resolved.append(path)
        raise _ResolvedError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "ensure_local_checkpoint", resolve)

    with pytest.raises(_ResolvedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(2, ep_size=2),
            hf_config=SimpleNamespace(base_model_ep_plan=_EP_PLAN),
        )

    assert resolved == ["org/model"]


def test_the_experts_are_counted_in_the_config_holding_the_plan(monkeypatch, gloo_world_of_one):
    """As the distribution plan's decision counts them: a VLM whose nested config holds the MoE and
    its plan has its experts counted there, not by an expert-ish field on its root config."""

    class _CheckedError(Exception):
        """Stops the load once its checks pass."""

    def resolve(path, **hub_options):
        raise _CheckedError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "ensure_local_checkpoint", resolve)
    text = SimpleNamespace(base_model_ep_plan=_EP_PLAN, num_experts=8)

    with pytest.raises(_CheckedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(4, ep_size=4),
            hf_config=SimpleNamespace(text_config=text, num_experts=3),
        )


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        *(
            (
                SimpleNamespace(quantization_config={"quant_method": method}),
                NotImplementedError,
                f"cannot dequantize a '{method}'",
            )
            for method in ("gptq", "modelopt", "compressed-tensors", "mxfp4")
        ),
        (SimpleNamespace(), ValueError, "declare base_model_ep_plan"),
        (
            SimpleNamespace(base_model_ep_plan={"layers.*.self_attn.q_proj": "colwise"}),
            ValueError,
            "styles no routed experts",
        ),
        *(
            (
                SimpleNamespace(
                    base_model_ep_plan={"layers.*.mlp.experts": "moe_tp_experts", **params}
                ),
                ValueError,
                r"shards no weights of layers\.\*\.mlp\.experts through grouped_gemm",
            )
            for params in ({}, {"layers.*.mlp.experts.gate_up_proj": "packed_colwise"})
        ),
        (
            SimpleNamespace(
                base_model_ep_plan={**_EP_PLAN, "layers.*.mlp.experts.down_proj": "packed_rowwise"}
            ),
            ValueError,
            r"styles layers\.\*\.mlp\.experts\.down_proj by a style this load drops",
        ),
    ],
    ids=[
        "gptq",
        "modelopt",
        "compressed-tensors",
        "mxfp4",
        "no-ep-plan",
        "no-routed-experts",
        "experts-container-only",
        "experts-sharded-otherwise",
        "experts-sharded-partly-otherwise",
    ],
)
def test_a_config_the_load_cannot_take_is_refused_before_anything_loads(
    ep_world_of_two, config, error, match
):
    """A format transformers' own quantizer would load as stored, for PTQ to treat as plain weights,
    or, MXFP4, dequantize into GPT-OSS's transposed expert layout whatever the model's, or a plan
    that leaves the experts whole on every rank: one naming no routed experts, or none of
    whose entries shards their weights by grouped_gemm, the only expert-weight style the load
    shards by, or one that styles some of their weights otherwise. The load refuses each before it
    reads anything."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert ep_world_of_two == []


def test_an_ep_size_that_does_not_divide_the_experts_is_refused_before_anything_loads(
    ep_world_of_two,
):
    """The all-to-all transport finds an expert's rank by dividing its id by this rank's share of
    the experts: with 3 experts over 2 ranks the ranks would disagree on who holds which, and send
    tokens to the wrong experts."""
    config = SimpleNamespace(base_model_ep_plan=_EP_PLAN, num_experts=3)

    with pytest.raises(ValueError, match="ep_size=2 must divide the 3 routed experts"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert ep_world_of_two == []


def test_a_plan_whose_expert_containers_match_no_module_is_refused(ep_world_of_two, monkeypatch):
    """A plan that styles the grouped-GEMM parameters a level below the experts names a container no
    module is. Loaded anyway, every module would get the dense state, and the experts' amax would
    reduce over ranks holding different experts."""

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return model_with_decoder_at("model.")

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: [])
    config = SimpleNamespace(
        base_model_ep_plan={"layers.*.mlp.experts.*.gate_up_proj": "grouped_gemm"}
    )

    with pytest.raises(ValueError, match="that no module of"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)


@pytest.mark.parametrize("declared", [True, False], ids=["on-the-config", "a-class-default"])
def test_the_model_keeps_the_plan_its_checkpoint_declares(
    ep_world_of_two, gloo_world_of_one, monkeypatch, declared
):
    """The load shards by the torch_a2a styles, but the config the model keeps goes back to the
    checkpoint's plan: an export writes it out, and nothing without ModelOpt knows those styles. A
    plan the config class ships (transformers' own, e.g. Qwen3-MoE's) stays off the instance."""

    class _PastError(Exception):
        """Raised by the load's next check."""

    model = model_with_decoder_at("model.")

    class _Loader:
        @staticmethod
        def from_pretrained(path, config, **kwargs):
            model.config = copy.deepcopy(config)  # transformers keeps a copy of the config
            return model

    def next_check(*args, **kwargs):
        raise _PastError

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_seed_parallel_states", next_check)

    class _Config(SimpleNamespace):
        base_model_ep_plan = _EP_PLAN

    # 8 experts over 2 ranks: the stand-in model holds this rank's 4.
    config = _Config(num_experts=8, **({"base_model_ep_plan": dict(_EP_PLAN)} if declared else {}))

    with pytest.raises(_PastError):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)

    assert model.config.base_model_ep_plan == _EP_PLAN
    assert ("base_model_ep_plan" in vars(model.config)) is declared


def test_a_loaded_model_keeps_its_sharding_styles_out_of_its_saved_config(
    ep_world_of_two, gloo_world_of_one, monkeypatch
):
    """What the loader leaves on a transformers model, as transformers treats it. Built with the
    torch_a2a styles, the model keeps them as ``model.tp_plan``, which calibration re-installs the
    all-to-all forwards by, and the expert-parallel mesh it re-installs them over, which this stand-in
    from_pretrained does not set; while its config goes back to the checkpoint's plan; and the experts
    implementation its forwards dispatch on, torch_a2a, stays out of the config it serializes,
    which an export writes."""
    if not Qwen3MoeConfig.base_model_ep_plan:
        pytest.skip("this transformers ships no expert-parallel plan for Qwen3-MoE")

    class _PastError(Exception):
        """Raised by the load's next check."""

    built = []

    class _Loader:
        @staticmethod
        def from_pretrained(path, config, **kwargs):
            # As transformers' expert-parallel load builds it: this rank's 2 of the 4 experts.
            config.distributed_config = DistributedConfig(enable_expert_parallel=True)
            share = copy.deepcopy(config)
            share.num_experts //= 2
            with torch.device("meta"):
                model = Qwen3MoeForCausalLM(share)
            model.config = config
            built.append(model)
            return model

    def next_check(*args, **kwargs):
        raise _PastError

    mesh = _Mesh()
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: mesh
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_seed_parallel_states", next_check)
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_experts=4,
        num_experts_per_tok=1,
    )

    with pytest.raises(_PastError):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)

    (model,) = built
    assert model.config.base_model_ep_plan == Qwen3MoeConfig.base_model_ep_plan
    assert "torch_a2a_experts" in set(model.tp_plan.values())
    assert model._device_mesh is mesh
    assert model.config._experts_implementation == "torch_a2a"
    assert not any("experts_implementation" in key for key in model.config.to_dict())


def test_the_weights_load_in_the_checkpoints_dtype_through_its_class(ep_world_of_two, monkeypatch):
    """The checkpoint's ``dtype`` reaches ``from_pretrained``, and the class that builds it is chosen
    with the trust the caller gave its code."""

    class _LoadedError(Exception):
        """Stops the load at from_pretrained."""

    loaded, classes = {}, []

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            raise _LoadedError

    def auto_class(hf_config, require_lm_head=True, trust_remote_code=False):
        classes.append(trust_remote_code)
        return _Loader

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", auto_class)
    config = SimpleNamespace(dtype=torch.float16, base_model_ep_plan=_EP_PLAN)

    with pytest.raises(_LoadedError):
        load_and_prepare_ep(
            "org/model", DistributionPlan(2, ep_size=2), trust_remote_code=True, hf_config=config
        )
    assert loaded["dtype"] is torch.float16
    assert classes == [True]


# ----------------------------------------------------------------------- _seed_parallel_states


def test_expert_containers_and_their_children_get_the_expert_state():
    """Everything else, the root and the containers' own parents included, gets the dense state; so
    does a sibling whose name merely starts with a container's (Qwen2-MoE keeps
    ``shared_expert_gate`` beside ``shared_expert``)."""
    model = model_with_decoder_at("model.")
    for layer in model.model.layers:
        layer.mlp.experts.weight_quantizer = nn.Identity()  # a child, as quantizer insertion adds
    model.model.layers[0].mlp.experts_gate = nn.Linear(2, 1)
    experts = {f"model.layers.{i}.mlp.experts": model.model.layers[i].mlp.experts for i in (0, 1)}
    dense, expert = object(), object()

    _seed_parallel_states(model, experts, dense_state=dense, expert_state=expert)

    states = {name: module._parallel_state for name, module in model.named_modules()}
    assert {name for name, state in states.items() if state is expert} == {
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.weight_quantizer",
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.experts.weight_quantizer",
    }
    assert all(state is dense for state in states.values() if state is not expert)


# ------------------------------------------------------------------------ expert_parallel_mesh


@pytest.mark.parametrize(
    ("plan", "is_expert_parallel"),
    [(DistributionPlan(2, ep_size=2), True), (None, False), (DistributionPlan(2), False)],
    ids=["expert-parallel", "no-plan", "replicated"],
)
def test_the_expert_parallel_mesh_comes_with_an_expert_parallel_plan_only(plan, is_expert_parallel):
    """An expert-parallel plan: the mesh transformers kept on the model. Any other model has none,
    even with a device mesh on it, as a load through from_pretrained's own TP can leave."""
    model = model_with_decoder_at("model.")
    if plan is not None:
        set_distribution_plan(model, plan)
    model._device_mesh = mesh = object()

    assert expert_parallel_mesh(model) is (mesh if is_expert_parallel else None)


def test_an_expert_parallel_model_without_its_mesh_is_refused():
    """A model recorded expert-parallel that load_and_prepare_ep did not load has no mesh for
    calibration to re-install the all-to-all forwards over; it is refused by name."""
    model = model_with_decoder_at("model.")
    set_distribution_plan(model, DistributionPlan(2, ep_size=2))

    with pytest.raises(RuntimeError, match="carries no device mesh"):
        expert_parallel_mesh(model)


# --------------------------------------------------------------------------- _ckpt_quant_method


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(quantization_config={"quant_method": "fp8"}), "fp8"),
        (SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4")), "mxfp4"),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
            ),
            "fp8",
        ),
        (
            SimpleNamespace(
                llm_config=SimpleNamespace(quantization_config={"quant_method": "fp8"})
            ),
            "fp8",
        ),
        (
            SimpleNamespace(
                quantization_config={"quant_method": "mxfp4"},
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
            ),
            "mxfp4",
        ),
        (SimpleNamespace(quantization_config=FineGrainedFP8Config()), "fp8"),
        (SimpleNamespace(quantization_config={"weight_block_size": [128, 128]}), None),
        (SimpleNamespace(text_config=None), None),
    ],
    ids=[
        "dict",
        "object",
        "vlm-text-config",
        "vlm-llm-config",
        "top-level-wins",
        "transformers-config-object",
        "no-method",
        "unquantized",
    ],
)
def test_quant_method_comes_from_the_config_or_a_vlms_text_config(config, expected):
    """The top-level config first, else a VLM's text or LLM config. A loaded model's config holds
    transformers' config object, not the JSON dict, and callers compare against ``"fp8"`` /
    ``"mxfp4"``: it reports the plain method name, not its enum."""
    assert _ckpt_quant_method(config) == expected


def test_from_pretrained_builds_its_mesh_from_the_ep_mesh_it_is_given():
    """Newer transformers ignores ``from_pretrained``'s ``device_mesh`` and builds a world-sized one,
    which shards the experts over every rank when EP is below the world size. During the load it gets
    the EP mesh instead, and its own builder is back afterwards, even when the load raises."""
    mixin = _transformers_mesh_builder()
    if mixin is None:
        pytest.skip("this transformers honors from_pretrained's device_mesh")
    builder = mixin.initialize_distributed_mesh
    ep_mesh = SimpleNamespace(device_type="cpu")
    config = DistributedConfig(enable_expert_parallel=True, tp_plan="auto", tp_size=2)
    with _experts_sharded_over(ep_mesh):
        _, device_map, device_mesh = PreTrainedModel.prepare_distribute_model(config)

    assert device_mesh is ep_mesh
    assert device_map == torch.device("cpu")
    assert mixin.initialize_distributed_mesh is builder
    with pytest.raises(RuntimeError), _experts_sharded_over(ep_mesh):
        raise RuntimeError("the load failed")
    assert mixin.initialize_distributed_mesh is builder


@pytest.mark.parametrize(
    ("fields", "builds_its_mesh", "config", "tp_plan"),
    [
        # transformers 5.6-5.9, 5.11, 5.12
        (["enable_expert_parallel"], False, {"enable_expert_parallel": True}, "auto"),
        # 5.10, 5.13-5.17
        (
            ["tp_size", "tp_plan", "enable_expert_parallel"],
            False,
            {"enable_expert_parallel": True, "tp_plan": "auto"},
            None,
        ),
        # 5.18 on
        (
            ["tp_size", "tp_plan", "enable_expert_parallel"],
            True,
            {"enable_expert_parallel": True, "tp_plan": "auto", "tp_size": 2},
            None,
        ),
        # no release so far
        (
            ["tp_size", "enable_expert_parallel"],
            True,
            {"enable_expert_parallel": True, "tp_size": 2},
            "auto",
        ),
    ],
    ids=["no-tp-plan-field", "tp-plan-field", "own-mesh-builder", "own-mesh-builder-no-tp-plan"],
)
def test_each_load_argument_goes_where_this_transformers_takes_it(
    ep_world_of_two, monkeypatch, fields, builds_its_mesh, config, tp_plan
):
    """``tp_plan="auto"`` goes on ``DistributedConfig`` where it has the field, else to
    ``from_pretrained``, never both; a release that builds its own mesh gets it sized to the EP
    degree. Each is decided by its own field, so a release with one and not the other loads too."""

    class _LoadedError(Exception):
        """Stops the load at from_pretrained."""

    loaded = {}

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            raise _LoadedError

    distributed_config = dataclasses.make_dataclass(
        "DistributedConfig", [(name, object, dataclasses.field(default=None)) for name in fields]
    )
    monkeypatch.setattr(hf_distributed, "DistributedConfig", distributed_config)
    monkeypatch.setattr(
        model_load_utils,
        "_transformers_mesh_builder",
        lambda: SimpleNamespace(initialize_distributed_mesh=None) if builds_its_mesh else None,
    )
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)

    with pytest.raises(_LoadedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(2, ep_size=2),
            hf_config=SimpleNamespace(base_model_ep_plan=_EP_PLAN),
        )

    given = {k: v for k, v in vars(loaded["distributed_config"]).items() if v is not None}
    assert given == config
    assert loaded.get("tp_plan") == tp_plan


# ----------------------------------------------------------------- one rank failing alone


class _PastTheMeshesError(Exception):
    """Raised by the meshes' stand-in: this rank got past the load's checks."""


def _meshes_in_step(*args, **kwargs):
    """The meshes' subgroups, as the collective they are: a rank in it waits for every peer."""
    torch.distributed.barrier()
    raise _PastTheMeshesError


def _one_rank_fails_job(rank, size, ckpt, failure, failing_rank, match):
    path, hf_config, patches = ckpt, None, []
    if failure != "unreadable-config":
        hf_config = SimpleNamespace(base_model_ep_plan=dict(_EP_PLAN), num_experts=4)
    if failure != "unseen-checkpoint":
        local = mock.patch.object(
            model_load_utils, "ensure_local_checkpoint", side_effect=lambda p, **k: (None, p)
        )
        patches.append(local)
    fails_here = rank == failing_rank

    def fail(*args, **kwargs):
        raise (MemoryError if failure == "load" else OSError)(match)

    if failure == "unreadable-config" and fails_here:
        path = ckpt + "-not-on-this-node"
    elif failure == "transformers" and fails_here:
        patches.append(mock.patch("transformers.__version__", "5.6.2"))
    elif failure == "quantized-config" and fails_here:
        hf_config.quantization_config = {"quant_method": "gptq"}
    elif failure == "unseen-checkpoint" and fails_here:
        # Rank 0 makes the checkpoint local; a rank on a node without that storage cannot see it.
        patches.append(mock.patch.object(hf_checkpoint_utils.os.path, "isdir", return_value=False))
    elif failure == "no-gpu" and not fails_here:
        # No rank has a GPU here: the others are given theirs.
        patches.append(mock.patch.object(model_load_utils, "_has_own_gpu", return_value=True))
    if failure in ("load", "record"):
        loaded = model_with_decoder_at("model.")
        load = fail if failure == "load" and fails_here else (lambda path, **kwargs: loaded)
        loader = SimpleNamespace(__name__="Loader", from_pretrained=load)
        record = fail if failure == "record" and fails_here else None
        patches += [
            mock.patch("torch.distributed.device_mesh.init_device_mesh", return_value=_Mesh()),
            mock.patch.object(model_load_utils, "_auto_class", return_value=loader),
            mock.patch.object(model_load_utils, "record_unplaced_source_keys", side_effect=record),
        ]
    else:
        patches.append(
            mock.patch("torch.distributed.device_mesh.init_device_mesh", _meshes_in_step)
        )
    refused = None
    with bounded_collectives(), ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        try:  # not pytest.raises: its failure would end the pooled worker instead of the job
            load_and_prepare_ep(path, DistributionPlan(2, ep_size=2), hf_config=hf_config)
        except Exception as error:
            refused = f"{type(error).__name__}: {error}"
    # The one rank's failure on every rank, before any rank went on alone into a collective. The GPU
    # check decides on every rank at once, from one gather, so no rank relays another's failure.
    assert refused is not None and match in refused, (rank, refused)
    assert fails_here or failure == "no-gpu" or f"rank {failing_rank} failed" in refused, refused


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"), reason="the EP load needs transformers 5.7"
)
@pytest.mark.parametrize(
    ("failure", "failing_rank", "match"),
    [
        ("unreadable-config", 1, "not-on-this-node"),
        ("transformers", 1, "requires transformers >= 5.7, but 5.6.2"),
        ("quantized-config", 1, "cannot dequantize a 'gptq'"),
        ("unseen-checkpoint", 1, "which this rank cannot see"),
        ("no-gpu", 1, "rank(s) [1] have none while the others have theirs"),
        ("load", 1, "the GPU ran out of memory"),
        ("record", 1, "a weight file this rank cannot read"),
    ],
)
def test_a_load_failing_on_one_rank_fails_every_rank(
    cpu_dist_workers, tmp_path, failure, failing_rank, match
):
    """One rank fails alone: reading the config, on its transformers, on its arguments, seeing the
    checkpoint rank 0 made local, without a GPU of its own where the others have theirs, or in the
    load. Every rank raises that failure, rather than leave the others waiting for it in the meshes
    (stood in by a bounded barrier) or the warmup's collectives."""
    Qwen3MoeConfig(num_experts=4, num_hidden_layers=1, hidden_size=8).save_pretrained(tmp_path)
    cpu_dist_workers.run(
        _one_rank_fails_job,
        ckpt=str(tmp_path),
        failure=failure,
        failing_rank=failing_rank,
        match=match,
    )
