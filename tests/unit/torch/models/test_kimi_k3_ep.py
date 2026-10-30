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

"""Expert-parallel adapter for Kimi-K3 (``kimi_k3``), whose kimi_linear backbone is remote code.

The checkpoint's modeling is not available here, so the tests build a stand-in remote module -- the
classes and functions the adapter patches, with the per-expert ``nn.ModuleList`` MoE the remote code
ships -- and check what the adapter does to it on tiny CPU tensors. Every process-wide patch is
undone after each test.
"""

import copy
import sys
import types
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.kimi_k3_remote import remote_module
from torch import nn

transformers = pytest.importorskip("transformers")

from packaging.version import Version

if Version(transformers.__version__) < Version("5.8"):
    pytest.skip("Kimi-K3 expert parallelism needs transformers >= 5.8", allow_module_level=True)

import transformers.conversion_mapping as cm
import transformers.core_model_loading as cml
import transformers.dynamic_module_utils as dmu
import transformers.modeling_utils as mu
import transformers.utils.generic as hf_generic
from transformers.core_model_loading import build_glob_alternation
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

from modelopt.torch.export.quant_aware_conversion import build_reverse_name_mapper
from modelopt.torch.models import _onfly_decompress, get_spec
from modelopt.torch.models.kimi_k3 import ep as kimi_k3
from modelopt.torch.utils import moe
from modelopt.torch.utils.moe import check_distributed_support

ROUTES = torch.tensor([[0, 3], [1, 2], [3, 1], [2, 0], [1, 3]])  # top-2 experts of 5 tokens


def _restore(monkeypatch, owner, name):
    """Have ``monkeypatch`` put ``owner.<name>`` back after the test, or remove it if it was absent."""
    if hasattr(owner, name):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    else:
        monkeypatch.setattr(owner, name, None, raising=False)
        monkeypatch.delattr(owner, name)


@pytest.fixture
def private_conversion_registry(monkeypatch):
    """Let a test register conversion mappings without leaking them into transformers' registry."""
    monkeypatch.setattr(cm, "_checkpoint_conversion_mapping_cache", None)  # rebuilt on first use
    if hasattr(cm, "USER_REGISTERED_MAPPINGS"):  # transformers >= 5.13
        monkeypatch.setattr(cm, "USER_REGISTERED_MAPPINGS", set(cm.USER_REGISTERED_MAPPINGS))


@pytest.fixture
def restore_loader_hooks(monkeypatch):
    """Undo the process-wide transformers hooks the on-the-fly decompressor installs."""
    for owner, name in ((cml, "spawn_materialize"), (mu, "get_model_conversion_mapping")):
        monkeypatch.setattr(owner, name, getattr(owner, name))


def _config(**overrides):
    """A tiny kimi_linear MoE config: 4 routed experts, top-2, run in a 6-wide latent space."""
    fields = {
        "num_experts": 4,
        "num_experts_per_token": 2,
        "hidden_size": 8,
        "routed_expert_hidden_size": 6,
        "moe_intermediate_size": 5,
        "hidden_act": "silu",
        "situ_beta": 1.5,
        "situ_linear_beta": 0.25,
        "latent_moe_use_norm": True,
        "num_shared_experts": 1,
        "_experts_implementation": "eager",
    }
    return SimpleNamespace(**{**fields, **overrides})


def _randomize(module, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen))


def _load(converters, checkpoint):
    """Route, collect and fuse ``checkpoint`` the way transformers' loader does."""
    per_target = {}
    for key, tensor in checkpoint.items():
        for conv in converters:
            target, pattern = conv.rename_source_key(key)
            if pattern is not None:
                break
        else:
            raise AssertionError(f"no converter claims {key}")
        future = Future()
        future.set_result(tensor)
        per_target.setdefault(target, copy.deepcopy(conv)).add_tensor(target, key, pattern, future)
    fused = {}
    for target, conv in per_target.items():
        fused.update(conv.convert(target))
    return fused


# --------------------------------------------------------------------------- the fused experts


def test_fused_experts_dispatch_through_the_experts_interface(monkeypatch):
    """``config._experts_implementation`` picks the forward (``torch_a2a`` under EP), so the module
    must route through transformers' ``ExpertsInterface``; and a stock implementation reading the
    fused params agrees with the eager loop, so the declared layout ([E, out, in], gated) is right."""
    routed = []
    batched_mm = ALL_EXPERTS_FUNCTIONS["batched_mm"]

    def probe(module, *args):
        routed.append(module)
        return batched_mm(module, *args)

    monkeypatch.setitem(ALL_EXPERTS_FUNCTIONS, "modelopt_test_probe", probe)
    config = _config(hidden_act="situ")
    experts = kimi_k3._build_fused_experts_class(remote_module())(
        config, config.routed_expert_hidden_size
    )
    _randomize(experts)
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randn(len(ROUTES), config.routed_expert_hidden_size, generator=gen)
    route_weights = torch.rand(ROUTES.shape, generator=gen)

    with torch.no_grad():
        eager = experts(tokens, ROUTES, route_weights)
        config._experts_implementation = "modelopt_test_probe"
        dispatched = experts(tokens, ROUTES, route_weights)

    assert routed == [experts]
    torch.testing.assert_close(dispatched, eager)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "latent_moe_use_norm": False,
            "routed_expert_hidden_size": None,
            "num_shared_experts": None,
        },
        {"hidden_act": "situ"},
    ],
    ids=["latent-norm-shared", "plain", "situ"],
)
def test_the_patched_block_reproduces_the_remote_per_expert_forward(overrides):
    """Swapping the per-expert ModuleList for the fused module keeps the block's output: the latent
    projections, the latent norm and the shared experts stay around the fused call, and w1 / w3 /
    w2 are the gate rows, the up rows and the down of the fused params. A ``situ`` model hands the
    full [gate; up] row to the remote ``SituAndMul``. A block built before the patch, as a model
    loaded earlier in the process holds, keeps its per-expert experts and its own forward."""
    remote = remote_module()
    config = _config(**overrides)
    reference = remote.KimiSparseMoeBlock(config)
    _randomize(reference)
    gen = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(2, 3, config.hidden_size, generator=gen)
    with torch.no_grad():
        expected = reference(hidden_states)

    kimi_k3._patch_kimi_moe_block(remote)
    with torch.no_grad():
        torch.testing.assert_close(reference(hidden_states), expected)
    block = remote.KimiSparseMoeBlock(config)
    assert not isinstance(block.experts, nn.ModuleList)
    dense = {k: v for k, v in reference.state_dict().items() if not k.startswith("experts.")}
    result = block.load_state_dict(dense, strict=False)
    assert sorted(result.missing_keys) == ["experts.down_proj", "experts.gate_up_proj"]
    assert not result.unexpected_keys
    with torch.no_grad():
        gate_up = [torch.cat([m.w1.weight, m.w3.weight]) for m in reference.experts]
        block.experts.gate_up_proj.copy_(torch.stack(gate_up))
        block.experts.down_proj.copy_(torch.stack([m.w2.weight for m in reference.experts]))
        torch.testing.assert_close(block(hidden_states), expected)


# ------------------------------------------------------------------- weight conversion and plan


@pytest.mark.usefixtures("private_conversion_registry")
def test_registered_conversions_fuse_w1_w3_into_gate_up_and_stack_w2():
    """transformers builds no weight conversions for remote code unless they are registered. The
    registered pair fuses the checkpoint's per-expert w1 (gate) and w3 (up) into
    ``gate_up_proj = [w1; w3]`` and stacks w2 into ``down_proj``, under the VLM's prefix."""
    kimi_k3._register_kimi_k3_conversions()
    converters = cm.get_checkpoint_conversion_mapping("kimi_linear")
    n_experts, inter, hidden = 3, 2, 4
    shapes = {"w1": (inter, hidden), "w3": (inter, hidden), "w2": (hidden, inter)}
    gen = torch.Generator().manual_seed(0)
    weights = {
        (name, e): torch.randn(shape, generator=gen)
        for e in range(n_experts)
        for name, shape in shapes.items()
    }
    prefix = "language_model.model.layers.1.block_sparse_moe.experts."

    fused = _load(converters, {f"{prefix}{e}.{name}.weight": w for (name, e), w in weights.items()})

    gate_up = [torch.cat([weights["w1", e], weights["w3", e]]) for e in range(n_experts)]
    assert torch.equal(fused[prefix + "gate_up_proj"], torch.stack(gate_up))
    down = [weights["w2", e] for e in range(n_experts)]
    assert torch.equal(fused[prefix + "down_proj"], torch.stack(down))


@pytest.mark.usefixtures("private_conversion_registry")
def test_packed_experts_export_under_the_checkpoints_names():
    """The decompressor rewrites the registered converters to read the packed experts, and the model
    keeps the rewritten ones as the conversions its load ran. The export names the experts back
    through them, so they reverse as the fusion they rewrote: to the checkpoint's ``w1``/``w3``/``w2``,
    not to ModelOpt's ``gate_proj``/``up_proj``/``down_proj``."""
    kimi_k3._register_kimi_k3_conversions()
    packed = [
        _onfly_decompress._compressedify(conv, None, None, ("weight_packed", "weight_scale"))
        for conv in cm.get_checkpoint_conversion_mapping("kimi_linear")
    ]
    expert = "model.layers.1.block_sparse_moe.experts.0."

    to_checkpoint = build_reverse_name_mapper(
        SimpleNamespace(_weight_conversions=packed), tensor_keys=True
    )

    exported = [
        to_checkpoint(f"{expert}{part}.weight") for part in ("gate_proj", "up_proj", "down_proj")
    ]
    assert exported == [f"{expert}{leaf}.weight" for leaf in ("w1", "w3", "w2")]


def test_the_registration_raises_without_the_conversion_api(monkeypatch):
    """Unregistered, the checkpoint's experts would load as MISSING: newly initialized, at full
    size on every rank. So a missing conversion API stops the load instead of warning."""
    monkeypatch.setitem(sys.modules, "transformers.conversion_mapping", None)

    with pytest.raises(ImportError):
        kimi_k3._register_kimi_k3_conversions()


@pytest.mark.usefixtures("private_conversion_registry")
def test_the_plan_the_conversions_and_the_fused_params_name_the_same_tensors():
    """Three separately written names must agree, or EP silently shards nothing: the fused module's
    parameters, the conversion targets that fill them, and the EP plan globs (matched the way
    transformers matches them). The router, latent projections and shared experts stay replicated."""
    remote = remote_module()
    kimi_k3._patch_kimi_moe_block(remote)
    layer = nn.Module()
    layer.block_sparse_moe = remote.KimiSparseMoeBlock(_config())
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    kimi_k3._register_kimi_k3_conversions()
    converters = cm.get_checkpoint_conversion_mapping("kimi_linear")
    plan = kimi_k3.BASE_MODEL_EP_PLAN
    matcher, glob_of_group, _ = build_glob_alternation(list(plan))

    def style(name):
        match = matcher.search(name)
        return plan[glob_of_group[match.lastgroup]] if match else None

    sharded = {name for name, _ in model.named_parameters() if style(name) is not None}

    assert sharded == {
        "layers.0.block_sparse_moe.experts.gate_up_proj",
        "layers.0.block_sparse_moe.experts.down_proj",
    }
    assert {style(name) for name in sharded} == {"grouped_gemm"}
    assert {name.removeprefix("layers.0.") for name in sharded} == {
        target for conv in converters for target in conv.target_patterns
    }
    assert style("layers.0.block_sparse_moe.experts") == "moe_tp_experts"


# ------------------------------------------------------------------------------------- prepare

_AUTO_MAP = {"AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration"}


@pytest.fixture
def remote_checkpoint(monkeypatch):
    """The checkpoint's remote code, a fresh stand-in that importing it registers; the shims
    prepare installs on transformers are undone after the test."""
    remote = remote_module("transformers_modules.stub.modeling_kimi_k3")

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        monkeypatch.setitem(sys.modules, remote.__name__, remote)
        return remote.KimiK3ForConditionalGeneration

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    _restore(monkeypatch, hf_generic, "OutputRecorder")
    return remote


def _prepare(config):
    return kimi_k3.prepare(config, ckpt_path="/ckpt", trust_remote_code=True)


@pytest.mark.usefixtures("private_conversion_registry", "remote_checkpoint")
def test_prepare_sets_the_plan_and_a2a_experts_and_hands_the_quant_config_to_the_decompressor(
    monkeypatch,
):
    """The plan goes on the text config (its globs are written against the language model) and is
    the one the spec states for a run that skips prepare, the experts forward is ``torch_a2a``, the
    kimi_linear fusion is registered, and the decompressor sees the compressed-tensors config before
    it is dropped (so transformers builds no competing quantizer)."""
    seen = []
    monkeypatch.setattr(
        _onfly_decompress,
        "install_onfly_decompress",
        lambda cfg: seen.append((cfg.quantization_config, cfg.text_config.quantization_config)),
    )
    quant = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}
    text = SimpleNamespace(quantization_config=quant.copy())
    config = SimpleNamespace(text_config=text, quantization_config=quant.copy(), auto_map=_AUTO_MAP)

    assert _prepare(config) is config

    assert text.base_model_ep_plan == kimi_k3.BASE_MODEL_EP_PLAN
    assert text.base_model_ep_plan is not kimi_k3.BASE_MODEL_EP_PLAN
    assert text.base_model_ep_plan == get_spec("kimi_k3").ep_spec.prepared_ep_plan
    assert not hasattr(config, "base_model_ep_plan")
    assert text._experts_implementation == "torch_a2a"
    registered = cm.get_checkpoint_conversion_mapping("kimi_linear")
    assert any("block_sparse_moe.experts.*.w1.weight" in c.source_patterns for c in registered)
    assert seen == [(quant, quant)]
    assert config.quantization_config is None
    assert text.quantization_config is None


@pytest.fixture
def torch_a2a_unregistered(monkeypatch):
    """The process as the plan decision finds it, before anything registered torch_a2a; what the
    test registers is dropped after it."""
    experts = type(ALL_EXPERTS_FUNCTIONS)
    registered = {k: v for k, v in experts._global_mapping.items() if k != "torch_a2a"}
    monkeypatch.setattr(experts, "_global_mapping", registered)
    monkeypatch.setattr(moe, "_TORCH_A2A_REGISTERED", False)


@pytest.mark.usefixtures(
    "private_conversion_registry", "remote_checkpoint", "torch_a2a_unregistered"
)
def test_a_model_builds_from_the_prepared_config_before_the_expert_parallel_load():
    """The plan decision builds the model from prepare's config on the meta device before the
    expert-parallel load registers torch_a2a, and transformers refuses to build a model whose experts
    implementation it has not registered: prepare registers the one it names."""
    text = SimpleNamespace()
    _prepare(SimpleNamespace(text_config=text, auto_map=_AUTO_MAP))

    # Kimi-K3's own model is the checkpoint's code; every transformers model runs the same check.
    config = transformers.Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=16,
    )
    config._experts_implementation = text._experts_implementation
    with torch.device("meta"):
        transformers.Qwen3MoeForCausalLM(config)


@pytest.mark.usefixtures("private_conversion_registry", "remote_checkpoint")
def test_prepare_keeps_a_plan_the_config_already_has(monkeypatch):
    """A plan the checkpoint's config already carries describes its own layout and wins."""
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    text = SimpleNamespace(base_model_ep_plan={"layers.*.custom": "grouped_gemm"})

    _prepare(SimpleNamespace(text_config=text, auto_map=_AUTO_MAP))

    assert text.base_model_ep_plan == {"layers.*.custom": "grouped_gemm"}


@pytest.mark.usefixtures("private_conversion_registry")
def test_prepare_with_a_checkpoint_patches_the_loaded_remote_module_once(monkeypatch):
    """Given a checkpoint, prepare imports the remote modeling through the class the config's
    ``auto_map`` names, then patches the module that *defines* the MoE block (found by class, not
    by file name, and not a VLM wrapper that merely imports it) -- a repeat prepare patches nothing
    twice -- and leaves the remote-code staging shim installed."""
    remote = remote_module()
    wrapper = types.ModuleType("kimi_k3_remote_stub_vlm")  # imports the block, defines nothing
    wrapper.KimiSparseMoeBlock = remote.KimiSparseMoeBlock
    loads = []

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        loads.append((class_ref, ckpt_path, kwargs))
        for module in (wrapper, remote):  # what importing the remote code does
            monkeypatch.setitem(sys.modules, module.__name__, module)
        return remote.KimiK3ForConditionalGeneration

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    staging = dmu.get_cached_module_file
    monkeypatch.setattr(dmu, "get_cached_module_file", staging)
    _restore(monkeypatch, hf_generic, "OutputRecorder")
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    auto_map = {"AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration"}

    for _ in range(2):
        config = SimpleNamespace(text_config=SimpleNamespace(), auto_map=auto_map)
        kimi_k3.prepare(config, ckpt_path="/ckpt", trust_remote_code=True)

    assert loads == [(auto_map["AutoModelForCausalLM"], "/ckpt", {"trust_remote_code": True})] * 2
    assert dmu.get_cached_module_file is not staging
    assert hasattr(hf_generic, "OutputRecorder")  # the remote code imports it from there
    # every incompatibility is patched on the module the checkpoint's code defined
    assert not isinstance(remote.KimiSparseMoeBlock(_config()).experts, nn.ModuleList)
    a_log = remote.KimiDeltaAttention(num_heads=3, head_dim=4).A_log
    assert (a_log.shape, a_log.dtype) == (
        (4,),
        torch.bfloat16,
    )  # the checkpoint's shape, same dtype
    remote.chunk_kda(1, 2, 3, 4, beta=torch.zeros(1), use_beta_sigmoid_in_kernel=True)
    _, beta, flag = remote.calls.pop()
    assert (beta.item(), flag) == (0.5, False)
    remote.create_causal_mask(
        config=None, input_embeds=None, attention_mask=None, past_key_values=None, cache_position=0
    )
    assert remote.KimiDynamicCache(seen_tokens=1).get_mask_sizes(2, 0) == (3, 0)
    vlm = remote.KimiK3ForConditionalGeneration()
    assert vlm.tie_weights(recompute_mapping=False) == "tied"
    assert [call for call in remote.calls if call[0] == "tie_weights"] == [("tie_weights", vlm)]


def test_prepare_fails_loudly_when_no_loaded_module_defines_the_moe_block(monkeypatch):
    """Skipping the patch silently would load the per-expert layout that EP cannot shard."""

    class KimiK3ForConditionalGeneration:
        __module__ = "transformers_modules.no_moe_block.modeling_kimi_k3"

    monkeypatch.setattr(
        kimi_k3, "_force_load_remote_modeling", lambda *args: KimiK3ForConditionalGeneration
    )

    with pytest.raises(RuntimeError, match="KimiSparseMoeBlock"):
        kimi_k3.prepare(
            SimpleNamespace(text_config=SimpleNamespace()),
            ckpt_path="/ckpt",
            trust_remote_code=True,
        )


@pytest.mark.usefixtures("private_conversion_registry")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # Without the checkpoint there is no modeling code to patch: the plan and the fusion would
        # name fused experts the unpatched per-expert modeling never builds.
        ({"trust_remote_code": True}, "needs the checkpoint's path"),
        # transformers' get_class_from_dynamic_module runs the checkpoint's code without checking
        # trust_remote_code.
        ({"ckpt_path": "/ckpt"}, "trust_remote_code=True"),
    ],
    ids=["no-checkpoint", "untrusted-checkpoint"],
)
def test_prepare_refuses_before_changing_anything(monkeypatch, kwargs, match):
    monkeypatch.setattr(
        kimi_k3,
        "_force_load_remote_modeling",
        lambda *args: pytest.fail("the checkpoint's code was imported"),
    )
    text = SimpleNamespace()

    with pytest.raises(ValueError, match=match):
        kimi_k3.prepare(SimpleNamespace(text_config=text), **kwargs)
    assert vars(text) == {}
    registered = cm.get_checkpoint_conversion_mapping("kimi_linear") or []
    assert not any("block_sparse_moe.experts.*.w1.weight" in c.source_patterns for c in registered)


def test_a_transformers_older_than_the_validated_one_is_refused_before_prepare(monkeypatch):
    """The spec states the release the adapter's patches were validated against, so a run on an
    older one is refused, with its way around, before prepare touches anything."""
    monkeypatch.setattr(transformers, "__version__", "5.12.0")

    with pytest.raises(RuntimeError, match=r"transformers >= 5\.13"):
        check_distributed_support("kimi_k3")


@pytest.mark.usefixtures("private_conversion_registry")
def test_each_checkpoint_revision_patches_its_own_remote_module(monkeypatch):
    """Two revisions of the remote code import as two packages, each defining its own MoE block. Each
    checkpoint's prepare patches its own revision's module, whichever was imported first, and a
    repeat for the first checkpoint replaces nothing that is already patched."""
    revisions = {
        f"/ckpt/{rev}": remote_module(f"transformers_modules.{rev}.modeling_kimi")
        for rev in ("rev_a", "rev_b")
    }

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        remote = revisions[ckpt_path]
        monkeypatch.setitem(sys.modules, remote.__name__, remote)
        return remote.KimiK3ForConditionalGeneration

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    _restore(monkeypatch, hf_generic, "OutputRecorder")
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    auto_map = {"AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration"}

    def prepare(ckpt_path):
        config = SimpleNamespace(text_config=SimpleNamespace(), auto_map=auto_map)
        kimi_k3.prepare(config, ckpt_path=ckpt_path, trust_remote_code=True)

    def patched_surface(remote):
        return (
            remote.KimiSparseMoeBlock.__init__,
            remote.KimiSparseMoeBlock.forward,
            remote.KimiDeltaAttention.__init__,
            remote.chunk_kda,
            remote.fused_recurrent_kda,
            remote.create_causal_mask,
            remote.KimiDynamicCache.get_mask_sizes,
            remote.KimiK3ForConditionalGeneration.tie_weights,
        )

    rev_a, rev_b = revisions["/ckpt/rev_a"], revisions["/ckpt/rev_b"]
    shipped_b = patched_surface(rev_b)
    prepare("/ckpt/rev_a")
    assert patched_surface(rev_b) == shipped_b
    prepare("/ckpt/rev_b")
    patched = patched_surface(rev_a), patched_surface(rev_b)
    prepare("/ckpt/rev_a")

    assert (patched_surface(rev_a), patched_surface(rev_b)) == patched
    assert all(new is not old for new, old in zip(patched_surface(rev_b), shipped_b))
    for remote in (rev_a, rev_b):
        assert not isinstance(remote.KimiSparseMoeBlock(_config()).experts, nn.ModuleList)
        assert (
            remote.KimiK3ForConditionalGeneration().tie_weights(recompute_mapping=False) == "tied"
        )


def test_the_remote_modeling_is_imported_through_the_class_the_auto_map_declares(monkeypatch):
    """``AutoModelForCausalLM`` is preferred, then ``AutoModel``; a config declaring neither is an
    error rather than a load that silently skips the patches."""
    loads = []
    monkeypatch.setattr(
        dmu, "get_class_from_dynamic_module", lambda ref, path, **kwargs: loads.append(ref)
    )
    monkeypatch.setattr(dmu, "get_cached_module_file", dmu.get_cached_module_file)
    _restore(monkeypatch, hf_generic, "OutputRecorder")

    for auto_map in (
        {"AutoModel": "remote.Base", "AutoModelForCausalLM": "remote.ForCausalLM"},
        {"AutoModel": "remote.Base"},
    ):
        kimi_k3._force_load_remote_modeling(SimpleNamespace(auto_map=auto_map), "/ckpt", False)

    assert loads == ["remote.ForCausalLM", "remote.Base"]
    config_only = SimpleNamespace(auto_map={"AutoConfig": "remote.Config"})
    with pytest.raises(RuntimeError, match="auto_map"):
        kimi_k3._force_load_remote_modeling(config_only, "/ckpt", False)


@pytest.mark.usefixtures("private_conversion_registry")
def test_prepare_lets_an_error_reading_the_quant_config_through(monkeypatch, remote_checkpoint):
    """compressed_tensors raises pydantic's ValidationError, a ValueError, for a quantization_config
    it cannot parse. Dropped after that, the config would load the experts newly initialized. The
    refusal comes before prepare patches the remote block or registers its converters, so a process
    that goes on after it is unchanged."""

    def install(cfg):
        raise ValueError("1 validation error for QuantizationConfig")

    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", install)
    config = SimpleNamespace(
        text_config=SimpleNamespace(), quantization_config={"format": "new"}, auto_map=_AUTO_MAP
    )

    with pytest.raises(ValueError, match="validation error"):
        _prepare(config)
    assert isinstance(remote_checkpoint.KimiSparseMoeBlock(_config()).experts, nn.ModuleList)
    registered = cm.get_checkpoint_conversion_mapping("kimi_linear") or []
    assert not any("block_sparse_moe.experts.*.w1.weight" in c.source_patterns for c in registered)


@pytest.mark.usefixtures("private_conversion_registry", "restore_loader_hooks", "remote_checkpoint")
def test_prepare_leaves_the_loader_alone_for_bf16_experts():
    """Without a compressed-tensors config the experts are plain bf16: prepare completes and leaves
    transformers' conversion mapping unwrapped, with or without ``compressed_tensors`` installed."""
    stock = mu.get_model_conversion_mapping

    _prepare(SimpleNamespace(text_config=SimpleNamespace(), auto_map=_AUTO_MAP))

    assert mu.get_model_conversion_mapping is stock


def test_the_kimi_k3_spec_routes_expert_parallelism_through_this_adapter():
    """kimi_k3 is remote code, so its whole EP setup is this adapter's ``prepare``: another Kimi
    adapter's would load it without its own patches."""
    assert get_spec("kimi_k3").ep_spec.prepare is kimi_k3.prepare
