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
import torch.nn.functional as F
from torch import nn

transformers = pytest.importorskip("transformers")

from packaging.version import Version

if Version(transformers.__version__) < Version("5.6"):
    pytest.skip("Kimi-K3 expert parallelism needs transformers >= 5.6", allow_module_level=True)

import transformers.conversion_mapping as cm
import transformers.core_model_loading as cml
import transformers.dynamic_module_utils as dmu
import transformers.modeling_utils as mu
import transformers.utils.generic as hf_generic
from transformers.core_model_loading import WeightConverter, build_glob_alternation
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

from modelopt.torch.models import _onfly_decompress, get_spec
from modelopt.torch.models.kimi_k3 import ep as kimi_k3

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
    for owner, name in (
        (WeightConverter, "add_tensor"),
        (WeightConverter, "materialize_tensors"),
        (cml, "_materialize_copy"),
        (mu, "get_model_conversion_mapping"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))


def _situ(gate_up, beta, linear_beta):
    """The stand-in ``situ`` gated activation over a full [gate; up] row."""
    gate, up = gate_up.chunk(2, dim=-1)
    return gate * torch.sigmoid(beta * gate) * (up + linear_beta)


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
        "use_latent_moe": True,
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


def _remote_module(name="kimi_k3_remote_stub"):
    """A stand-in for the remote modeling namespace: everything the adapter patches, as shipped."""
    remote = types.ModuleType(name)
    remote.calls = []

    def _get_situ_activation_params(config):
        return config.situ_beta, config.situ_linear_beta

    class SituAndMul(nn.Module):
        def __init__(self, beta, linear_beta):
            super().__init__()
            self.beta, self.linear_beta = beta, linear_beta

        def forward(self, gate_up):
            return _situ(gate_up, self.beta, self.linear_beta)

    class KimiBlockSparseMLP(nn.Module):
        """One expert as the checkpoint stores it: w1 = gate, w3 = up, w2 = down."""

        def __init__(self, config, hidden):
            super().__init__()
            inter = config.moe_intermediate_size
            self.w1 = nn.Linear(hidden, inter, bias=False)
            self.w3 = nn.Linear(hidden, inter, bias=False)
            self.w2 = nn.Linear(inter, hidden, bias=False)
            situ = config.hidden_act == "situ"
            self.situ_params = _get_situ_activation_params(config) if situ else None

        def forward(self, x):
            gate, up = self.w1(x), self.w3(x)
            if self.situ_params is None:
                return self.w2(F.silu(gate) * up)
            return self.w2(_situ(torch.cat([gate, up], dim=-1), *self.situ_params))

    class KimiMoEGate(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.top_k = config.num_experts_per_token
            self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            weights, idx = torch.topk(torch.sigmoid(flat @ self.weight.t()), self.top_k, dim=-1)
            return idx, weights

    class KimiSparseMoeBlock(nn.Module):
        """Latent projections around a per-expert Python loop, plus always-on shared experts."""

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.use_latent_moe = config.use_latent_moe
            self.latent_moe_use_norm = config.latent_moe_use_norm
            hidden = config.hidden_size
            width = config.routed_expert_hidden_size if config.use_latent_moe else hidden
            self.experts = nn.ModuleList(
                KimiBlockSparseMLP(config, width) for _ in range(config.num_experts)
            )
            self.gate = KimiMoEGate(config)
            if config.use_latent_moe:
                self.routed_expert_down_proj = nn.Linear(hidden, width, bias=False)
                self.routed_expert_up_proj = nn.Linear(width, hidden, bias=False)
                if config.latent_moe_use_norm:
                    self.routed_expert_norm = nn.LayerNorm(width)
            if config.num_shared_experts is not None:
                self.shared_experts = KimiBlockSparseMLP(config, hidden)

        def forward(self, hidden_states):
            idx, weights = self.gate(hidden_states)
            x = hidden_states.reshape(-1, hidden_states.shape[-1])
            if self.use_latent_moe:
                x = self.routed_expert_down_proj(x)
            y = torch.stack(
                [
                    sum(w * self.experts[int(e)](token) for e, w in zip(top_e, top_w))
                    for token, top_e, top_w in zip(x, idx, weights)
                ]
            )
            if self.use_latent_moe:
                if self.latent_moe_use_norm:
                    y = self.routed_expert_norm(y)
                y = self.routed_expert_up_proj(y)
            y = y.reshape(hidden_states.shape)
            if self.config.num_shared_experts is not None:
                y = y + self.shared_experts(hidden_states)
            return y

    class KimiDeltaAttention(nn.Module):
        def __init__(self, num_heads, head_dim):
            super().__init__()
            self.head_dim = head_dim
            self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.bfloat16))  # stale shape

    def chunk_kda(q, k, v, g, beta=None, use_beta_sigmoid_in_kernel=False):
        remote.calls.append(("chunk_kda", beta, use_beta_sigmoid_in_kernel))
        return "chunk_kda"

    def fused_recurrent_kda(q, k, v, g, beta=None, use_beta_sigmoid_in_kernel=False):
        remote.calls.append(("fused_recurrent_kda", beta, use_beta_sigmoid_in_kernel))
        return "fused_recurrent_kda"

    def create_causal_mask(
        config, inputs_embeds, attention_mask, past_key_values, position_ids=None
    ):
        # current transformers' signature: ``inputs_embeds``, and no ``cache_position``
        args = (config, inputs_embeds, attention_mask, past_key_values, position_ids)
        remote.calls.append(("create_causal_mask", *args))
        return "mask"

    class KimiDynamicCache:
        def __init__(self, seen_tokens):
            self.seen_tokens = seen_tokens

        def get_seq_length(self, layer_idx=0):
            remote.calls.append(("get_seq_length", layer_idx))
            return self.seen_tokens

        def get_mask_sizes(self, cache_position, layer_idx):  # wants a tensor, as older callers
            return cache_position.shape[0] + self.get_seq_length(layer_idx), 0

    class KimiK3ForConditionalGeneration:
        def tie_weights(self):  # the remote override, with the old no-kwarg signature
            remote.calls.append(("tie_weights", self))
            return "tied"

    for obj in (
        _get_situ_activation_params,
        SituAndMul,
        KimiBlockSparseMLP,
        KimiMoEGate,
        KimiSparseMoeBlock,
        KimiDeltaAttention,
        chunk_kda,
        fused_recurrent_kda,
        create_causal_mask,
        KimiDynamicCache,
        KimiK3ForConditionalGeneration,
    ):
        obj.__module__ = name  # defined in the remote module, as far as any lookup can tell
        setattr(remote, obj.__name__, obj)
    return remote


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


@pytest.mark.parametrize("hidden_act", ["silu", "situ"])
def test_fused_experts_match_a_per_expert_reference(hidden_act):
    """Per token: the routing-weighted sum over its top-k experts of down(act(gate x) * up x), with
    gate read from rows [0:I] and up from rows [I:2I] of ``gate_up_proj``. A ``situ`` model hands
    the full [gate; up] row to the remote ``SituAndMul``. Experts live in the latent width."""
    config = _config(hidden_act=hidden_act)
    experts = kimi_k3._build_fused_experts_class(_remote_module())(config)
    n, inter, latent = 4, config.moe_intermediate_size, config.routed_expert_hidden_size
    assert experts.gate_up_proj.shape == (n, 2 * inter, latent)
    assert experts.down_proj.shape == (n, latent, inter)
    _randomize(experts)
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randn(len(ROUTES), latent, generator=gen)
    route_weights = torch.rand(ROUTES.shape, generator=gen)

    def act(gate_up):
        if hidden_act == "situ":
            return _situ(gate_up, config.situ_beta, config.situ_linear_beta)
        return F.silu(gate_up[:inter]) * gate_up[inter:]

    with torch.no_grad():
        expected = torch.stack(
            [
                sum(
                    w * (experts.down_proj[e] @ act(experts.gate_up_proj[e] @ x))
                    for e, w in zip(top_e, top_w)
                )
                for x, top_e, top_w in zip(tokens, ROUTES, route_weights)
            ]
        )
        torch.testing.assert_close(experts(tokens, ROUTES, route_weights), expected)


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
    experts = kimi_k3._build_fused_experts_class(_remote_module())(config)
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
            "use_latent_moe": False,
            "latent_moe_use_norm": False,
            "routed_expert_hidden_size": None,
            "num_shared_experts": None,
        },
    ],
    ids=["latent-norm-shared", "plain"],
)
def test_the_patched_block_reproduces_the_remote_per_expert_forward(overrides):
    """Swapping the per-expert ModuleList for the fused module keeps the block's output: the latent
    projections, the latent norm and the shared experts stay around the fused call, and w1 / w3 /
    w2 are the gate rows, the up rows and the down of the fused params."""
    remote = _remote_module()
    config = _config(**overrides)
    reference = remote.KimiSparseMoeBlock(config)
    _randomize(reference)
    gen = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(2, 3, config.hidden_size, generator=gen)
    with torch.no_grad():
        expected = reference(hidden_states)

    kimi_k3._patch_kimi_moe_block(remote)
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


def test_patching_the_block_twice_wraps_it_once():
    """A second ``prepare`` must not stack another fused-experts allocation onto ``__init__``."""
    remote = _remote_module()
    kimi_k3._patch_kimi_moe_block(remote)
    patched = (remote.KimiSparseMoeBlock.__init__, remote.KimiSparseMoeBlock.forward)

    kimi_k3._patch_kimi_moe_block(remote)

    assert (remote.KimiSparseMoeBlock.__init__, remote.KimiSparseMoeBlock.forward) == patched


# ------------------------------------------------------------------- the linear-attention fixes


def test_kda_beta_is_squashed_in_torch_only_when_the_kernel_flag_asks_for_it():
    """fla-core ignores ``use_beta_sigmoid_in_kernel``, so both KDA entry points apply
    ``sigmoid(beta)`` themselves (in beta's dtype) and clear the flag; calls without the flag pass
    beta through untouched. Each entry point is wrapped once."""
    remote = _remote_module()
    kimi_k3._patch_kda_beta_sigmoid(remote)
    wrapped = (remote.chunk_kda, remote.fused_recurrent_kda)
    kimi_k3._patch_kda_beta_sigmoid(remote)
    assert (remote.chunk_kda, remote.fused_recurrent_kda) == wrapped
    beta = torch.tensor([-8.0, 0.0, 8.4], dtype=torch.bfloat16)

    assert remote.chunk_kda(1, 2, 3, 4, beta=beta, use_beta_sigmoid_in_kernel=True) == "chunk_kda"
    remote.fused_recurrent_kda(1, 2, 3, 4, beta=beta, use_beta_sigmoid_in_kernel=True)

    squashed = torch.sigmoid(beta.float()).to(torch.bfloat16)
    assert [name for name, *_ in remote.calls] == ["chunk_kda", "fused_recurrent_kda"]
    for _, seen, flag in remote.calls:
        assert flag is False
        assert seen.dtype == torch.bfloat16
        assert torch.equal(seen, squashed)

    remote.calls.clear()
    remote.chunk_kda(1, 2, 3, 4, beta=beta)
    remote.fused_recurrent_kda(1, 2, 3, 4, beta=beta, use_beta_sigmoid_in_kernel=False)
    assert [(seen is beta, flag) for _, seen, flag in remote.calls] == [(True, False)] * 2


def test_kda_beta_passed_positionally_is_squashed_too():
    """The fix must not depend on how the remote code spells the call."""
    remote = _remote_module()
    kimi_k3._patch_kda_beta_sigmoid(remote)
    beta = torch.tensor([-8.0, 8.4], dtype=torch.bfloat16)

    remote.chunk_kda(1, 2, 3, 4, beta, True)

    ((_, seen, flag),) = remote.calls
    assert flag is False
    assert torch.equal(seen, torch.sigmoid(beta.float()).to(torch.bfloat16))


def test_kda_a_log_takes_the_checkpoints_head_dim_shape():
    """The stale port allocates ``A_log`` as [num_heads]; the checkpoint stores [head_dim], which
    would fail to load (or be re-initialized). The reallocation keeps the dtype."""
    remote = _remote_module()
    kimi_k3._patch_kda_a_log(remote)

    attention = remote.KimiDeltaAttention(num_heads=3, head_dim=4)

    assert attention.A_log.shape == (4,)
    assert attention.A_log.dtype == torch.bfloat16


# ------------------------------------------------------------- transformers-version compatibility


def test_create_causal_mask_shim_adapts_the_remote_call_once():
    """The remote code calls ``create_causal_mask(input_embeds=..., cache_position=...)``; current
    transformers takes ``inputs_embeds`` and no ``cache_position``. The shim renames the one, drops
    what the installed signature does not accept, and wraps only once."""
    remote = _remote_module()
    kimi_k3._shim_create_causal_mask(remote)
    shim = remote.create_causal_mask
    kimi_k3._shim_create_causal_mask(remote)
    assert remote.create_causal_mask is shim

    mask = remote.create_causal_mask(
        config="config",
        input_embeds="embeds",
        attention_mask="mask_2d",
        cache_position="positions",
        past_key_values="cache",
        position_ids="ids",
    )

    assert mask == "mask"
    assert remote.calls == [("create_causal_mask", "config", "embeds", "mask_2d", "cache", "ids")]


def test_kimi_cache_mask_sizes_accept_a_query_length_or_a_cache_position():
    """transformers >= 5.13 passes the query length as an int; the remote cache expects a
    ``cache_position`` tensor. Either way kv_length is the query length plus the cached tokens."""
    remote = _remote_module()
    kimi_k3._patch_kimi_cache_get_mask_sizes(remote)
    cache = remote.KimiDynamicCache(seen_tokens=5)

    assert cache.get_mask_sizes(3, 2) == (8, 0)
    assert cache.get_mask_sizes(torch.arange(4), 1) == (9, 0)
    assert remote.calls == [("get_seq_length", 2), ("get_seq_length", 1)]


def test_tie_weights_shim_drops_the_new_kwargs_once(monkeypatch):
    """transformers >= 5.13 calls ``tie_weights(recompute_mapping=...)``; the remote VLM's override
    takes no kwargs. The shim drops them, delegates once per call, and wraps only once."""
    remote = _remote_module()
    monkeypatch.setitem(sys.modules, remote.__name__, remote)
    vlm = remote.KimiK3ForConditionalGeneration
    kimi_k3._shim_tie_weights_kwargs()
    shim = vlm.__dict__["tie_weights"]
    kimi_k3._shim_tie_weights_kwargs()
    assert vlm.__dict__["tie_weights"] is shim

    model = vlm()

    assert model.tie_weights(missing_keys=set(), recompute_mapping=False) == "tied"
    assert remote.calls == [("tie_weights", model)]


def test_tie_weights_shim_leaves_an_inherited_tie_weights_alone(monkeypatch):
    """Only an override defined on the VLM class has the old signature; an inherited
    ``tie_weights`` already takes the kwargs and must keep receiving them."""

    class PreTrainedStub:
        def tie_weights(self, **kwargs):
            return kwargs

    remote = types.ModuleType("kimi_k3_remote_inheriting_stub")
    vlm = type("KimiK3ForConditionalGeneration", (PreTrainedStub,), {"__module__": remote.__name__})
    remote.KimiK3ForConditionalGeneration = vlm
    monkeypatch.setitem(sys.modules, remote.__name__, remote)

    kimi_k3._shim_tie_weights_kwargs()

    assert vlm().tie_weights(recompute_mapping=False) == {"recompute_mapping": False}


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
def test_the_plan_the_conversions_and_the_fused_params_name_the_same_tensors():
    """Three separately written names must agree, or EP silently shards nothing: the fused module's
    parameters, the conversion targets that fill them, and the EP plan globs (matched the way
    transformers matches them). The router, latent projections and shared experts stay replicated."""
    remote = _remote_module()
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


@pytest.mark.usefixtures("private_conversion_registry")
def test_prepare_sets_the_plan_and_a2a_experts_and_hands_the_quant_config_to_the_decompressor(
    monkeypatch,
):
    """The plan goes on the text config (its globs are written against the language model), the
    experts forward is ``torch_a2a``, the kimi_linear fusion is registered, and the decompressor
    sees the compressed-tensors config before it is dropped (so transformers builds no competing
    quantizer)."""
    seen = []
    monkeypatch.setattr(
        _onfly_decompress,
        "install_onfly_decompress",
        lambda cfg: seen.append((cfg.quantization_config, cfg.text_config.quantization_config)),
    )
    quant = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}
    text = SimpleNamespace(quantization_config=quant.copy())
    config = SimpleNamespace(text_config=text, quantization_config=quant.copy())

    assert kimi_k3.prepare(config) is config

    assert text.base_model_ep_plan == kimi_k3.BASE_MODEL_EP_PLAN
    assert text.base_model_ep_plan is not kimi_k3.BASE_MODEL_EP_PLAN
    assert not hasattr(config, "base_model_ep_plan")
    assert text._experts_implementation == "torch_a2a"
    registered = cm.get_checkpoint_conversion_mapping("kimi_linear")
    assert any("block_sparse_moe.experts.*.w1.weight" in c.source_patterns for c in registered)
    assert seen == [(quant, quant)]
    assert config.quantization_config is None
    assert text.quantization_config is None


@pytest.mark.usefixtures("private_conversion_registry")
def test_prepare_keeps_a_plan_the_config_already_has(monkeypatch):
    """A plan the checkpoint's config already carries describes its own layout and wins."""
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    text = SimpleNamespace(base_model_ep_plan={"layers.*.custom": "grouped_gemm"})

    kimi_k3.prepare(SimpleNamespace(text_config=text))

    assert text.base_model_ep_plan == {"layers.*.custom": "grouped_gemm"}


@pytest.mark.usefixtures("private_conversion_registry")
def test_prepare_with_a_checkpoint_patches_the_loaded_remote_module_once(monkeypatch):
    """Given a checkpoint, prepare imports the remote modeling through the class the config's
    ``auto_map`` names, then patches the module that *defines* the MoE block (found by class, not
    by file name, and not a VLM wrapper that merely imports it) -- once per process -- and leaves
    the remote-code staging shim installed."""
    remote = _remote_module()
    wrapper = types.ModuleType("kimi_k3_remote_stub_vlm")  # imports the block, defines nothing
    wrapper.KimiSparseMoeBlock = remote.KimiSparseMoeBlock
    loads = []

    def import_remote_class(class_ref, ckpt_path, **kwargs):
        loads.append((class_ref, ckpt_path, kwargs))
        for module in (wrapper, remote):  # what importing the remote code does
            monkeypatch.setitem(sys.modules, module.__name__, module)

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", import_remote_class)
    staging = dmu.get_cached_module_file
    monkeypatch.setattr(dmu, "get_cached_module_file", staging)
    _restore(monkeypatch, hf_generic, "OutputRecorder")
    monkeypatch.setattr(kimi_k3, "_PATCHED", False)
    monkeypatch.setattr(_onfly_decompress, "install_onfly_decompress", lambda cfg: None)
    auto_map = {"AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration"}

    for _ in range(2):
        config = SimpleNamespace(text_config=SimpleNamespace(), auto_map=auto_map)
        kimi_k3.prepare(config, ckpt_path="/ckpt", trust_remote_code=True)

    assert loads == [(auto_map["AutoModelForCausalLM"], "/ckpt", {"trust_remote_code": True})]
    assert dmu.get_cached_module_file is not staging
    assert hasattr(hf_generic, "OutputRecorder")  # the remote code imports it from there
    # every incompatibility is patched on the module the checkpoint's code defined
    assert not isinstance(remote.KimiSparseMoeBlock(_config()).experts, nn.ModuleList)
    assert remote.KimiDeltaAttention(num_heads=3, head_dim=4).A_log.shape == (4,)
    remote.chunk_kda(1, 2, 3, 4, beta=torch.zeros(1), use_beta_sigmoid_in_kernel=True)
    _, beta, flag = remote.calls.pop()
    assert (beta.item(), flag) == (0.5, False)
    remote.create_causal_mask(
        config=None, input_embeds=None, attention_mask=None, past_key_values=None, cache_position=0
    )
    assert remote.KimiDynamicCache(seen_tokens=1).get_mask_sizes(2, 0) == (3, 0)
    assert remote.KimiK3ForConditionalGeneration().tie_weights(recompute_mapping=False) == "tied"


def test_prepare_fails_loudly_when_no_loaded_module_defines_the_moe_block(monkeypatch):
    """Skipping the patch silently would load the per-expert layout that EP cannot shard."""
    monkeypatch.setattr(kimi_k3, "_PATCHED", False)
    monkeypatch.setattr(kimi_k3, "_force_load_remote_modeling", lambda *args: None)

    with pytest.raises(RuntimeError, match="KimiSparseMoeBlock"):
        kimi_k3.prepare(SimpleNamespace(text_config=SimpleNamespace()), ckpt_path="/ckpt")


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


@pytest.mark.usefixtures("private_conversion_registry", "restore_loader_hooks")
def test_prepare_leaves_the_loader_alone_for_bf16_experts():
    """Without a compressed-tensors config the experts are plain bf16: prepare completes and leaves
    transformers' conversion mapping unwrapped. (Needs ``compressed_tensors`` importable: the
    decompressor imports it before it looks for the config.)"""
    pytest.importorskip("compressed_tensors")
    stock = mu.get_model_conversion_mapping

    kimi_k3.prepare(SimpleNamespace(text_config=SimpleNamespace()))

    assert mu.get_model_conversion_mapping is stock


def test_the_kimi_k3_spec_routes_expert_parallelism_through_this_adapter():
    """kimi_k3 is remote code, so its whole EP setup is this adapter's ``prepare``."""
    spec = get_spec("kimi_k3")
    assert spec.modeling_source == "remote_code"
    assert spec.ep_spec.prepare is kimi_k3.prepare
