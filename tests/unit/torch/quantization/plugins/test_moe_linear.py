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

"""Tests for _QuantMoELinear: expert-indexed MoE weights (Step-3.5 / Step-3.7 remote code)."""

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("transformers")

import modelopt.torch.quantization as mtq
from modelopt.torch.export.hf_export_handlers import _export_moe_linear
from modelopt.torch.export.registry import ExportContext, ExportModuleRegistry
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import (
    _is_expert_indexed_moe_linear,
    _QuantMoELinear,
    _reconstruct_fused_moe_linear,
    register_moe_linear_on_the_fly,
)

NUM_EXPERTS = 4
HIDDEN_SIZE = 32
MOE_INTERMEDIATE_SIZE = 16
TOP_K = 2


class _SyntheticMoELinear(nn.Module):
    """Mimics Step-3.5 / Step-3.7 ``MoELinear`` (verbatim layout from their remote code)."""

    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features) * 0.02)

    def forward(self, x, expert_id):
        return F.linear(x.float(), self.weight[expert_id].float())


class _SyntheticStepMoEMLP(nn.Module):
    """Mimics ``Step3p7MoEMLP``: a router plus three expert-indexed projections."""

    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.gate = nn.Linear(HIDDEN_SIZE, NUM_EXPERTS, bias=False)
        self.act_fn = nn.SiLU()
        self.up_proj = _SyntheticMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        self.gate_proj = _SyntheticMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        self.down_proj = _SyntheticMoELinear(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)

    def forward(self, hidden_states):
        tokens = hidden_states.view(-1, HIDDEN_SIZE)
        routing = F.softmax(self.gate(tokens).float(), dim=-1)
        weights, indices = torch.topk(routing, self.top_k, dim=-1)
        out = torch.zeros_like(tokens)
        for expert_id in range(self.num_experts):
            pos, token_idx = torch.where(indices == expert_id)
            if token_idx.numel() == 0:
                continue
            current = tokens[pos]
            gate = self.act_fn(self.gate_proj(current, expert_id))
            up = self.up_proj(current, expert_id)
            expert_out = self.down_proj(gate * up, expert_id)
            out.index_add_(0, pos, (expert_out * weights[pos, token_idx, None]).to(out.dtype))
        return out.view_as(hidden_states)


class _TinyStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        # register_moe_linear_on_the_fly gates on the Step-family model_type; real Step
        # checkpoints carry this in config.json.
        self.config = types.SimpleNamespace(model_type="step3p7")
        self.moe = _SyntheticStepMoEMLP()

    def forward(self, x):
        return self.moe(x)


@pytest.fixture(autouse=True)
def _unregister_synthetic_moe_linear():
    """Keep the on-the-fly registration from leaking into other tests."""
    yield
    if QuantModuleRegistry.get(_SyntheticMoELinear) is not None:
        QuantModuleRegistry.unregister(_SyntheticMoELinear)


def _moe_quant_cfg():
    """Per-tensor INT8 on the expert projections only — CPU-friendly, no kernels needed."""
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*moe*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*moe*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            {"quantizer_name": "*moe.gate.*", "enable": False},
        ],
        "algorithm": "max",
    }


def test_expert_indexed_moe_linear_is_detected():
    assert _is_expert_indexed_moe_linear(
        _SyntheticMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    )


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), id="plain_linear_2d_weight"),
        pytest.param(nn.LayerNorm(HIDDEN_SIZE), id="norm_1d_weight"),
    ],
)
def test_unrelated_modules_are_not_claimed(module):
    assert not _is_expert_indexed_moe_linear(module)


def test_module_with_3d_weight_but_other_forward_is_not_claimed():
    """A 3-D weight alone is not enough — the forward must take ``(x, expert_id)``."""

    class _NotExpertIndexed(_SyntheticMoELinear):
        def forward(self, x, top_k_index, top_k_weights):
            return x

    assert not _is_expert_indexed_moe_linear(
        _NotExpertIndexed(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    )


def test_disabled_quantizers_reproduce_bf16_weight_fp32_compute_parity():
    """Conversion must not change the model's output when every quantizer is disabled, and
    must not permanently promote expert storage to fp32 to get there.

    ``MoELinear.forward`` always promotes to fp32 for the matmul regardless of storage
    dtype (``F.linear(x.float(), self.weight[expert_id].float())``). A wrapper that
    instead downcasts the fp32 activation to the weight's original storage dtype (e.g.
    bf16) before the matmul silently changes the model even with quantization off. The
    opposite mistake -- promoting every expert's *storage* to fp32 in `_setup` to match --
    reproduces Step's numerics but doubles the model's expert-weight memory footprint for
    its entire lifetime (on Step-3.7's full routed-expert set, ~354 GiB); the promotion
    must be transient, scoped to the one expert actually being called.
    """
    torch.manual_seed(0)
    num_experts, in_features, out_features = 2, 4096, 1280
    module = _SyntheticMoELinear(num_experts, in_features, out_features)
    module.weight.data = module.weight.data.to(torch.bfloat16)
    module.config = types.SimpleNamespace(model_type="step3p7")  # satisfy the family gate
    x = torch.randn(8, in_features, dtype=torch.bfloat16)
    reference = module(x, 0)

    mtq.quantize(module, {"quant_cfg": [{"quantizer_name": "*", "enable": False}]})
    assert isinstance(module, _QuantMoELinear), "conversion did not happen; test is vacuous"

    # Storage stays at the checkpoint's own dtype -- only the matmul promotes, transiently.
    for expert in module.experts:
        assert expert.weight.dtype == torch.bfloat16

    converted = module(x, 0)
    assert torch.equal(converted, reference)

    # Reconstruction (export) must also see -- and keep -- the original storage dtype, not
    # a permanently-promoted one.
    _reconstruct_fused_moe_linear(module)
    assert module.weight.dtype == torch.bfloat16


def test_grouped_routing_module_is_not_claimed():
    """A grouped-GEMM MoE layer has the identical shape but no scalar-index contract.

    Moondream3's ``MoeFusedLinear`` carries the same 3-D weight and the same three
    attributes, but its second argument is a per-expert token-count *tensor*. Claiming it
    would make ``_QuantMoELinear.forward`` evaluate ``self.experts[m_sizes]`` and raise
    ``TypeError: only integer tensors of a single element can be converted to an index``
    on the first calibration forward.
    """

    class _MoeFusedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = NUM_EXPERTS
            self.in_features = HIDDEN_SIZE
            self.out_features = MOE_INTERMEDIATE_SIZE
            self.weight = nn.Parameter(
                torch.randn(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE) * 0.02
            )

        def forward(self, input, m_sizes):
            return input

    assert not _is_expert_indexed_moe_linear(_MoeFusedLinear())


def test_extra_forward_parameters_are_not_claimed():
    """The replacement forward is exactly ``(x, expert_id)``.

    Anything else the caller could pass — a keyword-only `router_state`, `*args`,
    `**kwargs` — would raise `TypeError` once the module is converted.
    """

    class _WithRouterState(_SyntheticMoELinear):
        def forward(self, x, expert_id, *, router_state=None):
            return x

    class _WithKwargs(_SyntheticMoELinear):
        def forward(self, x, expert_id, **kwargs):
            return x

    for cls in (_WithRouterState, _WithKwargs):
        assert not _is_expert_indexed_moe_linear(
            cls(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        ), cls.__name__


def test_transposed_weight_layout_is_not_claimed():
    """`[num_experts, in_features, out_features]` would rebuild each expert from wrong slices."""

    class _TransposedMoELinear(_SyntheticMoELinear):
        def __init__(self, num_experts, in_features, out_features):
            super().__init__(num_experts, in_features, out_features)
            self.weight = nn.Parameter(torch.randn(num_experts, in_features, out_features) * 0.02)

    assert not _is_expert_indexed_moe_linear(
        _TransposedMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    )


def test_offloaded_weights_are_refused_not_silently_corrupted():
    """Accelerate offload leaves `weight` on meta, with the value in the module's hook.

    Expanding that would copy meta storage into each expert and delete the key the hook
    restores into, exporting a checkpoint of zeros. Conversion must refuse instead.
    """
    pytest.importorskip("accelerate")
    from accelerate import cpu_offload  # local: accelerate is an optional dependency

    model = _TinyStepModel()
    cpu_offload(model.moe.up_proj, execution_device=torch.device("cpu"))
    assert model.moe.up_proj.weight.is_meta

    with pytest.raises(NotImplementedError, match="offloaded by Accelerate"):
        mtq.quantize(model, _moe_quant_cfg(), forward_loop=None)


def test_registration_is_not_gated_on_exact_revision_class_name():
    """Any Step-family root (matched by `model_type`/class-name convention, not an exact
    revision) registers its `MoELinear` modules -- Step-3.7 as readily as Step-3.5."""
    model = _TinyStepModel()
    assert QuantModuleRegistry.get(_SyntheticMoELinear) is None

    register_moe_linear_on_the_fly(model)

    assert issubclass(QuantModuleRegistry.get(_SyntheticMoELinear), _QuantMoELinear)


def test_non_step_model_with_identical_signature_is_not_registered():
    """A same-shape, same-signature module is not enough on its own to be claimed.

    The structural check in `_is_expert_indexed_moe_linear` cannot tell a real Step
    `MoELinear` apart from unrelated code that happens to reuse the `(x, expert_id)`
    parameter names with different semantics (a per-expert bias or post-scale, say) --
    `_QuantMoELinear` would silently drop that behavior. Registration is therefore also
    gated on the model being Step-family; a structurally identical module on a model that
    is not must not be registered.
    """

    class _ThirdPartyMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="not_step")
            self.moe = _SyntheticStepMoEMLP()

        def forward(self, x):
            return self.moe(x)

    model = _ThirdPartyMoEModel()
    register_moe_linear_on_the_fly(model)

    assert QuantModuleRegistry.get(_SyntheticMoELinear) is None


def test_local_hessian_calibration_fires_through_the_transient_weight_swap():
    """`forward` must keep calling `expert(x)` (`__call__`), not `.forward()` directly.

    `local_hessian_calibrate` registers a `forward_pre_hook` on each quantized Linear
    module and relies on standard `nn.Module.__call__` dispatch to fire it. A `forward`
    that bypassed `__call__` -- e.g. to reimplement the input/weight-quantize/output-quantize
    sequence inline instead of transiently swapping the expert's weight storage -- would
    silently skip this hook and leave the weight quantizer uncalibrated (amax stays None).
    """
    torch.manual_seed(0)
    model = _TinyStepModel()
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*moe*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*moe*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            {"quantizer_name": "*moe.gate.*", "enable": False},
        ],
        "algorithm": "local_hessian",
    }

    def forward_loop(m):
        for _ in range(3):
            m(torch.randn(2, 8, HIDDEN_SIZE))

    mtq.quantize(model, cfg, forward_loop=forward_loop)

    for expert in model.moe.up_proj.experts:
        assert expert.weight_quantizer.amax is not None


def test_expert_indexed_moe_is_quantized_and_reconstructed():
    """Each expert gets its own quantizers, and export folds them back to the 3-D layout."""
    torch.manual_seed(0)
    model = _TinyStepModel()
    reference_weight = model.moe.up_proj.weight.detach().clone()

    def forward_loop(m):
        m(torch.randn(2, 8, HIDDEN_SIZE))

    mtq.quantize(model, _moe_quant_cfg(), forward_loop=forward_loop)

    # Every expert of every projection carries its own calibrated quantizer pair.
    for proj in ("up_proj", "gate_proj", "down_proj"):
        experts = getattr(model.moe, proj).experts
        assert len(experts) == NUM_EXPERTS
        for expert in experts:
            assert expert.weight_quantizer.is_enabled
            assert expert.weight_quantizer.amax is not None
    # The router stays untouched.
    assert not model.moe.gate.weight_quantizer.is_enabled

    _reconstruct_fused_moe_linear(model)

    # Back to the original ``[num_experts, out_features, in_features]`` parameter, so the
    # exported keys match the hub checkpoint instead of per-expert names.
    up_proj = model.moe.up_proj
    assert not hasattr(up_proj, "experts")
    assert up_proj.weight.shape == reference_weight.shape
    assert torch.equal(up_proj.weight, reference_weight)


def test_export_handler_matches_a_differently_named_wrapper():
    """Export dispatch must key on the wrapper type, not the generated class name.

    The registration is structural, so a compatible remote-code class can be named
    anything; its generated class is then ``Quant<ThatName>``. If the export handler only
    matched the literal name ``QuantMoELinear``, such a module would skip
    ``_export_moe_linear`` and export without the input-amax fallback for experts that
    calibration never routed to.
    """
    model = _TinyStepModel()
    mtq.quantize(model, _moe_quant_cfg(), forward_loop=lambda m: m(torch.randn(2, 8, HIDDEN_SIZE)))

    converted = model.moe.up_proj
    # The generated name is derived from the model's own class, not from `MoELinear`.
    assert type(converted).__name__ == "Quant_SyntheticMoELinear"
    assert ExportModuleRegistry.match(converted) is _export_moe_linear


def test_export_handler_fills_input_amax_for_unrouted_experts():
    """The handler is what gives never-routed experts an input amax before export."""
    torch.manual_seed(0)
    model = _TinyStepModel()
    mtq.quantize(model, _moe_quant_cfg(), forward_loop=lambda m: m(torch.randn(2, 8, HIDDEN_SIZE)))

    experts = model.moe.up_proj.experts
    # Simulate an expert that calibration never routed a token to.
    experts[0].input_quantizer.reset_amax()
    assert experts[0].input_quantizer.amax is None
    assert any(e.input_quantizer.amax is not None for e in experts), "need a donor amax"

    _export_moe_linear("moe.up_proj", model.moe.up_proj, ExportContext(model, torch.float16))

    assert experts[0].input_quantizer.amax is not None
