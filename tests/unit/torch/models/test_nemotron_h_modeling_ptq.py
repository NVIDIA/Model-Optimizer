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

"""Tests for _QuantNemotronHExperts: Nemotron-H's non-gated fused experts, eager and grouped_mm."""

import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from transformers import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

import modelopt.torch.quantization as mtq
from modelopt.torch.export.moe_utils import _export_fused_experts
from modelopt.torch.export.quant_utils import get_quant_config, get_quantization_format
from modelopt.torch.models.nemotron_h.modeling_ptq import _QuantNemotronHExperts
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import (
    _is_fused_experts_module,
    normalize_experts_impl_on_the_fly,
)
from modelopt.torch.quantization.utils.calib_utils import GPTQHelper
from modelopt.torch.quantization.utils.core_utils import weight_attr_names

NUM_EXPERTS = 4
HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16
TOP_K = 2

# Per-tensor 8-bit inputs, per-channel 8-bit weights, on both projections.
QUANT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*up_proj_input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        {"quantizer_name": "*down_proj_input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        {"quantizer_name": "*up_proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*down_proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    ],
    "algorithm": "max",
}


class _ExpertsModel(nn.Module):
    """A tiny real ``NemotronHExperts`` on the ``impl`` experts backend, with random weights."""

    def __init__(self, impl):
        super().__init__()
        config = NemotronHConfig(
            n_routed_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            hidden_size=HIDDEN_DIM,
            moe_intermediate_size=INTERMEDIATE_DIM,
        )
        config._experts_implementation = impl
        self.experts = NemotronHExperts(config)
        with torch.no_grad():
            for param in self.experts.parameters():
                param.normal_(std=0.02)


def _batch():
    """Tokens routed round-robin, so every expert sees some."""
    seq = 16
    hidden = torch.randn(seq, HIDDEN_DIM)
    top_k_index = torch.stack(
        [torch.arange(seq) % NUM_EXPERTS, (torch.arange(seq) + 1) % NUM_EXPERTS], dim=1
    )
    top_k_weights = torch.softmax(torch.randn(seq, TOP_K), dim=1)
    return hidden, top_k_index, top_k_weights


def _forward_loop(model):
    torch.manual_seed(0)
    with torch.no_grad():
        model.experts(*_batch())


def _eager_and_grouped():
    torch.manual_seed(0)
    eager, grouped = _ExpertsModel("eager"), _ExpertsModel("grouped_mm")
    grouped.load_state_dict(eager.state_dict())
    return eager, grouped


def test_registered_wrapper_quantizes_up_and_down_proj():
    experts = _ExpertsModel("eager").experts
    assert _is_fused_experts_module(experts)

    converted = QuantModuleRegistry.convert(experts)

    assert isinstance(converted, _QuantNemotronHExperts)
    # A second quantize() finds the converted experts too, as it does gated ones by their shapes.
    assert _is_fused_experts_module(converted)
    assert len(converted.up_proj_weight_quantizers) == NUM_EXPERTS
    assert len(converted.down_proj_weight_quantizers) == NUM_EXPERTS
    assert not hasattr(converted, "gate_up_proj_input_quantizer")
    assert set(weight_attr_names(converted)) == {"up_proj", "down_proj"}


@pytest.mark.parametrize("impl", ["eager", "grouped_mm"])
def test_forward_passthrough_matches(impl):
    """With its quantizers disabled, the wrapper computes what the unconverted experts do."""
    torch.manual_seed(0)
    experts = _ExpertsModel(impl).experts
    reference = copy.deepcopy(experts)
    converted = QuantModuleRegistry.convert(experts)
    for module in converted.modules():
        if isinstance(module, TensorQuantizer):
            module.disable()

    with torch.no_grad():
        batch = _batch()
        torch.testing.assert_close(converted(*batch), reference(*batch))


def test_calibration_matches_eager():
    """grouped_mm calibrates every expert's quantizers to what eager does.

    The same weights and routing give the shared input quantizers the same activations and each
    per-expert weight quantizer the same slice; only down_proj's input passes through a GEMM, hence
    the tolerance on its amax.
    """
    eager, grouped = _eager_and_grouped()
    for model in (eager, grouped):
        mtq.quantize(model, QUANT_CFG, forward_loop=_forward_loop)
    e, g = eager.experts, grouped.experts
    assert (e._running_grouped_mm, g._running_grouped_mm) == (False, True)
    # grouped_mm went through `_grouped_linear`, one call per projection, not through per-expert
    # `F.linear` calls as eager does: the amaxes below are the grouped hook's.
    assert (e._intercept_count, g._intercept_count) == (2 * NUM_EXPERTS, 2)

    assert e.up_proj_input_quantizer.amax is not None
    torch.testing.assert_close(g.up_proj_input_quantizer.amax, e.up_proj_input_quantizer.amax)
    assert e.down_proj_input_quantizer.amax is not None
    torch.testing.assert_close(
        g.down_proj_input_quantizer.amax, e.down_proj_input_quantizer.amax, atol=1e-3, rtol=1e-2
    )
    for name in ("up_proj_weight_quantizers", "down_proj_weight_quantizers"):
        for idx, (eq, gq) in enumerate(zip(getattr(e, name), getattr(g, name))):
            assert eq.amax is not None, f"{name}[{idx}]"
            torch.testing.assert_close(gq.amax, eq.amax)


def test_gptq_hessians_match_eager(monkeypatch):
    """GPTQ gives each expert the Hessian of only its own tokens under grouped_mm, as eager."""
    hessians = {}
    free = GPTQHelper.free

    def recording(helper):
        hessians[helper.name] = (helper.n_samples, helper.hessian.clone())
        free(helper)

    monkeypatch.setattr(GPTQHelper, "free", recording)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 4, "block_sizes": {-1: 16}},
            },
        ],
        "algorithm": {"method": "gptq", "block_size": 16},
    }
    found = []
    for model in _eager_and_grouped():
        hessians.clear()
        mtq.quantize(model, copy.deepcopy(cfg), forward_loop=_forward_loop)
        found.append(dict(hessians))

    eager, grouped = found
    assert eager.keys() == grouped.keys() and len(eager) == 2 * NUM_EXPERTS
    for name, (n_samples, hessian) in eager.items():
        assert n_samples > 0 and grouped[name][0] == n_samples, name
        # The down-proj input flows through a GEMM, hence the tolerance.
        torch.testing.assert_close(grouped[name][1], hessian, rtol=1e-3, atol=1e-4)


def test_export_writes_up_and_down_proj_per_expert():
    """The export leaves up_proj whole: there is no gate half to split off."""
    model = _ExpertsModel("grouped_mm")
    mtq.quantize(model, QUANT_CFG, forward_loop=_forward_loop)
    experts = model.experts
    assert get_quantization_format(experts) is not None
    assert get_quant_config(model)["quantization"].get("quant_algo") is not None
    up_scale, down_scale = (
        (q.amax / q.maxbound).squeeze()
        for q in (experts.up_proj_input_quantizer, experts.down_proj_input_quantizer)
    )

    _export_fused_experts(experts, torch.float16)

    for idx in range(NUM_EXPERTS):
        expert = getattr(experts, str(idx))
        assert not hasattr(expert, "gate_proj")
        assert expert.up_proj.weight.shape == (INTERMEDIATE_DIM, HIDDEN_DIM)
        assert expert.down_proj.weight.shape == (HIDDEN_DIM, INTERMEDIATE_DIM)
        torch.testing.assert_close(expert.up_proj.input_scale, up_scale)
        torch.testing.assert_close(expert.down_proj.input_scale, down_scale)
    assert not hasattr(experts, "up_proj_weight_quantizers")
    # The fused parameters are released, not kept beside the per-expert copies.
    assert not hasattr(experts, "up_proj") and not hasattr(experts, "down_proj")


def test_unsupported_backend_is_redirected_to_grouped_mm():
    """Found through their registration, the experts leave a backend the quantizers cannot see."""
    model = _ExpertsModel("batched_mm")

    normalize_experts_impl_on_the_fly(model)

    assert model.experts.config._experts_implementation == "grouped_mm"
