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


"""CPU lifecycle and ownership contracts for native IQ scale repair."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.repair import plan_iq_repair
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback


def _config(entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


def _iq(path, name):
    return {"quantizer_name": path, "cfg": {"num_bits": name, "backend": "ggml"}}


@pytest.mark.parametrize("alias", ["parameter", "storage", "frozen_bias"])
def test_iq_plan_protects_frozen_owners_and_allows_shared_nonwrite_parameters(alias):
    model = nn.ModuleList([nn.Linear(256, 2), nn.Linear(256, 2)])
    entries = [_iq("0.weight_quantizer", "iq1_s")]
    if alias == "frozen_bias":
        entries.append(_iq("1.weight_quantizer", "iq1_s"))
    mtq.quantize(model, {**_config(entries), "algorithm": None})
    if alias == "parameter":
        model[1].weight = model[0].weight
    elif alias == "storage":
        model[1].weight = nn.Parameter(model[0].weight.detach())
    else:
        model[1].bias = model[0].bias
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    if alias == "frozen_bias":
        plan = plan_iq_repair([model])
        assert len(plan.bindings[0]) == 2
    else:
        with pytest.raises(ValueError):
            plan_iq_repair([model])
    assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == flags


@pytest.mark.parametrize(
    ("alias", "offload"),
    [
        ("parameter", False),
        ("storage", False),
        ("buffer", False),
        ("quantizer", False),
        ("parameter", True),
        ("storage", True),
        ("buffer", True),
        ("quantizer", True),
        ("resident_parameter", True),
        ("sparse_buffer", False),
        ("independent_sparse_buffer", False),
        ("numpy_parameter", False),
        ("numpy_buffer", False),
        ("numpy_disjoint", False),
    ],
)
def test_iq_plan_refuses_shared_weights_outside_selected_boundaries(alias, offload):
    model = nn.ModuleList([nn.Linear(256, 2, bias=False) for _ in range(2)])
    mtq.quantize(model, {**_config([_iq("0.weight_quantizer", "iq1_s")]), "algorithm": None})
    if alias.startswith("numpy_"):
        array = np.arange(1024, dtype=np.float32)
        model[0].weight = nn.Parameter(torch.from_numpy(array[:512]).reshape(2, 256))
        offset = 512 if alias == "numpy_disjoint" else 256
        other = torch.from_numpy(array[offset : offset + 512]).reshape(2, 256)
        assert other.untyped_storage().data_ptr() != model[0].weight.untyped_storage().data_ptr()
        if alias == "numpy_buffer":
            model[1].register_buffer("frozen_weight", other)
        else:
            model[1].weight = nn.Parameter(other)
        if alias == "numpy_disjoint":
            plan_iq_repair([model[0]], root_model=model)
            return
    elif alias in ("sparse_buffer", "independent_sparse_buffer"):
        values = model[0].weight.detach().view(-1)
        if alias == "independent_sparse_buffer":
            values = values.clone()
        model[1].register_buffer(
            "frozen_weight", torch.sparse_coo_tensor(torch.arange(512).unsqueeze(0), values, (512,))
        )
        if alias == "independent_sparse_buffer":
            plan_iq_repair([model[0]], root_model=model)
            return
    elif alias == "resident_parameter":
        resident_weight = model[1].weight.detach()
    elif alias == "quantizer":
        model[1].weight_quantizer = model[0].weight_quantizer
    elif alias == "buffer":
        model[1].register_buffer("frozen_weight", model[0].weight.detach())
    else:
        model[1].weight = (
            model[0].weight if alias == "parameter" else nn.Parameter(model[0].weight.detach())
        )
    before = copy.deepcopy(model.state_dict())
    if offload:
        pytest.importorskip("transformers")
        pytest.importorskip("accelerate").cpu_offload(
            model, execution_device="cpu", offload_buffers=True
        )
    if alias == "resident_parameter":
        model[0].weight = nn.Parameter(resident_weight)
        before["0.weight"] = before["1.weight"]
    parameters = dict(model.named_parameters(remove_duplicate=False))
    flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
    with pytest.raises(ValueError):
        plan_iq_repair([model[0]], root_model=model)
    assert all(
        parameter is parameters[name]
        for name, parameter in model.named_parameters(remove_duplicate=False)
    )
    assert {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters(remove_duplicate=False)
    } == flags
    with enable_weight_access_and_writeback(model, model, writeback=False):
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert not hasattr(model[0].weight_quantizer, "_local_repair_state")


def test_planning_units_keeps_expert_weights_lazy(monkeypatch):
    pytest.importorskip("transformers")
    # Transformers is optional; use its actual quantized fused expert banks when available.
    from _test_utils.torch.transformers_models import get_tiny_mixtral

    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

    model = get_tiny_mixtral(
        hidden_size=256,
        intermediate_size=16,
        num_hidden_layers=2,
        num_local_experts=3,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
    )
    mtq.quantize(
        model,
        {
            **_config([_iq("*experts.gate_up_proj_weight_quantizers*", "iq2_xxs")]),
            "algorithm": None,
        },
    )
    banks = [module for module in model.modules() if isinstance(module, _QuantFusedExperts)]
    assert len(banks) == 2
    materializations = [0, 0]
    expected = [
        sum(q.is_enabled for q in bank.iter_weight_quantizers_for_calibration()) for bank in banks
    ]
    for index, bank in enumerate(banks):
        original = bank.iter_weights_for_calibration

        def measured(iterate=original, owner=index):
            for pair in iterate():
                materializations[owner] += 1
                yield pair

        monkeypatch.setattr(bank, "iter_weights_for_calibration", measured)
    plan = plan_iq_repair(banks)
    assert materializations == [0, 0]
    assert [len(bindings) for bindings in plan.bindings] == expected
