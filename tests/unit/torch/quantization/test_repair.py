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

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields
from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair


def _config(entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


def _iq(path, name):
    return {"quantizer_name": path, "cfg": {"num_bits": name, "backend": "ggml"}}


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
def test_iq_fields_finalize_and_reload(name, tmp_path):
    torch.manual_seed(17)
    model = nn.Linear(256, 4, bias=False)
    x = torch.randn(3, 256)
    mtq.quantize(model, _config([_iq("weight_quantizer", name)]), lambda m: m(x))
    fmt = IQ_FORMAT_REGISTRY[name]
    if name == "iq1_s":
        # Native FP16 scales include negative zero, whose sign must survive a
        # zero-update projection even though floating-point equality cannot see it.
        native = fmt.pack(model.weight, model.weight_quantizer)
        global_scale, local, _ = decode_scale_fields(native, fmt)
        global_scale[0] = -0.0
        native = replace_scale_fields(native, fmt, global_scale, local)
        with torch.no_grad():
            model.weight.copy_(
                fmt.dequantize(native, torch.tensor(model.weight.shape), dtype=model.weight.dtype)
            )
        pin_packed_weight(model.weight_quantizer, name, native, source=model.weight)
    before = model(x).detach()
    original = fmt.pack(model.weight, model.weight_quantizer).clone()
    if name == "iq1_s":
        assert torch.signbit(decode_scale_fields(original, fmt)[0][0])
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        assert torch.equal(model.weight_quantizer._local_repair_state.native_payload(), original)
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
        optimizer = torch.optim.SGD(fields, lr=0.01)
        model(x).square().mean().backward()
        assert model.weight.grad is None
        assert any(p.grad is not None and p.grad.count_nonzero() for p in fields)
        optimizer.step()
        policy.project_()
        changed = model(x).detach()
        assert not torch.equal(changed, before)
        policy.finalize_()
    finally:
        policy.close()
    payload = pinned_packed_weight(model.weight_quantizer, name)
    global_scale, local, _ = decode_scale_fields(original, fmt)
    assert torch.equal(replace_scale_fields(payload, fmt, global_scale, local), original)
    torch.testing.assert_close(model(x), changed, rtol=0, atol=0)
    mto.save(model, tmp_path / "student.pt")
    restored = mto.restore(nn.Linear(256, 4, bias=False), tmp_path / "student.pt")
    torch.testing.assert_close(restored(x), changed, rtol=0, atol=0)
    assert torch.equal(pinned_packed_weight(restored.weight_quantizer, name), payload)
    assert all("repair" not in name for name, _ in restored.named_modules())


def test_iq_policy_refuses_aliases_and_restores_on_failure():
    model = nn.Sequential(nn.Linear(256, 2, bias=False))
    x = torch.randn(1, 256)
    mtq.quantize(model, _config([_iq("*weight_quantizer", "iq1_m")]), lambda m: m(x))
    alias = nn.Sequential(model[0], model[0])
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        plan_iq_repair([alias])
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    other = copy.deepcopy(model)
    other[0].weight = nn.Parameter(model[0].weight.detach())
    with pytest.raises(ValueError):
        plan_iq_repair([model[0], other[0]])
    policy = IQRepairPolicy()
    try:
        policy.prepare(model)
        raise RuntimeError("replay failed")
    except RuntimeError:
        policy.close()
    assert model[0].weight.requires_grad
    torch.testing.assert_close(model(x), F.linear(x, model[0].weight_quantizer(model[0].weight)))
    assert all("repair" not in n for n, _ in model.named_modules())


def test_preparing_units_streams_each_expert_and_returns_new_fields(monkeypatch):
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
    policy = IQRepairPolicy()
    try:
        fields = [parameter for index in range(2) for parameter in policy.prepare_plan(plan, index)]
        observed = (len(fields) == len({id(parameter) for parameter in fields}), materializations)
        assert observed == (True, expected)
        assert fields and all(parameter.requires_grad for parameter in fields)
        policy.finalize_()
        assert materializations == [2 * count for count in expected]
    finally:
        policy.close()
