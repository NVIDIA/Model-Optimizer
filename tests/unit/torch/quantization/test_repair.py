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

import builtins
import copy
import io
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import repair_formats
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields
from modelopt.torch.quantization.nn import QuantModule, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback
from modelopt.torch.quantization.utils.weight_cache import (
    disable_weight_cache,
    enable_weight_cache,
    weight_cache_scope,
)


def _config(entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


def _iq(path, name):
    return {"quantizer_name": path, "cfg": {"num_bits": name, "backend": "ggml"}}


@pytest.mark.parametrize("width", [256, 300])
@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
def test_iq_fields_finalize_and_reload(name, width, tmp_path):
    torch.manual_seed(17)
    model = nn.Linear(width, 4, bias=False)
    x = torch.randn(3, width)
    entry = _iq("weight_quantizer", name)
    if width == 300:
        entry["cfg"]["block_sizes"] = {-1: 256}
    mtq.quantize(model, _config([entry]), lambda m: m(x))
    fmt = IQ_FORMAT_REGISTRY[name]
    if name == "iq1_s" and width == 256:
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
    quantizer = model.weight_quantizer
    processed = quantizer._process_for_blockquant(model.weight) if width == 300 else model.weight
    original = fmt.pack(processed, quantizer).clone().reshape(4, -1, fmt.block_bytes)
    parameters = {id(parameter) for parameter in model.parameters()}
    if name == "iq1_s" and width == 256:
        assert torch.signbit(decode_scale_fields(original, fmt)[0][0])
    policy = IQRepairPolicy()
    enable_weight_cache(model)
    try:
        fields = policy.prepare(model)
        assert torch.equal(model.weight_quantizer._local_repair_state.native_payload(), original)
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
        assert {id(parameter) for parameter in model.parameters()} == parameters
        optimizer = torch.optim.SGD(fields, lr=0.01)
        with weight_cache_scope(model):
            model(x).square().mean().backward()
        assert model.weight.grad is None
        assert any(p.grad is not None and p.grad.count_nonzero() for p in fields)
        optimizer.step()
        policy.project_()
        changed = model(x).detach()
        assert not torch.equal(changed, before)
        if width == 300:
            decoded = fmt.dequantize(
                model.weight_quantizer._local_repair_state.native_payload(),
                torch.tensor((4, 512)),
                dtype=model.weight.dtype,
            )
            assert decoded[:, width:].count_nonzero()
            torch.testing.assert_close(
                changed, torch.nn.functional.linear(x, decoded[:, :width]), rtol=0, atol=0
            )
        checkpoint = io.BytesIO()
        mto.save(model, checkpoint)
        checkpoint.seek(0)
        active_restored = mto.restore(nn.Linear(width, 4, bias=False), checkpoint)
        torch.testing.assert_close(active_restored(x), before, rtol=0, atol=0)
        assert not hasattr(active_restored.weight_quantizer, "_local_repair_state")
        assert "_fake_quantize" not in active_restored.weight_quantizer.__dict__
        policy.finalize_()
    finally:
        policy.close()
        disable_weight_cache(model)
    payload = pinned_packed_weight(model.weight_quantizer, name)
    global_scale, local, _ = decode_scale_fields(original, fmt)
    assert torch.equal(replace_scale_fields(payload, fmt, global_scale, local), original)
    torch.testing.assert_close(model(x), changed, rtol=0, atol=0)
    processed = quantizer._process_for_blockquant(model.weight) if width == 300 else model.weight
    assert torch.equal(fmt.pack(processed, quantizer).reshape_as(payload), payload)
    mto.save(model, tmp_path / "student.pt")
    restored = mto.restore(nn.Linear(width, 4, bias=False), tmp_path / "student.pt")
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
    quantizer = model[0].weight_quantizer
    baseline = model(x).detach()
    original_forward = quantizer._fake_quantize
    try:
        fields = policy.prepare(model)
        with torch.no_grad():
            for field in fields:
                field.add_(0.5)
        assert not torch.equal(model(x), baseline)
        with pytest.raises(RuntimeError):
            model(x[..., :-1])
    finally:
        policy.close()
    assert model[0].weight.requires_grad
    assert not hasattr(quantizer, "_local_repair_state")
    assert not hasattr(quantizer, "_local_repair_active")
    assert quantizer._fake_quantize == original_forward
    torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)

    class NativeOnlyProjection(nn.Linear):
        @property
        def allow_real_quant_gemm(self):
            return getattr(self, "_gemm_enabled", True)

        @allow_real_quant_gemm.setter
        def allow_real_quant_gemm(self, enabled):
            if not enabled and not getattr(self, "accept_disable", False):
                raise ValueError("This projection requires native GEMM")
            self._gemm_enabled = enabled

    model.append(NativeOnlyProjection(2, 2, bias=False))
    model[-1].weight.requires_grad_(False)
    quantizer.enable_calib()
    expected = model(x).detach()
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    layout = (quantizer.axis, quantizer.block_sizes, quantizer._calibrator._axis)
    gemm = {
        name: ("allow_real_quant_gemm" in module.__dict__, module.allow_real_quant_gemm)
        for name, module in model.named_modules()
        if hasattr(module, "allow_real_quant_gemm")
    }
    with pytest.raises(ValueError):
        policy.prepare(model)
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == flags
    assert quantizer._if_calib
    assert (quantizer.axis, quantizer.block_sizes, quantizer._calibrator._axis) == layout
    assert not hasattr(quantizer, "_local_repair_properties")
    assert quantizer._fake_quantize == original_forward
    assert {
        name: ("allow_real_quant_gemm" in module.__dict__, module.allow_real_quant_gemm)
        for name, module in model.named_modules()
        if hasattr(module, "allow_real_quant_gemm")
    } == gemm
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    policy.close()
    model[-1].accept_disable = True
    try:
        policy.prepare(model)
        assert not model[-1].allow_real_quant_gemm
    finally:
        policy.close()
    assert model[-1].allow_real_quant_gemm


@pytest.mark.parametrize("change", ["replace_quantizer", "prepare_twice", "rotate_back"])
@pytest.mark.parametrize("sequential", [False, True])
def test_iq_plan_rejects_stale_or_active_bindings_without_mutating_caller(change, sequential):
    model = nn.Linear(256, 2, bias=False)
    x = torch.randn(1, 256)
    mtq.quantize(model, _config([_iq("weight_quantizer", "iq1_m")]), lambda m: m(x))
    quantizer = model.weight_quantizer
    if sequential:
        disabled = copy.deepcopy(quantizer)
        disabled.disable()
        model.weight_quantizer = SequentialQuantizer(quantizer, disabled)
    plan = plan_iq_repair([model])
    policy = IQRepairPolicy()
    try:
        if change == "replace_quantizer":
            previous = quantizer
            model.weight_quantizer = copy.deepcopy(model.weight_quantizer)
            quantizer = model.weight_quantizer[0] if sequential else model.weight_quantizer
            expected_grad = True
        elif change == "prepare_twice":
            fields = policy.prepare_plan(plan, 0)
            previous = quantizer._local_repair_state
            expected_grad = False
        else:
            previous = quantizer
            quantizer.set_from_attribute_config({"rotate": {"enable": True, "mode": "rotate_back"}})
            expected_grad = True
        with pytest.raises(ValueError):
            policy.prepare_plan(plan, 0)
        assert model.weight.requires_grad is expected_grad
        if change != "prepare_twice":
            assert not hasattr(previous, "_local_repair_state")
            assert not hasattr(quantizer, "_local_repair_state")
        else:
            assert quantizer._local_repair_state is previous
            model(x).square().mean().backward()
            assert any(field.grad is not None and field.grad.count_nonzero() for field in fields)
    finally:
        policy.close()
    assert model.weight.requires_grad


@pytest.mark.parametrize("alias", ["parameter", "storage", "frozen_bias"])
@pytest.mark.parametrize("timing", ["before_plan", "before_prepare", "before_finalize"])
def test_iq_plan_protects_frozen_owners_and_allows_shared_nonwrite_parameters(alias, timing):
    model = nn.ModuleList([nn.Linear(256, 2), nn.Linear(256, 2)])
    entries = [_iq("0.weight_quantizer", "iq1_s")]
    if alias == "frozen_bias":
        entries.append(_iq("1.weight_quantizer", "iq1_s"))
    mtq.quantize(model, {**_config(entries), "algorithm": None})
    plan = plan_iq_repair([model]) if timing != "before_plan" else None
    policy = IQRepairPolicy()
    committed = False
    try:
        fields = policy.prepare_plan(plan, 0) if timing == "before_finalize" else None
        if alias == "parameter":
            model[1].weight = model[0].weight
        elif alias == "storage":
            model[1].weight = nn.Parameter(model[0].weight.detach())
        else:
            model[1].bias = model[0].bias
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        if alias != "frozen_bias":
            with pytest.raises(ValueError):
                if timing == "before_finalize":
                    policy.finalize_()
                elif plan is None:
                    plan_iq_repair([model])
                else:
                    policy.prepare_plan(plan, 0)
        else:
            if fields is None:
                fields = policy.prepare(model) if plan is None else policy.prepare_plan(plan, 0)
            inputs = torch.randn(1, 256)
            sum(module(inputs).square().mean() for module in model).backward()
            assert all(field.grad is not None for field in fields)
            if timing == "before_finalize":
                with torch.no_grad():
                    for field in fields:
                        field.add_(0.25)
                expected = [module(inputs).detach() for module in model]
                policy.finalize_()
                committed = True
                assert model[0].bias is model[1].bias
                for module, output in zip(model, expected, strict=True):
                    torch.testing.assert_close(module(inputs), output, rtol=0, atol=0)
    finally:
        policy.close()
    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
        if not committed or name.endswith("bias")
    )
    assert all(parameter.requires_grad for parameter in model.parameters())


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
            inputs = torch.randn(1, 256)
            before = model[0](inputs).detach()
            policy = IQRepairPolicy()
            try:
                policy.prepare_plan(plan_iq_repair([model[0]], root_model=model), 0)
                policy.finalize_()
            finally:
                policy.close()
            torch.testing.assert_close(model[0](inputs), before, rtol=0, atol=0)
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


@pytest.mark.parametrize(
    ("overlap", "timing", "selected_count", "copied_frozen"),
    [
        (False, "prepare", 1, False),
        (True, "prepare", 1, False),
        (True, "finalize", 1, False),
        (False, "prepare", 2, False),
        (True, "prepare", 2, False),
        (True, "finalize", 2, False),
        (False, "prepare", 1, True),
        (True, "prepare", 1, True),
        (True, "finalize", 1, True),
    ],
)
def test_iq_sliced_experts_require_disjoint_write_views(
    overlap, timing, selected_count, copied_frozen
):
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 256) * 0.02)

    class QuantExperts(QuantModule):
        def _setup(self):
            self.overlap = overlap and timing == "prepare"
            self.copied_frozen = copied_frozen and timing == "prepare"
            self.weight_quantizers = nn.ModuleList([TensorQuantizer() for _ in range(2)])
            for quantizer in self.weight_quantizers:
                quantizer.set_from_attribute_config({"num_bits": "iq1_s", "backend": "ggml"})

        def iter_weight_quantizers_for_calibration(self):
            yield from self.weight_quantizers

        def iter_weights_for_calibration(self):
            yield self.weight[:2], self.weight_quantizers[0]
            start = 1 if self.overlap else 2
            frozen = self.weight[start : start + 2]
            yield frozen.clone() if self.copied_frozen else frozen, self.weight_quantizers[1]

        def forward(self, inputs):
            return torch.cat(
                [
                    nn.functional.linear(inputs, q(weight))
                    for weight, q in self.iter_weights_for_calibration()
                ],
                dim=-1,
            )

    model = QuantExperts.convert(Experts())
    if selected_count == 1:
        model.weight_quantizers[1].disable()
    inputs = torch.randn(2, 256)
    baseline = model(inputs).detach()
    before = copy.deepcopy(model.state_dict())
    policy = IQRepairPolicy()
    try:
        if (overlap or copied_frozen) and timing == "prepare":
            with pytest.raises(ValueError):
                policy.prepare(model)
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
            torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
        else:
            fields = policy.prepare(model)
            model(inputs).square().mean().backward()
            assert all(field.grad is not None and field.grad.count_nonzero() for field in fields)
            with torch.no_grad():
                for field in fields:
                    field.add_(0.5)
            expected = model(inputs).detach()
            assert not torch.equal(expected, baseline)
            if overlap or copied_frozen:
                model.overlap, model.copied_frozen = overlap, copied_frozen
                with pytest.raises(ValueError):
                    policy.finalize_()
                torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
                model.overlap = model.copied_frozen = False
            policy.finalize_()
            torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
            if selected_count == 1:
                torch.testing.assert_close(model.weight[2:], before["weight"][2:], rtol=0, atol=0)
    finally:
        policy.close()
    assert model.weight.requires_grad
    assert all(not hasattr(q, "_local_repair_state") for q in model.weight_quantizers)


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_iq_preparation_requires_preservable_preprocessing_dtype(scale_dtype):
    model = nn.Linear(256, 2, bias=False, dtype=torch.bfloat16)
    mtq.quantize(model, {**_config([_iq("weight_quantizer", "iq1_s")]), "algorithm": None})
    model.weight_quantizer.pre_quant_scale = torch.linspace(0.5, 1.5, 256, dtype=scale_dtype)
    inputs = torch.randn(2, 256, dtype=scale_dtype)
    baseline = model(inputs).detach()
    before = copy.deepcopy(model.state_dict())
    policy = IQRepairPolicy()
    try:
        if scale_dtype != model.weight.dtype:
            with pytest.raises(ValueError):
                policy.prepare(model)
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
            assert model.weight_quantizer._enable_pre_quant_scale
            torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
        else:
            fields = policy.prepare(model)
            model(inputs).square().mean().backward()
            assert any(field.grad is not None and field.grad.count_nonzero() for field in fields)
            with torch.no_grad():
                for field in fields:
                    field.add_(0.5)
            expected = model(inputs).detach()
            policy.finalize_()
            torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    finally:
        policy.close()
    assert model.weight.requires_grad
    assert not hasattr(model.weight_quantizer, "_local_repair_state")


@pytest.mark.parametrize("separate_policies", [False, True])
def test_iq_policy_restores_first_flags_of_shared_frozen_modules(separate_policies):
    shared = nn.Linear(2, 2)
    units = nn.ModuleList([nn.Sequential(nn.Linear(256, 2), shared) for _ in range(2)])
    mtq.quantize(
        units,
        {**_config([_iq("*.0.weight_quantizer", "iq1_s")]), "algorithm": None},
    )
    shared.weight_quantizer.enable()
    shared.weight_quantizer._if_calib = True
    shared.allow_real_quant_gemm = True
    flags = {parameter: parameter.requires_grad for parameter in units.parameters()}
    policies = [IQRepairPolicy() for _ in range(2 if separate_policies else 1)]
    try:
        plan = plan_iq_repair(units)
        for index, unit in enumerate(units):
            fields = policies[index if separate_policies else 0].prepare_plan(plan, index)
            unit(torch.randn(1, 256)).sum().backward()
            assert all(field.grad is not None for field in fields)
        if separate_policies:
            policies[0].close()
            assert not shared.weight.requires_grad
            assert not shared.bias.requires_grad
            assert not shared.weight_quantizer._if_calib
            assert not shared.allow_real_quant_gemm
            units[1](torch.randn(1, 256)).sum().backward()
            assert shared.weight.grad is None and shared.bias.grad is None
            assert shared.weight_quantizer._calibrator.compute_amax() is None
        checkpoint = io.BytesIO()
        mto.save(units, checkpoint)
        checkpoint.seek(0)
        restored_shared = nn.Linear(2, 2)
        skeleton = nn.ModuleList(
            [nn.Sequential(nn.Linear(256, 2), restored_shared) for _ in range(2)]
        )
        restored = mto.restore(skeleton, checkpoint)
        assert restored_shared.weight_quantizer._if_calib
        restored[0](torch.randn(1, 256))
        assert restored_shared.weight_quantizer._calibrator.compute_amax() is not None
    finally:
        for policy in policies:
            policy.close()
    assert all(parameter.requires_grad == flag for parameter, flag in flags.items())
    assert shared.weight_quantizer._if_calib
    assert shared.allow_real_quant_gemm


def test_iq_resident_nonoffload_hook_needs_no_optional_offload_plugin(monkeypatch):
    model = nn.Linear(256, 2, bias=False)
    mtq.quantize(model, {**_config([_iq("weight_quantizer", "iq1_s")]), "algorithm": None})
    model._hf_hook = SimpleNamespace(offload=False)
    original_import = builtins.__import__

    def without_offload_plugin(name, *args, **kwargs):
        if name.endswith("plugins.accelerate"):
            raise ImportError("Optional offload dependencies are unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_offload_plugin)
    inputs = torch.randn(2, 256)
    baseline = model(inputs).detach()
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        model(inputs).square().mean().backward()
        assert all(field.grad is not None and field.grad.count_nonzero() for field in fields)
        with torch.no_grad():
            for field in fields:
                field.add_(0.5)
        expected = model(inputs).detach()
        assert not torch.equal(expected, baseline)
        policy.finalize_()
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    finally:
        policy.close()
    assert model.weight.requires_grad


@pytest.mark.parametrize(
    ("width", "block_size", "preexisting_layout"),
    [(256, None, False), (256, 256, False), (300, 256, False), (300, 256, True)],
)
def test_iq_prepare_close_preserves_unconsumed_layout(width, block_size, preexisting_layout):
    torch.manual_seed(122)
    model = nn.Sequential(nn.Linear(width, 2, bias=False), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[0.002, 0.003], [0.2, 0.3]]))
    entries = [
        _iq("0.weight_quantizer", "iq1_s"),
        {"quantizer_name": "1.weight_quantizer", "cfg": {"num_bits": 8}},
    ]
    mtq.quantize(model, {**_config(entries), "algorithm": None})
    for layer in model:
        layer.weight_quantizer.set_from_attribute_config({"block_sizes": {-1: block_size}})
    x = torch.randn(3, width)
    if preexisting_layout:
        model(x)
    sibling = copy.deepcopy(model)
    layouts = [
        (
            layer.weight_quantizer.axis,
            layer.weight_quantizer.block_sizes,
            layer.weight_quantizer._calibrator._axis,
        )
        for layer in model
    ]
    if not preexisting_layout:
        assert layouts == [(None, {-1: block_size}, None)] * 2
    original_metadata = copy.deepcopy(mto.modelopt_state(model))
    original_weights = copy.deepcopy(model.state_dict())
    policy = IQRepairPolicy()
    try:
        policy.prepare(model)
        model[0].weight_quantizer._local_repair_properties.update(
            {"_num_bits": 8, "_unexpected_repair_property": True}
        )
        checkpoint = io.BytesIO()
        mto.save(model, checkpoint)
        checkpoint.seek(0)
        restored = mto.restore(
            nn.Sequential(nn.Linear(width, 2, bias=False), nn.Linear(2, 2, bias=False)), checkpoint
        )
        assert mto.modelopt_state(restored) == original_metadata
        torch.testing.assert_close(restored.state_dict(), original_weights, rtol=0, atol=0)
        model(x)
        checkpoint = io.BytesIO()
        mto.save(model, checkpoint)
        checkpoint.seek(0)
        restored = mto.restore(
            nn.Sequential(nn.Linear(width, 2, bias=False), nn.Linear(2, 2, bias=False)), checkpoint
        )
        for layer, layout in zip(restored, layouts, strict=True):
            quantizer = layer.weight_quantizer
            assert (quantizer.axis, quantizer.block_sizes, quantizer._calibrator._axis) == layout
    finally:
        policy.close()
    for layer, sibling_layer, layout in zip(model, sibling, layouts, strict=True):
        quantizer = layer.weight_quantizer
        assert (quantizer.axis, quantizer.block_sizes, quantizer._calibrator._axis) == layout
        assert getattr(quantizer, "_consumed_block_sizes", None) == getattr(
            sibling_layer.weight_quantizer, "_consumed_block_sizes", None
        )
    expected = sibling(x).detach()
    for current in (model, restored):
        torch.testing.assert_close(current(x), expected, rtol=0, atol=0)
    for current in (sibling, model, restored):
        for layer in current:
            layer.weight_quantizer.enable_calib()
        current(x)
    for index, layer in enumerate(sibling):
        expected_amax = layer.weight_quantizer._calibrator.compute_amax()
        assert expected_amax.shape == ((4, 1) if index == 0 and width == 300 else (2, 1))
        for current in (model, restored):
            quantizer = current[index].weight_quantizer
            assert quantizer._calibrator._axis == layer.weight_quantizer._calibrator._axis
            torch.testing.assert_close(
                quantizer._calibrator.compute_amax(), expected_amax, rtol=0, atol=0
            )


@pytest.mark.parametrize("failure", ["invalid_scale", "publication", "recovery"])
@pytest.mark.parametrize("sequential", [False, True])
def test_iq_finalization_failure_rolls_back_and_can_retry(monkeypatch, failure, sequential):
    model = nn.ModuleList([nn.Linear(256, 2, bias=False) for _ in range(3)])
    mtq.quantize(model, {**_config([_iq("*weight_quantizer", "iq1_s")]), "algorithm": None})
    model[2].weight_quantizer.disable_quant()
    if sequential:
        for layer in model[:2]:
            disabled = copy.deepcopy(layer.weight_quantizer)
            disabled.disable_quant()
            layer.weight_quantizer = SequentialQuantizer(layer.weight_quantizer, disabled)
    before = copy.deepcopy(model.state_dict())
    original = repair_formats.pin_packed_weight
    policy = IQRepairPolicy()
    recovery_attempts = publications = 0

    @contextmanager
    def access(*, writeback, rollback):
        nonlocal recovery_attempts, publications
        if failure == "recovery":
            if publications:
                recovery_attempts += 1
                if recovery_attempts <= 2:
                    raise RuntimeError("recovery entry failed")
                if recovery_attempts == 4:
                    # A fresh publication cannot replace the original snapshots before recovery.
                    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
            else:
                publications += 1
        yield

    try:
        fields = policy.prepare(model)
        quantizers = [binding.quantizer for binding, *_ in policy.adapters]
        with torch.no_grad():
            for field in fields:
                field.add_(0.5)
        if failure == "invalid_scale":
            policy.adapters[1][1].global_delta.data.fill_(float("nan"))
            error = ValueError
        else:

            def fail_after_publication(quantizer, *args, **kwargs):
                original(quantizer, *args, **kwargs)
                if quantizer is policy.adapters[1][0].quantizer:
                    raise RuntimeError("pin publication failed")

            monkeypatch.setattr(repair_formats, "pin_packed_weight", fail_after_publication)
            error = RuntimeError
        with pytest.raises(error):
            policy.finalize_(weight_access=access)
        if failure == "recovery":
            assert any(not torch.equal(model[i].weight, before[f"{i}.weight"]) for i in range(2))
            with pytest.raises(RuntimeError):
                policy.close()
            with pytest.raises(RuntimeError):
                policy.finalize_(weight_access=access)
            assert recovery_attempts == 2
            monkeypatch.setattr(repair_formats, "pin_packed_weight", original)
            policy.finalize_(weight_access=access)
            assert recovery_attempts == 4
            assert not policy.adapters
            assert all(not quantizer._enable_pre_quant_scale for quantizer in quantizers)
            return
        after = model.state_dict()
        assert after.keys() == before.keys()
        assert all(torch.equal(value, before[name]) for name, value in after.items())
        assert len(policy.adapters) == 2
        assert all(binding.quantizer._enable_pre_quant_scale for binding, *_ in policy.adapters)
        assert all(
            hasattr(binding.quantizer, "_local_repair_state") for binding, *_ in policy.adapters
        )
        monkeypatch.setattr(repair_formats, "pin_packed_weight", original)
        with torch.no_grad():
            for field in fields:
                field.fill_(0.5)
        policy.finalize_()
    finally:
        policy.close()
    assert all(
        not torch.equal(module.weight, before[f"{index}.weight"])
        for index, module in enumerate(model[:2])
    )
    torch.testing.assert_close(model[2].weight, before["2.weight"], rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(not quantizer._enable_pre_quant_scale for quantizer in quantizers)
    assert not hasattr(model[2].weight_quantizer, "_local_repair_state")


def test_iq_finalization_refuses_a_changed_native_format():
    model = nn.Linear(256, 2, bias=False)
    mtq.quantize(model, {**_config([_iq("weight_quantizer", "iq1_m")]), "algorithm": None})
    x = torch.randn(1, 256)
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        with torch.no_grad():
            for field in fields:
                field.add_(0.25)
        expected = model(x).detach()
        model.weight_quantizer.set_from_attribute_config({"num_bits": "iq2_s"})
        before = model.weight.detach().clone()
        with pytest.raises(ValueError):
            policy.finalize_()
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        model.weight_quantizer.set_from_attribute_config({"num_bits": "iq1_m"})
        policy.finalize_()
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    finally:
        policy.close()


@pytest.mark.parametrize("nested", [False, True])
def test_iq_cpu_offload_preserves_repaired_outputs_and_flags(nested):
    pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    layer = nn.Linear(256, 2, bias=False)
    unit = nn.Sequential(layer) if nested else layer
    model = nn.Sequential(unit) if nested else unit
    mtq.quantize(model, {**_config([_iq("*weight_quantizer", "iq1_m")]), "algorithm": None})
    native = IQ_FORMAT_REGISTRY["iq1_m"].pack(layer.weight, layer.weight_quantizer).clone()
    pin_packed_weight(layer.weight_quantizer, "iq1_m", native, source=layer.weight)
    accelerate.cpu_offload(model, execution_device="cpu", offload_buffers=True)
    plan = plan_iq_repair([unit], root_model=model)
    policy = IQRepairPolicy()
    x = torch.randn(1, 256)
    try:
        with enable_weight_access_and_writeback(unit, model, writeback=False):
            assert not layer.weight.is_meta
            fields = policy.prepare_plan(plan, 0)
            assert torch.equal(layer.weight_quantizer._local_repair_state.native_payload(), native)
        assert layer.weight.is_meta
        model(x).sum().backward()
        assert any(field.grad is not None and field.grad.count_nonzero() for field in fields)
        with torch.no_grad():
            for field in fields:
                field.add_(0.25)
        expected = model(x).detach()
        policy.finalize_(
            weight_access=lambda **kwargs: enable_weight_access_and_writeback(unit, model, **kwargs)
        )
    finally:
        policy.close()
    assert layer.weight.requires_grad
    assert layer.weight.is_meta
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


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
    assert [len(bindings) for bindings in plan.bindings] == expected

    policy = IQRepairPolicy()
    try:
        fields = [parameter for index in range(2) for parameter in policy.prepare_plan(plan, index)]
        observed = (len(fields) == len({id(parameter) for parameter in fields}), materializations)
        assert observed == (True, expected)
        assert fields and all(parameter.requires_grad for parameter in fields)
        policy.finalize_()
        # Commit validates the views without retaining them before its write pass.
        assert materializations == [3 * count for count in expected]
    finally:
        policy.close()


def test_iq_plan_refuses_uninspectable_storage():
    class StorageLessTensor(torch.Tensor):
        def untyped_storage(self):
            raise NotImplementedError("No physical storage")

    model = nn.Linear(256, 2, bias=False)
    mtq.quantize(model, {**_config([_iq("*weight_quantizer", "iq1_s")]), "algorithm": None})
    model.weight = nn.Parameter(model.weight.as_subclass(StorageLessTensor))
    with pytest.raises(ValueError, match="cannot inspect physical storage"):
        plan_iq_repair([model])


def _iq_sharded_ownership_job(rank, size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from modelopt.torch.quantization.utils.core_utils import _fsdp2_persistent_parameters

    mesh = init_device_mesh("cpu", (size,))
    for tied in (False, True):
        model = nn.Sequential(*(nn.Linear(256, 4, bias=False) for _ in range(2)))
        mtq.quantize(model, {**_config([_iq("*weight_quantizer", "iq1_s")]), "algorithm": None})
        if tied:
            model[1].weight = model[0].weight
        model[0].weight.requires_grad_(False)
        fully_shard(model, mesh=mesh)
        persistent = _fsdp2_persistent_parameters(model)
        assert all(isinstance(persistent[owner, "weight"], DTensor) for owner in model)
        if tied:
            assert persistent[model[0], "weight"] is persistent[model[1], "weight"]
            with pytest.raises(ValueError, match=r"ownership|storage"):
                plan_iq_repair([model[0]], root_model=model)
        else:
            plan = plan_iq_repair(list(model), root_model=model)
            for index, owner in enumerate(model):
                flags = [
                    entry for entry in plan.parameter_flags[index] if entry[:2] == (owner, "weight")
                ]
                assert len(flags) == 1
                assert flags[0][2] is persistent[owner, "weight"]
                assert flags[0][3] is (index != 0)


def test_iq_plan_inspects_persistent_shards_and_shared_registrations(cpu_dist_workers):
    cpu_dist_workers.run(_iq_sharded_ownership_job)


def test_iq_resident_nonoffload_hook_needs_no_optional_offload_plugin(monkeypatch):
    model = nn.Linear(256, 2, bias=False)
    mtq.quantize(model, {**_config([_iq("weight_quantizer", "iq1_s")]), "algorithm": None})
    model._hf_hook = SimpleNamespace(offload=False)
    original_import = builtins.__import__

    def without_offload_plugin(name, *args, **kwargs):
        if name.endswith("plugins.accelerate"):
            raise ImportError("Optional offload dependencies are unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_offload_plugin)
    original = model.weight
    plan = plan_iq_repair([model])
    assert len(plan.bindings[0]) == 1
    assert model.weight is original and model.weight.requires_grad


def test_legacy_first_projection_quantizer_is_detected_by_export():
    from modelopt.torch.export.quant_utils import (
        get_quantization_format,
        get_weight_block_size,
        uses_iq_quantization,
    )
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.quantization.utils.core_utils import (
        representative_weight_quantizer,
        weight_attr_names,
    )

    module = nn.Module()
    module._first_proj_attr = "up_proj"
    module.up_proj = nn.Parameter(torch.randn(2, 256))
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits="iq1_s", backend="ggml", block_sizes={-1: 256})
    )
    module.gate_up_proj_weight_quantizers = nn.ModuleList([quantizer])
    assert list(weight_attr_names(module)) == ["up_proj"]
    assert representative_weight_quantizer(module, "up_proj") is quantizer
    assert get_quantization_format(module) == "iq1_s"
    assert get_weight_block_size(module, "up_proj") == 256
    assert uses_iq_quantization(module)


@pytest.mark.parametrize("native_iq", [False, True])
def test_iq_plan_refuses_an_unsupported_grouped_iq_container(native_iq):
    from modelopt.torch.quantization.nn import GroupedQuantizer

    model = nn.Sequential(nn.Linear(256, 2, bias=False), nn.Linear(256, 2, bias=False))
    entries = [_iq("1.weight_quantizer", "iq1_s")]
    if native_iq:
        entries.append(_iq("0.weight_quantizer", "iq1_s"))
    else:
        entries.append({"quantizer_name": "0.weight_quantizer", "cfg": {"num_bits": 8}})
    mtq.quantize(model, {**_config(entries), "algorithm": None})
    model[0].weight_quantizer = GroupedQuantizer(model[0].weight_quantizer)
    original = model[0].weight
    if native_iq:
        with pytest.raises(ValueError, match="unambiguous native weight quantizer"):
            plan_iq_repair([model])
    else:
        plan = plan_iq_repair([model])
        assert len(plan.bindings[0]) == 1
        assert plan.bindings[0][0].module is model[1]
    assert model[0].weight is original and model[0].weight.requires_grad
