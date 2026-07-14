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

"""Tests for the registries dispatching unified Hugging Face export handlers."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export import hf_export_handlers, unified_export_hf
from modelopt.torch.export.hf_export_handlers import (
    _export_bmm_experts,
    _export_fused_experts_module,
    _export_moe_linear,
    _export_quant_embedding,
    _export_quant_linear,
    _prepare_bmm_experts,
    _prepare_dbrx_experts,
    _prepare_fused_experts,
    _prepare_iterable_experts,
)
from modelopt.torch.export.registry import (
    ExportContext,
    ExportModuleRegistry,
    PrepareMoEInputsRegistry,
    _ExportHandlerRegistryCls,
)
from modelopt.torch.export.unified_export_hf import _process_quantized_modules


class _Experts(nn.Module):
    pass


def _make_dynamic_subclass(base: type[nn.Module], prefix: str = "Quant") -> type[nn.Module]:
    """Mimic _DMRegistryCls's on-the-fly class generation (e.g. QuantLinear)."""
    return type(f"{prefix}{base.__name__}", (base,), {})


def test_unified_export_imports_builtin_handlers():
    assert unified_export_hf._hf_export_handlers is hf_export_handlers


def test_class_key_matches_generated_subclass_via_mro():
    registry = _ExportHandlerRegistryCls()

    @registry.register(nn.Linear)
    def linear_handler(name, module, ctx):
        pass

    quant_linear_cls = _make_dynamic_subclass(nn.Linear)
    module = nn.Linear(2, 2)
    module.__class__ = quant_linear_cls

    assert registry.match(module) is linear_handler
    assert registry.match(nn.Linear(2, 2)) is linear_handler
    assert registry.match(nn.Embedding(2, 2)) is None


def test_name_key_matches_original_and_generated_class():
    registry = _ExportHandlerRegistryCls()

    @registry.register("_Experts")
    def experts_handler(name, module, ctx):
        pass

    raw = _Experts()
    generated = _Experts()
    generated.__class__ = _make_dynamic_subclass(_Experts)

    # The original class name appears in the generated class's MRO.
    assert registry.match(raw) is experts_handler
    assert registry.match(generated) is experts_handler
    assert registry.match(nn.Linear(2, 2)) is None


def test_keys_and_predicate_are_both_required():
    registry = _ExportHandlerRegistryCls()

    @registry.register("_Experts", predicate=lambda module: hasattr(module, "experts"))
    def experts_handler(name, module, ctx):
        pass

    without_experts = _Experts()
    with_experts = _Experts()
    with_experts.experts = nn.ModuleList([nn.Linear(2, 2)])

    assert registry.match(without_experts) is None
    assert registry.match(with_experts) is experts_handler


def test_first_registered_entry_wins():
    registry = _ExportHandlerRegistryCls()

    @registry.register(predicate=lambda module: isinstance(module, nn.Linear))
    def specific_handler(name, module, ctx):
        pass

    @registry.register(nn.Module)
    def generic_handler(name, module, ctx):
        pass

    assert registry.match(nn.Linear(2, 2)) is specific_handler
    assert registry.match(nn.Embedding(2, 2)) is generic_handler


def test_registration_order_is_independent_between_registries():
    prepare_registry = _ExportHandlerRegistryCls()
    export_registry = _ExportHandlerRegistryCls()

    def handler_a(name, module, ctx):
        pass

    def handler_b(name, module, ctx):
        pass

    def handler_c(name, module, ctx):
        pass

    for handler in [handler_a, handler_b, handler_c]:
        prepare_registry.register(nn.Module)(handler)
    for handler in [handler_b, handler_a, handler_c]:
        export_registry.register(nn.Module)(handler)

    assert [handler for _, _, handler in prepare_registry._entries] == [
        handler_a,
        handler_b,
        handler_c,
    ]
    assert [handler for _, _, handler in export_registry._entries] == [
        handler_b,
        handler_a,
        handler_c,
    ]

    module = nn.Linear(2, 2)
    assert prepare_registry.match(module) is handler_a
    assert export_registry.match(module) is handler_b


def test_register_requires_key_or_predicate():
    registry = _ExportHandlerRegistryCls()

    def handler(name, module, ctx):
        pass

    with pytest.raises(AssertionError):
        registry.register()(handler)


def test_prepend_registers_before_existing_entries():
    registry = _ExportHandlerRegistryCls()

    @registry.register(predicate=lambda module: True)
    def catch_all_handler(name, module, ctx):
        pass

    @registry.register(nn.Linear, prepend=True)
    def specific_handler(name, module, ctx):
        pass

    assert registry.match(nn.Linear(2, 2)) is specific_handler
    assert registry.match(nn.Embedding(2, 2)) is catch_all_handler


def test_reregistering_same_handler_replaces_entry_in_place():
    registry = _ExportHandlerRegistryCls()

    @registry.register(nn.Linear)
    def first_handler(name, module, ctx):
        pass

    @registry.register(predicate=lambda module: True)
    def catch_all_handler(name, module, ctx):
        pass

    # Simulate a module reload changing the same function's registration:
    # the entry is replaced in place, keeping its winning position.
    registry.register(nn.Embedding)(first_handler)
    assert len(registry._entries) == 2
    assert registry.match(nn.Linear(2, 2)) is catch_all_handler
    assert registry.match(nn.Embedding(2, 2)) is first_handler


def _named_module(name: str, base_name: str | None = None, **attrs) -> nn.Module:
    """Create a module instance whose class (and optionally base class) has a given name."""
    base = type(base_name, (nn.Module,), {}) if base_name else nn.Module
    module = type(name, (base,), {})()
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def test_builtin_dispatch_covers_all_handler_shapes():
    # Step-3.5 QuantMoELinear wrapper: exact leaf-class name plus experts attr.
    moe_linear = _named_module("QuantMoELinear", experts=nn.ModuleList([nn.Linear(2, 2)]))
    assert ExportModuleRegistry.match(moe_linear) is _export_moe_linear
    assert PrepareMoEInputsRegistry.match(moe_linear) is None
    assert ExportModuleRegistry.match(_named_module("QuantMoELinear")) is None

    # DBRX experts container: the usual generated class name...
    dbrx = _named_module("QuantDbrxExperts", base_name="DbrxExperts")
    assert PrepareMoEInputsRegistry.match(dbrx) is _prepare_dbrx_experts
    assert ExportModuleRegistry.match(dbrx) is None
    # ...and the _DMRegistryCls collision-fallback name, where only the mixin
    # class name ("_QuantDbrxExperts") remains recognizable in the MRO.
    fallback = _named_module(
        "transformers_modules_modeling_dbrx_QuantDbrxExperts", base_name="_QuantDbrxExperts"
    )
    assert PrepareMoEInputsRegistry.match(fallback) is _prepare_dbrx_experts

    # DBRX preparation remains type-specific even if a future DBRX variant gains
    # plural quantizers; the independent export registry takes the fused path.
    fused_dbrx = _named_module(
        "QuantDbrxExperts",
        base_name="DbrxExperts",
        gate_up_proj_weight_quantizers=nn.ModuleList(),
    )
    assert PrepareMoEInputsRegistry.match(fused_dbrx) is _prepare_dbrx_experts
    assert ExportModuleRegistry.match(fused_dbrx) is _export_fused_experts_module

    # Structural fused-experts matching wins over BMM name matching independently
    # in both registries.
    fused = _named_module(
        "QuantGptOssExperts",
        base_name="GptOssExperts",
        gate_up_proj_weight_quantizers=nn.ModuleList(),
    )
    assert PrepareMoEInputsRegistry.match(fused) is _prepare_fused_experts
    assert ExportModuleRegistry.match(fused) is _export_fused_experts_module

    # Structural fused matching also takes precedence over the iterable fallback.
    fused_iterable = nn.ModuleList()
    fused_iterable.gate_up_proj_weight_quantizers = nn.ModuleList()
    assert PrepareMoEInputsRegistry.match(fused_iterable) is _prepare_fused_experts
    assert ExportModuleRegistry.match(fused_iterable) is _export_fused_experts_module

    # BMM-style experts match through raw or quant-generated class names.
    for bmm in [
        _named_module("GptOssExperts"),
        _named_module("QuantLlama4TextExperts", base_name="Llama4TextExperts"),
    ]:
        assert PrepareMoEInputsRegistry.match(bmm) is _prepare_bmm_experts
        assert ExportModuleRegistry.match(bmm) is _export_bmm_experts

    opaque = _named_module("OpaqueExperts")
    assert PrepareMoEInputsRegistry.match(opaque) is None
    assert ExportModuleRegistry.match(opaque) is None


@pytest.mark.parametrize(
    "iterable_type",
    [nn.ModuleList, nn.Sequential, nn.ModuleDict, nn.ParameterList],
)
def test_iterable_modules_only_match_preparation_registry(iterable_type):
    iterable = iterable_type()
    assert PrepareMoEInputsRegistry.match(iterable) is _prepare_iterable_experts
    assert ExportModuleRegistry.match(iterable) is None


def test_builtin_registry_dispatches_quantized_modules():
    model = ToyModel(dims=[16, 32, 16])
    mtq.quantize(model, partial_fp8_config, lambda module: module(torch.randn(2, 4, 16)))

    quantized = [module for module in model.modules() if type(module).__name__ == "QuantLinear"]
    assert quantized, "expected at least one quantized linear in the toy model"
    for module in quantized:
        assert ExportModuleRegistry.match(module) is _export_quant_linear

    # Plain (unquantized) modules match no export handler.
    assert ExportModuleRegistry.match(nn.Linear(2, 2)) is None

    embedding = nn.Embedding(4, 4)
    embedding.weight_quantizer = None
    assert ExportModuleRegistry.match(embedding) is _export_quant_embedding


def test_process_quantized_modules_exports_via_registry():
    model = ToyModel(dims=[16, 32, 16])
    mtq.quantize(model, partial_fp8_config, lambda module: module(torch.randn(2, 4, 16)))

    _process_quantized_modules(model, torch.float16)

    state_dict = model.state_dict()
    fp8_weights = [key for key in state_dict if key.endswith("weight_scale")]
    assert fp8_weights, "expected weight_scale buffers registered by the linear handler"
    for key in fp8_weights:
        weight = state_dict[key.replace("weight_scale", "weight")]
        assert weight.dtype == torch.float8_e4m3fn


def test_export_context_caches_are_per_instance():
    model = nn.Linear(2, 2)
    ctx_a = ExportContext(model=model, dtype=torch.float16)
    ctx_b = ExportContext(model=model, dtype=torch.float16)
    ctx_a.tied_cache[123] = model
    assert ctx_b.tied_cache == {}
    assert ctx_b.moe_tied_cache == {}
