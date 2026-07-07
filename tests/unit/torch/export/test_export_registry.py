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

"""Tests for the export module registry dispatching the unified HF export."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export.registry import (
    ExportContext,
    ExportModuleRegistry,
    ModuleExporter,
    _ExportModuleRegistryCls,
)
from modelopt.torch.export.unified_export_hf import (
    _BmmExpertsExporter,
    _DbrxExpertsExporter,
    _FusedExpertsExporter,
    _IterableExpertsExporter,
    _MoELinearExporter,
    _process_quantized_modules,
    _QuantEmbeddingExporter,
    _QuantLinearExporter,
)


class _Experts(nn.Module):
    pass


def _make_dynamic_subclass(base: type[nn.Module], prefix: str = "Quant") -> type[nn.Module]:
    """Mimic _DMRegistryCls's on-the-fly class generation (e.g. QuantLinear)."""
    return type(f"{prefix}{base.__name__}", (base,), {})


def test_class_key_matches_generated_subclass_via_mro():
    registry = _ExportModuleRegistryCls()

    @registry.register(nn.Linear)
    class _LinearHandler(ModuleExporter):
        pass

    quant_linear_cls = _make_dynamic_subclass(nn.Linear)
    module = nn.Linear(2, 2)
    module.__class__ = quant_linear_cls

    assert isinstance(registry.match(module), _LinearHandler)
    assert isinstance(registry.match(nn.Linear(2, 2)), _LinearHandler)
    assert registry.match(nn.Embedding(2, 2)) is None


def test_name_key_matches_original_and_generated_class():
    registry = _ExportModuleRegistryCls()

    @registry.register("_Experts")
    class _ExpertsHandler(ModuleExporter):
        pass

    raw = _Experts()
    generated = _Experts()
    generated.__class__ = _make_dynamic_subclass(_Experts)

    # The original class name appears in the generated class's MRO.
    assert isinstance(registry.match(raw), _ExpertsHandler)
    assert isinstance(registry.match(generated), _ExpertsHandler)
    assert registry.match(nn.Linear(2, 2)) is None


def test_keys_and_predicate_are_both_required():
    registry = _ExportModuleRegistryCls()

    @registry.register("_Experts", predicate=lambda m: hasattr(m, "experts"))
    class _ExpertsHandler(ModuleExporter):
        pass

    without_experts = _Experts()
    with_experts = _Experts()
    with_experts.experts = nn.ModuleList([nn.Linear(2, 2)])

    assert registry.match(without_experts) is None
    assert isinstance(registry.match(with_experts), _ExpertsHandler)


def test_first_registered_entry_wins():
    registry = _ExportModuleRegistryCls()

    @registry.register(predicate=lambda m: isinstance(m, nn.Linear))
    class _SpecificHandler(ModuleExporter):
        pass

    @registry.register(nn.Module)
    class _GenericHandler(ModuleExporter):
        pass

    assert isinstance(registry.match(nn.Linear(2, 2)), _SpecificHandler)
    assert isinstance(registry.match(nn.Embedding(2, 2)), _GenericHandler)


def test_register_requires_key_or_predicate():
    registry = _ExportModuleRegistryCls()
    with pytest.raises(AssertionError):
        registry.register()(ModuleExporter)


def test_prepend_registers_before_existing_entries():
    registry = _ExportModuleRegistryCls()

    @registry.register(predicate=lambda m: True)
    class _CatchAllHandler(ModuleExporter):
        pass

    @registry.register(nn.Linear, prepend=True)
    class _SpecificHandler(ModuleExporter):
        pass

    assert isinstance(registry.match(nn.Linear(2, 2)), _SpecificHandler)
    assert isinstance(registry.match(nn.Embedding(2, 2)), _CatchAllHandler)


def test_reregistering_same_class_replaces_entry_in_place():
    registry = _ExportModuleRegistryCls()

    @registry.register(nn.Linear)
    class _FirstHandler(ModuleExporter):
        pass

    @registry.register(predicate=lambda m: True)
    class _CatchAllHandler(ModuleExporter):
        pass

    # Simulate a module reload re-running the decorator on the same class:
    # the entry is replaced in place, keeping its (winning) position.
    registry.register(nn.Linear)(_FirstHandler)
    assert len(registry._entries) == 2
    assert isinstance(registry.match(nn.Linear(2, 2)), _FirstHandler)


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
    assert isinstance(ExportModuleRegistry.match(moe_linear), _MoELinearExporter)
    # Without the experts attr the entry must not match.
    assert not isinstance(
        ExportModuleRegistry.match(_named_module("QuantMoELinear")), _MoELinearExporter
    )

    # DBRX experts container: the usual generated class name...
    assert isinstance(
        ExportModuleRegistry.match(_named_module("QuantDbrxExperts", base_name="DbrxExperts")),
        _DbrxExpertsExporter,
    )
    # ...and the _DMRegistryCls collision-fallback name, where only the mixin
    # class name ("_QuantDbrxExperts") remains recognizable in the MRO.
    fallback = _named_module(
        "transformers_modules_modeling_dbrx_QuantDbrxExperts", base_name="_QuantDbrxExperts"
    )
    assert isinstance(ExportModuleRegistry.match(fallback), _DbrxExpertsExporter)

    # Fused experts (plural per-expert weight quantizers) must win over the BMM
    # name keys even when the class name also matches a BMM entry.
    fused = _named_module(
        "QuantGptOssExperts",
        base_name="GptOssExperts",
        gate_up_proj_weight_quantizers=nn.ModuleList(),
    )
    assert isinstance(ExportModuleRegistry.match(fused), _FusedExpertsExporter)

    # BMM-style experts by class name: raw and quant-generated variants.
    assert isinstance(
        ExportModuleRegistry.match(_named_module("GptOssExperts")), _BmmExpertsExporter
    )
    assert isinstance(
        ExportModuleRegistry.match(
            _named_module("QuantLlama4TextExperts", base_name="Llama4TextExperts")
        ),
        _BmmExpertsExporter,
    )

    # Iterable containers (e.g. Mixtral's ModuleList of experts) hit the catch-all.
    assert isinstance(ExportModuleRegistry.match(nn.ModuleList()), _IterableExpertsExporter)

    # Opaque experts containers match nothing — the prepass turns this into
    # an actionable NotImplementedError.
    assert ExportModuleRegistry.match(_named_module("OpaqueExperts")) is None


def test_builtin_registry_dispatches_quantized_modules():
    model = ToyModel(dims=[16, 32, 16])
    mtq.quantize(model, partial_fp8_config, lambda m: m(torch.randn(2, 4, 16)))

    quantized = [m for m in model.modules() if type(m).__name__ == "QuantLinear"]
    assert quantized, "expected at least one quantized linear in the toy model"
    for module in quantized:
        assert isinstance(ExportModuleRegistry.match(module), _QuantLinearExporter)

    # Plain (unquantized) modules match no handler.
    assert ExportModuleRegistry.match(nn.Linear(2, 2)) is None

    # A quantized embedding dispatches to the embedding handler.
    embedding = nn.Embedding(4, 4)
    embedding.weight_quantizer = None
    assert isinstance(ExportModuleRegistry.match(embedding), _QuantEmbeddingExporter)


def test_process_quantized_modules_exports_via_registry():
    model = ToyModel(dims=[16, 32, 16])
    mtq.quantize(model, partial_fp8_config, lambda m: m(torch.randn(2, 4, 16)))

    _process_quantized_modules(model, torch.float16)

    state_dict = model.state_dict()
    fp8_weights = [k for k in state_dict if k.endswith("weight_scale")]
    assert fp8_weights, "expected weight_scale buffers registered by the linear exporter"
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
