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

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, layerwise_fp8_config, partial_fp8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint, hf_export_handlers, unified_export_hf
from modelopt.torch.export.hf_export_handlers import (
    _export_bmm_experts,
    _export_fused_experts_module,
    _export_quant_embedding,
    _export_quant_linear,
    _prepare_bmm_experts,
    _prepare_fused_experts,
    _prepare_iterable_experts,
)
from modelopt.torch.export.layer_utils import is_quantlinear
from modelopt.torch.export.layerwise_export import LAYERWISE_EXPORTER_ATTR
from modelopt.torch.export.registry import (
    ExportContext,
    ExportModuleRegistry,
    ModuleExportHandler,
    PrepareMoEInputsRegistry,
    _ExportHandlerRegistryCls,
    _PrepareMoEInputsRegistryCls,
)
from modelopt.torch.export.unified_export_hf import _process_quantized_modules, _start_export
from modelopt.torch.models import specs
from modelopt.torch.opt import ModelOptSession


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


def test_each_phase_takes_the_first_handler_that_defines_it():
    """A handler that takes no part in a phase does not hide a later one that does."""
    registry = _ExportHandlerRegistryCls()

    def prepare(name, module, ctx):
        pass

    @registry.register(nn.Module)
    class PrepareOnly(ModuleExportHandler):
        on_prepare_moe_inputs = staticmethod(prepare)

    @registry.register(nn.Linear)
    def pack(name, module, ctx):
        pass

    module = nn.Linear(2, 2)
    assert registry.match(module, ModuleExportHandler.on_prepare_moe_inputs) is prepare
    assert registry.match(module) is pack
    assert registry.match(module, ModuleExportHandler.on_post_pack) is None
    # A phase is its method, not its name.
    with pytest.raises(ValueError):
        registry.match(module, "on_pack")


def test_a_handler_defines_only_the_export_phases():
    """A misspelt phase would never run, so the handler is refused when it registers."""
    registry = _ExportHandlerRegistryCls()

    class Misspelt(ModuleExportHandler):
        def on_postpack(self, name, module, ctx):
            pass

    with pytest.raises(ValueError, match="on_postpack"):
        registry.register(nn.Linear)(Misspelt)


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

    # So is a reloaded class's, though its phases changed: the old phase no longer runs.
    def handler_class(*phases):
        def take_part(self, name, module, ctx):
            pass

        return type("Reloaded", (ModuleExportHandler,), dict.fromkeys(phases, take_part))

    registry.register(nn.Conv1d)(handler_class("on_pack", "on_post_pack"))
    registry.register(nn.Conv1d)(handler_class("on_pack"))
    assert len(registry._entries) == 3
    assert registry.match(nn.Conv1d(2, 2, 1), ModuleExportHandler.on_post_pack) is None


def _named_module(name: str, base_name: str | None = None, **attrs) -> nn.Module:
    """Create a module instance whose class (and optionally base class) has a given name."""
    base = type(base_name, (nn.Module,), {}) if base_name else nn.Module
    module = type(name, (base,), {})()
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def test_builtin_dispatch_covers_all_handler_shapes(monkeypatch):
    # Structural fused-experts matching wins over BMM name matching, in each phase.
    fused = _named_module(
        "QuantGptOssExperts",
        base_name="GptOssExperts",
        gate_up_proj_weight_quantizers=nn.ModuleList(),
    )
    prepare = ModuleExportHandler.on_prepare_moe_inputs
    assert ExportModuleRegistry.match(fused, prepare) is _prepare_fused_experts
    assert ExportModuleRegistry.match(fused) is _export_fused_experts_module

    # Structural fused matching also takes precedence over the iterable fallback.
    fused_iterable = nn.ModuleList()
    fused_iterable.gate_up_proj_weight_quantizers = nn.ModuleList()
    assert ExportModuleRegistry.match(fused_iterable, prepare) is _prepare_fused_experts
    assert ExportModuleRegistry.match(fused_iterable) is _export_fused_experts_module

    # BMM-style experts, as the models' specs name them, match through raw or quant-generated
    # class names.
    for bmm in [
        _named_module("GptOssExperts"),
        _named_module("QuantLlama4TextExperts", base_name="Llama4TextExperts"),
    ]:
        assert ExportModuleRegistry.match(bmm, prepare) is _prepare_bmm_experts
        assert ExportModuleRegistry.match(bmm) is _export_bmm_experts
    # Those of a model registered after the export was imported too.
    later = specs.ModelSpec(
        model_type="later", export_spec=specs.ExportSpec(bmm_experts_names=("LaterExperts",))
    )
    monkeypatch.setitem(specs._SPECS, later.model_type, later)
    assert ExportModuleRegistry.match(_named_module("LaterExperts")) is _export_bmm_experts
    # And no longer once that spec is replaced, though the specs stay as many.
    replaced = dataclasses.replace(
        later, export_spec=specs.ExportSpec(bmm_experts_names=("OtherExperts",))
    )
    monkeypatch.setitem(specs._SPECS, later.model_type, replaced)
    assert ExportModuleRegistry.match(_named_module("LaterExperts")) is None
    assert ExportModuleRegistry.match(_named_module("OtherExperts")) is _export_bmm_experts

    opaque = _named_module("OpaqueExperts")
    assert ExportModuleRegistry.match(opaque, prepare) is None
    assert ExportModuleRegistry.match(opaque) is None


def test_the_deprecated_preparation_registry_registers_a_phase_of_the_export_registry():
    """``PrepareMoEInputsRegistry``, public since 0.46.0, still registers and matches what prepares an
    MoE block's inputs, now the ``on_prepare_moe_inputs`` phase of the export's handlers, with a
    deprecation warning."""
    registry = _ExportHandlerRegistryCls()
    preparation = _PrepareMoEInputsRegistryCls(registry)

    def prepare(name, module, ctx):
        pass

    with pytest.deprecated_call():
        preparation.register(_Experts)(prepare)
    with pytest.deprecated_call():
        assert preparation.match(_Experts()) is prepare
    assert registry.match(_Experts(), ModuleExportHandler.on_prepare_moe_inputs) is prepare
    assert registry.match(_Experts()) is None
    assert PrepareMoEInputsRegistry._registry is ExportModuleRegistry
    # The same function registered as a packing handler too, as two registries once allowed, is
    # one more handler, not a replacement.
    registry.register(nn.Linear)(prepare)
    assert registry.match(nn.Linear(2, 2)) is prepare
    assert registry.match(_Experts(), ModuleExportHandler.on_prepare_moe_inputs) is prepare


@pytest.mark.parametrize(
    "iterable_type",
    [nn.ModuleList, nn.Sequential, nn.ModuleDict, nn.ParameterList],
)
def test_iterable_modules_take_part_only_in_preparing_moe_inputs(iterable_type):
    iterable = iterable_type()
    assert (
        ExportModuleRegistry.match(iterable, ModuleExportHandler.on_prepare_moe_inputs)
        is _prepare_iterable_experts
    )
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

    with ModelOptSession().bind(model), _start_export(model, torch.float16):
        _process_quantized_modules()

    state_dict = model.state_dict()
    fp8_weights = [key for key in state_dict if key.endswith("weight_scale")]
    assert fp8_weights, "expected weight_scale buffers registered by the linear handler"
    for key in fp8_weights:
        weight = state_dict[key.replace("weight_scale", "weight")]
        assert weight.dtype == torch.float8_e4m3fn


def test_export_context_carries_no_resolver():
    # Tied-weight dedup is driven by the driver's resolver (fed to sync_tied_input_amax /
    # postprocess_state_dict), not by handlers, so ExportContext no longer builds or holds
    # a resolver. Building one per context would be dead work (handlers never read it) and,
    # for large models, an avoidable O(#modules x #patterns x #params) alias-map pass.
    ctx = ExportContext(model=nn.Linear(2, 2), dtype=torch.float16)
    assert not hasattr(ctx, "resolver")


def test_an_export_runs_in_the_workflows_modelopt_session():
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type="llama")

    # The workflow's session holds the export's context while it runs. The session comes bound to
    # the model, as the export's public entry point binds it, and binding read the model's type.
    with ModelOptSession().bind(model):
        with _start_export(model, torch.float16) as outer:
            assert ExportContext.current() is outer
            assert outer.model is model
            assert outer.model_type == "llama"
            # An export started within another gives the workflow back the outer one when it ends.
            with _start_export(model, torch.bfloat16) as inner:
                assert ExportContext.current() is inner
            assert ExportContext.current() is outer
        # The export ends with its block.
        with pytest.raises(RuntimeError):
            ExportContext.current()

    # Outside an export, there is no export context to find.
    with ModelOptSession(), pytest.raises(RuntimeError):
        ExportContext.current()


def test_every_export_finishes_the_modules_it_packed(tmp_path, monkeypatch):
    """on_post_pack runs on every module the export packed, under its name in the whole model, the
    same in a layerwise and an offloaded (streaming) export as in a whole-model one: the tail's
    modules, packed after the layers, too."""
    pytest.importorskip("transformers")
    # Local import: transformers and its test fixtures are optional dependencies.
    from _test_utils.torch.transformers_models import create_tiny_llama_dir, get_tiny_llama
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoModelForCausalLM

    monkeypatch.setattr(ExportModuleRegistry, "_entries", list(ExportModuleRegistry._entries))
    finished = []

    @ExportModuleRegistry.register(predicate=is_quantlinear, prepend=True)
    class _Finished(ModuleExportHandler):
        def on_post_pack(self, name, module, ctx):
            finished.append(name)

    def tiny_llama():
        torch.manual_seed(0)
        model = get_tiny_llama(num_hidden_layers=2).eval()
        model.config.architectures = ["LlamaForCausalLM"]
        return model

    def calibrate(model):
        model(torch.randint(0, 32, (1, 16)))

    model = tiny_llama()
    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, calibrate)
    export_hf_checkpoint(model, export_dir=tmp_path / "whole")
    whole = sorted(finished)
    assert "lm_head" in whole and "model.layers.1.self_attn.q_proj" in whole

    finished.clear()
    model = tiny_llama()
    mtq.quantize(model, layerwise_fp8_config(tmp_path / "layerwise"), calibrate)
    getattr(model, LAYERWISE_EXPORTER_ATTR).finalize()
    assert sorted(finished) == whole

    # Offloaded to disk, each module is packed in a window of its own, the tail's lm_head too, by
    # the streaming export and by a layerwise one.
    def offloaded(name):
        source = create_tiny_llama_dir(tmp_path / name / "source", num_hidden_layers=2)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(source))
        on_disk = ["model.layers.0", "model.layers.1", "model.norm", "lm_head"]
        model = load_checkpoint_and_dispatch(
            model,
            source,
            device_map={
                "model.embed_tokens": "cpu",
                "model.rotary_emb": "cpu",
                **dict.fromkeys(on_disk, "disk"),
            },
            offload_folder=str(tmp_path / name / "offload"),
        )
        model.config.architectures = ["LlamaForCausalLM"]
        return model

    finished.clear()
    model = offloaded("streamed")
    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, calibrate)
    export_hf_checkpoint(model, export_dir=tmp_path / "streamed" / "export")
    assert sorted(finished) == whole

    finished.clear()
    model = offloaded("layerwise-offloaded")
    mtq.quantize(
        model, layerwise_fp8_config(tmp_path / "layerwise-offloaded" / "export"), calibrate
    )
    getattr(model, LAYERWISE_EXPORTER_ATTR).finalize()
    assert sorted(finished) == whole
