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

"""The registry dispatching per-module logic for the unified HF export path.

This mirrors the registration-and-dispatch idiom of
:class:`QuantModuleRegistry <modelopt.torch.quantization.nn.modules.quant_module.QuantModuleRegistry>`,
but not its mechanism: quantization registers replacement classes and converts modules
in place, whereas export registers handlers that emit compressed weights and scale
buffers for a module without changing its class.

A handler (:class:`ModuleExportHandler`) defines what the export does to one kind of module
in each phase it takes part in: preparing an MoE block's inputs, packing, and after packing.
Registering a handler for a new module type replaces what previously required editing
if/elif chains inside ``unified_export_hf.py``.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from modelopt.torch.opt.session import ModelOptSession

if TYPE_CHECKING:
    from modelopt.torch.export.hf_export_handlers import _HeldWindow
    from modelopt.torch.quantization.utils.core_utils import ModuleNames

__all__ = [
    "ExportContext",
    "ExportHandler",
    "ExportModuleRegistry",
    "ModuleExportHandler",
    "PrepareMoEInputsRegistry",
]


@dataclass
class ExportContext:
    """The state of one export of a model, which the export's steps and handlers share.

    The workflow's :class:`~modelopt.torch.opt.session.ModelOptSession` holds it while the export
    runs (``runtime.export_context``), and each step of the export gets it there with
    :meth:`current`; the handlers are handed it. A layerwise export's is there from the exporter's
    binding, before calibration, until the export that finalizes it ends.

    Tied-weight dedup is not a handler concern: the driver builds one name-based
    :class:`TiedWeightMap` and feeds it to ``sync_tied_input_amax`` and
    ``postprocess_state_dict`` directly. Both dense and fused-MoE tied weights are packed
    independently and their duplicate keys are dropped by name there, so the context
    carries no tied-weight map (handlers never consulted it).
    """

    model: nn.Module
    dtype: torch.dtype
    is_modelopt_qlora: bool = False
    model_type: str | None = None
    """The model's HF model type (``model.config.model_type``), used to resolve the
    model's spec in ``modelopt.torch.models``: an export takes it from the workflow's ModelOpt
    session, which read it off the model when binding it. ``None`` means unknown: spec lookups then
    fail loudly instead of guessing."""

    keep_fused_experts: bool = False
    """Marks the distributed no-gather export, whose handlers pack while the model is still sharded:
    an FSDP2-owned weight is packed inside its unit's weight-update window, and fused MoE experts are
    folded in place and kept fused and sharded rather than split into whole per-expert tensors,
    which :func:`~modelopt.torch.export.moe_utils.split_fused_experts_state_dict` does later."""

    module_names: "ModuleNames | None" = None
    """``model``'s name maps, built once per export by the first handler that looks a module up
    (:func:`~modelopt.torch.quantization.utils.module_name_maps`)."""

    hold_fsdp2_windows: bool = False
    """Whether the handlers hold an FSDP2 unit's weight-update window open across its modules: set
    only inside :func:`~modelopt.torch.export.hf_export_handlers.held_fsdp2_weight_windows`, which
    closes the last window. Elsewhere each module is packed in a window of its own."""

    fsdp2_window: "_HeldWindow | None" = None
    """The FSDP2 unit's weight-update window the handlers hold open across its modules."""

    @classmethod
    def current(cls) -> "ExportContext":
        """The context of the export in progress in the active ModelOpt session.

        An export's steps and handlers run only inside it, so outside one this raises.
        """
        export = ModelOptSession.active().runtime.export_context
        if export is None:
            raise RuntimeError(
                "No export is in progress: export steps and handlers run only inside one."
            )
        return export


ExportHandler = Callable[[str, nn.Module, ExportContext], None]
"""One phase of a :class:`ModuleExportHandler`, called as ``(name, module, ctx)`` with the export's
:class:`ExportContext`."""

_Registered = TypeVar("_Registered", bound=Callable[..., Any])


class ModuleExportHandler:
    """What the export does to one kind of module, in each phase it takes part in.

    A subclass registered with :data:`ExportModuleRegistry` overrides the phases it takes part
    in, as methods or static functions called ``(name, module, ctx)`` with the export's
    :class:`ExportContext`; a phase it does not override is one it takes no part in.
    A phase is named by its method on this class, as in
    ``ExportModuleRegistry.match(module, ModuleExportHandler.on_post_pack)``. A phase may run on
    one rank alone, as an FSDP2 export packs each unit on the rank that writes it, so it must not
    start a collective.

    Usage::

        @ExportModuleRegistry.register("MyFusedExperts", prepend=True)
        class _MyFusedExperts(ModuleExportHandler):
            def on_pack(self, name, module, ctx): ...

            def on_post_pack(self, name, module, ctx): ...

    A model's own handlers live in its package under ``modelopt.torch.models`` and register there,
    ahead of the built-in ones (``prepend=True``), once a model needs them: its ``modeling_ptq``
    must not import the export when it is imported. Quantization imports it through its Hugging Face
    plugin and the export imports quantization, a cycle that resolves at module scope only because
    ``modelopt.torch`` imports quantization's core before its plugins.
    """

    def on_prepare_moe_inputs(self, name: str, module: nn.Module, ctx: ExportContext) -> None:
        """Before packing, for an MoE block whose ``experts`` container the handler matches.

        ``module`` is the block. The export refuses an MoE block whose experts no handler prepares.
        """

    def on_pack(self, name: str, module: nn.Module, ctx: ExportContext) -> None:
        """In the walk that packs each module's quantized weights in place."""

    def on_post_pack(self, name: str, module: nn.Module, ctx: ExportContext) -> None:
        """Once the export has packed the module and every module under it.

        For example, to stack experts it quantized one by one back into one weight.
        """


# The phases a handler can take part in, in the order an export runs them: ModuleExportHandler's
# methods, which name them.
_PHASES = tuple(
    phase for name, phase in vars(ModuleExportHandler).items() if name.startswith("on_")
)

_Phase = Callable[[ModuleExportHandler, str, nn.Module, ExportContext], None]
"""A phase, named by its method on :class:`ModuleExportHandler`."""


def _phases_of(handler: type) -> dict[_Phase, ExportHandler]:
    """The phases a registered handler class takes part in, each with its function.

    A class taking part in none is refused, as is one defining an ``on_*`` method that is not a
    phase, such as a misspelt one, which would never run.
    """
    if not issubclass(handler, ModuleExportHandler):
        raise TypeError(f"{handler.__qualname__} is not a ModuleExportHandler.")
    names = [phase.__name__ for phase in _PHASES]
    unknown = sorted({name for name in dir(handler) if name.startswith("on_")} - set(names))
    overridden = [phase for phase in _PHASES if getattr(handler, phase.__name__) is not phase]
    if unknown or not overridden:
        raise ValueError(
            f"{handler.__qualname__} defines {unknown or 'no phase'}; a handler defines some of "
            f"{names}."
        )
    instance = handler()
    return {phase: getattr(instance, phase.__name__) for phase in overridden}


class _ExportHandlerRegistryCls:
    """Ordered registry mapping modules to the handlers of the export's phases.

    An entry can match a module by any combination of:

    - a class key: the registered class appears in ``type(module).__mro__``, so
      dynamically generated quantized classes (e.g. ``QuantLinear``) match through
      their original base class;
    - a class-name string key: the string equals the ``__name__`` of a class in the
      MRO — for classes that cannot be imported statically (trust_remote_code models
      or on-the-fly generated quantized classes);
    - a predicate on the module instance, for structural detection.

    When keys and a predicate are both given, both must match. Each phase takes the first entry,
    in registration order, that matches the module and defines that phase: more specific handlers
    must be registered before generic ones, and a handler that takes no part in a phase never hides
    one that does. External handlers can use ``prepend=True`` to take precedence over built-in
    entries.
    """

    def __init__(self) -> None:
        self._entries: list[
            tuple[
                tuple[type | str, ...],
                Callable[[nn.Module], bool] | None,
                dict[_Phase, ExportHandler],
                tuple[str, ...],
            ]
        ] = []

    def register(
        self,
        *keys: type | str,
        predicate: Callable[[nn.Module], bool] | None = None,
        prepend: bool = False,
    ) -> Callable[[_Registered], _Registered]:
        """Return a decorator registering a :class:`ModuleExportHandler` subclass.

        A plain function registers as a handler's ``on_pack``, as before handlers had phases.
        Re-registering the same class or function (e.g. on module reload) replaces its existing
        entry in place instead of appending a duplicate.
        """
        assert keys or predicate is not None, "register() requires at least one key or a predicate"

        def decorator(handler: _Registered) -> _Registered:
            registered: Any = handler
            phases: dict[_Phase, ExportHandler] = (
                _phases_of(registered)
                if isinstance(registered, type)
                else {ModuleExportHandler.on_pack: registered}
            )
            self._add(keys, predicate, phases, registered, prepend)
            return handler

        return decorator

    def _add(
        self,
        keys: tuple[type | str, ...],
        predicate: Callable[[nn.Module], bool] | None,
        phases: dict[_Phase, ExportHandler],
        registered: Any,
        prepend: bool,
    ) -> None:
        # A class is one handler, whatever phases it takes part in, so re-registering it replaces it.
        # A function registered for two phases, as a plain one and through the deprecated
        # PrepareMoEInputsRegistry, is two, one per registry before they merged.
        identity = (registered.__module__, registered.__qualname__)
        if not isinstance(registered, type):
            identity += tuple(phase.__name__ for phase in phases)
        entry = (keys, predicate, phases, identity)
        for i, existing in enumerate(self._entries):
            if existing[3] == identity:
                self._entries[i] = entry
                return
        if prepend:
            self._entries.insert(0, entry)
        else:
            self._entries.append(entry)

    def match(
        self, module: nn.Module, phase: _Phase = ModuleExportHandler.on_pack
    ) -> ExportHandler | None:
        """Return ``phase`` of the first registered handler that matches ``module`` and defines it.

        ``phase`` is a phase's method on :class:`ModuleExportHandler`, by default ``on_pack``, the
        one there was before handlers had phases. Anything else raises ``ValueError``.
        """
        if phase not in _PHASES:
            raise ValueError(
                f"{phase!r} is not one of the export's phases, "
                f"{[known.__qualname__ for known in _PHASES]}."
            )
        mro = type(module).__mro__
        mro_names: set[str] | None = None  # built once an entry with keys takes part in the phase
        for keys, predicate, phases, _ in self._entries:
            hook = phases.get(phase)
            if hook is None:
                continue
            if keys:
                if mro_names is None:
                    mro_names = {cls.__name__ for cls in mro}
                if not any(
                    key in mro_names if isinstance(key, str) else key in mro for key in keys
                ):
                    continue
            if predicate is not None and not predicate(module):
                continue
            return hook
        return None


# The registry of the export's handlers: a handler's ``on_prepare_moe_inputs`` matches an MoE
# block's ``.experts`` container and is handed the block; its other phases match, and are handed,
# the module itself.
ExportModuleRegistry = _ExportHandlerRegistryCls()


class _PrepareMoEInputsRegistryCls:
    """Deprecated: the registry of MoE input preparation, now a phase of the export's handlers.

    It registers a function as the ``on_prepare_moe_inputs`` phase of a handler in the registry it
    serves, and matches that phase there, until it goes.
    """

    def __init__(self, registry: _ExportHandlerRegistryCls) -> None:
        self._registry = registry

    def register(
        self,
        *keys: type | str,
        predicate: Callable[[nn.Module], bool] | None = None,
        prepend: bool = False,
    ) -> Callable[[ExportHandler], ExportHandler]:
        """Return a decorator registering a function that prepares an MoE block's inputs.

        It matches the block's ``experts`` container and is handed the block.
        """
        _warn_prepare_registry_deprecated()
        assert keys or predicate is not None, "register() requires at least one key or a predicate"

        def decorator(handler: ExportHandler) -> ExportHandler:
            phases: dict[_Phase, ExportHandler] = {
                ModuleExportHandler.on_prepare_moe_inputs: handler
            }
            self._registry._add(keys, predicate, phases, handler, prepend)
            return handler

        return decorator

    def match(self, module: nn.Module) -> ExportHandler | None:
        """Return the MoE input preparation of the first handler that matches ``module``, or None."""
        _warn_prepare_registry_deprecated()
        return self._registry.match(module, ModuleExportHandler.on_prepare_moe_inputs)


def _warn_prepare_registry_deprecated() -> None:
    warnings.warn(
        "PrepareMoEInputsRegistry is deprecated and will be removed in 0.51.0: register a "
        "ModuleExportHandler defining on_prepare_moe_inputs with ExportModuleRegistry, and match "
        "it with ExportModuleRegistry.match(module, ModuleExportHandler.on_prepare_moe_inputs).",
        DeprecationWarning,
        stacklevel=3,
    )


# Deprecated: registers and matches the ``on_prepare_moe_inputs`` phase of ExportModuleRegistry's
# handlers.
PrepareMoEInputsRegistry = _PrepareMoEInputsRegistryCls(ExportModuleRegistry)
