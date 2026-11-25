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

"""Registries dispatching per-module logic for the unified HF export path.

This mirrors the registration-and-dispatch idiom of
:class:`QuantModuleRegistry <modelopt.torch.quantization.nn.modules.quant_module.QuantModuleRegistry>`,
but not its mechanism: quantization registers replacement classes and converts modules
in place, whereas export registers functions that emit compressed weights and scale
buffers for a module without changing its class.

Preparation and export use separate registries because they have independent matching
precedence. Registering a handler for a new module type replaces what previously required
editing if/elif chains inside ``unified_export_hf.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from modelopt.torch.export.hf_export_handlers import _HeldWindow
    from modelopt.torch.quantization.utils.core_utils import ModuleNames

__all__ = [
    "ExportContext",
    "ExportFormat",
    "ExportHandler",
    "ExportModuleRegistry",
    "PrepareMoEInputsRegistry",
]


@dataclass
class ExportContext:
    """Shared state for a single export invocation, passed to every handler call.

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
    model's spec in ``modelopt.torch.models``. ``None`` means unknown: spec lookups then
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


ExportHandler = Callable[[str, nn.Module, ExportContext], None]


@dataclass(frozen=True, kw_only=True)
class ExportFormat:
    """The format in which an :data:`ExportModuleRegistry` handler writes the weights it exports."""

    quantization_format: str
    """A format of :mod:`~modelopt.torch.export.quant_format`, such as ``w4a16_nvfp4``."""

    block_size: int = 0
    """The number of weights per scale block, the ``group_size`` of the quantization config, which
    a blocked format such as ``w4a16_nvfp4`` requires; 0 for a format without blocks."""

    def __post_init__(self) -> None:
        if not isinstance(self.quantization_format, str) or not self.quantization_format:
            raise ValueError(
                f"quantization_format must name a format, got {self.quantization_format!r}"
            )
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size < 0
        ):
            raise ValueError(f"block_size must be an int >= 0, got {self.block_size!r}")
        # Check the declaration against the config entry that export writes for it, which also
        # checks the block size of a GGML format. quant_utils imports this module.
        from .quant_utils import process_layer_quant_config

        config = process_layer_quant_config(
            {
                "layer.quantization": self.quantization_format,
                "layer.awq_block_size": self.block_size,
            }
        )
        # A name that export does not know passes through to the quant_algo unchanged.
        if config["quant_algo"] == self.quantization_format:
            raise ValueError(
                f"{self.quantization_format!r} is not a format that export writes a config for"
            )
        if ("group_size" in config) != (self.block_size > 0):
            raise ValueError(
                f"block_size must be {'> 0' if 'group_size' in config else '0'} for "
                f"{self.quantization_format}, got {self.block_size}"
            )


class _ExportHandlerRegistryCls:
    """Ordered, first-match-wins registry mapping modules to handler functions.

    An entry can match a module by any combination of:

    - a class key: the registered class appears in ``type(module).__mro__``, so
      dynamically generated quantized classes (e.g. ``QuantLinear``) match through
      their original base class;
    - a class-name string key: the string equals the ``__name__`` of a class in the
      MRO — for classes that cannot be imported statically (trust_remote_code models
      or on-the-fly generated quantized classes);
    - a predicate on the module instance, for structural detection.

    When keys and a predicate are both given, both must match. Entries are tried in
    registration order and the first match wins, so more specific handlers must be
    registered before generic ones. External handlers can use ``prepend=True`` to take
    precedence over built-in entries.
    """

    def __init__(self) -> None:
        self._entries: list[
            tuple[
                tuple[type | str, ...],
                Callable[[nn.Module], bool] | None,
                ExportHandler,
            ]
        ] = []

    def register(
        self,
        *keys: type | str,
        predicate: Callable[[nn.Module], bool] | None = None,
        prepend: bool = False,
    ) -> Callable[[ExportHandler], ExportHandler]:
        """Return a decorator registering a handler function.

        Re-registering the same handler (e.g. on module reload) replaces its existing
        entry in place instead of appending a duplicate.

        Usage::

            @ExportModuleRegistry.register(
                "Llama4TextExperts",
                "GptOssExperts",
            )
            def _export_bmm_experts(name, module, ctx): ...
        """
        assert keys or predicate is not None, "register() requires at least one key or a predicate"

        def decorator(handler: ExportHandler) -> ExportHandler:
            entry = (keys, predicate, handler)
            identity = (handler.__module__, handler.__qualname__)
            for i, (_, _, existing) in enumerate(self._entries):
                if (existing.__module__, existing.__qualname__) == identity:
                    self._entries[i] = entry
                    return handler
            if prepend:
                self._entries.insert(0, entry)
            else:
                self._entries.append(entry)
            return handler

        return decorator

    def match(self, module: nn.Module) -> ExportHandler | None:
        """Return the first registered handler matching ``module``, or ``None``."""
        mro = type(module).__mro__
        mro_names = {cls.__name__ for cls in mro}
        for keys, predicate, handler in self._entries:
            if keys and not any(
                key in mro_names if isinstance(key, str) else key in mro for key in keys
            ):
                continue
            if predicate is not None and not predicate(module):
                continue
            return handler
        return None


class _ExportModuleRegistryCls(_ExportHandlerRegistryCls):
    """The registry of the whole-model export walk, whose handlers can declare their format."""

    def __init__(self) -> None:
        super().__init__()
        # The format each handler declares, by the identity that re-registering it replaces it by.
        self._declared: dict[tuple[str, str], ExportFormat] = {}

    def register(
        self,
        *keys: type | str,
        predicate: Callable[[nn.Module], bool] | None = None,
        prepend: bool = False,
        export_format: ExportFormat | None = None,
    ) -> Callable[[ExportHandler], ExportHandler]:
        """Return a decorator registering a handler function.

        Keys, ``predicate`` and ``prepend`` match as in the other registries. ``export_format``
        (an :class:`ExportFormat`) declares the format in which the handler writes the quantized
        weights of the modules it exports. Export then reads the format and block size of such a
        module, and with them its quantization config, from this entry instead of from the module's
        quantizers, as long as the module's weight quantizer is enabled. The entry applies only to
        the modules it matches first, the ones its handler exports, so register it with
        ``prepend=True``.

        Declare the format for weights that a custom quantization backend (see
        :func:`~modelopt.torch.quantization.nn.modules.tensor_quantizer.register_quant_backend`)
        quantizes with a ``num_bits`` of its own, such as a string, which tells export nothing of
        how to store them, so that it raises ``NotImplementedError: Unsupported quantizer``. The
        declared format covers the whole module, its input quantization included: ``w4a16_nvfp4``
        for NVFP4 weights with unquantized inputs, ``nvfp4`` with NVFP4 inputs, and, as format
        detection gives, ``nvfp4_awq`` for either when the input quantizer holds an AWQ
        ``pre_quant_scale``: export decides from an AWQ format whether to fold that scale and unify
        it across the layers that share an input.

        The handler writes the packed weight and every scale of the format, the input scale
        included. Export itself writes the input quantizer's ``pre_quant_scale``. The handler
        receives each module after the usual export preprocessing, which can change it: it can
        merge the ``amax`` of layers that share an input, such as q/k/v and an MoE expert's gate and
        up projections, and fold a pre-quant scale (AWQ) into the preceding layernorm or linear,
        which rescales weights and recalibrates their ``amax``. Under
        :attr:`ExportContext.keep_fused_experts`, the distributed export that packs a still-sharded
        model, the handler must pack as the built-in handlers do, FSDP2-owned weights inside their
        unit's weight-update window and fused experts kept fused, or raise. The export of an
        offloaded model calls the handler only for the decoder layers and the modules it
        materializes, and raises for another such module, such as an untied ``lm_head`` kept
        resident.

        ``export_hf_vllm_fq_checkpoint``, which writes the weights the backend fake-quantizes,
        reads the declared format and does not call the handler. It folds a fake-quant
        ``TensorQuantizer`` into the weight and raises for any other weight quantizer of these
        modules, such as a ``SequentialQuantizer``, and, with ``inplace_mem_efficient=True``, which
        folds only the decoder layers, for these modules outside them. The exporters that pack
        weights with the built-in math, such as the Megatron export, raise for these modules.

        Usage::

            register_quant_backend("my_backend", my_fake_quant)


            @ExportModuleRegistry.register(
                predicate=has_my_backend_weights,
                prepend=True,
                export_format=ExportFormat(
                    quantization_format=QUANTIZATION_W4A16_NVFP4, block_size=16
                ),
            )
            def _export_my_backend(name, module, ctx): ...
        """
        if export_format is not None and not isinstance(export_format, ExportFormat):
            raise TypeError(f"export_format must be an ExportFormat, got {export_format!r}")
        register = super().register(*keys, predicate=predicate, prepend=prepend)

        def decorator(handler: ExportHandler) -> ExportHandler:
            identity = (handler.__module__, handler.__qualname__)
            self._declared.pop(identity, None)
            if export_format is not None:
                self._declared[identity] = export_format
            return register(handler)

        return decorator

    def declared_format(self, module: nn.Module) -> ExportFormat | None:
        """Return the format that the handler exporting ``module`` declares, or ``None``."""
        # Format detection calls this for every quantized layer: match only if a format is declared.
        if not self._declared:
            return None
        handler = self.match(module)
        if handler is None:
            return None
        return self._declared.get((handler.__module__, handler.__qualname__))


# Matches an MoE block's ``.experts`` container; the handler receives the enclosing block.
PrepareMoEInputsRegistry = _ExportHandlerRegistryCls()
# Matches and passes the same module to the handler during the whole-model export walk.
ExportModuleRegistry = _ExportModuleRegistryCls()
