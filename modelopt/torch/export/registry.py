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
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from modelopt.torch.utils.distributed import is_fsdp2_model

from .model_utils import _reorder_canonical_first

__all__ = [
    "ExportContext",
    "ExportHandler",
    "ExportModuleRegistry",
    "PrepareMoEInputsRegistry",
]


@dataclass
class ExportContext:
    """Shared state for a single export invocation, passed to every handler call.

    ``duplicate_weight_map`` maps each duplicate source-parameter name to the canonical
    name that should survive export. It is snapshotted before export mutates Parameters.
    """

    model: nn.Module
    dtype: torch.dtype
    is_modelopt_qlora: bool = False
    duplicate_weight_map: dict[str, str] = field(init=False, default_factory=dict)
    weight_locations: dict[str, tuple[nn.Module, str]] = field(init=False, default_factory=dict)
    weight_formats: dict[str, str | None] = field(init=False, default_factory=dict)
    skipped_weight_names: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        # FSDP2 replaces Parameters while resharding. Keep its existing behavior of writing
        # tied packed weights independently instead of relying on pre-reshard identities.
        if is_fsdp2_model(self.model):
            return

        names_by_source_id: dict[int, list[str]] = {}
        for module_name, module in self.model.named_modules(remove_duplicate=False):
            for weight_name, parameter in module._parameters.items():
                if parameter is None:
                    continue
                full_name = f"{module_name}.{weight_name}" if module_name else weight_name
                self.weight_locations[full_name] = (module, weight_name)
                names_by_source_id.setdefault(id(parameter), []).append(full_name)

        for names in names_by_source_id.values():
            if len(names) < 2:
                continue
            # Reuse the existing HF canonical-name policy. For models without an explicit
            # canonical side this preserves traversal order and keeps the first name.
            ordered = list(_reorder_canonical_first(dict.fromkeys(names), self.model))
            canonical = ordered[0]
            self.duplicate_weight_map.update(dict.fromkeys(ordered[1:], canonical))

    def duplicate_of(self, weight_name: str) -> str | None:
        """Return the canonical source weight name, or ``None`` if it is not duplicated."""
        return self.duplicate_weight_map.get(weight_name)

    def mark_skipped(self, *weight_names: str) -> None:
        """Record duplicate source weights omitted from packing and the state dict."""
        self.skipped_weight_names.update(weight_names)


ExportHandler = Callable[[str, nn.Module, ExportContext], None]


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


# Matches an MoE block's ``.experts`` container; the handler receives the enclosing block.
PrepareMoEInputsRegistry = _ExportHandlerRegistryCls()
# Matches and passes the same module to the handler during the whole-model export walk.
ExportModuleRegistry = _ExportHandlerRegistryCls()
