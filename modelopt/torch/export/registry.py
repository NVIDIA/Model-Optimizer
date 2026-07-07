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

"""Registry dispatching per-module export logic for the unified HF export path.

This mirrors the registration-and-dispatch idiom of
:class:`QuantModuleRegistry <modelopt.torch.quantization.nn.modules.quant_module.QuantModuleRegistry>`,
but not its mechanism: quantization registers replacement classes and converts modules
in place, whereas export registers :class:`ModuleExporter` handlers that emit compressed
weights and scale buffers for a module without changing its class.

Handlers are matched per module during the export walk. Registering a handler for a new
module type replaces what previously required editing if/elif chains inside
``unified_export_hf.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = ["ExportContext", "ExportModuleRegistry", "ModuleExporter"]


@dataclass
class ExportContext:
    """Shared state for a single export invocation, passed to every handler call.

    The tied-weight dedup caches must be scoped to one export invocation: a
    process-global cache would carry stale entries whose ``data_ptr`` keys can be
    recycled by PyTorch's allocator across exports, causing silent false-positive
    aliasing. ``tied_cache`` (int keys) holds dense Linear / per-expert wrapper
    dedup; ``moe_tied_cache`` (tuple keys) holds MoE fused-experts module dedup.
    """

    model: nn.Module
    dtype: torch.dtype
    is_modelopt_qlora: bool = False
    tied_cache: dict[int, nn.Module] = field(default_factory=dict)
    moe_tied_cache: dict[tuple[int, int], nn.Module] = field(default_factory=dict)


class ModuleExporter:
    """Base class for per-module export handlers.

    Subclasses are registered on :data:`ExportModuleRegistry` and dispatched during the
    export walk. Both hooks default to no-ops so a handler only implements the phases
    that apply to its module type.
    """

    def prepare_moe_inputs(self, name: str, moe_module: nn.Module, ctx: ExportContext) -> None:
        """Fill missing expert input-quantizer amax values before fusion and compression.

        Called once per MoE block whose ``.experts`` container matched this entry.
        ``moe_module`` is the MoE block itself, not the experts container.
        """

    def export(self, name: str, module: nn.Module, ctx: ExportContext) -> None:
        """Emit compressed weights and scale buffers for ``module``, in place."""


class _ExportModuleRegistryCls:
    """Ordered, first-match-wins registry mapping modules to :class:`ModuleExporter`.

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
    registered before generic ones. The built-in handlers end with broad structural
    catch-alls (e.g. any iterable module), so external handlers should register with
    ``prepend=True`` to take precedence over them.
    """

    def __init__(self) -> None:
        self._entries: list[
            tuple[tuple[type | str, ...], Callable[[nn.Module], bool] | None, ModuleExporter]
        ] = []

    def register(
        self,
        *keys: type | str,
        predicate: Callable[[nn.Module], bool] | None = None,
        prepend: bool = False,
    ):
        """Return a decorator registering a :class:`ModuleExporter` subclass.

        Re-registering the same exporter class (e.g. on module reload) replaces its
        existing entry in place instead of appending a duplicate.

        Usage::

            @ExportModuleRegistry.register("Llama4TextExperts", "GptOssExperts")
            class _BmmExpertsExporter(ModuleExporter): ...
        """
        assert keys or predicate is not None, "register() requires at least one key or a predicate"

        def decorator(exporter_cls: type[ModuleExporter]) -> type[ModuleExporter]:
            entry = (keys, predicate, exporter_cls())
            identity = (exporter_cls.__module__, exporter_cls.__qualname__)
            for i, (_, _, existing) in enumerate(self._entries):
                if (type(existing).__module__, type(existing).__qualname__) == identity:
                    self._entries[i] = entry
                    return exporter_cls
            if prepend:
                self._entries.insert(0, entry)
            else:
                self._entries.append(entry)
            return exporter_cls

        return decorator

    def match(self, module: nn.Module) -> ModuleExporter | None:
        """Return the first registered exporter matching ``module``, or None."""
        mro = type(module).__mro__
        mro_names = {cls.__name__ for cls in mro}
        for keys, predicate, exporter in self._entries:
            if keys and not any(
                key in mro_names if isinstance(key, str) else key in mro for key in keys
            ):
                continue
            if predicate is not None and not predicate(module):
                continue
            return exporter
        return None


ExportModuleRegistry = _ExportModuleRegistryCls()
