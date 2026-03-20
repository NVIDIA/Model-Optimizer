# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Forward-hook based feature extraction for layer-wise distillation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn
    from torch import Tensor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Capture intermediate module outputs via ``register_forward_hook``.

    Supports arbitrary dotted module paths (e.g. ``"blocks.5"``,
    ``"blocks.5.self_attn"``).  Per-module *output transforms* convert
    non-Tensor outputs (tuples, dataclasses) into a plain Tensor before
    storage.

    Args:
        model: The ``nn.Module`` to hook into.
        module_paths: Dotted module paths to capture.
        output_transforms: Per-path callables that extract a ``Tensor`` from
            the module's forward output.  Keyed by module path.
        default_transform: Fallback applied when a path has no entry in
            *output_transforms*.  ``None`` means the raw output must already
            be a ``Tensor`` (or a tuple whose first element is a ``Tensor``).
    """

    def __init__(
        self,
        model: nn.Module,
        module_paths: list[str],
        output_transforms: dict[str, Callable] | None = None,
        default_transform: Callable | None = None,
    ) -> None:
        self._features: dict[str, Tensor] = {}
        self._hooks: list[nn.utils.hooks.RemovableHook] = []
        self._transforms = output_transforms or {}
        self._default_transform = default_transform

        modules = dict(model.named_modules())
        for path in module_paths:
            if path not in modules:
                available = [n for n, _ in model.named_modules() if n][:30]
                raise ValueError(
                    f"Module '{path}' not found in model. Available (first 30): {available}"
                )
            hook = modules[path].register_forward_hook(self._make_hook(path))
            self._hooks.append(hook)

        logger.debug("FeatureExtractor: hooked %d modules", len(module_paths))

    def _make_hook(self, path: str):
        transform = self._transforms.get(path, self._default_transform)

        def hook_fn(_module, _input, output):
            if transform is not None:
                self._features[path] = transform(output)
            elif isinstance(output, tuple):
                self._features[path] = output[0]
            else:
                self._features[path] = output

        return hook_fn

    def get_features(self) -> dict[str, Tensor]:
        """Return captured features.  Keys are module paths."""
        return self._features

    def clear(self) -> None:
        """Release stored tensors (call after each training step)."""
        self._features.clear()

    def remove(self) -> None:
        """Permanently unregister all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()
