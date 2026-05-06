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

"""Shared mixin for HuggingFace speculative decoding model classes."""

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from .modeling_fakebase import _BASE_MODEL_PATHS, _EMBED_TOKENS_PATHS, _LM_HEAD_PATHS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # At type-check time, pretend the mixin is an nn.Module so attribute access
    # like self.get_submodule(...) type-checks. At runtime it remains a plain
    # mixin (object) and gets nn.Module via the sibling base class in the MRO.
    _Host: TypeAlias = torch.nn.Module
else:
    _Host = object

__all__ = ["HFSpecDecMixin"]


class HFSpecDecMixin(_Host, ABC):
    """Mixin providing HuggingFace base-model discovery for speculative decoding plugins.

    Provides shared properties and methods for locating base-model submodules
    (backbone, embeddings, lm_head), plus NVTX profiling and torch.compile helpers.

    Must be used with multiple inheritance alongside an algorithm-specific base
    (EagleModel, DFlashModel, etc.) that inherits from DynamicModule.

    Lifecycle:
        Base-model paths are discovered automatically inside ``modify()`` via the
        MRO hook below — subclasses only need to call ``super().modify(config)``
        and the ``_base_model`` / ``_base_model_embeddings`` / ``_base_model_lm_head``
        properties are ready to use in the rest of the subclass's ``modify()`` body.

    Example::

        @EagleDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
        class HFEagleModel(HFSpecDecMixin, EagleModel): ...
    """

    # -- Host-supplied attributes (declared for type checkers) --

    # Provided by the host (e.g., PreTrainedModel.config).
    config: Any
    # Set by ``_find_base_model_parts()``.
    base_model_path: str
    base_model_embeddings_path: str
    base_model_lm_head_path: str

    # -- Class attributes (subclasses may override) --

    # List of (method_name, compile_kwargs) for _activate_torch_compile().
    # Example: [("_eagle_forward", {"mode": "max-autotune"}), ("_eagle_loss", {"fullgraph": True})]
    _compile_targets: list[tuple[str, dict]] = []

    # Set to True in subclass ``modify()`` to enable NVTX ranges.
    _enable_nvtx: bool = False

    # -- Properties: base model access --

    @property
    def _base_model(self) -> torch.nn.Module:
        return self.get_submodule(self.base_model_path)

    @property
    def _base_model_embeddings(self) -> torch.nn.Module:
        return self.get_submodule(self.base_model_embeddings_path)

    @property
    def _base_model_lm_head(self) -> torch.nn.Module:
        return self.get_submodule(self.base_model_lm_head_path)

    @property
    def _base_llm_config(self):
        """Return the LLM config for the base model, handling VLM nesting."""
        return (
            getattr(self.config, "text_config", None)
            or getattr(self.config, "llm_config", None)
            or self.config
        )

    # -- Lifecycle hook --

    def modify(self, config):
        """Run base-class ``modify``, then auto-discover base-model paths.

        Subclasses only need to call ``super().modify(config)`` first; the base-model
        properties are then ready to use in the rest of the subclass's ``modify()`` body.
        """
        super().modify(config)
        self._find_base_model_parts()

    # -- Methods: model discovery --

    def _find_base_model_parts(self):
        """Find model parts from different models and set base_{part}_path attributes.

        Iterates over candidate submodule paths from modeling_fakebase to locate the
        base model backbone, embedding layer, and LM head.

        Raises:
            ValueError: If any required model part cannot be found.
        """
        for name, paths in {
            "base_model_path": _BASE_MODEL_PATHS,
            "base_model_embeddings_path": _EMBED_TOKENS_PATHS,
            "base_model_lm_head_path": _LM_HEAD_PATHS,
        }.items():
            for path in paths:
                try:
                    self.get_submodule(path)
                    setattr(self, name, path)
                    logger.debug("Found %s at %s", name, path)
                    break
                except Exception:
                    continue
            else:
                raise ValueError(f"Part {name} not found in model")

    # -- Methods: profiling & compilation --

    def _nvtx_range(self, name):
        """Optionally create an NVTX range for profiling.

        Enabled when the subclass sets ``self._enable_nvtx = True`` in ``modify()``.
        """
        if not self._enable_nvtx:
            return contextlib.nullcontext()
        try:
            import torch.cuda.nvtx as nvtx

            return nvtx.range(name)
        except Exception as e:
            print(f"Failed to create NVTX range {name}: {e}")
            return contextlib.nullcontext()

    def _activate_torch_compile(self):
        """Apply ``torch.compile`` to methods listed in ``_compile_targets``.

        Each entry is ``(method_name, extra_kwargs)`` passed to ``torch.compile(..., dynamic=False)``.
        Failures fall back to eager mode silently.
        """
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True  # Allow fallback to eager mode

        for name, kwargs in self._compile_targets:
            try:
                setattr(self, name, torch.compile(getattr(self, name), dynamic=False, **kwargs))
            except Exception:  # noqa: PERF203
                print(f"Disabling torch.compile for {name} due to compilation error.")

    # -- Required interface --

    @abstractmethod
    def get_exporter(self):
        """Return the exporter for the draft model."""

    @abstractmethod
    def get_dummy_inputs(self) -> dict:
        """Construct dummy inputs for the export forward pass.

        Used by unified HF quantization export to drive a fake forward when the
        model's ``forward`` signature is non-standard (e.g. takes ``base_model_outputs``).
        Subclasses that don't yet support this path should raise ``NotImplementedError``
        with a clear message so callers fail loudly rather than silently.
        """
