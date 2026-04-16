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

# mypy: disable-error-code="attr-defined,misc"

import contextlib

import torch
from torch.nn import CrossEntropyLoss

from .modeling_fakebase import _BASE_MODEL_PATHS, _EMBED_TOKENS_PATHS, _LM_HEAD_PATHS

__all__ = ["HFSpecDecMixin"]


class HFSpecDecMixin:
    """Mixin providing HuggingFace base-model discovery for speculative decoding plugins.

    Provides shared properties and methods for locating base-model submodules
    (backbone, embeddings, lm_head) and running the base-model forward pass.

    Must be used with multiple inheritance alongside an algorithm-specific base
    (EagleModel, DFlashModel, etc.) that inherits from DynamicModule.

    Example::

        @EagleDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
        class HFEagleModel(HFSpecDecMixin, EagleModel): ...
    """

    # -- Class attributes (subclasses may override) --

    # List of (method_name, compile_kwargs) for _activate_torch_compile().
    # Example: [("_eagle_forward", {"mode": "max-autotune"}), ("_eagle_loss", {"fullgraph": True})]
    _compile_targets: list[tuple[str, dict]] = []

    # -- Properties: base model access --

    @property
    def _base_model(self):
        return self.get_submodule(self.base_model_path)

    @property
    def _base_model_embeddings(self):
        return self.get_submodule(self.base_model_embeddings_path)

    @property
    def _base_model_lm_head(self):
        return self.get_submodule(self.base_model_lm_head_path)

    @property
    def _base_llm_config(self):
        """Return the LLM config for the base model, handling VLM nesting."""
        return (
            getattr(self.config, "text_config", None)
            or getattr(self.config, "llm_config", None)
            or self.config
        )

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
                    submodule = self.get_submodule(path)
                    assert isinstance(submodule, torch.nn.Module)
                    setattr(self, name, path)
                    break
                except Exception:
                    continue
            else:
                raise ValueError(f"Part {name} not found in model")

    # -- Methods: base model forward --

    def _base_model_forward(self, input_ids, attention_mask, freeze=True, labels=None, **kwargs):
        """Run the base model forward pass with optional freeze and base-model loss.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            freeze: If True, run under torch.no_grad().
            labels: Optional labels for computing base model CE loss.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            (outputs, base_loss) tuple where outputs is the raw model output and
            base_loss is the cross-entropy loss (None if freeze=True or labels=None).
        """
        ctx = torch.no_grad() if freeze else contextlib.nullcontext()
        with ctx:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
            base_loss = None
            if not freeze and labels is not None:
                loss_fct = CrossEntropyLoss()
                base_loss = loss_fct(
                    outputs.logits.view(-1, outputs.logits.shape[-1]),
                    labels.view(-1),
                )
        return outputs, base_loss

    # -- Methods: profiling & compilation --

    def _nvtx_range(self, name):
        """Optionally create an NVTX range for profiling.

        Enabled when the subclass sets ``self._enable_nvtx = True`` in ``modify()``.
        """
        if not getattr(self, "_enable_nvtx", False):
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

    # -- Methods: export interface (subclasses must override) --

    def get_dummy_inputs(self) -> dict:
        """Construct dummy inputs for export forward pass. Subclasses must override."""
        raise NotImplementedError

    def get_exporter(self):
        """Return the exporter for the draft model. Subclasses must override."""
        raise NotImplementedError
