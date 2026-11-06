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

"""Thin architecture contracts; optimization and native-format math stay shared."""

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

from torch import nn
from torch.distributed.tensor import DTensor, Shard

from modelopt.torch.quantization.repair_partition import requires_full_context_activations

from ....local_units import ModuleLocalDistillationAdapter, _has_quantized_weights
from ..output_head import _FULL_REPAIR_LOGITS, repair_head_context
from ..packing import document_batching

__all__ = ["TransformersFamilyAdapter"]


@dataclass(frozen=True)
class TransformersFamilyAdapter(ModuleLocalDistillationAdapter):
    """Describe semantic units without replacing a model's actual forward math.

    Custom HF implementations can provide explicit decoder paths and override
    unit_attributes for hybrid blocks. Registered module names remain checkpoint
    ownership names; property aliases are never used as independent boundaries.
    functional_children names decoder-relative children read directly by their parent.
    """

    name: str = "generic"
    decoder_paths: tuple[str, ...] = ()
    isolated_documents: bool = True
    functional_children: tuple[str, ...] = ()
    additional_fsdp_children: tuple[str, ...] = ()
    batch_packed_documents: bool = False

    def output_head(self, model):
        """Resolve the real text task's output projection, including wrapped tasks."""
        target = self.teacher_forward_target(model)
        getter = getattr(target, "get_output_embeddings", None)
        if getter is None:
            return None
        try:
            return getter()
        except NotImplementedError:
            return None

    @contextmanager
    def repair_forward_context(self, model, units):
        """Omit unselected output-head math while running the normal task lifecycle."""
        # Calibration hooks do not mask added padding; keep their original inputs.
        with (
            document_batching(bool(units) and self.document_batching_supported(model)),
            repair_head_context(model, units, self.output_head(model)),
        ):
            yield

    def forward_teacher(self, teacher, args, kwargs):
        """Head repair needs full-token targets; other units avoid unused logits."""
        if _FULL_REPAIR_LOGITS.get():
            parameters = inspect.signature(self.teacher_forward_target(teacher).forward).parameters
            kwargs = {
                **kwargs,
                **{key: 0 for key in ("logits_to_keep", "num_logits_to_keep") if key in parameters},
            }
        return super().forward_teacher(teacher, args, kwargs)

    def transform_logits(self, model, logits):
        """Apply task-owned post-projection math; custom heads override this contract."""
        config = getattr(self.teacher_forward_target(model), "config", None)
        if callable(getattr(config, "get_text_config", None)):
            config = config.get_text_config()
        if (
            getattr(config, "final_logit_softcapping", None) not in (None, 0)
            or getattr(config, "output_multiplier", 1) != 1
        ):
            raise ValueError("This task needs a family-owned transform_logits implementation.")
        return logits

    def decoder_layers(self, model):
        """Resolve the text decoder through HF's public roots or explicit paths."""
        if self.decoder_paths:
            for path in self.decoder_paths:
                try:
                    layers = model.get_submodule(path)
                except AttributeError:
                    continue
                if isinstance(layers, (nn.ModuleList, nn.ModuleDict)):
                    return layers
            raise ValueError(f"No decoder stack matches the {self.name} family adapter.")
        queue, seen = [model], set()
        get_decoder = getattr(model, "get_decoder", None)
        if callable(get_decoder):
            try:
                decoder = get_decoder()
            except (AttributeError, NotImplementedError):
                decoder = None
            if isinstance(decoder, nn.Module):
                queue.insert(0, decoder)
        while queue:
            root = queue.pop(0)
            if id(root) in seen:
                continue
            seen.add(id(root))
            for attribute in ("layers", "h", "block"):
                layers = getattr(root, attribute, None)
                if isinstance(layers, (nn.ModuleList, nn.ModuleDict)):
                    return layers
            attributes = dict.fromkeys(
                (
                    "thinker",
                    "language_model",
                    "text_model",
                    "decoder",
                    "model",
                    "backbone",
                    "transformer",
                    getattr(root, "base_model_prefix", ""),
                )
            )
            queue.extend(
                child
                for attribute in attributes
                if attribute
                and isinstance(child := getattr(root, attribute, None), nn.Module)
                and id(child) not in seen
            )
        raise ValueError(
            "Cannot discover HF transformer layers; provide explicit units or a family adapter "
            "with decoder_paths."
        )

    def unit_attributes(self, block, unit_types):
        """Map shared semantic names onto registered children in ordinary decoders."""
        result = []
        for role in unit_types:
            if role in ("mlp", "moe"):
                for attribute in ("mlp", "moe", "ffn", "block_sparse_moe"):
                    child = block._modules.get(attribute)
                    if child is not None and (role == "mlp" or self.is_moe(child)):
                        result.append(attribute)
            else:
                aliases = {
                    "attention": ("self_attn", "attn", "attention"),
                    "gdn": ("linear_attn",),
                    "mamba": ("mamba",),
                }
                result.extend(aliases.get(role, (role,)))
        return list(dict.fromkeys(result))

    @staticmethod
    def is_moe(module):
        """Recognize routed feed-forward boundaries, excluding dense MLPs."""
        return any(hasattr(module, name) for name in ("experts", "routed_experts"))

    def unit_paths(self, model, selection, *, quantized_only=True):
        """Select semantic children; unquantized paths also support PTQ setup."""
        names = {id(module): path for path, module in model.named_modules()}
        layers = self.decoder_layers(model)
        result = []
        for block in layers.values() if isinstance(layers, nn.ModuleDict) else layers:
            path = names[id(block)]
            if "decoder" in selection.unit_types:
                if not quantized_only or _has_quantized_weights(block):
                    result.append(path)
                continue
            for attribute in self.unit_attributes(block, selection.unit_types):
                child = block._modules.get(attribute)
                if child is not None and (not quantized_only or _has_quantized_weights(child)):
                    result.append(names[id(child)])
        return list(dict.fromkeys(result))

    def weight_patterns(self, model, selection):
        """Select unit-owned projections for PTQ while leaving routing decisions intact."""
        result = []
        paths = selection.units
        if paths is None:
            paths = self.unit_paths(model, selection, quantized_only=False)
        for path in paths:
            for name, module in model.get_submodule(path).named_modules():
                if any(part in ("gate", "router") for part in name.split(".")):
                    continue
                if any(parameter.ndim >= 2 for parameter in module.parameters(recurse=False)):
                    result.append(".".join(filter(None, (path, name, "*weight_quantizer*"))))
        return list(dict.fromkeys(result))

    def prepare_batch(self, model, batch):
        """Validate independent document inputs before invoking the actual task."""
        if any(
            batch.get(name) is not None
            for name in ("past_key_values", "past_key_value", "cache_params")
        ):
            raise ValueError("Independent repair microbatches cannot consume generation caches.")
        return batch

    def _default_unit_paths(self, model, selection):
        return self.unit_paths(model, selection)

    def replay_unit(self, module, args, kwargs, replay_state=None):
        """Execute the model's own math; exceptional side-state belongs to its family."""
        return module(*args, **kwargs)

    def objective_tp_mesh(self, module, target):
        """Describe feature-sharded projection outputs; custom units can override this."""
        weight = module._parameters.get("weight")
        if not isinstance(module, nn.Linear) or not isinstance(weight, DTensor):
            return None
        axes = weight.device_mesh.mesh_dim_names or ()
        if "tp" not in axes:
            return None
        placement = weight.placements[axes.index("tp")]
        if not isinstance(placement, Shard) or placement.dim % weight.ndim != 0:
            return None
        mesh = weight.device_mesh["tp"]
        if target.shape[-1] == weight.shape[0] or mesh.size() == 1:
            return None
        if target.shape[-1] * mesh.size() != weight.shape[0]:
            raise ValueError("A custom projection output needs a family-owned TP objective layout.")
        return mesh

    def document_isolation_required(self, model):
        """Retain native window/recurrent masks when a causal family enables them."""
        if self.isolated_documents:
            return True
        config = getattr(model, "config", None)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            config = get_text_config()
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            return any(role not in ("full_attention", "attention") for role in layer_types)
        return bool(getattr(config, "sliding_window", None)) and getattr(
            config, "use_sliding_window", True
        )

    def document_batches(self, model, batch):
        """Preserve recurrence and compression boundaries in packed training data."""
        from ..packing import batch_documents, document_batches

        documents = document_batches(batch)
        if self.document_batching_supported(model):
            return batch_documents(documents, batch["input_ids"].numel())
        return documents

    def document_batching_supported(self, model):
        """Batch only models whose output statistics do not depend on other rows."""
        if not self.batch_packed_documents or requires_full_context_activations(model):
            return False
        config = getattr(model, "config", None)
        if callable(getattr(config, "get_text_config", None)):
            config = config.get_text_config()
        pending = [getattr(config, name, None) for name in ("rope_scaling", "rope_parameters")]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if value.get("rope_type", value.get("type")) in ("dynamic", "longrope"):
                    return False
                pending.extend(value.values())
        return True
