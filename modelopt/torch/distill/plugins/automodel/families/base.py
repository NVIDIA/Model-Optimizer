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

"""Shared native descriptor behavior; family exceptions live beside their specs."""

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

from ....local_units import _has_quantized_weights, _valid_token_mask
from ..runners import native_fast_replay

__all__ = ["NativeFamilyAdapter"]


@dataclass(frozen=True)
class NativeFamilyAdapter:
    """Model-owned boundaries and execution hooks for one native family."""

    name: str
    explicit_padding_mask: bool = False
    packed_document_key: str | None = None
    unit_runner: Callable | None = native_fast_replay

    def units(self, model, selection, *, quantized_only=True):
        """Resolve actual registered children, including property-backed aliases."""
        aliases = {
            "mlp": ("mlp", "moe", "ffn"),
            "moe": ("mlp", "moe", "ffn"),
            "attention": ("self_attn", "attn", "attention"),
            "self_attn": ("self_attn", "attn"),
            "gdn": ("linear_attn",),
            "linear_attn": ("linear_attn",),
        }
        # This is also AutoModel's FSDP and layerwise-calibration boundary,
        # excluding unrelated vision/audio stacks with similarly named MLPs.
        from modelopt.torch.quantization.plugins.automodel import _native_decoder_layers

        # PP embedding/head-only stages own no default quantized repair units.
        if quantized_only and not _has_quantized_weights(model):
            return []
        layers = _native_decoder_layers(model)
        if layers is None:
            return []
        names = {id(module): path for path, module in model.named_modules()}
        candidates = []
        for block in layers:
            path = names[id(block)]
            if "decoder" in selection.unit_types:
                if not quantized_only or _has_quantized_weights(block):
                    candidates.append(path)
                continue
            attributes = list(
                dict.fromkeys(
                    attribute
                    for role in selection.unit_types
                    for attribute in aliases.get(role, (role,))
                )
            )
            attributes = self.unit_attributes(block, selection, attributes)
            for attribute in attributes:
                # Use registration names, never a property alias that disagrees
                # with FSDP/DCP ownership (DeepSeek V4.1's mlp is really ffn).
                child = block._modules.get(attribute)
                if child is not None and (not quantized_only or _has_quantized_weights(child)):
                    candidates.append((path + "." if path else "") + attribute)
        return candidates

    def unit_attributes(self, block, selection, attributes):
        """Extend semantic aliases for a family's hybrid blocks."""
        return attributes

    def mlp_weight_patterns(self, model):
        """Select decoder MLP/expert weights while preserving routers and towers."""
        patterns = []
        for path in self.units(
            model, SimpleNamespace(unit_types=("mlp", "moe")), quantized_only=False
        ):
            module = model.get_submodule(path)
            owners = [path]
            if "experts" in module._modules:
                owners = [
                    path + "." + name
                    for name in ("experts", "shared_experts", "shared_expert")
                    if name in module._modules
                ]
            for owner in owners:
                patterns.extend((owner + ".*weight_quantizer", owner + ".*weight_quantizers.*"))
        return patterns

    def packed_batch(self, batch):
        """Keep the token padding mask separate from a block-causal mask."""
        documents = batch.get("_packed_seq_ids")
        if self.packed_document_key is not None and documents is not None:
            batch = {**batch, self.packed_document_key: documents, "attention_mask": documents > 0}
            batch.pop("_packed_seq_ids")
        if self.explicit_padding_mask and batch.get("attention_mask") is not None:
            if batch["attention_mask"].ndim == 4 and "padding_mask" not in batch:
                valid = _valid_token_mask(batch)
                if valid is not None:
                    return {**batch, "padding_mask": ~valid}
        return batch

    def packed_cp_batch(self, batch, *, cp_size=1):
        """Prepare compact native CP metadata when the family needs it."""
        return batch

    def cp_loss_mask(self, valid):
        """Choose the native sharder mask contract without altering validity."""
        return valid

    def configure_context_parallel(self, model, mesh_context):
        """Optional family-owned native transport setup."""

    def snapshot_unit_state(self, module):
        """Optional per-forward side inputs owned by a native family."""
        return None

    def replay_unit(self, module, args, kwargs, replay_state=None):
        """Native correctness baseline; families may supply a validated runner."""
        if self.unit_runner is not None:
            return self.unit_runner(module, args, kwargs)
        return module(*args, **kwargs)
