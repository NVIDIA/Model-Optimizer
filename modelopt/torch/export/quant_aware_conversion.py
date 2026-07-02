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

"""Quantization-aware reverse weight conversion for unified HF export.

Background
----------
``transformers`` may apply a ``conversion_mapping`` when loading a model, so the
in-memory parameter names differ from the original model-hub checkpoint (e.g. fused
``mlp.gate_up_proj``, renamed MoE leaves, reordered ``model``/``language_model``
prefix). On save, ``transformers`` reverses this via ``revert_weight_conversion`` so
the on-disk names match the hub checkpoint again.

ModelOpt's unified export disables that reverse (it raises ``IndexError`` on 0-d
scalar scale tensors such as ``weight_scale_2``/``input_scale``), so a quantized
export emits the *in-memory* (post-conversion) names — violating the unified
checkpoint contract that names stay aligned with the original hub checkpoint.

This module performs the reverse in a quantization-aware way: it carries each
weight's companion scale tensors (``weight_scale``, ``weight_scale_2``,
``input_scale``, ``weight_scale_inv``, ``bias``) through the rename and un-fuse
operations.

Scope
-----
Two reverse primitives cover the conversion_mapping cases:

* **Rename** — a key-level string substitution. Because a quantized linear stores
  every tensor under ``<module>.<leaf>``, renaming the module substring rewrites the
  weight and all its scale siblings together with no tensor manipulation.
* **Split** — un-fuse an output-dim concatenation (e.g. dense ``gate_up_proj`` ->
  ``gate_proj`` + ``up_proj``). ``weight``/``weight_scale``/``weight_scale_inv``/
  ``bias`` are chunked along the fused (output) dim; 0-d scalar ``weight_scale_2``/
  ``input_scale`` are duplicated to each part (they are per-tensor and shared).

MoE experts need only **Rename**: ModelOpt's export already expands the fused,
stacked in-memory experts (``experts.gate_up_proj`` of shape ``[E, 2F, H]``) into
per-expert 2-D linears (``experts.<i>.gate_proj`` / ``up_proj`` / ``down_proj``)
before save, so the reverse just maps those per-expert leaf names back to the hub
leaves (e.g. ``gate_proj`` -> ``w1``, ``up_proj`` -> ``w3``, ``down_proj`` -> ``w2``).

Reverse rules are derived from the model's conversion mapping via transformers'
``reverse_transform()``. Any op shape not covered raises
:class:`QuantConversionUnsupportedError` so the caller falls back to the legacy
(in-memory-name) behavior rather than emit a silently-wrong checkpoint.
"""

import re
from dataclasses import dataclass

import torch

__all__ = [
    "QuantConversionUnsupportedError",
    "RenameRule",
    "SplitRule",
    "apply_reverse_rules",
    "revert_weight_conversion_quant_aware",
]

# Tensor leaves that belong to a single quantized linear module. A rename of the
# parent module path applies uniformly to all of these.
_LEAF_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
    ".weight_scale_inv",
    ".input_scale",
    ".bias",
)

# Leaves that are per-tensor scalars (0-d) and must be *duplicated*, not split, when
# a fused module is un-fused.
_SCALAR_LEAF_SUFFIXES = (".weight_scale_2", ".input_scale")


class QuantConversionUnsupportedError(Exception):
    """Raised when a conversion op cannot be reversed quant-aware (caller falls back)."""


@dataclass(frozen=True)
class RenameRule:
    """Reverse of a ``WeightRenaming``: ``re.sub(pattern, repl, key)`` on every key."""

    pattern: str
    repl: str


@dataclass(frozen=True)
class SplitRule:
    """Reverse of an output-dim ``Concatenate``: un-fuse one module into ``parts``.

    Args:
        fused_suffix: module suffix of the fused tensor, e.g. ``".gate_up_proj"``.
        part_suffixes: ordered replacements, e.g. ``(".gate_proj", ".up_proj")``.
        dim: the fused (output) dim along which ``weight``/``weight_scale``/``bias``
            are chunked. NVFP4 ``weight`` is ``[out, in//2]`` and ``weight_scale`` is
            ``[out, in//block]`` so the output dim is ``0`` for both.
    """

    fused_suffix: str
    part_suffixes: tuple[str, ...]
    dim: int = 0


def _split_leaf_tensor(leaf: str, tensor: torch.Tensor, n: int, idx: int, dim: int):
    """Return the ``idx``-th of ``n`` parts of ``tensor`` for tensor leaf ``leaf``."""
    if leaf in _SCALAR_LEAF_SUFFIXES or tensor.dim() == 0:
        # Per-tensor scalar shared across the fused parts -> duplicate.
        return tensor.clone()
    size = tensor.size(dim)
    if size % n != 0:
        raise QuantConversionUnsupportedError(
            f"cannot split leaf '{leaf}' of size {size} along dim {dim} into {n} parts"
        )
    return tensor.chunk(n, dim=dim)[idx].clone()


def _apply_split_rule(state_dict: dict[str, torch.Tensor], rule: SplitRule) -> None:
    """Un-fuse all modules matching ``rule.fused_suffix`` in place."""
    n = len(rule.part_suffixes)
    # Collect (module_path, leaf, key) for every tensor under a fused module.
    fused_keys: list[tuple[str, str, str]] = []
    for key in state_dict:
        for leaf in _LEAF_SUFFIXES:
            if key.endswith(rule.fused_suffix + leaf):
                module = key[: -len(leaf)][: -len(rule.fused_suffix)]
                fused_keys.append((module, leaf, key))
                break

    for module, leaf, key in fused_keys:
        tensor = state_dict.pop(key)
        # A 3-D expert tensor here means stacked experts (MergeModulelist) — out of scope.
        if leaf == ".weight" and tensor.dim() >= 3:
            raise QuantConversionUnsupportedError(
                f"stacked 3-D expert tensor '{key}' (ndim={tensor.dim()}) is not supported; "
                "un-stacking experts + their scales is a follow-up"
            )
        for idx, part in enumerate(rule.part_suffixes):
            state_dict[module + part + leaf] = _split_leaf_tensor(leaf, tensor, n, idx, rule.dim)


def apply_reverse_rules(
    state_dict: dict[str, torch.Tensor],
    split_rules: list[SplitRule],
    rename_rules: list[RenameRule],
) -> dict[str, torch.Tensor]:
    """Apply quant-aware reverse conversion: splits first, then renames.

    Splits run on the in-memory (post-conversion) names; renames then map the
    resulting keys back to the original hub names. Renames are applied in order.
    """
    out = dict(state_dict)
    for rule in split_rules:
        _apply_split_rule(out, rule)

    compiled = [(re.compile(r.pattern), r.repl) for r in rename_rules]
    renamed: dict[str, torch.Tensor] = {}
    for key, value in out.items():
        new_key = key
        for pattern, repl in compiled:
            new_key = pattern.sub(repl, new_key)
        if new_key in renamed:
            raise QuantConversionUnsupportedError(f"rename collision on '{new_key}'")
        renamed[new_key] = value
    return renamed


def revert_weight_conversion_quant_aware(model, state_dict: dict[str, torch.Tensor]):
    """Reverse a transformers conversion_mapping on a quantized state dict.

    Builds reverse rules from the model's conversion mapping and applies them
    carrying companion scale tensors. Raises :class:`QuantConversionUnsupportedError`
    when the mapping uses an op that cannot be reversed quant-aware yet, so the
    caller can fall back to the legacy behavior.
    """
    split_rules, rename_rules = _build_reverse_rules(model)
    if not split_rules and not rename_rules:
        return state_dict
    return apply_reverse_rules(state_dict, split_rules, rename_rules)


def _build_reverse_rules(model) -> tuple[list[SplitRule], list[RenameRule]]:
    """Derive reverse rules from the model's transformers conversion mapping.

    Returns empty rule lists when no mapping applies (export unchanged). Uses
    transformers' own ``reverse_transform()`` to get correctly-reversed name patterns
    (so anchored regex renamings reverse properly), then translates them:

    * ``WeightRenaming`` -> :class:`RenameRule` (carries scale siblings for free).
    * Expert ``WeightConverter`` (reverse contains ``SplitModulelist``): ModelOpt's
      export already expands fused experts into per-expert 2-D linears, so only the
      per-expert leaf names need mapping back (e.g. ``gate_proj`` -> ``w1``). Emitted
      as rename rules -- no tensor manipulation.
    * Dense fusing ``WeightConverter`` (reverse is ``Chunk`` only): the fused tensor
      survives in the state dict, so it is un-fused via a :class:`SplitRule`.

    Raises :class:`QuantConversionUnsupportedError` for any op shape not covered, so
    the caller falls back to the legacy (in-memory-name) behavior.
    """
    try:
        conversions = getattr(model, "_weight_conversions", None)
        if conversions is None:
            from transformers.conversion_mapping import get_model_conversion_mapping

            conversions = get_model_conversion_mapping(model, add_legacy=False)
    except Exception as exc:  # transformers without conversion_mapping, or API drift
        raise QuantConversionUnsupportedError(f"could not read conversion mapping: {exc}") from exc

    if not conversions:
        return [], []

    from transformers.core_model_loading import (
        Chunk,
        SplitModulelist,
        WeightConverter,
        WeightRenaming,
    )

    split_rules: list[SplitRule] = []
    rename_rules: list[RenameRule] = []
    for conv in conversions:
        rev = conv.reverse_transform()  # hub<-in-memory; reversed name patterns + ops
        if isinstance(rev, WeightRenaming):
            for pattern, repl in zip(_as_list(rev.source_patterns), _as_list(rev.target_patterns)):
                rename_rules.append(RenameRule(pattern=pattern, repl=repl))
        elif isinstance(rev, WeightConverter):
            ops = list(rev.operations)
            if any(isinstance(op, SplitModulelist) for op in ops):
                # Expert converter: ModelOpt already un-stacked/un-fused experts to
                # per-expert 2-D linears, so only per-expert leaf names remain to map.
                rename_rules.extend(_expert_leaf_renames(rev))
            elif ops and all(isinstance(op, Chunk) for op in ops):
                # Dense fused linear survives in the state dict -> un-fuse (split).
                split_rules.append(_dense_split_rule(rev, ops))
            else:
                raise QuantConversionUnsupportedError(
                    f"unsupported reverse ops: {[type(o).__name__ for o in ops]}"
                )
        else:
            raise QuantConversionUnsupportedError(f"unsupported conversion: {type(rev).__name__}")
    return split_rules, rename_rules


# ModelOpt's export splits a fused ``gate_up_proj`` into these per-expert linears,
# in this order (see modelopt.torch.export.layer_utils.get_expert_linear_names).
_FUSED_EXPERT_PART_NAMES = {"gate_up_proj": ["gate_proj", "up_proj"]}


def _expert_leaf_renames(rev) -> list[RenameRule]:
    """Per-expert leaf renames for an expert converter (ModelOpt pre-expands experts).

    ``rev`` reverses hub<-in-memory, so ``rev.source_patterns`` is the fused in-memory
    leaf (e.g. ``.experts.gate_up_proj``) and ``rev.target_patterns`` the hub leaves
    (e.g. ``.experts.*.w1.weight``, ``.experts.*.w3.weight``). ModelOpt exports the
    fused leaf as per-expert parts, mapped back to the hub leaves positionally.
    """
    src_leaf = _leaf(_as_list(rev.source_patterns)[0])
    hub_leaves = [_leaf(t) for t in _as_list(rev.target_patterns)]
    part_leaves = _FUSED_EXPERT_PART_NAMES.get(src_leaf, [src_leaf])
    if len(part_leaves) != len(hub_leaves):
        raise QuantConversionUnsupportedError(
            f"expert converter arity mismatch: {part_leaves} vs {hub_leaves}"
        )
    return [
        RenameRule(rf"(\.experts\.\d+\.){re.escape(part)}\b", rf"\g<1>{hub}")
        for part, hub in zip(part_leaves, hub_leaves)
    ]


def _dense_split_rule(rev, ops) -> SplitRule:
    """Un-fuse a dense (non-expert) fused linear that survives in the state dict."""
    fused = _leaf_suffix(_as_list(rev.source_patterns)[0])
    parts = tuple(_leaf_suffix(t) for t in _as_list(rev.target_patterns))
    dim = next((op.dim for op in ops if hasattr(op, "dim")), 0)
    return SplitRule(fused_suffix=fused, part_suffixes=parts, dim=dim)


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _leaf(pattern: str) -> str:
    """Bare leaf name from a conversion pattern, e.g. ``.experts.*.w1.weight`` -> ``w1``."""
    p = pattern
    for suffix in _LEAF_SUFFIXES:
        if p.endswith(suffix):
            p = p[: -len(suffix)]
            break
    return p.rstrip(".*").rsplit(".", 1)[-1]


def _leaf_suffix(pattern: str) -> str:
    """Leaf name as a module suffix, e.g. ``.gate_proj``."""
    return "." + _leaf(pattern)
