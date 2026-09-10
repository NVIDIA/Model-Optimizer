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

"""Shared helpers for the published-checkpoint recipe backfill.

Both the online scanner (``scan_collection.py``) and the offline verifier
(``verify_recipes.py``) build the same *module map* for a checkpoint: every
quantizable module of the published checkpoint, keyed by a **canonical pattern**
with the integer indices factored out, mapped to the quantization format it
carries.

Canonical pattern
-----------------
``model.layers.7.mlp.experts.31.up_proj`` becomes
``model.layers.{}.mlp.experts.{}.up_proj`` with index tuple ``(7, 31)``.  Index
tuples are stored as one compact range string per dimension when the observed
tuples form a full cross product (the normal case), and as an explicit list
otherwise, so the snapshot stays small without losing per-layer precision --
which matters for checkpoints such as ``nvidia/Mistral-Medium-3.5-128B-NVFP4``
whose edge decoder layers use a different format than the interior ones.
"""

from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from itertools import product

__all__ = [
    "BF16",
    "canonical_pattern",
    "compress_indices",
    "expand_indices",
    "is_quantizable_module",
    "module_map_from_quant_config",
    "parse_range",
    "render_range",
]

#: Format string used for a module that exists but carries no quantizer output.
BF16 = "BF16"

_INDEX_RE = re.compile(r"(?<=\.)\d+(?=\.|$)")

# Module leaf/segment names that never carry a ModelOpt weight quantizer even
# though they own a ``.weight`` tensor: normalisation layers, embeddings and the
# rotary caches. Everything else that owns a ``.weight`` is treated as a
# candidate ``nn.Linear`` / ``nn.Conv1d``.
_NON_QUANTIZABLE_RE = re.compile(
    r"(^|\.)("
    r"[a-z0-9_]*norm[a-z0-9_]*"  # input_layernorm, q_norm, model.norm, post_attention_layernorm
    r"|ln_[a-z0-9_]+|[a-z0-9_]+_ln"
    r"|[a-z0-9_]*embed[a-z0-9_]*|[a-z0-9_]*embedding[a-z0-9_]*"
    r"|wte|wpe|rotary_emb|alibi"
    r")$"
)


#: Quantizable modules whose scales the HF exporter does not write out, so their
#: format can only be read from ``hf_quant_config.json``.
_SCALELESS_EXPORT_RE = re.compile(r"(^|\.)conv1d$")


def is_quantizable_module(name: str) -> bool:
    """Whether a module that owns a ``.weight`` tensor can carry a quantizer.

    Name-based because a safetensors *index* carries no tensor shapes. The rule
    only has to be right about the modules a recipe's wildcards can reach; the
    verifier cross-checks it against the checkpoint's own ``exclude_modules`` and
    ``quantized_layers``, so a module wrongly ruled out here shows up as a
    mismatch rather than silently passing.
    """
    return not _NON_QUANTIZABLE_RE.search(name)


def canonical_pattern(name: str) -> tuple[str, tuple[int, ...]]:
    """Split ``a.3.b.7.c`` into ``("a.{}.b.{}.c", (3, 7))``."""
    indices = tuple(int(m.group()) for m in _INDEX_RE.finditer(name))
    return _INDEX_RE.sub("{}", name), indices


def render_range(values: list[int]) -> str:
    """Compress a sorted int list into ``"0-3,7,10-12"``."""
    if not values:
        return ""
    out: list[str] = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = v
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(out)


def parse_range(spec: str) -> list[int]:
    """Inverse of :func:`render_range`."""
    if not spec:
        return []
    values: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    return values


def compress_indices(tuples: set[tuple[int, ...]]) -> list[str] | list[list[int]]:
    """Compress a set of index tuples.

    Returns one range string per dimension when the tuples form the full cross
    product of their per-dimension values, otherwise the explicit sorted list of
    tuples (as lists, so it round-trips through JSON).
    """
    if not tuples:
        return []
    ndim = len(next(iter(tuples)))
    if ndim == 0:
        return []
    dims = [sorted({t[d] for t in tuples}) for d in range(ndim)]
    size = 1
    for dim in dims:
        size *= len(dim)
    if size == len(tuples):
        return [render_range(dim) for dim in dims]
    return [list(t) for t in sorted(tuples)]


def expand_indices(spec: list[str] | list[list[int]]) -> set[tuple[int, ...]]:
    """Inverse of :func:`compress_indices`."""
    if not spec:
        return {()}
    if isinstance(spec[0], list):
        return {tuple(t) for t in spec}
    return set(product(*(parse_range(s) for s in spec)))


def _ancestors(name: str) -> list[str]:
    """``a.b.c`` -> ``["a.b.c", "a.b", "a"]`` (the module and every parent)."""
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def _match_selectors(modules: list[str], selectors: list[str]) -> set[str]:
    """Resolve ``exclude_modules`` / ``quantized_layers`` keys against real modules.

    A selector matches a module when it is the module itself, a ``*``-suffixed
    prefix of it (the form ModelOpt's exporter emits, e.g.
    ``model.layers.30.self_attn*``), or a parent module of it (``mlp.experts``
    standing for every linear underneath).

    Matching is driven from each module's own ancestors rather than by scanning
    the selector list, so a checkpoint with a million modules and thousands of
    selectors (``nvidia/Kimi-K3-NVFP4``) stays linear in the module count.
    """
    exact = {s for s in selectors if "*" not in s}
    # ``foo*`` is segment-aligned in every export seen so far; anything else falls
    # back to a single compiled alternation.
    aligned = {s[:-1] for s in selectors if s.endswith("*") and "*" not in s[:-1]}
    other = [s for s in selectors if "*" in s and s not in exact and s[:-1] not in aligned]
    other_re = re.compile("|".join(fnmatch.translate(s) for s in other)) if other else None

    hit = set()
    for m in modules:
        if any(a in exact or a in aligned for a in _ancestors(m)) or (
            other_re is not None and other_re.match(m)
        ):
            hit.add(m)
    return hit


def module_map_from_quant_config(
    modules: list[str], quantization: dict, scale_formats: dict[str, str] | None = None
) -> dict[str, str]:
    """Map every quantizable module to the format the published checkpoint gives it.

    *modules* is the full list of ``.weight``-owning module names from the
    checkpoint's safetensors index. *quantization* is the ``quantization`` block
    of ``hf_quant_config.json``. *scale_formats* is the per-module format implied
    by the exported scale tensors, used to confirm the config-derived answer when
    the checkpoint ships them.
    """
    quantizable = [m for m in modules if is_quantizable_module(m)]
    result = dict.fromkeys(quantizable, BF16)

    per_layer = quantization.get("quantized_layers") or {}
    if per_layer:
        algo_of = {
            selector: (cfg["quant_algo"] if isinstance(cfg, dict) else str(cfg))
            for selector, cfg in per_layer.items()
        }
        # One pass over the modules rather than one pass per selector: a checkpoint
        # can list thousands of ``quantized_layers`` keys against a million modules.
        exact = {s: a for s, a in algo_of.items() if "*" not in s}
        aligned = {s[:-1]: a for s, a in algo_of.items() if s.endswith("*") and "*" not in s[:-1]}
        globbed = [(s, a) for s, a in algo_of.items() if s not in exact and s[:-1] not in aligned]
        for m in quantizable:
            for ancestor in _ancestors(m):
                algo = exact.get(ancestor) or aligned.get(ancestor)
                if algo:
                    result[m] = algo
                    break
            else:
                for selector, algo in globbed:
                    if fnmatch.fnmatch(m, selector):
                        result[m] = algo
                        break
    else:
        algo = quantization.get("quant_algo")
        if algo:
            excluded = _match_selectors(quantizable, quantization.get("exclude_modules") or [])
            for m in quantizable:
                if m not in excluded:
                    result[m] = algo

    if scale_formats:
        # Reconcile the config with the exported scale tensors.
        #
        # ``exclude_modules`` is not always exhaustive -- an older export lists only
        # what the producing script chose to name, so a ``mlp.gate`` router that was
        # never quantized can be missing from it and look quantized here. For a
        # quantized ``nn.Linear`` the exporter always writes ``weight_scale``, so the
        # absence of scales is proof the module stayed in the source dtype.
        #
        # The exception is modules the exporter does not scale even when their
        # quantizer is on -- ``nn.Conv1d`` in a linear-attention block, which
        # ``hf_quant_config.json`` does list (e.g. ``linear_attn.conv1d`` on
        # ``nvidia/Qwen3.8-2.4T-A95B-NVFP4``). For those the config is kept.
        for m in quantizable:
            observed = scale_formats.get(m, BF16)
            if observed != BF16:
                # Quantized. The config names the format precisely (it distinguishes
                # per-tensor FP8 from block-scaled ``FP8_PB_WO``, and W4A16 NVFP4 from
                # W4A4); the scale tensors cannot. Fall back to the scale-derived name
                # only when the config does not mention the module at all.
                result[m] = result[m] if result[m] != BF16 else observed
            elif result[m] != BF16 and not _SCALELESS_EXPORT_RE.search(m):
                result[m] = BF16
    return result


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")


def drop_unbuilt_layers(
    module_map: dict[str, str], num_hidden_layers: int | None, num_nextn_predict_layers: int
) -> tuple[dict[str, str], list[str]]:
    """Remove decoder layers the HF model class never instantiates.

    A checkpoint that declares ``num_hidden_layers: 61`` with
    ``num_nextn_predict_layers: 1`` ships weights for ``model.layers.61`` -- the
    MTP / next-token-prediction block -- but the model class builds layers 0-60
    only. Those tensors are therefore in the file yet outside the quantized model,
    and a recipe is not wrong for leaving them alone. Dropping them keeps the
    comparison to what PTQ could actually reach.
    """
    if not num_hidden_layers or num_nextn_predict_layers <= 0:
        return module_map, []
    unbuilt = set(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers))
    kept, dropped = {}, []
    for name, fmt in module_map.items():
        match = _LAYER_INDEX_RE.search(name)
        if match and int(match.group(1)) in unbuilt and fmt == BF16:
            dropped.append(name)
        else:
            kept[name] = fmt
    return kept, dropped


def pack_module_map(module_map: dict[str, str]) -> dict[str, dict[str, list]]:
    """Group a module -> format map into ``{pattern: {format: <index spec>}}``."""
    grouped: dict[str, dict[str, set[tuple[int, ...]]]] = defaultdict(lambda: defaultdict(set))
    for name, fmt in module_map.items():
        pattern, indices = canonical_pattern(name)
        grouped[pattern][fmt].add(indices)
    return {
        pattern: {fmt: compress_indices(tuples) for fmt, tuples in sorted(fmts.items())}
        for pattern, fmts in sorted(grouped.items())
    }


def unpack_module_map(packed: dict[str, dict[str, list]]) -> dict[str, str]:
    """Inverse of :func:`pack_module_map`."""
    out: dict[str, str] = {}
    for pattern, fmts in packed.items():
        for fmt, spec in fmts.items():
            for indices in expand_indices(spec):
                name = pattern
                for i in indices:
                    name = name.replace("{}", str(i), 1)
                out[name] = fmt
    return out
