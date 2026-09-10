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

"""Check that a recipe reproduces a published checkpoint's quantization layout.

Replays a recipe's ``quantize.quant_cfg`` -- ``fnmatch`` wildcards applied in
order, later entries winning, exactly as
:func:`modelopt.torch.quantization.conversion.set_quantizer_by_cfg` does -- over
the module names of the published checkpoint recorded in
``published_checkpoints.json``, then diffs the resulting per-module format
against the format the checkpoint actually ships.

This runs fully offline against the checked-in snapshot, so it needs neither the
Hub nor the (often several-hundred-GB) weights.

Usage::

    python tools/recipe_backfill/verify_recipes.py                    # every mapped recipe
    python tools/recipe_backfill/verify_recipes.py --checkpoint Qwen3-8B-NVFP4
    python tools/recipe_backfill/verify_recipes.py --show-diff        # per-module mismatches
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from checkpoint_scan import BF16, expand_indices, parse_range

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).parent / "published_checkpoints.json"
RECIPE_MAP = Path(__file__).parent / "recipe_map.json"

#: Quantizer attributes a quantized ``nn.Linear`` / ``nn.Conv1d`` carries.
LINEAR_QUANTIZERS = ("weight_quantizer", "input_quantizer", "output_quantizer")


#: ``QuantizerAttributeConfig`` normalises ``e4m3`` to ``(4, 3)`` on dump; the YAML
#: uses the mnemonic. Map both onto the mnemonic so a format key is stable.
_FLOAT_BITS = {(4, 3): "e4m3", (2, 1): "e2m1", (3, 2): "e3m2", (5, 2): "e5m2", (8, 0): "e8m0"}


def _bits_name(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _FLOAT_BITS.get(tuple(value), str(tuple(value)))
    return str(value)


def _numerics_key(cfg: dict | None) -> tuple:
    """Reduce a resolved quantizer cfg to the fields that name a format."""
    if cfg is None:
        return ()
    block = cfg.get("block_sizes") or {}
    return (
        _bits_name(cfg.get("num_bits")),
        block.get(-1) or block.get("-1"),
        block.get(-2) or block.get("-2"),
        block.get("type"),
        _bits_name(block.get("scale_bits")),
        bool(block.get("four_over_six")),
    )


#: (num_bits, block, block2, type, scale_bits, 4o6) -> short format name.
_WEIGHT_FORMATS = {
    ("e2m1", 16, None, "dynamic", "e4m3", False): "NVFP4",
    ("e2m1", 16, None, "static", "e4m3", False): "NVFP4",
    ("e2m1", 16, None, "static", "e4m3", True): "NVFP4",
    ("e2m1", 32, None, "dynamic", "e4m3", False): "NVFP4_BS32",
    ("e2m1", 32, None, "dynamic", "e8m0", False): "MXFP4",
    ("e4m3", 32, None, "dynamic", "e8m0", False): "MXFP8",
    ("e4m3", None, None, None, "None", False): "FP8",
    ("e4m3", 128, 128, None, "None", False): "FP8_BLOCK_SCALES",
    ("e4m3", 128, None, None, "None", False): "FP8_BLOCK_SCALES",
    ("8", None, None, None, "None", False): "INT8",
    ("4", 128, None, "static", "None", False): "INT4",
}


def _format_name(weight_cfg: dict | None, input_cfg: dict | None) -> str:
    """Name the (weight, activation) pair the way ``hf_quant_config.json`` does."""
    if weight_cfg is None:
        return BF16
    w = _WEIGHT_FORMATS.get(_numerics_key(weight_cfg))
    if w is None:
        return f"UNKNOWN{_numerics_key(weight_cfg)}"
    if input_cfg is None:
        return {
            "NVFP4": "W4A16_NVFP4",
            "FP8": "FP8_WO",
            "MXFP4": "MXFP4_WO",
            # ``hf_quant_config.json`` calls block-scaled FP8 weight-only FP8_PB_WO.
            "FP8_BLOCK_SCALES": "FP8_PB_WO",
        }.get(w, f"{w}_WO")
    return w


def replay(quant_cfg: list[dict], quantizer_names: list[str]) -> dict[str, dict | None]:
    """Replay a ``quant_cfg`` list over quantizer names; return name -> cfg (None = off).

    Mirrors :func:`set_quantizer_by_cfg`: ``fnmatch`` wildcards, entries applied in
    order with later ones overriding, ``cfg`` replacing attributes wholesale while a
    bare ``enable`` only toggles.
    """
    state: dict[str, dict | None] = dict.fromkeys(quantizer_names)
    attrs: dict[str, dict] = {}
    for entry in quant_cfg:
        pattern = entry["quantizer_name"]
        cfg = entry.get("cfg")
        enable = entry.get("enable", cfg is not None)
        if entry.get("parent_class"):
            # ``parent_class`` restricts to a torch class we cannot see from names
            # alone. Every shipped use disables a non-Linear class (BatchNorm,
            # LeakyReLU, Embedding), which owns none of the names replayed here.
            continue
        for name in fnmatch.filter(quantizer_names, pattern):
            if cfg is None:
                state[name] = attrs.get(name, {}) if enable else None
            else:
                resolved = cfg if isinstance(cfg, dict) else {"sequential": cfg}
                attrs[name] = resolved
                state[name] = resolved if enable else None
    return state


def quantizer_name(module: str, kind: str, naming: str = "child") -> str:
    """The name of a module's ``kind`` quantizer under the given naming convention.

    ``child`` (the default) is what a quantized ``nn.Linear`` produces:
    ``...up_proj.weight_quantizer``. ``fused_parameter`` is what a module that keeps
    several weights as plain ``nn.Parameter`` objects produces -- DeepSeek-V4's
    ``Expert.w1/w2/w3`` register ``w1_weight_quantizer`` and friends on the *expert*
    (see ``examples/deepseek/deepseek_v4/ptq.py``), so the recipe's wildcards are
    written against that shape instead.
    """
    if naming == "child":
        return f"{module}.{kind}"
    parent, _, leaf = module.rpartition(".")
    return f"{parent}.{leaf}_{kind}" if parent else f"{leaf}_{kind}"


#: A module index inside a wildcard: a whole dotted segment of digits. Written this way
#: so ``conv1d`` and friends -- a digit that is part of a name, not an index -- do not
#: count.
_INDEX_IN_PATTERN_RE = re.compile(r"(?<=\.)\d+(?=[.*]|$)")
_INDEX_SEGMENT_RE = re.compile(r"(?<=\.)\d+(?=\.|$)")


def index_depth(quant_cfg: list[dict]) -> int:
    """How many leading index positions a recipe's wildcards can tell apart.

    ``0`` -- no wildcard pins an index, so every index is interchangeable and one
    representative per index-collapsed name answers for the whole family.
    ``1`` -- some wildcard pins a decoder layer (``*layers.87.mlp*``); layers must be
    kept apart but the expert index underneath still is not.
    ``-1`` -- an index is pinned somewhere else, so nothing may be collapsed.
    """
    depth = 0
    for entry in quant_cfg:
        pattern = entry["quantizer_name"]
        for match in _INDEX_IN_PATTERN_RE.finditer(pattern):
            if not pattern[: match.start()].endswith("layers."):
                return -1
            depth = 1
    return depth


def collapse_indices(name: str, depth: int) -> str:
    """Replace every index past *depth* with ``N``.

    ``layers.7.experts.31.up_proj`` at depth 1 becomes ``layers.7.experts.N.up_proj``:
    the layer stays distinguishable, the expert does not. This is what keeps a
    million-module checkpoint down to a few dozen ``fnmatch`` calls.
    """
    if depth < 0:
        return name
    seen = 0

    def repl(match: re.Match) -> str:
        nonlocal seen
        seen += 1
        return match.group() if seen <= depth else "N"

    return _INDEX_SEGMENT_RE.sub(repl, name)


def recipe_module_formats(
    quant_cfg: list[dict], modules: list[str], naming: str = "child"
) -> dict[str, str]:
    """Per-module format the recipe produces for *modules*."""
    depth = index_depth(quant_cfg)

    def key(name: str) -> str:
        return collapse_indices(name, depth)

    names = sorted({key(quantizer_name(m, q, naming)) for m in modules for q in LINEAR_QUANTIZERS})
    state = replay(quant_cfg, names)
    format_cache: dict[tuple[int, int], str] = {}

    def named(weight_cfg: dict | None, input_cfg: dict | None) -> str:
        # The replay reuses the same cfg objects across thousands of quantizers, so
        # identity is enough of a key to skip re-deriving the format name.
        key_ = (id(weight_cfg), id(input_cfg))
        if key_ not in format_cache:
            format_cache[key_] = _format_name(weight_cfg, input_cfg)
        return format_cache[key_]

    return {
        m: named(
            state[key(quantizer_name(m, "weight_quantizer", naming))],
            state[key(quantizer_name(m, "input_quantizer", naming))],
        )
        for m in modules
    }


def recipe_kv_mode(quant_cfg: list[dict], attention_modules: list[str]) -> tuple[str | None, bool]:
    """The KV-cache format the recipe produces, and whether it pins a constant amax.

    ``None`` means the recipe leaves the KV cache alone.
    """
    if not attention_modules:
        return None, False
    names = [f"{m}.{q}" for m in attention_modules for q in ("k_bmm_quantizer", "v_bmm_quantizer")]
    state = replay(quant_cfg, names)
    live = [cfg for cfg in state.values() if cfg is not None]
    if not live:
        return None, False
    formats = {_format_name(cfg, cfg) for cfg in live}
    constant = all(c.get("use_constant_amax") or c.get("constant_amax") for c in live)
    return (formats.pop() if len(formats) == 1 else "MIXED"), constant


@cache
def _load_recipe_quant_cfg_cached(recipe_rel: str) -> tuple | None:
    cfg = load_recipe_quant_cfg(recipe_rel)
    return None if cfg is None else tuple(cfg)


def load_recipe_quant_cfg(recipe_rel: str) -> list[dict] | None:
    """Load a recipe by its ``modelopt_recipes``-relative path and dump its quant_cfg.

    Returns ``None`` for a recipe whose per-module assignment is decided by a search
    rather than by the YAML (``auto_quantize``): its ``quantize`` block is only the
    starting point, so replaying it would not describe the published checkpoint.
    """
    from modelopt.recipe import load_recipe

    recipe = load_recipe(recipe_rel)
    if getattr(recipe, "auto_quantize", None) is not None:
        return None
    return list(recipe.quantize.model_dump()["quant_cfg"])


def _attention_modules(modules: list[str]) -> list[str]:
    """Attention parents that can own KV-cache BMM quantizers."""
    out = set()
    for m in modules:
        parent, _, leaf = m.rpartition(".")
        if leaf.startswith(("k_proj", "v_proj", "kv_a_proj", "kv_b_proj", "qkv_proj", "wkv")):
            out.add(parent)
    return sorted(out)


def published_modules(packed: dict[str, dict[str, list]], depth: int) -> list[tuple[str, str, int]]:
    """Published ``(module, format, module_count)`` triples to check.

    Modules that a recipe's wildcards cannot tell apart -- see :func:`index_depth` --
    collapse to a single representative, with ``module_count`` recording how many real
    modules it stands for so the diff output still reports true counts.

    The collapse is computed from the packed per-dimension ranges rather than by
    expanding them, so a checkpoint with a million expert linears costs the same as a
    dense one.
    """
    out: list[tuple[str, str, int]] = []
    for pattern, formats in packed.items():
        for fmt, spec in formats.items():
            if spec and isinstance(spec[0], list):
                # Explicit tuple list (the format is not a clean cross product).
                groups: dict[tuple[int, ...], int] = {}
                for indices in sorted(expand_indices(spec)):
                    gk = indices if depth < 0 else indices[:depth]
                    groups[gk] = groups.get(gk, 0) + 1
                representatives = [(list(gk) + [0] * 0, count) for gk, count in groups.items()]
                # Rebuild a full index tuple for the representative name.
                rep_of = {}
                for indices in sorted(expand_indices(spec)):
                    rep_of.setdefault(indices if depth < 0 else indices[:depth], indices)
                representatives = [(list(rep_of[gk]), count) for gk, count in groups.items()]
            else:
                dims = [parse_range(r) for r in spec]
                if depth < 0:
                    representatives = [(list(t), 1) for t in sorted(expand_indices(spec))]
                else:
                    lead, rest = dims[:depth], dims[depth:]
                    tail = [d[0] for d in rest]
                    tail_count = 1
                    for d in rest:
                        tail_count *= len(d)
                    representatives = [
                        (list(head) + tail, tail_count) for head in _cross_product(lead)
                    ]
            for indices, count in representatives:
                name = pattern
                for i in indices:
                    name = name.replace("{}", str(i), 1)
                out.append((name, fmt, count))
    return out


def _cross_product(dims: list[list[int]]) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = [()]
    for dim in dims:
        result = [(*prev, value) for prev in result for value in dim]
    return result


def verify(entry: dict, mapping: dict, show_diff: bool = False) -> tuple[bool, str]:
    """Verify one recipe against one published checkpoint entry."""
    recipe_rel = mapping["recipe"]
    cached = _load_recipe_quant_cfg_cached(recipe_rel)
    quant_cfg = None if cached is None else list(cached)
    if quant_cfg is None:
        return (
            True,
            "    auto_quantize recipe: per-module formats come from the search, not the YAML",
        )

    packed = entry["modules"]
    # ``ignore_modules`` excuses modules whose published format is the *source*
    # checkpoint's rather than PTQ output -- e.g. a speculative-decoding block the
    # recipe deliberately leaves alone, which therefore still carries the source
    # model's native block-FP8 weights. Each use must carry a ``note``.
    ignore = mapping.get("ignore_modules") or []
    if ignore:
        packed = {
            pattern: formats
            for pattern, formats in packed.items()
            if not any(fnmatch.fnmatch(pattern, g) for g in ignore)
        }

    triples = published_modules(packed, index_depth(quant_cfg))
    naming = mapping.get("quantizer_naming", "child")
    produced = recipe_module_formats(quant_cfg, [t[0] for t in triples], naming)

    diffs = [(m, want, produced[m], n) for m, want, n in triples if want != produced[m]]

    kv_published = entry.get("kv_cache_quant_algo")
    kv_published = None if (kv_published or "").lower() == "none" else kv_published
    attention = _attention_modules([t[0] for t in triples])
    kv_recipe, kv_constant = recipe_kv_mode(quant_cfg, attention)
    kv_ok = (kv_published or None) == (kv_recipe or None)

    # ``hf_quant_config.json`` records only "FP8", not whether the scale was
    # calibrated or pinned. ``use_constant_amax`` pins amax to the E4M3 max, so a cast
    # KV cache exports ``k_scale == 1.0`` exactly; a calibrated one essentially never
    # does. That is what separates ``kv_fp8_cast`` from ``kv_fp8``.
    k_scale = (entry.get("scale_samples") or {}).get("k_scale")
    if k_scale is None and kv_published and not entry.get("kv_scale_modules"):
        # KV declared quantized but no scale tensor shipped: a runtime that finds no
        # ``k_scale`` uses 1.0, which is what a constant-amax (cast) KV cache produces.
        k_scale = 1.0
    kv_mode_ok = True
    if kv_ok and kv_recipe is not None and k_scale is not None:
        kv_mode_ok = kv_constant == (k_scale == 1.0)

    lines = []
    if diffs:
        by_kind: dict[tuple[str, str], tuple[int, list[str]]] = {}
        for m, want, got, n in diffs:
            count, examples = by_kind.get((want, got), (0, []))
            by_kind[(want, got)] = (count + n, [*examples, m])
        for (want, got), (count, mods) in sorted(by_kind.items(), key=lambda kv: -kv[1][0]):
            lines.append(f"    {count:>7} modules: published={want} recipe={got}")
            if show_diff:
                lines += [f"              {m}" for m in mods[:8]]
                if len(mods) > 8:
                    lines.append(f"              ... and {len(mods) - 8} more patterns")
    if not kv_ok:
        lines.append(f"    KV cache: published={kv_published} recipe={kv_recipe}")
    if not kv_mode_ok:
        want = "cast (constant amax)" if k_scale == 1.0 else "calibrated"
        got = "cast (constant amax)" if kv_constant else "calibrated"
        lines.append(f"    KV scale: published k_scale={k_scale} implies {want}, recipe is {got}")
    return (not diffs and kv_ok and kv_mode_ok), "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="substring filter on the checkpoint id")
    ap.add_argument("--show-diff", action="store_true")
    args = ap.parse_args()

    snapshot = json.loads(SNAPSHOT.read_text())
    entries = {e["id"]: e for e in snapshot["checkpoints"]}
    mapping = json.loads(RECIPE_MAP.read_text())["recipes"]

    failures = 0
    for checkpoint_id, spec in sorted(mapping.items()):
        if args.checkpoint and args.checkpoint not in checkpoint_id:
            continue
        spec = {"recipe": spec} if isinstance(spec, str) else spec
        entry = entries.get(checkpoint_id)
        if entry is None or "modules" not in entry:
            print(f"SKIP {checkpoint_id}: not in snapshot")
            continue
        ok, detail = verify(entry, spec, args.show_diff)
        approximate = spec.get("approximate")
        status = "PASS" if ok else ("WARN" if approximate else "FAIL")
        print(f"{status} {checkpoint_id}\n       {spec['recipe']}")
        if not ok:
            if approximate:
                print(f"       approximate: {approximate}")
            else:
                failures += 1
            print(detail)
    print(f"\n{failures} failing" if failures else "\nall recipes reproduce their checkpoints")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
