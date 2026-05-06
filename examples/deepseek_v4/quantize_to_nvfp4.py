# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply calibrated amax to produce an NVFP4 DS-V4 checkpoint in the
**original HF 64-shard release layout**.

Operates directly on the original HF-style checkpoint
(``models/DeepSeek-V4-Pro/``) — 64 ``model-{k:05d}-of-{N:05d}.safetensors``
plus ``model.safetensors.index.json`` — and produces a new directory with
the same layout where every routed-expert weight has been quantized to
NVFP4. DS-native key naming is preserved (the original release already uses
it; our amax dump already matches).

Inputs:
  * ``--amax_path``: the per-rank amax dump from ``ptq.py``
    (``amax_dict_rank{i}-mp{mp}.pt``). The MP count is auto-detected from
    the filenames. Routed experts are rank-sharded in the calibration run,
    so amax keys do not collide across ranks; we union-merge.
  * ``--source_ckpt``: the original HF 64-shard release — the same directory
    ``inference/convert.py`` reads forward. We do **not** need the MP-
    sharded derivative here; we go straight from original → NVFP4.

Outputs:
  * ``--output_ckpt``: a directory with 64 shards identical to the source
    EXCEPT: every routed-expert weight is NVFP4 packed, the paired
    ``.scale`` sibling is dropped, and three new sibling keys are added per
    expert weight:

        <path>.weight          NVFP4-packed uint8, shape (out, in//2)
        <path>.weight_scale    per-block scale (E4M3), shape (out, in//16)
        <path>.weight_scale_2  per-tensor scale (FP32 scalar)
        <path>.input_scale     per-tensor activation scale (FP32 scalar)

  * An updated ``model.safetensors.index.json`` reflecting dropped/added keys.
  * An ``hf_quant_config.json`` manifest listing the NVFP4-quantized layers.
  * Ancillary files (``config.json``, ``tokenizer.json``, ``LICENSE``,
    ``encoding/``, ``inference/``, ...) are **hard-linked** from the source
    — no duplication.

Uncalibrated-expert handling: some routed experts receive zero tokens during
calibration (common in V4's first ``n_hash_layers`` with deterministic
``tid2eid`` routing, occasionally in score-routed layers, entire MTP block
when MTP is inactive). For those:
  * ``weight_amax`` is synthesized from the dequantized BF16 weight
    (``bf16.abs().max()``).
  * ``input_amax`` falls back to the max observed on any other expert in the
    same ``(block, projection)`` bucket.
  * If an entire bucket has no observation (e.g. inactive MTP), the fallback
    is the constant ``1.0`` and the case is logged.

Usage (single compute node, CPU-default; dequant+requant math is cheap
relative to shard I/O):

    srun --container-image=dsv4-ready.sqsh ... \\
        python quantize_to_nvfp4.py \\
            --amax_path   /home/mxin/.../amax-nvfp4-experts \\
            --source_ckpt /home/mxin/mxin/dsv4/models/DeepSeek-V4-Pro \\
            --output_ckpt /home/mxin/.../DeepSeek-V4-Pro-nvfp4-experts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor
except ImportError:
    _MO = Path("/home/mxin/mxin/dsv4/sources/Model-Optimizer")
    sys.path.insert(0, str(_MO))
    from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor


# Routed-expert weights in both regular MoE layers and the MTP block(s).
_EXPERT_WEIGHT_RE = re.compile(
    r"^(?:mtp\.\d+|layers\.\d+)\.ffn\.experts\.\d+\.w[123]\.weight$"
)

_AMAX_KEY_RE = re.compile(
    r"^(?P<block>(?:mtp\.\d+|layers\.\d+))\.ffn\.experts\.(?P<eid>\d+)\.(?P<proj>w[123])"
    r"_(?P<which>weight|input)_quantizer\._amax$"
)

_MP_FILE_RE = re.compile(r"^amax_dict_rank\d+-mp(?P<mp>\d+)\.pt$")
_HF_SHARD_RE = re.compile(r"^model-(?P<idx>\d+)-of-(?P<total>\d+)\.safetensors$")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _amax_to_nvfp4_scale_2(amax: torch.Tensor) -> torch.Tensor:
    """``amax / (fp4_max * fp8_max) = amax / (6 * 448)``; returns a 0-d fp32 scalar."""
    return (amax.float() / (6.0 * 448.0)).to(torch.float32).reshape(())


# ---------------------------------------------------------------------------
# amax loading
# ---------------------------------------------------------------------------


def _discover_mp_from_amax_dir(amax_path: Path) -> int:
    """Auto-detect MP from amax filenames. Require all files to agree so a
    stale cross-run directory (e.g. both mp4 and mp8 dumps present) fails
    loud instead of silently merging half of each run."""
    files = sorted(amax_path.glob("amax_dict_rank*-mp*.pt"))
    assert files, f"no amax dumps in {amax_path}"
    mps = set()
    for f in files:
        m = _MP_FILE_RE.match(f.name)
        assert m, f"unexpected amax filename: {f.name}"
        mps.add(int(m.group("mp")))
    assert len(mps) == 1, (
        f"amax dir {amax_path} contains multiple MP values {sorted(mps)}; "
        f"clean out stale dumps or pass --world_size explicitly."
    )
    return mps.pop()


def _load_merged_amax(
    amax_path: Path, world_size: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
    """Union-merge amax across ranks, fuse w1/w3 input amax per expert, and
    compute a per-``(block, proj)`` input-amax fallback."""
    mp = world_size if world_size is not None else _discover_mp_from_amax_dir(amax_path)
    _log(f"[load] using MP={mp} ({'explicit --world_size' if world_size else 'auto-detected'})")

    merged: dict[str, torch.Tensor] = {}
    for r in range(mp):
        fp = amax_path / f"amax_dict_rank{r}-mp{mp}.pt"
        assert fp.exists(), f"missing {fp}"
        rank_state = torch.load(str(fp), map_location="cpu", weights_only=True)
        for k, v in rank_state.items():
            assert _AMAX_KEY_RE.match(k), f"unexpected amax key: {k!r}"
            assert k not in merged, f"amax collision across ranks: {k!r}"
            merged[k] = v
    _log(f"[load] merged {len(merged)} amax entries")

    # w1/w3 share the same input x → fuse per expert.
    fused = 0
    for k in list(merged.keys()):
        if k.endswith("w1_input_quantizer._amax"):
            k3 = k.replace("w1_input", "w3_input")
            if k3 in merged:
                shared = torch.maximum(merged[k], merged[k3])
                merged[k] = shared
                merged[k3] = shared
                fused += 1
    _log(f"[load] fused w1/w3 input amax on {fused} experts")

    buckets: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for k, v in merged.items():
        m = _AMAX_KEY_RE.match(k)
        assert m is not None
        if m.group("which") == "input":
            buckets[(m.group("block"), m.group("proj"))].append(v)
    input_fallback = {
        key: torch.stack([t.reshape(-1) for t in vals]).flatten().max()
        for key, vals in buckets.items()
    }
    _log(f"[load] input-fallback buckets: {len(input_fallback)} populated")
    return merged, input_fallback


def _lookup_amax(
    amax: dict[str, torch.Tensor], expert_path: str, which: str
) -> torch.Tensor | None:
    return amax.get(f"{expert_path}_{which}_quantizer._amax")


# ---------------------------------------------------------------------------
# Per-shard rewrite
# ---------------------------------------------------------------------------


def _quantize_weight_nvfp4(
    mxfp4_weight: torch.Tensor,
    mxfp4_scale: torch.Tensor,
    weight_amax: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """MXFP4 + UE8M0 → BF16 → NVFP4 packed. Synthesizes ``weight_amax`` from
    the dequantized BF16 tensor when ``None`` is passed."""
    bf16 = MXFP4QTensor.dequantize_packed(
        mxfp4_weight.to(device), mxfp4_scale.to(device),
        block_size=32, dtype=torch.bfloat16,
    )
    synthesized = weight_amax is None
    if synthesized:
        weight_amax = bf16.abs().max()
    weight_scale_2 = _amax_to_nvfp4_scale_2(weight_amax.to(device))
    q_tensor, weight_scale, _ = NVFP4QTensor.quantize(
        bf16, 16, None, weight_scale_2, try_tensorrt=False
    )
    return q_tensor._quantized_data, weight_scale, weight_scale_2, synthesized


def convert_shard(
    src_shard: Path,
    dst_shard: Path,
    amax: dict[str, torch.Tensor],
    input_fallback: dict[tuple[str, str], torch.Tensor],
    device: str,
    stats: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Rewrite one HF-style shard. Returns ``(added_keys, removed_keys)`` so
    the caller can update ``model.safetensors.index.json``."""
    out: dict[str, torch.Tensor] = {}
    added: list[str] = []
    removed: list[str] = []

    with safe_open(str(src_shard), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        expert_weight_keys = [k for k in all_keys if _EXPERT_WEIGHT_RE.match(k)]
        scale_siblings = {
            k[: -len(".weight")] + ".scale"
            for k in expert_weight_keys
            if k[: -len(".weight")] + ".scale" in all_keys
        }

        for key in all_keys:
            if key in scale_siblings:
                removed.append(key)
                continue

            if _EXPERT_WEIGHT_RE.match(key):
                expert_path = key[: -len(".weight")]
                scale_key = expert_path + ".scale"
                assert scale_key in all_keys, f"no paired scale for {key}"

                m = re.match(
                    r"^(?P<block>(?:mtp\.\d+|layers\.\d+))\.ffn\.experts\.\d+\.(?P<proj>w[123])$",
                    expert_path,
                )
                assert m is not None
                block = m.group("block")
                proj = m.group("proj")
                block_kind = (
                    "mtp" if block.startswith("mtp")
                    else ("hash" if int(block.split(".")[1]) < stats["_n_hash_layers"] else "score")
                )

                weight_amax = _lookup_amax(amax, expert_path, "weight")
                input_amax = _lookup_amax(amax, expert_path, "input")
                used_fallback_input = False
                if input_amax is None:
                    input_amax = input_fallback.get((block, proj))
                    used_fallback_input = input_amax is not None
                if input_amax is None:
                    input_amax = torch.tensor(1.0)
                    stats[f"input_total_fallback_{block_kind}"] += 1

                w = f.get_tensor(key)
                s = f.get_tensor(scale_key)
                packed, weight_scale, weight_scale_2, weight_synth = _quantize_weight_nvfp4(
                    w, s, weight_amax, device=device
                )
                input_scale = _amax_to_nvfp4_scale_2(input_amax).to(weight_scale_2.device)

                out[key] = packed.cpu()
                out[expert_path + ".weight_scale"] = weight_scale.cpu()
                out[expert_path + ".weight_scale_2"] = weight_scale_2.cpu()
                out[expert_path + ".input_scale"] = input_scale.cpu()
                added.extend([
                    expert_path + ".weight_scale",
                    expert_path + ".weight_scale_2",
                    expert_path + ".input_scale",
                ])

                stats["experts_total"] += 1
                stats[f"experts_{block_kind}"] += 1
                if weight_synth:
                    stats[f"weight_synth_{block_kind}"] += 1
                if used_fallback_input:
                    stats[f"input_fallback_{block_kind}"] += 1
            else:
                out[key] = f.get_tensor(key)
                stats["passthrough"] += 1

    save_file(out, str(dst_shard))
    return added, removed


# ---------------------------------------------------------------------------
# Aux + manifest
# ---------------------------------------------------------------------------


# Top-level names never hard-linked from source (rewritten or excluded).
_SKIP_TOP_LEVEL = {
    "model.safetensors.index.json",  # rewritten
    "config.json",                   # rewritten (drop stale quantization_config)
    ".cache",                        # HF download sidecars referencing old shards
}
# Subdir names to skip anywhere in the walk.
_SKIP_SUBDIR_NAMES = {"__pycache__"}


def _hard_link_aux(src: Path, dst: Path) -> None:
    """Hard-link everything that isn't a shard file, rewritten metadata, or
    a cache/__pycache__ directory. Recurses into legit subdirectories
    (``encoding/``, ``inference/`` etc.) preserving structure."""
    assert os.stat(src).st_dev == os.stat(dst).st_dev, (
        "hard links require same filesystem"
    )
    for item in src.iterdir():
        if item.name in _SKIP_TOP_LEVEL:
            continue
        if _HF_SHARD_RE.match(item.name):
            continue
        target = dst / item.name
        if item.is_file():
            if target.exists():
                target.unlink()
            os.link(item, target)
        elif item.is_dir():
            target.mkdir(exist_ok=True)
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d not in _SKIP_SUBDIR_NAMES]
                rel = Path(root).relative_to(item)
                (target / rel).mkdir(parents=True, exist_ok=True)
                for fname in files:
                    # Never pull stale shards / indexes from inside subdirs.
                    if fname == "model.safetensors.index.json" or _HF_SHARD_RE.match(fname):
                        continue
                    src_f = Path(root) / fname
                    dst_f = target / rel / fname
                    if dst_f.exists():
                        dst_f.unlink()
                    os.link(src_f, dst_f)


def _rewrite_config_json(src_dir: Path, dst_dir: Path) -> None:
    """Copy ``config.json`` to the output with the stale ``quantization_config``
    stanza removed (it described the source FP8+UE8M0 format; the output is
    mixed NVFP4 experts + unchanged-elsewhere, with the real per-layer info
    in ``hf_quant_config.json``). Matches V3's
    ``remove_quantization_config_from_original_config`` pattern."""
    src = src_dir / "config.json"
    dst = dst_dir / "config.json"
    cfg = json.loads(src.read_text())
    cfg.pop("quantization_config", None)
    dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def _write_index_and_manifest(
    output_ckpt: Path,
    src_index: dict,
    shard_updates: dict[str, tuple[list[str], list[str]]],
    quantized_layer_names: list[str],
) -> None:
    """Update ``model.safetensors.index.json`` with dropped ``.scale`` keys
    and added NVFP4 scale keys. Write the modelopt-style manifest."""
    weight_map = dict(src_index["weight_map"])
    for shard_name, (added, removed) in shard_updates.items():
        for k in removed:
            weight_map.pop(k, None)
        for k in added:
            weight_map[k] = shard_name
    new_index = {"metadata": src_index.get("metadata", {}), "weight_map": weight_map}
    (output_ckpt / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2)
    )
    _log(f"[index] wrote model.safetensors.index.json ({len(weight_map)} keys)")

    cfg = {
        "producer": {
            "name": "modelopt",
            "version": "dsv4-nvfp4-experts",
        },
        "quantization": {
            # Mixed precision — top-level algo is None and consumers must
            # consult quantized_layers / exclude_modules for specifics.
            "quant_algo": None,
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                name: {"quant_algo": "NVFP4", "awq_block_size": 16}
                for name in quantized_layer_names
            },
            "exclude_modules": [
                # Attention path
                "*.attn.*", "*.attn_norm.*",
                # Shared expert + router (untouched FP8)
                "*.ffn.shared_experts.*", "*.ffn.gate.*",
                "*.ffn_norm.*",
                # Embeddings + head
                "embed.weight", "head.weight",
                # Hyper-connection params
                "*.hc_*",
                # MTP auxiliary projections/norms (non-expert)
                "*.h_proj.*", "*.e_proj.*",
                "*.enorm.*", "*.hnorm.*",
                "mtp.*.norm.*", "norm.weight",
            ],
        },
    }
    (output_ckpt / "hf_quant_config.json").write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--amax_path", type=Path, required=True)
    p.add_argument(
        "--source_ckpt", type=Path, required=True,
        help="original HF 64-shard release (e.g. /.../DeepSeek-V4-Pro/) — NOT the MP derivative",
    )
    p.add_argument("--output_ckpt", type=Path, required=True)
    p.add_argument(
        "--world_size", type=int, default=None,
        help="override MP auto-detect (useful if multiple-MP dumps exist in the amax dir)",
    )
    p.add_argument(
        "--device", default="cpu",
        help="device for MXFP4 dequant + NVFP4 quant ('cpu' safe; 'cuda' faster)",
    )
    p.add_argument(
        "--n_hash_layers", type=int, default=3,
        help="diagnostic only — labels hash-routed layers in stats",
    )
    args = p.parse_args()

    args.output_ckpt.mkdir(parents=True, exist_ok=True)

    src_index_path = args.source_ckpt / "model.safetensors.index.json"
    assert src_index_path.exists(), (
        f"{src_index_path} not found — --source_ckpt must be the original "
        f"HF 64-shard release, not the MP-sharded derivative"
    )
    src_index = json.loads(src_index_path.read_text())

    amax, input_fallback = _load_merged_amax(args.amax_path, world_size=args.world_size)

    shards = sorted(args.source_ckpt.glob("model-*-of-*.safetensors"))
    assert shards, f"no HF-style shards in {args.source_ckpt}"
    _log(f"[config] {len(shards)} input shards  device={args.device}")

    stats: dict[str, int] = defaultdict(int)
    stats["_n_hash_layers"] = args.n_hash_layers
    shard_updates: dict[str, tuple[list[str], list[str]]] = {}

    for idx, src in enumerate(shards):
        dst = args.output_ckpt / src.name
        _log(f"[shard {idx + 1}/{len(shards)}] {src.name}")
        added, removed = convert_shard(src, dst, amax, input_fallback, args.device, stats)
        shard_updates[src.name] = (added, removed)

    stats.pop("_n_hash_layers", None)
    _log("[stats]")
    for k in sorted(stats.keys()):
        _log(f"  {k:40s} {stats[k]}")

    quantized: set[str] = set()
    for _added, _removed in shard_updates.values():
        for a in _added:
            if a.endswith(".input_scale"):
                quantized.add(a[: -len(".input_scale")])

    _write_index_and_manifest(args.output_ckpt, src_index, shard_updates, sorted(quantized))
    _log(f"[config] rewriting config.json (dropping stale quantization_config)")
    _rewrite_config_json(args.source_ckpt, args.output_ckpt)
    _log(f"[aux] hard-linking ancillary files from {args.source_ckpt}")
    _hard_link_aux(args.source_ckpt, args.output_ckpt)
    _log(f"[done] {args.output_ckpt}  ({len(quantized)} quantized expert layers)")


if __name__ == "__main__":
    main()
