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
  * ``config.json`` keeps the source FP8 quantization metadata, adds
    ``"moe_quant_algo": "NVFP4"`` for DeepSeek-V4 loaders, and embeds the
    same NVFP4 MoE layer manifest emitted to ``hf_quant_config.json``.
  * An ``hf_quant_config.json`` manifest listing the NVFP4-quantized layers.
    MTP expert weights are left in the source format by default; pass
    ``--include_mtp_experts`` to convert them too.
  * Ancillary files (``tokenizer.json``, ``LICENSE``, ``encoding/``,
    ``inference/``, ...) are linked from the source when possible, with a
    copy fallback across filesystems.

Uncalibrated-expert handling: some routed experts receive zero tokens during
calibration (common in V4's first ``n_hash_layers`` with deterministic
``tid2eid`` routing, occasionally in score-routed layers, entire MTP block
when MTP is inactive). For those:
  * ``weight_amax`` is synthesized from the dequantized BF16 weight
    (``bf16.abs().max()``).
  * ``input_amax`` falls back to the max observed on calibrated routed experts
    for the same projection. If no calibrated expert exists for that
    projection, export fails.

Usage (single compute node, CPU-default; dequant+requant math is cheap
relative to shard I/O):

    srun --container-image=dsv4-ready.sqsh ... \\
        python quantize_to_nvfp4.py \\
            --amax_path   /path/to/amax-nvfp4-experts \\
            --source_ckpt /path/to/DeepSeek-V4-Pro \\
            --output_ckpt /path/to/DeepSeek-V4-Pro-nvfp4-experts
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor

# Routed-expert weights in regular MoE layers and optionally the MTP block(s).
_EXPERT_WEIGHT_RE = re.compile(r"^(?:mtp\.\d+|layers\.\d+)\.ffn\.experts\.\d+\.w[123]\.weight$")
_EXPERT_PROJ_RE = re.compile(r"^(?P<experts>(?:mtp\.\d+|layers\.\d+)\.ffn\.experts)\.\d+\.w[123]$")

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


def _fuse_w1_w3_amax(merged: dict[str, torch.Tensor], which: str) -> int:
    fused = 0
    for k in list(merged.keys()):
        if k.endswith(f"w1_{which}_quantizer._amax"):
            k3 = k.replace(f"w1_{which}", f"w3_{which}")
            if k3 in merged:
                shared = torch.maximum(merged[k], merged[k3])
                merged[k] = shared
                merged[k3] = shared
                fused += 1
    return fused


def _load_merged_amax(
    amax_path: Path,
    world_size: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Union-merge amax across ranks, fuse w1/w3 amax per expert, and compute
    input-amax fallbacks from calibrated experts."""
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

    # w1/w3 share the same input x and vLLM consumes one fused GEMM1 weight scale.
    input_fused = _fuse_w1_w3_amax(merged, "input")
    weight_fused = _fuse_w1_w3_amax(merged, "weight")
    _log(f"[load] fused w1/w3 input amax on {input_fused} experts")
    _log(f"[load] fused w1/w3 weight amax on {weight_fused} experts")

    input_by_proj: dict[str, list[torch.Tensor]] = defaultdict(list)
    for k, v in merged.items():
        m = _AMAX_KEY_RE.match(k)
        assert m is not None
        if m.group("which") == "input":
            input_by_proj[m.group("proj")].append(v)
    input_fallback = {
        proj: torch.stack([t.reshape(-1) for t in vals]).flatten().max()
        for proj, vals in input_by_proj.items()
    }
    _log(f"[load] input-fallback projections: {len(input_fallback)} populated")
    return merged, input_fallback


def _lookup_amax(
    amax: dict[str, torch.Tensor], expert_path: str, which: str
) -> torch.Tensor | None:
    return amax.get(f"{expert_path}_{which}_quantizer._amax")


def _dequantize_mxfp4_to_bf16(
    mxfp4_weight: torch.Tensor, mxfp4_scale: torch.Tensor, device: str
) -> torch.Tensor:
    block_size = 32
    packed = mxfp4_weight.to(device).contiguous().view(torch.uint8)
    scale = mxfp4_scale.to(device).contiguous().view(torch.uint8)
    original_shape = torch.Size((*packed.shape[:-1], packed.shape[-1] * 2))
    assert packed.shape[:-1] == scale.shape[:-1] and (
        2 * packed.shape[-1] == scale.shape[-1] * block_size
    ), f"Incompatible MXFP4 shapes: weight {tuple(packed.shape)} vs scale {tuple(scale.shape)}"
    return MXFP4QTensor(original_shape, torch.bfloat16, packed).dequantize(
        dtype=torch.bfloat16,
        scale=scale,
        block_sizes=[block_size],
    )


def _synthesize_weight_amax(
    mxfp4_weight: torch.Tensor, mxfp4_scale: torch.Tensor, device: str
) -> torch.Tensor:
    return _dequantize_mxfp4_to_bf16(mxfp4_weight, mxfp4_scale, device).abs().max().cpu()


def _quantize_weight_nvfp4(
    mxfp4_weight: torch.Tensor,
    mxfp4_scale: torch.Tensor,
    weight_amax: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """MXFP4 + UE8M0 → BF16 → NVFP4 packed. Synthesizes ``weight_amax`` from
    the dequantized BF16 tensor when ``None`` is passed."""
    bf16 = _dequantize_mxfp4_to_bf16(mxfp4_weight, mxfp4_scale, device)
    synthesized = weight_amax is None
    if synthesized:
        weight_amax = bf16.abs().max()
    assert weight_amax is not None
    weight_scale_2 = _amax_to_nvfp4_scale_2(weight_amax.to(device))
    q_tensor, weight_scale, _ = NVFP4QTensor.quantize(
        bf16, 16, None, weight_scale_2, try_tensorrt=False
    )
    return q_tensor._quantized_data, weight_scale, weight_scale_2, synthesized


def _build_w13_weight_amax_overrides(
    f,
    expert_weight_keys: list[str],
    amax: dict[str, torch.Tensor],
    device: str,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Return shared w1/w3 amax overrides so fused GEMM1 has one scale."""
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for key in expert_weight_keys:
        expert_path = key[: -len(".weight")]
        base, proj = expert_path.rsplit(".", 1)
        if proj in {"w1", "w3"}:
            groups[base][proj] = expert_path

    overrides: dict[str, torch.Tensor] = {}
    synthesized_paths: set[str] = set()
    for paths in groups.values():
        if "w1" not in paths or "w3" not in paths:
            continue

        values: list[torch.Tensor] = []
        for proj in ("w1", "w3"):
            expert_path = paths[proj]
            weight_amax = _lookup_amax(amax, expert_path, "weight")
            if weight_amax is None:
                weight_amax = _synthesize_weight_amax(
                    f.get_tensor(expert_path + ".weight"),
                    f.get_tensor(expert_path + ".scale"),
                    device,
                )
                synthesized_paths.add(expert_path)
            values.append(weight_amax.reshape(()))

        shared = torch.maximum(values[0], values[1])
        overrides[paths["w1"]] = shared
        overrides[paths["w3"]] = shared
    return overrides, synthesized_paths


def convert_shard(
    src_shard: Path,
    dst_shard: Path,
    amax: dict[str, torch.Tensor],
    input_fallback: dict[str, torch.Tensor],
    device: str,
    stats: dict[str, int],
    include_mtp_experts: bool,
) -> tuple[list[str], list[str]]:
    """Rewrite one HF-style shard and return index deltas."""
    out: dict[str, torch.Tensor] = {}
    added: list[str] = []
    removed: list[str] = []

    with safe_open(str(src_shard), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        expert_weight_keys = [
            k
            for k in all_keys
            if _EXPERT_WEIGHT_RE.match(k) and (include_mtp_experts or not k.startswith("mtp."))
        ]
        expert_weight_key_set = set(expert_weight_keys)
        w13_weight_amax, w13_synth_paths = _build_w13_weight_amax_overrides(
            f, expert_weight_keys, amax, device
        )
        scale_siblings = {
            k[: -len(".weight")] + ".scale"
            for k in expert_weight_keys
            if k[: -len(".weight")] + ".scale" in all_keys
        }

        for key in all_keys:
            if key in scale_siblings:
                removed.append(key)
                continue

            if key in expert_weight_key_set:
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
                    "mtp"
                    if block.startswith("mtp")
                    else ("hash" if int(block.split(".")[1]) < stats["_n_hash_layers"] else "score")
                )

                weight_amax = _lookup_amax(amax, expert_path, "weight")
                if expert_path in w13_weight_amax:
                    weight_amax = w13_weight_amax[expert_path]
                input_amax = _lookup_amax(amax, expert_path, "input")
                used_fallback_input = False
                if input_amax is None:
                    input_amax = input_fallback.get(proj)
                    used_fallback_input = input_amax is not None
                if input_amax is None:
                    raise RuntimeError(
                        f"missing input amax for {expert_path} and no calibrated "
                        "input fallback is available"
                    )

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
                added.extend(
                    [
                        expert_path + ".weight_scale",
                        expert_path + ".weight_scale_2",
                        expert_path + ".input_scale",
                    ]
                )

                stats["experts_total"] += 1
                stats[f"experts_{block_kind}"] += 1
                if weight_synth or expert_path in w13_synth_paths:
                    stats[f"weight_synth_{block_kind}"] += 1
                if used_fallback_input:
                    stats[f"input_fallback_{block_kind}"] += 1
            else:
                out[key] = f.get_tensor(key)
                stats["passthrough"] += 1

    save_file(out, str(dst_shard))
    return added, removed


# Top-level names never hard-linked from source (rewritten or excluded).
_SKIP_TOP_LEVEL = {
    "model.safetensors.index.json",  # rewritten
    "config.json",  # rewritten (mark hybrid FP8 + NVFP4 MoE)
    ".cache",  # HF download sidecars referencing old shards
}
# Subdir names to skip anywhere in the walk.
_SKIP_SUBDIR_NAMES = {"__pycache__"}


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError as e:
        copy_errnos = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            getattr(errno, "EOPNOTSUPP", errno.EXDEV),
            getattr(errno, "ENOTSUP", errno.EXDEV),
        }
        if e.errno not in copy_errnos:
            raise
        shutil.copy2(src, dst)


def _hard_link_aux(src: Path, dst: Path) -> None:
    """Link everything that isn't a shard file, rewritten metadata, or
    a cache/__pycache__ directory. Recurses into legit subdirectories
    (``encoding/``, ``inference/`` etc.) preserving structure.

    Falls back to copying when source and destination are on different
    filesystems, which is common with container mounts.
    """
    for item in src.iterdir():
        if item.name in _SKIP_TOP_LEVEL:
            continue
        if _HF_SHARD_RE.match(item.name):
            continue
        target = dst / item.name
        if item.is_file():
            if target.exists():
                target.unlink()
            _link_or_copy(item, target)
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
                    _link_or_copy(src_f, dst_f)


def _build_moe_quantization(
    quantized_layer_names: list[str], include_mtp_experts: bool
) -> dict[str, Any]:
    return {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "quantized_layers": {
            name: {"quant_algo": "NVFP4", "group_size": 16} for name in quantized_layer_names
        },
        "exclude_modules": [
            # Attention path and shared experts remain in the source FP8 format.
            "*.attn.*",
            "*.ffn.shared_experts.*",
            # LM head remains unconverted.
            "head",
            # MTP remains in source format by default.
            *([] if include_mtp_experts else ["mtp.*"]),
        ],
    }


def _build_hf_quant_config(
    quantized_layer_names: list[str], include_mtp_experts: bool
) -> dict[str, Any]:
    return {
        "producer": {
            "name": "modelopt",
            "version": "dsv4-nvfp4-experts",
        },
        "quantization": _build_moe_quantization(quantized_layer_names, include_mtp_experts),
    }


def _build_nvfp4_config_groups() -> dict[str, Any]:
    return {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "targets": ["Linear"],
        }
    }


def _rewrite_config_json(
    src_dir: Path,
    dst_dir: Path,
    quantized_layer_names: list[str],
    include_mtp_experts: bool,
) -> None:
    """Copy ``config.json`` to the output and mark the MoE branch as NVFP4.

    DeepSeek-V4 mixed checkpoints remain FP8 for dense/attention paths, so
    ``quant_method`` must stay ``fp8``. The NVFP4 MoE manifest is duplicated
    into ``quantization_config`` for loaders that prefer config.json over the
    sibling ``hf_quant_config.json``.
    """
    src = src_dir / "config.json"
    dst = dst_dir / "config.json"
    cfg = json.loads(src.read_text())
    quant_cfg = cfg.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}

    hf_quant_config = _build_hf_quant_config(quantized_layer_names, include_mtp_experts)
    moe_quantization = hf_quant_config["quantization"]
    quant_cfg.setdefault("activation_scheme", "dynamic")
    quant_cfg["quant_method"] = "fp8"
    quant_cfg.setdefault("weight_block_size", [128, 128])
    quant_cfg["moe_quant_algo"] = "NVFP4"
    quant_cfg["producer"] = hf_quant_config["producer"]
    quant_cfg["quant_algo"] = moe_quantization["quant_algo"]
    quant_cfg["kv_cache_quant_algo"] = moe_quantization["kv_cache_quant_algo"]
    quant_cfg["group_size"] = moe_quantization["group_size"]
    quant_cfg["config_groups"] = _build_nvfp4_config_groups()
    quant_cfg["quantized_layers"] = moe_quantization["quantized_layers"]
    quant_cfg["exclude_modules"] = moe_quantization["exclude_modules"]
    quant_cfg["ignore"] = moe_quantization["exclude_modules"]
    cfg["quantization_config"] = quant_cfg
    dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def _write_index_and_manifest(
    output_ckpt: Path,
    src_index: dict,
    shard_updates: dict[str, tuple[list[str], list[str]]],
    quantized_layer_names: list[str],
    include_mtp_experts: bool,
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
    (output_ckpt / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    _log(f"[index] wrote model.safetensors.index.json ({len(weight_map)} keys)")

    cfg = _build_hf_quant_config(quantized_layer_names, include_mtp_experts)
    (output_ckpt / "hf_quant_config.json").write_text(json.dumps(cfg, indent=2))


def _routed_experts_prefix(expert_proj: str) -> str:
    match = _EXPERT_PROJ_RE.match(expert_proj)
    assert match is not None, f"unexpected expert projection path: {expert_proj}"
    return match.group("experts")


def _validate_paths(source_ckpt: Path, output_ckpt: Path) -> None:
    source_resolved = source_ckpt.resolve()
    output_resolved = output_ckpt.resolve()
    if (
        output_resolved == source_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError(
            "--source_ckpt and --output_ckpt must be disjoint directories; "
            f"got source={source_ckpt}, output={output_ckpt}"
        )


def _prepare_output_dir(output_ckpt: Path, overwrite: bool) -> None:
    if output_ckpt.exists():
        if not output_ckpt.is_dir():
            raise ValueError(f"--output_ckpt exists and is not a directory: {output_ckpt}")
        if any(output_ckpt.iterdir()):
            if not overwrite:
                raise ValueError(
                    f"--output_ckpt is not empty: {output_ckpt}; pass --overwrite to replace it"
                )
            for item in output_ckpt.iterdir():
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    output_ckpt.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--amax_path", type=Path, required=True)
    p.add_argument(
        "--source_ckpt",
        type=Path,
        required=True,
        help="original HF 64-shard release (e.g. /.../DeepSeek-V4-Pro/) — NOT the MP derivative",
    )
    p.add_argument("--output_ckpt", type=Path, required=True)
    p.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="override MP auto-detect (useful if multiple-MP dumps exist in the amax dir)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="device for MXFP4 dequant + NVFP4 quant ('cpu' safe; 'cuda' faster)",
    )
    p.add_argument(
        "--n_hash_layers",
        type=int,
        default=3,
        help="diagnostic only — labels hash-routed layers in stats",
    )
    p.add_argument(
        "--include_mtp_experts",
        action="store_true",
        help="also convert MTP routed-expert weights to NVFP4; default leaves MTP in source format",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty output checkpoint directory",
    )
    args = p.parse_args()

    _validate_paths(args.source_ckpt, args.output_ckpt)

    src_index_path = args.source_ckpt / "model.safetensors.index.json"
    assert src_index_path.exists(), (
        f"{src_index_path} not found — --source_ckpt must be the original "
        f"HF 64-shard release, not the MP-sharded derivative"
    )
    src_index = json.loads(src_index_path.read_text())

    amax, input_fallback = _load_merged_amax(args.amax_path, world_size=args.world_size)

    shards = sorted(args.source_ckpt.glob("model-*-of-*.safetensors"))
    assert shards, f"no HF-style shards in {args.source_ckpt}"
    _prepare_output_dir(args.output_ckpt, args.overwrite)
    _log(f"[config] {len(shards)} input shards  device={args.device}")

    stats: dict[str, int] = defaultdict(int)
    stats["_n_hash_layers"] = args.n_hash_layers
    shard_updates: dict[str, tuple[list[str], list[str]]] = {}

    for idx, src in enumerate(shards):
        dst = args.output_ckpt / src.name
        _log(f"[shard {idx + 1}/{len(shards)}] {src.name}")
        added, removed = convert_shard(
            src,
            dst,
            amax,
            input_fallback,
            args.device,
            stats,
            include_mtp_experts=args.include_mtp_experts,
        )
        shard_updates[src.name] = (added, removed)

    stats.pop("_n_hash_layers", None)
    _log("[stats]")
    for k in sorted(stats.keys()):
        _log(f"  {k:40s} {stats[k]}")

    quantized: set[str] = set()
    for _added, _removed in shard_updates.values():
        for a in _added:
            if a.endswith(".input_scale"):
                quantized.add(_routed_experts_prefix(a[: -len(".input_scale")]))

    _write_index_and_manifest(
        args.output_ckpt,
        src_index,
        shard_updates,
        sorted(quantized),
        include_mtp_experts=args.include_mtp_experts,
    )
    _log("[config] rewriting config.json (marking moe_quant_algo=NVFP4)")
    _rewrite_config_json(
        args.source_ckpt,
        args.output_ckpt,
        sorted(quantized),
        include_mtp_experts=args.include_mtp_experts,
    )
    _log(f"[aux] linking ancillary files from {args.source_ckpt}")
    _hard_link_aux(args.source_ckpt, args.output_ckpt)
    _log(f"[done] {args.output_ckpt}  ({len(quantized)} quantized routed-expert modules)")


if __name__ == "__main__":
    main()
