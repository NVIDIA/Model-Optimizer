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

"""Kimi-K3: MXFP4 routed experts -> NVFP4, plus FP8-family attention.

``moonshotai/Kimi-K3`` ships 2.8T parameters in a hybrid checkpoint:

* routed experts (92 MoE layers x 896 experts x {w1,w2,w3}) are **MXFP4**,
  stored compressed-tensors style as ``<proj>.weight_packed`` (E2M1 nibbles,
  2 per byte) + ``<proj>.weight_scale`` (uint8 E8M0, one per 32-element block);
* everything else -- MLA/KDA attention, LatentMoE, shared experts, the dense
  layer-0 MLP, ``lm_head``, the MoonViT tower -- is **BF16**.

This script rewrites the 96-shard release in place-for-place fashion:

1. **Routed experts: closed-form MXFP4 -> NVFP4 cast** (``--cast_mxfp4_to_nvfp4``).
   NVFP4 uses the same E2M1 nibble grid with 16-element blocks and a two-level
   scale ``per_block_scale (E4M3) * scale_2 (fp32)``. Pinning
   ``scale_2 = 2^m`` (``m = k_max - 8``) and ``per_block_scale = 2^(k_j - m)``
   makes ``per_block_scale * scale_2 == 2^k_j`` exactly, so every NVFP4 nibble
   equals the source MXFP4 nibble verbatim -- bit-exact for every block whose
   ``k_j`` lands in E4M3's window (``k_max - k_j <= 17``). Blocks outside the
   window fall back to a data-derived per-block amax. The numerics are shared
   with the GPT-OSS cast (``examples/hf_ptq/cast_mxfp4_to_nvfp4.py``, PR #1372)
   via ``modelopt.torch.quantization.utils.numeric_utils``.

   As in DeepSeek-V4, ``w1``/``w3`` feed one fused GEMM1 and therefore must
   share a single ``scale_2``, so ``k_max`` is taken over both projections.

2. **Attention: BF16 -> block FP8** (``--attn_fp8_pb``), MXFP8
   (``--attn_mxfp8``), or static per-tensor FP8 (``--attn_fp8``). Block FP8
   uses one fp32 scale per 128x128 weight tile and dynamic per-token activation
   scaling. MXFP8 uses one E8M0 scale per 32 weights
   and dynamic activation scaling, so it remains calibration-free without a
   coarse static activation scale. Static FP8 weight scales are derived as
   ``amax(|W|) / 448`` and activation ``input_scale`` is pinned to
   ``--input_scale``. Both cover the MLA projections (``q_a_proj``, ``q_b_proj``,
   ``kv_a_proj_with_mqa``, ``kv_b_proj``), the KDA projections (``q_proj``,
   ``k_proj``, ``v_proj``) and the per-layer ``o_proj`` / ``g_proj`` -- ~36B
   parameters, 72GB of BF16 down to 36GB. The vLLM-fused KDA projections
   include ``b_proj`` and ``f_a_proj`` and are quantized consistently with the
   other members of ``in_proj_qkvgfab``. Short convolutions (``*_conv1d``),
   ``A_log``, ``dt_bias``, every norm, and the MoE router ``gate`` stay BF16.
   ``f_b_proj`` stays BF16 for per-tensor FP8/MXFP8, but is included in the
   block-FP8 recipe because its TP-local shape is already 128-tile aligned.

   Not covered here, but the obvious next candidates if more memory is needed:
   the LatentMoE ``routed_expert_{up,down}_proj`` (~9.5GB) and
   ``shared_experts.*`` (~24GB), both of which are replicated rather than
   sharded under expert parallelism.

**No calibration anywhere.** Expert ``input_scale`` is pinned to
``--input_scale`` (default 1.0) rather than derived from an amax dump, so the
whole conversion is a closed-form tensor transform: no forward pass, no
dataset, no GPU required. ``--device cpu`` is the default and is what the
shipped configuration uses; the work is dominated by shard I/O.

Usage (CPU partition, no GPU needed; ``--jobs`` shards convert in parallel):

    python quantize_to_nvfp4.py \\
        --source_ckpt /path/to/Kimi-K3 \\
        --output_ckpt /path/to/Kimi-K3-NVFP4 \\
        --recipe huggingface/models/moonshotai/Kimi-K3/ptq/nvfp4_experts-fp8_pb_attention \\
        --jobs 8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt import __version__ as modelopt_version
from modelopt.recipe import load_recipe
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.shard_cast_utils import (
    build_w13_amax_overrides,
    build_w13_kmax_overrides,
    dequantize_mxfp4_to_bf16,
    link_aux_files,
    mxfp4_kmax,
    prepare_output_dir,
    quantize_mxfp4_to_nvfp4,
    quantize_mxfp4_to_nvfp4_lossless,
    resolve_checkpoint_file,
    validate_aux_files,
    validate_paths,
)
from modelopt.torch.export.shard_cast_utils import log as _log
from modelopt.torch.quantization.qtensor import FP8QTensor, MXFP8QTensor
from modelopt.torch.quantization.utils.numeric_utils import E2M1_MAX, E4M3_MAX

# --------------------------------------------------------------------------
# Kimi-K3 tensor schema (from model.safetensors.index.json, 497220 tensors).
# --------------------------------------------------------------------------
_LM = r"language_model\.model\.layers\.\d+"

# Routed experts: MXFP4 source pair <base>.weight_packed + <base>.weight_scale.
_EXPERT_PACKED_RE = re.compile(
    rf"^(?P<base>{_LM}\.block_sparse_moe\.experts\.\d+\.w[123])\.weight_packed$"
)
# Prefix reported in the quantized-layer manifest (one entry per expert bank).
_EXPERT_BANK_RE = re.compile(rf"^(?P<bank>{_LM}\.block_sparse_moe\.experts)\.\d+\.w[123]$")

# Attention projections eligible for FP8/MXFP8. Everything else under
# ``self_attn`` (conv1d kernels, A_log, dt_bias, o_norm, layernorms) stays BF16.
# vLLM fuses q/k/v/g/f_a/b into ``in_proj_qkvgfab``, so those six must use one
# quantization algorithm. ``f_b_proj`` remains BF16 except in the block-FP8
# recipe.
_ATTN_QUANT_PROJ = frozenset(
    {
        # KDA (69 layers)
        "q_proj",
        "k_proj",
        "v_proj",
        # MLA (24 layers)
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        # present on all 93 layers
        "o_proj",
        "g_proj",
        # vLLM fuses both into in_proj_qkvgfab with q/k/v/g.
        "b_proj",
        "f_a_proj",
    }
)
# ``f_b_proj`` is a small KDA state-control projection. It remains BF16 for
# per-tensor FP8/MXFP8, but its TP-local [1536, 128] shape is naturally aligned
# for 128x128 block FP8 and the prior standalone-FP8 recipe found it safe.
_ATTN_FP8_PB_PROJ = _ATTN_QUANT_PROJ | {"f_b_proj"}
_ATTN_WEIGHT_RE = re.compile(rf"^(?P<base>{_LM}\.self_attn\.(?P<proj>[a-z_0-9]+))\.weight$")

_FP8_MAX = 448.0
_FP8_PB_BLOCK = 128
_NVFP4_BLOCK = 16  # NVFP4 block size (elements)

_PUBLISHED_RECIPE = "huggingface/models/moonshotai/Kimi-K3/ptq/nvfp4_experts-fp8_pb_attention"


def _conversion_settings_from_recipe(recipe_path: str) -> dict[str, Any]:
    """Translate the supported Kimi-K3 recipe into streaming-converter settings."""
    quantize = load_recipe(recipe_path).quantize.model_dump()
    return _conversion_settings_from_quantize_config(quantize)


def _conversion_settings_from_quantize_config(quantize: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete recipe contract and return converter settings."""
    algorithm = quantize.get("algorithm")
    if (
        not isinstance(algorithm, dict)
        or algorithm.get("method") != "max"
        or not isinstance(algorithm.get("layerwise"), dict)
        or algorithm["layerwise"].get("enable") is not False
        or algorithm.get("skip_forward_without_activation_calib") is not True
    ):
        raise ValueError("recipe must use the calibration-free max algorithm")

    quant_cfg = quantize["quant_cfg"]
    by_name = {
        entry["quantizer_name"]: entry
        for entry in quant_cfg
        if isinstance(entry, dict) and "quantizer_name" in entry
    }

    expert_weight_name = "*block_sparse_moe.experts.*weight_quantizer"
    expert_input_name = "*block_sparse_moe.experts.*input_quantizer"
    expected_enabled = {
        expert_weight_name,
        expert_input_name,
        *(f"*self_attn.{projection}*weight_quantizer" for projection in _ATTN_FP8_PB_PROJ),
    }
    enabled = {
        entry["quantizer_name"]
        for entry in quant_cfg
        if isinstance(entry, dict) and entry.get("enable") is True
    }
    unexpected_enabled = enabled - expected_enabled
    if unexpected_enabled:
        raise ValueError(
            "recipe enables quantizers the streaming converter does not support: "
            f"{sorted(unexpected_enabled)}"
        )
    if not any(
        entry.get("quantizer_name") == "*"
        and entry.get("parent_class") is None
        and entry.get("enable") is False
        for entry in quant_cfg
        if isinstance(entry, dict)
    ):
        raise ValueError("recipe must disable all quantizers by default")

    try:
        expert_weight = by_name[expert_weight_name]["cfg"]
        expert_input = by_name[expert_input_name]["cfg"]
    except KeyError as exc:
        raise ValueError("recipe must configure Kimi-K3 routed experts") from exc

    if (
        expert_weight.get("num_bits") != (2, 1)
        or expert_weight.get("block_sizes", {}).get(-1) != _NVFP4_BLOCK
    ):
        raise ValueError("recipe routed-expert weights must use block-16 NVFP4")

    constant_amax = expert_input.get("constant_amax")
    if constant_amax is None:
        raise ValueError("recipe routed-expert inputs must set constant_amax")
    input_scale = float(constant_amax) / (E2M1_MAX * E4M3_MAX)

    if enabled != expected_enabled:
        raise ValueError(
            f"recipe is missing required quantizers: {sorted(expected_enabled - enabled)}"
        )

    for projection in _ATTN_FP8_PB_PROJ:
        name = f"*self_attn.{projection}*weight_quantizer"
        try:
            cfg = by_name[name]["cfg"]
        except KeyError as exc:
            raise ValueError(f"recipe must configure {projection} attention weights") from exc
        if cfg.get("num_bits") != (4, 3) or cfg.get("block_sizes") != {-1: 128, -2: 128}:
            raise ValueError(f"recipe {projection} weights must use 128x128 block FP8")

    return {
        "cast_mxfp4_to_nvfp4": True,
        "attn_fp8": False,
        "attn_mxfp8": False,
        "attn_fp8_pb": True,
        "input_scale": input_scale,
    }


# --------------------------------------------------------------------------
# BF16 -> per-tensor FP8 (attention)
# --------------------------------------------------------------------------
# Projections that serving stacks fuse into a single Linear, and therefore
# must share one per-tensor scale. SGLang builds ``fused_qkvg_proj`` over the
# wide KDA projections and ``fused_qkv_a_proj_with_mqa`` over the MLA pair.
#
# Sharing is not merely cosmetic. The KDA fusion is a MergedColumnParallelLinear,
# which keeps a per-shard scale array and requantizes to the max across shards,
# so independent scales would survive there. The MLA fusion is a *ReplicatedLinear*
# with a single scalar scale and no shard structure to requantize over, so two
# independent scales silently misinterpret half the fused weight. Deriving one
# amax per group up front makes the checkpoint correct under either layout.
_FUSED_ATTN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"),
    ("q_a_proj", "kv_a_proj_with_mqa"),  # MLA
)
_PROJ_TO_GROUP = {proj: i for i, g in enumerate(_FUSED_ATTN_GROUPS) for proj in g}


def _fused_attn_group_key(layer_prefix: str, proj: str) -> str | None:
    """``<layer>.self_attn`` + group index, or None when the proj is standalone."""
    idx = _PROJ_TO_GROUP.get(proj)
    return None if idx is None else f"{layer_prefix}#{idx}"


def compute_fused_attn_amax(shards: list[Path], device: str) -> dict[str, torch.Tensor]:
    """One shared ``amax`` per fused attention group, over all shards.

    Members of a group are not guaranteed to land in the same shard, so this is
    a separate pass over the attention weights (~36B parameters) before any
    conversion happens. It reads only BF16 attention tensors, not the experts.
    """
    group_amax: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                m = _ATTN_WEIGHT_RE.match(key)
                if not m or m.group("proj") not in _ATTN_QUANT_PROJ:
                    continue
                gk = _fused_attn_group_key(m.group("base").rsplit(".", 1)[0], m.group("proj"))
                if gk is None:
                    continue
                amax = f.get_tensor(key).to(device).float().abs().max().cpu()
                prev = group_amax.get(gk)
                group_amax[gk] = amax if prev is None else torch.maximum(prev, amax)
    _log(f"[fused-attn] shared amax computed for {len(group_amax)} fused groups")
    return group_amax


def _quantize_weight_fp8(
    weight: torch.Tensor, device: str, amax: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data-free per-tensor FP8: ``weight_scale = amax(|W|) / 448``.

    ``amax`` overrides the tensor's own maximum so every member of a fused
    group is quantized against one shared scale. Returns
    ``(fp8_weight, weight_scale)``. A degenerate all-zero weight would give a
    zero scale, which would make dequantization ill-defined, so the scale is
    floored at the smallest positive normal fp32 value.
    """
    w = weight.to(device).to(torch.float32)
    a = w.abs().max() if amax is None else amax.to(w.device).float()
    weight_scale = (a / _FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny).reshape(())
    q = (w / weight_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, weight_scale.to(torch.float32)


def _quantize_weight_mxfp8(weight: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Calibration-free MXFP8 with one E8M0 scale per 32 weights."""
    q_tensor, weight_scale = MXFP8QTensor.quantize(weight.to(device))
    return q_tensor._quantized_data, weight_scale


def _quantize_weight_fp8_pb(weight: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Calibration-free 128x128 FP8 weight blocks.

    ModelOpt crops the serialized FP8 weight back to its logical shape while
    retaining scales for the zero-padded edge tiles. Unified HF checkpoints
    store those scales as ``[out_block, 1, in_block, 1]``.
    """
    q_tensor, weight_scale = FP8QTensor.quantize(
        weight.to(device),
        block_sizes={-2: _FP8_PB_BLOCK, -1: _FP8_PB_BLOCK},
    )
    return (
        q_tensor._quantized_data.contiguous(),
        weight_scale[:, None, :, None].to(torch.float32).contiguous(),
    )


# --------------------------------------------------------------------------
# Shard conversion
# --------------------------------------------------------------------------
def convert_shard(
    src_shard: Path,
    dst_shard: Path,
    device: str,
    cast: bool,
    attn_fp8: bool,
    input_scale_value: float,
    fused_attn_amax: dict[str, torch.Tensor] | None = None,
    attn_mxfp8: bool = False,
    attn_fp8_pb: bool = False,
) -> dict[str, Any]:
    """Rewrite one HF-style shard and return its conversion report."""
    out: dict[str, torch.Tensor] = {}
    stats: dict[str, int] = defaultdict(int)
    banks: set[str] = set()
    attn_modules: set[str] = set()

    input_scale = torch.tensor(input_scale_value, dtype=torch.float32).reshape(())

    with safe_open(str(src_shard), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        key_set = set(all_keys)

        expert_bases = [
            m.group("base") for m in (_EXPERT_PACKED_RE.match(k) for k in all_keys) if m
        ]
        # Source E8M0 scale tensors are consumed and replaced by NVFP4 scales.
        expert_scale_keys = {b + ".weight_scale" for b in expert_bases}
        expert_packed_keys = {b + ".weight_packed" for b in expert_bases}

        if cast:
            w13_kmax = build_w13_kmax_overrides(
                expert_bases,
                lambda base: f.get_tensor(base + ".weight_scale"),
                device,
            )
            w13_weight_amax = {}
        else:
            w13_kmax = {}
            w13_weight_amax = build_w13_amax_overrides(
                expert_bases,
                lambda base: (
                    dequantize_mxfp4_to_bf16(
                        f.get_tensor(base + ".weight_packed"),
                        f.get_tensor(base + ".weight_scale"),
                        device,
                    )
                    .abs()
                    .max()
                ),
            )

        for key in all_keys:
            # Source MXFP4 E8M0 scales are rewritten below alongside the packed
            # weight; skip them here so they are not emitted twice.
            if key in expert_scale_keys:
                continue

            if key in expert_packed_keys:
                base = key[: -len(".weight_packed")]
                scale_key = base + ".weight_scale"
                assert scale_key in key_set, f"no paired weight_scale for {key}"

                w = f.get_tensor(key)
                s = f.get_tensor(scale_key)

                if cast:
                    k_max = w13_kmax.get(base)
                    if k_max is None:
                        k_max = mxfp4_kmax(s, device)
                    packed, weight_scale, weight_scale_2, n_blk, n_lossless = (
                        quantize_mxfp4_to_nvfp4_lossless(w, s, k_max, device)
                    )
                    stats["cast_blocks_total"] += n_blk
                    stats["cast_blocks_lossless"] += n_lossless
                    if n_lossless < n_blk:
                        stats["cast_oor_tensors"] += 1
                else:
                    packed, weight_scale, weight_scale_2, _ = quantize_mxfp4_to_nvfp4(
                        w,
                        s,
                        w13_weight_amax.get(base),
                        device,
                    )

                out[base + ".weight"] = packed.cpu()
                out[base + ".weight_scale"] = weight_scale.cpu()
                out[base + ".weight_scale_2"] = weight_scale_2.cpu()
                out[base + ".input_scale"] = input_scale.clone()

                stats["experts_converted"] += 1
                bank = _EXPERT_BANK_RE.match(base)
                assert bank is not None, f"unexpected expert path: {base}"
                banks.add(bank.group("bank"))
                continue

            attn = _ATTN_WEIGHT_RE.match(key)
            if attn_fp8_pb and attn and attn.group("proj") in _ATTN_FP8_PB_PROJ:
                base = attn.group("base")
                q, weight_scale = _quantize_weight_fp8_pb(f.get_tensor(key), device)
                out[key] = q.cpu()
                out[base + ".weight_scale"] = weight_scale.cpu()
                stats["attn_fp8_pb_converted"] += 1
                attn_modules.add(base)
                continue

            if attn_fp8 and attn and attn.group("proj") in _ATTN_QUANT_PROJ:
                base = attn.group("base")
                gk = _fused_attn_group_key(base.rsplit(".", 1)[0], attn.group("proj"))
                shared = (fused_attn_amax or {}).get(gk) if gk else None
                q, weight_scale = _quantize_weight_fp8(f.get_tensor(key), device, shared)
                if shared is not None:
                    stats["attn_fp8_fused_shared_scale"] += 1
                out[key] = q.cpu()
                out[base + ".weight_scale"] = weight_scale.cpu()
                out[base + ".input_scale"] = input_scale.clone()
                stats["attn_fp8_converted"] += 1
                attn_modules.add(base)
                continue

            if attn_mxfp8 and attn and attn.group("proj") in _ATTN_QUANT_PROJ:
                base = attn.group("base")
                q, weight_scale = _quantize_weight_mxfp8(f.get_tensor(key), device)
                out[key] = q.cpu()
                out[base + ".weight_scale"] = weight_scale.cpu()
                stats["attn_mxfp8_converted"] += 1
                attn_modules.add(base)
                continue

            out[key] = f.get_tensor(key)
            stats["passthrough"] += 1

    save_file(out, str(dst_shard))
    return {
        "shard": src_shard.name,
        "tensor_bytes": sum(t.numel() * t.element_size() for t in out.values()),
        "stats": dict(stats),
        "banks": sorted(banks),
        "attn_modules": sorted(attn_modules),
    }


def _convert_shard_task(job: dict[str, Any]) -> dict[str, Any]:
    """ProcessPoolExecutor entry point (module-level so it is picklable)."""
    torch.set_num_threads(job["threads"])
    result = convert_shard(
        Path(job["src"]),
        Path(job["dst"]),
        job["device"],
        job["cast"],
        job["attn_fp8"],
        job["input_scale"],
        job.get("fused_attn_amax"),
        job.get("attn_mxfp8", False),
        job.get("attn_fp8_pb", False),
    )
    _log(f"[shard] {result['shard']} done: {result['stats']}")
    return result


# --------------------------------------------------------------------------
# Ancillary files / config / index
# --------------------------------------------------------------------------
_SKIP_TOP_LEVEL = {
    "model.safetensors.index.json",  # rewritten
    "config.json",  # rewritten (mark hybrid NVFP4 MoE + FP8 attention)
    "hf_quant_config.json",  # rewritten
    "conversion_report.json",  # rewritten
    ".kimi_k3_conversion",  # temporary distributed-conversion rendezvous
    ".cache",  # HF download sidecars referencing old shards
}
_SKIP_SUBDIR_NAMES = {"__pycache__"}


# Modules deliberately left in BF16. Kept explicit so a reader can see that the
# router gate, the KDA state parameters, the norms, the LatentMoE/shared-expert
# projections, the dense layer-0 MLP and the whole vision stack are untouched.
_EXCLUDE_MODULES = [
    "*block_sparse_moe.gate*",
    "*block_sparse_moe.routed_expert_up_proj*",
    "*block_sparse_moe.routed_expert_down_proj*",
    "*block_sparse_moe.routed_expert_norm*",
    "*block_sparse_moe.shared_experts*",
    "*self_attn.q_conv1d*",
    "*self_attn.k_conv1d*",
    "*self_attn.v_conv1d*",
    # f_b_proj is a small KDA state-control linear and stays BF16 except in the
    # block-FP8 recipe. b_proj and f_a_proj are quantized because vLLM folds
    # them into in_proj_qkvgfab.
    "*self_attn.f_b_proj*",
    "*self_attn.o_norm*",
    "*self_attn.q_a_layernorm*",
    "*self_attn.kv_a_layernorm*",
    "*_res_proj*",
    "*_res_norm*",
    "*layernorm*",
    "language_model.model.layers.0.mlp.*",
    "language_model.lm_head",
    "vision_tower*",
    "mm_projector*",
]


def _module_name_aliases(name: str) -> list[str]:
    """Every prefix spelling a loader may probe for one HF module path.

    Manifest keys are HF checkpoint paths, but loaders look modules up by their
    runtime prefix, and for Kimi-K3 the two differ twice over. SGLang's
    ``kimi_k3.py`` builds the language model with ``prefix=""`` even though the
    attribute is ``language_model``, so lookups arrive as ``model.layers.N...``;
    and its ``WeightsMapper`` rewrites ``block_sparse_moe`` -> ``mlp`` but does
    not strip ``language_model.``. vLLM's K3 mapper strips
    ``language_model.model.`` entirely and probes ``layers.N...``. A manifest
    keyed only on HF names therefore matches nothing, every module falls back
    to the unquantized method, and a 2.8T MoE tries to allocate BF16 experts.
    Emitting all spellings costs a few KB and keeps the checkpoint loadable
    under either convention.
    """
    names = {name}
    if "block_sparse_moe" in name:
        names.add(name.replace("block_sparse_moe", "mlp"))
    runtime_names: set[str] = set()
    for n in names:
        if n.startswith("language_model.model."):
            suffix = n[len("language_model.model.") :]
            runtime_names.add("model." + suffix)
            runtime_names.add(suffix)
        elif n.startswith("language_model."):
            runtime_names.add(n[len("language_model.") :])
    names.update(runtime_names)
    return sorted(names)


def _fused_attention_module_names(attn_modules: list[str]) -> set[str]:
    """Return K3 runtime fused-linear prefixes implied by HF projection names.

    Different engines fuse the attention projections under different names, so
    emit every spelling and let the loader match whichever it builds.

    SGLang (use_full_rank_gate): the wide [q, k, v, g] projections fuse into
    ``fused_qkvg_proj`` while b / f_a / f_b stay standalone -- a clean split for
    our FP8 policy, which quantizes exactly q/k/v/g and leaves the KDA
    state-control projections BF16. The MLA pair fuses into
    ``fused_qkv_a_proj_with_mqa`` (matches deepseek_v2).

    vLLM names the KDA fusion ``in_proj_qkvgfab`` and the MLA fusion
    ``fused_qkv_a_proj``; the former also folds in b/f_a, so a mixed-precision
    fused module there is vLLM's own concern.
    """
    fused: set[str] = set()
    for name in attn_modules:
        parent, proj = name.rsplit(".", 1)
        if proj in {"q_proj", "k_proj", "v_proj", "g_proj"}:
            fused.add(parent + ".fused_qkvg_proj")  # SGLang KDA
        if proj in {"q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"}:
            fused.add(parent + ".in_proj_qkvgfab")  # vLLM KDA
        if proj in {"q_a_proj", "kv_a_proj_with_mqa"}:
            fused.add(parent + ".fused_qkv_a_proj_with_mqa")  # SGLang MLA
            fused.add(parent + ".fused_qkv_a_proj")  # vLLM MLA
    return fused


def _build_hf_quant_config(
    expert_banks: list[str],
    attn_modules: list[str],
    attn_fp8: bool,
    attn_mxfp8: bool = False,
    attn_fp8_pb: bool = False,
) -> dict[str, Any]:
    quantized_layers: dict[str, dict[str, Any]] = {}
    for name in expert_banks:
        for alias in _module_name_aliases(name):
            quantized_layers[alias] = {"quant_algo": "NVFP4", "group_size": _NVFP4_BLOCK}
    attention_algo = (
        "FP8" if attn_fp8 else "MXFP8" if attn_mxfp8 else "FP8_PB_WO" if attn_fp8_pb else None
    )
    if attention_algo:
        for name in attn_modules:
            for alias in _module_name_aliases(name):
                quantized_layers[alias] = {"quant_algo": attention_algo}
        # vLLM's K3 model names these packed modules directly. Its mixed
        # resolver does not currently infer either K3-specific fused name from
        # the individual HF projection policies.
        for name in _fused_attention_module_names(attn_modules):
            for alias in _module_name_aliases(name):
                quantized_layers[alias] = {"quant_algo": attention_algo}

    exclude_modules: list[str] = []
    for pattern in _EXCLUDE_MODULES:
        if attn_fp8_pb and pattern == "*self_attn.f_b_proj*":
            continue
        exclude_modules.extend(_module_name_aliases(pattern))

    return {
        "producer": {"name": "modelopt", "version": modelopt_version},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "group_size": _NVFP4_BLOCK,
            "quantized_layers": quantized_layers,
            "exclude_modules": list(dict.fromkeys(exclude_modules)),
        },
    }


def _rewrite_config_json(src: Path, dst_dir: Path, hf_quant_config: dict[str, Any]) -> None:
    """Copy ``config.json`` and replace the source MXFP4 manifest.

    The source ``quantization_config`` is compressed-tensors ``mxfp4-pack-quantized``
    and describes weights this script has just replaced, so leaving it in place
    would make a loader dequantize the NVFP4 experts as MXFP4. It is replaced
    wholesale by the ModelOpt mixed-precision manifest.
    """
    cfg = json.loads(src.read_text())
    quant_cfg = convert_hf_quant_config_format(hf_quant_config)
    # ``convert_hf_quant_config_format`` targets the llm-compressor layout and
    # stamps ``quant_method="modelopt"``. Loaders gate their mixed-precision
    # path on ``"modelopt_mixed"`` specifically (SGLang:
    # ``ModelOptMixedPrecisionConfig.override_quantization_method``), so with
    # the generic value the manifest is silently ignored and every quantized
    # module falls back to the unquantized method -- which for a 2.8T MoE means
    # allocating BF16 experts and OOMing at load.
    if quant_cfg.get("quant_algo") == "MIXED_PRECISION":
        quant_cfg["quant_method"] = "modelopt_mixed"
    # ``text_config`` carries the source MXFP4 manifest for this model.
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        text_cfg.pop("quantization_config", None)
    cfg["quantization_config"] = quant_cfg
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def _write_index_and_manifest(
    output_ckpt: Path,
    src_index: dict,
    results: list[dict[str, Any]],
    hf_quant_config: dict[str, Any],
    attn_fp8: bool,
    attn_mxfp8: bool = False,
    attn_fp8_pb: bool = False,
) -> None:
    converted_shards = {r["shard"] for r in results}
    weight_map: dict[str, str] = {}
    for key, shard in src_index["weight_map"].items():
        if shard not in converted_shards:
            continue

        expert = _EXPERT_PACKED_RE.match(key)
        if expert:
            base = expert.group("base")
            weight_map[base + ".weight"] = shard
            weight_map[base + ".weight_scale_2"] = shard
            weight_map[base + ".input_scale"] = shard
            continue

        weight_map[key] = shard
        attn = _ATTN_WEIGHT_RE.match(key)
        attention_projections = _ATTN_FP8_PB_PROJ if attn_fp8_pb else _ATTN_QUANT_PROJ
        if (
            (attn_fp8 or attn_mxfp8 or attn_fp8_pb)
            and attn
            and attn.group("proj") in attention_projections
        ):
            base = attn.group("base")
            weight_map[base + ".weight_scale"] = shard
            if attn_fp8:
                weight_map[base + ".input_scale"] = shard

    # Source expert ``weight_scale`` keys remain in ``weight_map`` and now
    # describe NVFP4's per-16-element E4M3 scale rather than MXFP4's per-32
    # E8M0 scale.
    metadata = dict(src_index.get("metadata", {}))
    metadata["total_size"] = sum(r["tensor_bytes"] for r in results)
    new_index = {"metadata": metadata, "weight_map": weight_map}
    (output_ckpt / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    _log(f"[index] wrote model.safetensors.index.json ({len(weight_map)} keys)")

    (output_ckpt / "hf_quant_config.json").write_text(json.dumps(hf_quant_config, indent=2))


def _write_json_atomic(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _conversion_fingerprint(args: argparse.Namespace, shards: list[Path]) -> dict[str, Any]:
    """Return the settings that must match across every conversion rank."""
    return {
        "source_ckpt": str(args.source_ckpt.resolve()),
        "shards": [str(shard.resolve()) for shard in shards],
        "device": args.device,
        "cast_mxfp4_to_nvfp4": args.cast_mxfp4_to_nvfp4,
        "attn_fp8": args.attn_fp8,
        "attn_mxfp8": args.attn_mxfp8,
        "attn_fp8_pb": args.attn_fp8_pb,
        "input_scale": args.input_scale,
    }


def _validate_fingerprint(
    published: dict[str, Any] | None,
    expected: dict[str, Any],
    description: str,
) -> None:
    published = published or {}
    mismatched = sorted(
        key for key in published.keys() | expected.keys() if published.get(key) != expected.get(key)
    )
    if mismatched:
        raise ValueError(f"{description} conversion fingerprint differs in: {mismatched}")


def _wait_for(
    predicate,
    description: str,
    timeout_s: float,
    poll_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description} after {timeout_s:.0f}s")
        time.sleep(poll_s)


def _rank0_ready(
    ready_path: Path,
    run_id: str,
    world_size: int,
    rank: int,
    fingerprint: dict[str, Any],
) -> bool:
    """Check that rank 0 published matching rendezvous settings."""
    if not ready_path.exists():
        return False
    ready = json.loads(ready_path.read_text())
    if ready.get("run_id") != run_id:
        return False
    published_world_size = ready.get("world_size")
    if published_world_size != world_size:
        raise ValueError(
            f"rank {rank} has --world_size {world_size}, but rank 0 published "
            f"{published_world_size} for run {run_id}"
        )
    _validate_fingerprint(ready.get("fingerprint"), fingerprint, f"rank {rank}")
    return True


def _rank_report_ready(
    report_path: Path,
    run_id: str,
    rank: int,
    fingerprint: dict[str, Any],
) -> bool:
    """Check that one rank atomically published a report for this conversion."""
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if report.get("run_id") != run_id:
        return False
    if report.get("rank") != rank:
        raise ValueError(f"expected rank {rank} report, got rank {report.get('rank')}")
    _validate_fingerprint(report.get("fingerprint"), fingerprint, f"rank {rank} report")
    return True


def _merge_rank_reports(
    rank_reports: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], set[str], set[str]]:
    results: list[dict[str, Any]] = []
    totals: dict[str, int] = defaultdict(int)
    banks: set[str] = set()
    attn_modules: set[str] = set()
    for report in rank_reports:
        results.extend(report["results"])
        for k, v in report["stats"].items():
            totals[k] += v
        banks.update(report["banks"])
        attn_modules.update(report["attn_modules"])
    return results, dict(totals), banks, attn_modules


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source_ckpt", type=Path, required=True, help="original HF Kimi-K3 release")
    p.add_argument("--output_ckpt", type=Path, required=True)
    p.add_argument(
        "--recipe",
        help=(
            "Kimi-K3 checkpoint-mirror recipe (recommended: "
            f"{_PUBLISHED_RECIPE}); cannot be combined with the low-level format flags"
        ),
    )
    p.add_argument(
        "--cast_mxfp4_to_nvfp4",
        action="store_true",
        help=(
            "closed-form bit-exact cast of the source MXFP4 routed-expert weights to "
            "NVFP4 (pin scale_2 = 2^(k_max-8), per-block scale = 2^(k_j-m) from the "
            "source E8M0 scales) instead of dequantizing and re-quantizing from data"
        ),
    )
    attention = p.add_mutually_exclusive_group()
    attention.add_argument(
        "--attn_fp8",
        action="store_true",
        help="quantize MLA/KDA attention projections to data-free per-tensor FP8",
    )
    attention.add_argument(
        "--attn_mxfp8",
        action="store_true",
        help="quantize MLA/KDA attention projections to calibration-free MXFP8",
    )
    attention.add_argument(
        "--attn_fp8_pb",
        action="store_true",
        help=(
            "quantize MLA/KDA attention projections to calibration-free 128x128 "
            "block FP8 weights with dynamic per-token activations"
        ),
    )
    p.add_argument(
        "--input_scale",
        type=float,
        default=None,
        help="fixed activation input_scale for every quantized module (default: 1.0)",
    )
    p.add_argument("--device", default="cpu", help="'cpu' (default) or 'cuda'")
    p.add_argument("--jobs", type=int, default=8, help="shards converted in parallel per rank")
    p.add_argument(
        "--threads_per_job", type=int, default=8, help="torch intra-op threads per worker"
    )
    p.add_argument(
        "--rank",
        type=int,
        default=0,
        help="distributed shard rank (use one rank per node; default 0)",
    )
    p.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="number of distributed shard ranks sharing --output_ckpt (default 1)",
    )
    p.add_argument(
        "--run_id",
        default=os.environ.get("SLURM_JOB_ID", "local"),
        help="rendezvous identifier shared by distributed ranks (defaults to SLURM_JOB_ID)",
    )
    p.add_argument(
        "--sync_timeout",
        type=float,
        default=24 * 60 * 60,
        help="seconds rank 0 waits for distributed ranks (default 24 hours)",
    )
    p.add_argument("--limit_shards", type=int, default=0, help="smoke test: convert only N shards")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.recipe:
        if (
            args.cast_mxfp4_to_nvfp4
            or args.attn_fp8
            or args.attn_mxfp8
            or args.attn_fp8_pb
            or args.input_scale is not None
        ):
            p.error("--recipe cannot be combined with the low-level format flags or --input_scale")
        try:
            settings = _conversion_settings_from_recipe(args.recipe)
        except (OSError, ValueError) as exc:
            p.error(f"invalid Kimi-K3 recipe: {exc}")
        for name, value in settings.items():
            setattr(args, name, value)
    if args.input_scale is None:
        args.input_scale = 1.0

    input_scale = float(args.input_scale)
    if not math.isfinite(input_scale) or input_scale <= 0:
        p.error("--input_scale must be finite and > 0")
    args.input_scale = input_scale
    if args.jobs <= 0:
        p.error("--jobs must be > 0")
    if args.threads_per_job <= 0:
        p.error("--threads_per_job must be > 0")
    if args.world_size <= 0:
        p.error("--world_size must be > 0")
    if not 0 <= args.rank < args.world_size:
        p.error("--rank must satisfy 0 <= rank < world_size")
    if args.limit_shards < 0:
        p.error("--limit_shards must be >= 0")
    if args.sync_timeout <= 0:
        p.error("--sync_timeout must be > 0")
    if args.device.startswith("cuda") and args.jobs > 1:
        p.error("--device cuda requires --jobs 1 so workers do not contend for one GPU")

    validate_paths(args.source_ckpt, args.output_ckpt)
    src_index_path = resolve_checkpoint_file(
        args.source_ckpt,
        "model.safetensors.index.json",
    )
    src_config_path = resolve_checkpoint_file(args.source_ckpt, "config.json")
    src_index = json.loads(src_index_path.read_text())

    shards = sorted(args.source_ckpt.glob("model-*-of-*.safetensors"))
    assert shards, f"no HF-style shards in {args.source_ckpt}"
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    validate_aux_files(
        args.source_ckpt,
        skip_top_level=_SKIP_TOP_LEVEL,
        skip_dir_names=_SKIP_SUBDIR_NAMES | _SKIP_TOP_LEVEL,
        skip_file=lambda path: path.suffix == ".safetensors",
    )
    fingerprint = _conversion_fingerprint(args, shards)

    marker_dir = args.output_ckpt / ".kimi_k3_conversion"
    ready_path = marker_dir / "ready.json"
    if args.rank == 0:
        prepare_output_dir(args.output_ckpt, args.overwrite)
        marker_dir.mkdir()
        _write_json_atomic(
            ready_path,
            {
                "run_id": args.run_id,
                "world_size": args.world_size,
                "fingerprint": fingerprint,
            },
        )
    else:
        _wait_for(
            lambda: _rank0_ready(
                ready_path,
                args.run_id,
                args.world_size,
                args.rank,
                fingerprint,
            ),
            f"rank-0 rendezvous for run {args.run_id}",
            args.sync_timeout,
        )

    assigned_shards = shards[args.rank :: args.world_size]
    _log(
        f"[config] rank={args.rank}/{args.world_size}  "
        f"shards={len(assigned_shards)}/{len(shards)}  device={args.device}  jobs={args.jobs}  "
        f"cast={args.cast_mxfp4_to_nvfp4}  attn_fp8={args.attn_fp8}  "
        f"attn_mxfp8={args.attn_mxfp8}  attn_fp8_pb={args.attn_fp8_pb}  "
        f"input_scale={args.input_scale}"
    )

    # Members of a fused attention group may live in different shards, so the
    # shared amax is derived in a pass over all shards before any conversion.
    fused_attn_amax = compute_fused_attn_amax(shards, args.device) if args.attn_fp8 else {}

    jobs = [
        {
            "src": str(s),
            "dst": str(args.output_ckpt / s.name),
            "device": args.device,
            "cast": args.cast_mxfp4_to_nvfp4,
            "attn_fp8": args.attn_fp8,
            "attn_mxfp8": args.attn_mxfp8,
            "attn_fp8_pb": args.attn_fp8_pb,
            "input_scale": args.input_scale,
            "threads": args.threads_per_job,
            "fused_attn_amax": fused_attn_amax,
        }
        for s in assigned_shards
    ]

    if args.jobs > 1 and len(jobs) > 1:
        # Use a 'spawn' pool, not the default 'fork'. The --attn_fp8 path runs
        # compute_fused_attn_amax() in the parent first, which initializes
        # torch's intra-op thread pool; forking after that leaves the workers
        # with an inherited-but-dead thread state and they hang at 0% CPU
        # before doing any work. 'spawn' starts each worker from a clean
        # interpreter, at the cost of re-importing torch per worker.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            results = list(ex.map(_convert_shard_task, jobs))
    else:
        results = [_convert_shard_task(j) for j in jobs]

    local_totals: dict[str, int] = defaultdict(int)
    local_banks: set[str] = set()
    local_attn_modules: set[str] = set()
    for r in results:
        for k, v in r["stats"].items():
            local_totals[k] += v
        local_banks.update(r["banks"])
        local_attn_modules.update(r["attn_modules"])

    rank_report = {
        "run_id": args.run_id,
        "rank": args.rank,
        "fingerprint": fingerprint,
        "results": results,
        "stats": dict(local_totals),
        "banks": sorted(local_banks),
        "attn_modules": sorted(local_attn_modules),
    }
    rank_path = marker_dir / f"rank-{args.rank:05d}.json"
    _write_json_atomic(rank_path, rank_report)

    if args.rank != 0:
        _log(f"[rank {args.rank}] complete; rank 0 will finalize checkpoint metadata")
        return

    rank_paths = [marker_dir / f"rank-{rank:05d}.json" for rank in range(args.world_size)]

    def all_ranks_done() -> bool:
        return all(
            _rank_report_ready(path, args.run_id, rank, fingerprint)
            for rank, path in enumerate(rank_paths)
        )

    _wait_for(all_ranks_done, f"{args.world_size} rank reports", args.sync_timeout)
    rank_reports = [json.loads(path.read_text()) for path in rank_paths]
    for rank, report in enumerate(rank_reports):
        if report.get("run_id") != args.run_id or report.get("rank") != rank:
            raise RuntimeError(f"rank {rank} report changed after rendezvous validation")
        _validate_fingerprint(report.get("fingerprint"), fingerprint, f"rank {rank} report")
    results, totals, banks, attn_modules = _merge_rank_reports(rank_reports)
    if len(results) != len(shards):
        raise RuntimeError(f"expected {len(shards)} converted shards, got {len(results)}")

    _log("[stats]")
    for k in sorted(totals):
        _log(f"  {k:32s} {totals[k]}")

    if args.cast_mxfp4_to_nvfp4:
        tot = totals.get("cast_blocks_total", 0)
        loss = totals.get("cast_blocks_lossless", 0)
        pct = 100.0 * loss / tot if tot else 100.0
        _log(f"[cast] lossless MXFP4->NVFP4 blocks: {loss}/{tot} ({pct:.6f}%)")
        _log(f"[cast] tensors with out-of-range blocks: {totals.get('cast_oor_tensors', 0)}")

    hf_quant_config = _build_hf_quant_config(
        sorted(banks),
        sorted(attn_modules),
        args.attn_fp8,
        args.attn_mxfp8,
        args.attn_fp8_pb,
    )
    _write_index_and_manifest(
        args.output_ckpt,
        src_index,
        results,
        hf_quant_config,
        args.attn_fp8,
        args.attn_mxfp8,
        args.attn_fp8_pb,
    )
    attention_algo = (
        "FP8"
        if args.attn_fp8
        else "MXFP8"
        if args.attn_mxfp8
        else "FP8_PB_WO"
        if args.attn_fp8_pb
        else "BF16"
    )
    _log(
        f"[config] rewriting config.json "
        f"(MIXED_PRECISION: NVFP4 experts + {attention_algo} attention)"
    )
    _rewrite_config_json(src_config_path, args.output_ckpt, hf_quant_config)
    _log(f"[aux] linking ancillary files from {args.source_ckpt}")
    link_aux_files(
        args.source_ckpt,
        args.output_ckpt,
        skip_top_level=_SKIP_TOP_LEVEL,
        skip_dir_names=_SKIP_SUBDIR_NAMES | _SKIP_TOP_LEVEL,
        skip_file=lambda path: path.suffix == ".safetensors",
    )
    _write_json_atomic(
        args.output_ckpt / "conversion_report.json",
        {
            "run_id": args.run_id,
            "recipe": args.recipe,
            "world_size": args.world_size,
            "conversion_fingerprint": fingerprint,
            "shards": len(results),
            "stats": totals,
            "expert_banks": len(banks),
            "attention_modules": len(attn_modules),
            "cast_mxfp4_to_nvfp4": args.cast_mxfp4_to_nvfp4,
            "attn_fp8": args.attn_fp8,
            "attn_mxfp8": args.attn_mxfp8,
            "attn_fp8_pb": args.attn_fp8_pb,
            "attention_quant_algo": attention_algo,
            "input_scale": args.input_scale,
        },
    )
    shutil.rmtree(marker_dir)
    _log(
        f"[done] {args.output_ckpt}  "
        f"({len(banks)} expert banks NVFP4, "
        f"{len(attn_modules)} attention modules {attention_algo})"
    )


if __name__ == "__main__":
    main()
