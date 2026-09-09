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

"""Closed-form cast from MXFP4 source to NVFP4 weight quantizer state.

Reads ``*_scales`` (E8M0 per-MXFP4-block exponents) and ``*_blocks`` (packed
E2M1 nibbles) from a Hugging Face checkpoint with
``quantization_config.quant_method == "mxfp4"`` (e.g. OpenAI's gpt-oss family)
and produces, per source layer:

* a per-tensor ``global_amax = 6 * 448 * 2^(k_max - 8)`` that pins NVFP4's
  ``scale_2`` to ``2^m`` (an exact power of 2, exactly representable in E4M3);
* a per-NVFP4-block ``_amax`` that is bit-exact (``6 * 2^k_j``) for blocks
  whose ``k_j`` lands in E4M3's representable window, and falls back to the
  data-derived ``max(|w_block|) = max_nibble * 2^k_j`` for out-of-range blocks
  (where the per-block scale would clamp anyway).

Together these guarantee NVFP4 dequant matches MXFP4 dequant bit-for-bit on
every in-range block, and minimizes per-block error on out-of-range ones.
Reads only the scales (~150 MB for gpt-oss-20b) plus the packed nibbles for
out-of-range blocks; runs in seconds.
"""

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

from modelopt.torch.quantization.nn.modules.tensor_quantizer import NVFP4StaticQuantizer
from modelopt.torch.quantization.utils.numeric_utils import (
    E2M1_MAX,
    E8M0_BIAS,
    mxfp4_to_nvfp4_global_amax,
    mxfp4_to_nvfp4_per_block_amax,
)


@contextmanager
def _shard_reader():
    """Yield a ``read(key, shard) -> tensor`` closure with cached safetensors handles.

    Each unique shard is opened lazily on first read and closed deterministically
    when the context exits, so callers don't need to manage the handle cache or
    the surrounding ``ExitStack`` themselves.
    """
    with ExitStack() as stack:
        handles: dict[Path, safe_open] = {}

        def read(key: str, shard: Path) -> torch.Tensor:
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
            return handles[shard].get_tensor(key)

        yield read


def quantizer_name_from_blocks_key(blocks_key: str) -> str:
    """Map ``<base>_blocks`` -> ``<base>_weight_quantizer``.

    OpenAI's MXFP4 checkpoint convention stores packed weights as
    ``<name>_blocks`` and scales as ``<name>_scales``. modelopt's
    ``GptOssExperts`` wrapper attaches the weight quantizer at
    ``<name>_weight_quantizer``.
    """
    assert blocks_key.endswith("_blocks"), f"Unexpected key {blocks_key!r}"
    return blocks_key[: -len("_blocks")] + "_weight_quantizer"


def _collect_keys_with_suffix(ckpt_dir: Path, suffix: str) -> dict[str, Path]:
    """Return ``{tensor_name: shard_path}`` for every key ending with ``suffix``."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as f:
            index = json.load(f)
        return {
            k: ckpt_dir / shard for k, shard in index["weight_map"].items() if k.endswith(suffix)
        }
    shards = list(ckpt_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(
            f"Expected model.safetensors.index.json or a single .safetensors file in {ckpt_dir}"
        )
    out: dict[str, Path] = {}
    with safe_open(shards[0], framework="pt") as f:
        # ``safe_open`` is not a dict; ``.keys()`` is its iterator.
        for k in f.keys():  # noqa: SIM118
            if k.endswith(suffix):
                out[k] = shards[0]
    return out


def _collect_scales_keys(ckpt_dir: Path) -> dict[str, Path]:
    """Return ``{tensor_name: shard_path}`` for every ``*_scales`` key."""
    return _collect_keys_with_suffix(ckpt_dir, "_scales")


def build_amax_map(checkpoint_dir: str | Path) -> dict[str, dict]:
    """Walk the source MXFP4 checkpoint and build the per-layer amax map.

    Args:
        checkpoint_dir: Path to a Hugging Face checkpoint directory whose
            ``quantization_config.quant_method`` is ``"mxfp4"`` (OpenAI layout
            with ``*_blocks`` + ``*_scales`` tensors).

    Returns:
        ``{quantizer_name: {"global_amax": float, "k_min": int, "k_max": int,
                            "m": int, "n_total_blocks": int,
                            "n_lossless_blocks": int, "pct_lossless": float,
                            "n_zero_blocks": int}}``

        Quantizer names match ``model.named_modules()`` after modelopt
        instrumentation (e.g. ``model.layers.0.mlp.experts.gate_up_proj_weight_quantizer``).

    Raises:
        SystemExit: if no ``*_scales`` tensors are found (not an MXFP4 checkpoint).
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    scales_keys = _collect_scales_keys(ckpt_dir)
    if not scales_keys:
        raise SystemExit(
            f"No '*_scales' tensors found in {ckpt_dir}. "
            "This requires an MXFP4 HF checkpoint with the OpenAI layout."
        )

    amax_map: dict[str, dict] = {}
    with _shard_reader() as read:
        for tensor_key, shard in sorted(scales_keys.items()):
            scales = read(tensor_key, shard)

            global_amax, info = mxfp4_to_nvfp4_global_amax(scales)

            blocks_key = tensor_key[: -len("_scales")] + "_blocks"
            qname = quantizer_name_from_blocks_key(blocks_key)
            amax_map[qname] = {"global_amax": global_amax, **info}

    return amax_map


def force_weight_quantizers_static(quant_cfg: list) -> None:
    """Force every weight-quantizer entry's ``block_sizes`` to ``type='static'``.

    The MXFP4 -> NVFP4 cast needs the per-block weight ``_amax`` to be recorded
    by max-cal (so it can be paired with the pinned global_amax later). Setting
    ``block_sizes['type'] = 'static'`` makes ``is_static_block_quant`` True so
    static NVFP4 finalization picks the entry up automatically at the end of
    max_calibrate.
    """
    for i, entry in enumerate(quant_cfg):
        qname = entry.get("quantizer_name", "")
        cfg = entry.get("cfg") or {}
        bs = cfg.get("block_sizes")
        if "weight_quantizer" in qname and isinstance(bs, dict):
            quant_cfg[i] = {**entry, "cfg": {**cfg, "block_sizes": {**bs, "type": "static"}}}


def apply_to_model(
    model: "torch.nn.Module",
    source_checkpoint_path: str | Path,
) -> None:
    """Closed-form cast: bit-exact MXFP4 -> NVFP4 weight conversion.

    Reads the source MXFP4 ``*_scales`` from ``source_checkpoint_path`` and
    overrides two buffers on each matching NVFP4 weight quantizer:

    1. ``global_amax`` = ``6 * 448 * 2^(k_max - 8)`` (closed-form scalar —
       pins ``scale_2 = 2^m``).
    2. ``_amax`` = ``6 * 2^k_j`` per NVFP4 block (closed-form per-block — pins
       ``per_block_scale = 2^(k_j - m)``, exactly representable in E4M3).

    Together these guarantee that ``per_block_scale * scale_2 = 2^k_j`` exactly,
    so the NVFP4 dequant produces ``nibble * 2^k_j`` — the same value as the
    MXFP4 dequant. End-to-end the weight conversion is bit-exact for every
    block whose ``k_j`` lands in E4M3's representable range (``k_max - k_j <= 17``).

    The weight quantizer is expected to be an :class:`NVFP4StaticQuantizer`
    (:func:`max_calibrate` auto-promotes static-block NVFP4 weight quantizers
    at the end of calibration). Both ``_amax`` (per-block from max-cal) and
    ``_global_amax`` (per-tensor from the auto-promotion) get overwritten.
    """
    ckpt_dir = Path(source_checkpoint_path)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    scales_keys = _collect_scales_keys(ckpt_dir)
    if not scales_keys:
        raise SystemExit(
            f"No '*_scales' tensors found in {ckpt_dir}. "
            "This requires an MXFP4 HF checkpoint with the OpenAI layout."
        )

    blocks_keys = _collect_keys_with_suffix(ckpt_dir, "_blocks")

    name_to_module = dict(model.named_modules())
    matched = 0
    missed: list[str] = []

    n_total_layers = 0
    n_lossless_layers = 0
    grand_total_blocks = 0
    grand_lossless_blocks = 0

    with _shard_reader() as read:
        for tensor_key, shard in sorted(scales_keys.items()):
            scales = read(tensor_key, shard)

            global_amax_value, info = mxfp4_to_nvfp4_global_amax(scales)
            n_total_layers += 1
            if info["pct_lossless"] >= 100.0:
                n_lossless_layers += 1
            grand_total_blocks += info["n_total_blocks"]
            grand_lossless_blocks += info["n_lossless_blocks"]

            blocks_key = tensor_key[: -len("_scales")] + "_blocks"
            qname = quantizer_name_from_blocks_key(blocks_key)
            blocks_shard = blocks_keys.get(blocks_key)
            assert blocks_shard is not None, (
                f"{tensor_key}: no paired '{blocks_key}' tensor found in source checkpoint."
            )

            weight_quantizer = name_to_module.get(qname)
            if weight_quantizer is None:
                missed.append(qname)
                continue

            # The cast assumes ``max_calibrate`` already promoted this quantizer
            # to NVFP4StaticQuantizer (with ``_amax`` populated per-block by
            # static-block max-cal and ``_global_amax`` set by the auto-promote).
            # Anything else means the qformat or quant_cfg disabled this module's
            # weight quantization — surface that loudly so we don't silently no-op.
            assert isinstance(weight_quantizer, NVFP4StaticQuantizer), (
                f"{qname}: expected NVFP4StaticQuantizer (set by max_calibrate's "
                f"auto-promote), got {type(weight_quantizer).__name__}. The cast "
                "requires the matching quantizer to be enabled with static-block "
                "NVFP4 (num_bits=(2,1), scale_bits=(4,3))."
            )
            existing = getattr(weight_quantizer, "_amax", None)
            assert isinstance(existing, torch.Tensor) and existing.numel() > 1, (
                f"{qname}: NVFP4StaticQuantizer must have a per-block ``_amax`` "
                f"buffer populated by max_calibrate. Got: {existing!r}."
            )

            # Pick the device from the existing per-block ``_amax`` buffer.
            device = existing.device

            global_amax = torch.tensor(float(global_amax_value), dtype=torch.float32, device=device)
            # Fully-lossless layers don't need the packed ``*_blocks`` tensor —
            # the per-block amax is just ``6 * 2^k_j`` from ``scales`` alone, and
            # avoiding the (16x larger) block read is the main I/O win the
            # closed-form path is designed for.
            if info["pct_lossless"] >= 100.0:
                k = scales.to(torch.int32) - E8M0_BIAS
                per_block_amax = (
                    (E2M1_MAX * torch.exp2(k.float()))
                    .repeat_interleave(2, dim=-1)
                    .to(dtype=torch.float32, device=device)
                )
            else:
                blocks = read(blocks_key, blocks_shard)
                per_block_amax = mxfp4_to_nvfp4_per_block_amax(blocks, scales).to(
                    dtype=torch.float32, device=device
                )
            # Numel must match — calibration may store ``_amax`` flat (e.g. (N, 1))
            # while we compute it in natural (E, F, num_blocks) layout. The static
            # export path reshapes via ``.view(expected_shape)``, so we just need
            # element count to agree, then reshape for the in-place copy.
            assert existing.numel() == per_block_amax.numel(), (
                f"{qname}: ``_amax`` element count {existing.numel()} does not "
                f"match the cast-computed count {per_block_amax.numel()}. The "
                "block layout from calibration disagrees with the source MXFP4 "
                "scales — check that the qformat block_size is 16 and the source "
                "checkpoint is the same MXFP4 model."
            )

            # global_amax via the NVFP4StaticQuantizer property setter (writes to
            # the canonical ``_global_amax`` buffer).
            weight_quantizer.global_amax = global_amax
            # _amax: in-place buffer copy, reshaping our value to the calibrator's
            # storage layout (numel verified above).
            with torch.no_grad():
                existing.data.copy_(per_block_amax.view_as(existing))

            matched += 1

    print(
        f"[cast_mxfp4_to_nvfp4] overrode {matched}/{n_total_layers} weight quantizers from {source_checkpoint_path}"
    )
    if missed:
        print(
            f"[cast_mxfp4_to_nvfp4] warning: {len(missed)} layers had no matching module. "
            f"First few: {missed[:5]}"
        )
    layer_pct = 100.0 * n_lossless_layers / n_total_layers if n_total_layers else 100.0
    block_pct = 100.0 * grand_lossless_blocks / grand_total_blocks if grand_total_blocks else 100.0
    print(
        f"[cast_mxfp4_to_nvfp4] lossless layers: {n_lossless_layers}/{n_total_layers} ({layer_pct:.2f}%)"
    )
    print(
        f"[cast_mxfp4_to_nvfp4] lossless blocks: {grand_lossless_blocks}/{grand_total_blocks} ({block_pct:.4f}%)"
    )
