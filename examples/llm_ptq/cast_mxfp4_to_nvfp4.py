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


E8M0_BIAS = 127  # E8M0 stores k_j as uint8 with bias 127
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_KMAX = 8
E4M3_KMIN = -9  # E4M3 represents 2^k exactly for k in [-9, 8]
# E2M1 magnitude grid indexed by the low 3 bits of an FP4 nibble.
_E2M1_MAGNITUDE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Cache of the E2M1 magnitude lookup table per (device, dtype) so we don't
# rebuild it for every layer in a batched cast.
_E2M1_MAG_CACHE: "dict[tuple, torch.Tensor]" = {}


def _e2m1_magnitude_table(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return ``_E2M1_MAGNITUDE`` as a tensor on the requested device, cached."""
    key = (device, dtype)
    cached = _E2M1_MAG_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(_E2M1_MAGNITUDE, dtype=dtype, device=device)
        _E2M1_MAG_CACHE[key] = cached
    return cached


def compute_global_amax_for_scales(e8m0_scales: torch.Tensor) -> tuple[float, dict]:
    """Closed-form per-tensor ``global_amax``: ``m = k_max - 8``, ``global_amax = 6 * 448 * 2^m``.

    Args:
        e8m0_scales: uint8 tensor of E8M0 scales for one MXFP4 source layer.

    Returns:
        global_amax: scalar (float) — pins NVFP4 scale_2 to 2^m.
        info: diagnostic dict with k_min, k_max, m, lossless-block stats.
    """
    # k_j = e8m0 - 127. MXFP4 quantize emits e8m0=0 (=> k=-127) for all-zero
    # blocks; treat those as "ignore me" when computing k_max.
    k = e8m0_scales.to(torch.int32) - E8M0_BIAS
    nonzero_mask = e8m0_scales > 0
    if nonzero_mask.any():
        k_nonzero = k[nonzero_mask]
        k_min = int(k_nonzero.min().item())
        k_max = int(k_nonzero.max().item())
    else:
        k_min = k_max = 0

    m = k_max - E4M3_KMAX
    global_amax = E2M1_MAX * E4M3_MAX * float(2.0**m)

    # A block is lossless under this cast iff k_max - k_j <= 17 (its k_j - m sits
    # in E4M3's [-9, 8] window). All-zero blocks are trivially lossless because
    # their reconstruction is 0 regardless of the snapped scale.
    n_total = e8m0_scales.numel()
    in_range = (k >= (k_max - 17)) | (~nonzero_mask)
    n_lossless = int(in_range.sum().item())
    pct_lossless = 100.0 * n_lossless / n_total if n_total else 100.0

    return global_amax, {
        "k_min": k_min,
        "k_max": k_max,
        "m": m,
        "n_total_blocks": n_total,
        "n_lossless_blocks": n_lossless,
        "pct_lossless": pct_lossless,
        "n_zero_blocks": int((~nonzero_mask).sum().item()),
    }


def compute_per_block_amax_for_mxfp4(
    blocks: torch.Tensor, e8m0_scales: torch.Tensor
) -> torch.Tensor:
    """Hybrid per-NVFP4-block amax for MXFP4 -> NVFP4 cast.

    Each MXFP4 block of 32 elements has one E8M0 exponent ``k_j``. Two cases
    based on whether ``k_j`` fits in NVFP4's E4M3 scale grid (with
    ``m = k_max - 8`` chosen by ``compute_global_amax_for_scales``):

    - **In-range** (``k_j - m`` in ``[-9, 8]``): ``6 * 2^k_j`` (closed-form
      ideal). The resulting per-block scale ``2^(k_j - m)`` is exactly
      representable in E4M3 — no rounding loss — and
      ``round_to_E2M1(value / 2^k_j)`` yields the original MXFP4 nibble
      verbatim. Bit-exact reconstruction.

    - **Out of range** (``|k_j - m| > 8/9``): ``max_nibble * 2^k_j``, i.e.
      ``max(|w_block|)`` where ``w`` is the MXFP4-dequantized block. This is
      the data-derived per-block amax. The per-block scale will still get
      clamped at the E4M3 boundary, but data-derived amax keeps the post-clamp
      scale closer to the block's actual magnitude than the closed-form ideal
      would, which reduces re-bucketing error for OOR blocks where
      ``max_nibble < 6``.

    Two NVFP4 blocks of 16 share each MXFP4 block's ``k_j``, so the result is
    expanded by ``repeat_interleave(2, dim=-1)``.

    Args:
        blocks: uint8 tensor of packed E2M1 nibbles, shape
            ``(..., num_mxfp4_blocks, 16)`` (16 bytes per 32-element MXFP4 block).
        e8m0_scales: uint8 tensor of E8M0 scales, shape
            ``(..., num_mxfp4_blocks)``.

    Returns:
        float32 tensor of shape ``(..., 2 * num_mxfp4_blocks)``.
    """
    if blocks.shape[-1] != 16 or blocks.shape[:-1] != e8m0_scales.shape:
        raise ValueError(
            f"shape mismatch: blocks {tuple(blocks.shape)} "
            "(expected (..., num_mxfp4_blocks, 16)) "
            f"vs scales {tuple(e8m0_scales.shape)}"
        )

    k = e8m0_scales.to(torch.int32) - E8M0_BIAS  # (..., num_mxfp4_blocks)
    pow2_k = torch.exp2(k.float())
    closed_form_ideal = E2M1_MAX * pow2_k  # (..., num_mxfp4_blocks)

    # ``m = k_max - 8`` over non-zero blocks. Compute via masked ``amax`` so
    # ``m`` stays a 0-d tensor and we avoid a GPU->CPU sync just to get a
    # Python int. All-zero scales fall through with the -E8M0_BIAS sentinel,
    # which leaves every block trivially in-range (closed_form_ideal == 0 there).
    nonzero = e8m0_scales > 0
    sentinel = torch.full_like(k, -E8M0_BIAS)
    k_max = torch.where(nonzero, k, sentinel).amax()
    delta = k - (k_max - E4M3_KMAX)
    in_range = (delta >= E4M3_KMIN) & (delta <= E4M3_KMAX)

    # Fast path: if every block fits E4M3's [-9, 8] window the per-block amax
    # is just the closed-form ideal, and we can skip the per-byte nibble scan
    # over the block tensor (which is 16x larger than the scales). For typical
    # MXFP4 checkpoints (e.g. gpt-oss-20b) this is the only path ever taken.
    if bool(in_range.all()):
        return closed_form_ideal.repeat_interleave(2, dim=-1)

    # OOR fallback: data-derived per-block amax = max(|w_block|) after MXFP4
    # dequant = ``max_nibble * 2^k_j``. The MXFP4 nibble is sign-magnitude with
    # sign in bit 3 and magnitude index in bits 0-2; we extract per-byte
    # magnitudes, take the byte-wise max, then reduce across the 16 bytes to
    # get the largest magnitude index in the 32-element block.
    low = blocks & 0x07
    high = (blocks >> 4) & 0x07
    max_idx = torch.maximum(low, high).amax(dim=-1).long()
    max_nibble = _e2m1_magnitude_table(blocks.device)[max_idx]
    data_derived = max_nibble * pow2_k

    per_block_amax_mxfp4 = torch.where(in_range, closed_form_ideal, data_derived)
    # Each MXFP4 block of 32 splits into two NVFP4 blocks of 16 sharing k_j.
    return per_block_amax_mxfp4.repeat_interleave(2, dim=-1)


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

            global_amax, info = compute_global_amax_for_scales(scales)

            blocks_key = tensor_key[: -len("_scales")] + "_blocks"
            qname = quantizer_name_from_blocks_key(blocks_key)
            amax_map[qname] = {"global_amax": global_amax, **info}

    return amax_map


def force_weight_quantizers_static(quant_cfg: list) -> None:
    """Force every weight-quantizer entry's ``block_sizes`` to ``type='static'``.

    The MXFP4 -> NVFP4 cast needs the per-block weight ``_amax`` to be recorded
    by max-cal (so it can be paired with the pinned global_amax later). Setting
    ``block_sizes['type'] = 'static'`` makes ``is_static_block_quant`` True so
    ``promote_nvfp4_static_quantizers`` picks the entry up automatically at the
    end of max_calibrate.
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

            global_amax_value, info = compute_global_amax_for_scales(scales)
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
                per_block_amax = compute_per_block_amax_for_mxfp4(blocks, scales).to(
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
