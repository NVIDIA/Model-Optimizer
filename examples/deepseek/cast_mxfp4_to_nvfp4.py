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

"""Re-emit a DeepSeek NVFP4-experts checkpoint with the MXFP4 -> NVFP4 closed-form cast.

Reads the original MXFP4 source checkpoint (e.g. ``deepseek-ai/DeepSeek-V4-Flash``)
and an existing NVFP4-experts checkpoint produced by the standard PTQ flow
(here ``flash-nvfp4-experts-v3``) and writes a new NVFP4 checkpoint
(``flash-nvfp4-experts-v3.5``) where the *expert* weight quantization is
replaced by the closed-form cast, run as a Triton GPU kernel:

  * ``weight_scale_2 = 2^m`` per tensor; ``m = max(k_max_w1, k_max_w3) - 8``
    for w1/w3 pairs (so SwiGLU fusion sees a shared global scale), and
    ``m = k_max - 8`` otherwise.
  * ``weight_scale   = 2^(k_j - m)`` as E4M3 — bit-exact for blocks where
    ``k_j - m`` lands in E4M3's representable window, clamped + re-quantized
    for out-of-range (OOR) blocks.
  * ``weight``         — bit-exact copy of the MXFP4 nibbles for in-range
    blocks; re-quantized only for OOR blocks.

Everything else (input_scale, MXFP8 attn/shared_experts tensors, BF16 norms,
config files, etc.) is copied byte-for-byte from v3.

Per-tensor MSE between MXFP4 dequant and NVFP4 dequant is reported for both
the original v3 export and the new v3.5 export, so the cast's reconstruction
gain is visible at the end of the run. v3.5's MSE is exactly 0 by
construction for any tensor flagged ``is_lossless`` in the pre-pass.

Requires CUDA + Triton (the cast is GPU-only).

Example:

  python examples/deepseek/cast_mxfp4_to_nvfp4.py \\
      --mxfp4_src ~/data_3/hf-local/deepseek-ai/DeepSeek-V4-Flash \\
      --nvfp4_src ~/Workspace/models/dsv4/flash-nvfp4-experts-v3 \\
      --nvfp4_dst ~/Workspace/models/dsv4/flash-nvfp4-experts-v3.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

import torch
import triton
import triton.language as tl
from tqdm import tqdm

# --- Self-contained safetensors I/O ----------------------------------------
# DeepSeek-V4 MXFP4 checkpoints store block scales as ``F8_E8M0`` (an MX-format
# dtype). ``safetensors < 0.6`` doesn't recognise that dtype — calling
# ``safe_open`` on those shards raises ``InvalidHeaderDeserialization``. Since
# E8M0 is just an unsigned 8-bit exponent (byte-for-byte equivalent to uint8),
# we read it (and every other tensor in the shard) ourselves through a small
# raw reader and write the output shard the same way, preserving the exact
# dtype strings end-to-end.
_DTYPE_INFO: dict[str, tuple[int, "torch.dtype | None"]] = {
    # name: (bytes_per_element, torch_dtype_or_None)
    "BOOL": (1, torch.bool),
    "U8": (1, torch.uint8),
    "I8": (1, torch.int8),
    "U16": (2, None),
    "I16": (2, torch.int16),
    "F16": (2, torch.float16),
    "BF16": (2, torch.bfloat16),
    "U32": (4, None),
    "I32": (4, torch.int32),
    "F32": (4, torch.float32),
    "U64": (8, None),
    "I64": (8, torch.int64),
    "F64": (8, torch.float64),
    "F8_E4M3": (1, torch.float8_e4m3fn),
    "F8_E5M2": (1, torch.float8_e5m2),
    "F8_E8M0": (1, torch.uint8),  # E8M0 is byte-for-byte uint8.
}


def _torch_dtype_to_st_name(dtype: torch.dtype) -> str:
    for name, (_, td) in _DTYPE_INFO.items():
        if td is dtype:
            return name
    raise ValueError(f"Unsupported torch dtype for safetensors save: {dtype}")


class SafeShardReader:
    """Tensor-only safetensors reader that preserves raw dtype names.

    Bypasses ``safetensors.safe_open`` so we can load shards containing dtypes
    the installed safetensors library doesn't know about (e.g. ``F8_E8M0``
    in DeepSeek-V4 MXFP4 shards under safetensors 0.5.x). E8M0 is byte-for-byte
    ``uint8``, so it rides as a torch ``uint8`` tensor; the original safetensors
    dtype name is returned alongside the tensor by ``get_named_tensor`` and
    forwarded to the writer so the round-trip preserves the dtype label.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self._header = json.loads(self._fh.read(n).decode("utf-8"))
        self._data_start = 8 + n
        self._meta = self._header.pop("__metadata__", None)

    def __enter__(self) -> "SafeShardReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def keys(self) -> list[str]:
        return list(self._header.keys())

    def get_dtype_name(self, key: str) -> str:
        return self._header[key]["dtype"]

    def get_shape(self, key: str) -> list[int]:
        return list(self._header[key]["shape"])

    def get_tensor(self, key: str, device: torch.device | str = "cpu") -> torch.Tensor:
        info = self._header[key]
        start, end = info["data_offsets"]
        dtype_name = info["dtype"]
        shape = info["shape"]
        if dtype_name not in _DTYPE_INFO:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")
        _, td = _DTYPE_INFO[dtype_name]
        if td is None:
            raise ValueError(
                f"safetensors dtype {dtype_name} has no torch equivalent."
            )
        self._fh.seek(self._data_start + start)
        # ``frombuffer`` requires a writable buffer; ``bytearray(...)`` of the
        # ``read`` result hands the resulting tensor its own memory so the
        # backing file can be closed independently.
        buf = bytearray(self._fh.read(end - start))
        t = torch.frombuffer(buf, dtype=td)
        t = t.view(*shape) if shape else t.view(())
        return t.to(device) if device != "cpu" else t

    def get_named_tensor(
        self, key: str, device: torch.device | str = "cpu"
    ) -> tuple[str, torch.Tensor]:
        """Return ``(safetensors_dtype_name, tensor)`` so callers can round-trip."""
        return self.get_dtype_name(key), self.get_tensor(key, device=device)


def save_safetensors(
    path: Path, entries: dict[str, tuple[str, torch.Tensor]]
) -> None:
    """Write a safetensors file from torch tensors, preserving original dtype names.

    Each entry: ``key -> (dtype_name, tensor)``. ``dtype_name`` is the
    safetensors dtype label to emit in the JSON header (e.g. ``"F8_E8M0"``
    for tensors that ride as ``uint8``); the tensor's torch ``element_size()``
    must match that label's byte width.

    Streams each tensor's storage straight to the file via the buffer
    protocol — no per-tensor ``bytes()`` materialization, no whole-shard
    payload buffer.
    """
    # Pass 1: build header from tensor shapes + sizes, no I/O.
    header: dict[str, dict] = {}
    offset = 0
    for key, (dtype_name, t) in entries.items():
        if dtype_name not in _DTYPE_INFO:
            raise ValueError(f"{key}: unknown safetensors dtype {dtype_name!r}")
        elem_size = _DTYPE_INFO[dtype_name][0]
        if t.element_size() != elem_size:
            raise ValueError(
                f"{key}: tensor dtype {t.dtype} (element_size {t.element_size()}) "
                f"is not byte-compatible with safetensors dtype {dtype_name} "
                f"(element_size {elem_size})"
            )
        nbytes = t.numel() * t.element_size()
        header[key] = {
            "dtype": dtype_name,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires the header byte length to be 8-byte aligned; the
    # spec's pad character is implementation-defined and spaces are
    # conventional.
    pad = (-len(header_bytes)) % 8
    header_bytes = header_bytes + b" " * pad

    # Pass 2: write header, then stream each tensor's bytes straight to file.
    # ``t.flatten().view(torch.uint8)`` reinterprets storage as uint8 with no
    # copy (works for any dtype, including fp8). ``.numpy()`` is a zero-copy
    # view; ``f.write(numpy_array)`` then issues a single C-level memcpy via
    # the buffer protocol — no intermediate ``bytes()`` materialization.
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for key, (_, t) in entries.items():
            t = t.detach().contiguous().cpu()
            f.write(t.flatten().view(torch.uint8).numpy())

# --- MXFP4 / NVFP4 constants ------------------------------------------------
E8M0_BIAS = 127
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_KMAX = 8
E4M3_KMIN = -9
NVFP4_BLOCK_SIZE = 16  # NVFP4 weight quantizer block size
MXFP4_BLOCK_SIZE = 32  # MXFP4 source block size

# Magnitude grid for the low 3 bits of an E2M1 nibble. Sign sits in bit 3.
_E2M1_MAGNITUDE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


# --- Helpers ----------------------------------------------------------------
# --- Triton kernel: per-NVFP4-block MXFP4 -> NVFP4 cast --------------------
@triton.jit
def _e2m1_decode(nib):
    """Decode an E2M1 nibble (sign in bit 3, magnitude in bits 0-2) to FP32.

    Magnitude grid: [0, 0.5, 1, 1.5, 2, 3, 4, 6]. exp=0 is subnormal
    (value = mantissa * 0.5), exp >= 1 is normal
    (value = (1 + mantissa/2) * 2^(exp-1)).
    """
    sign = (nib >> 3) & 1
    mag = nib & 0x7
    mant = mag & 1
    exp = mag >> 1
    sub = mant.to(tl.float32) * 0.5
    norm = (1.0 + mant.to(tl.float32) * 0.5) * tl.exp2((exp - 1).to(tl.float32))
    v = tl.where(exp == 0, sub, norm)
    return tl.where(sign == 1, -v, v)


@triton.jit
def _e2m1_encode(val):
    """Round signed FP32 to an E2M1 nibble (round-half-up at boundaries).

    Boundaries between adjacent magnitudes: ``[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]``.
    Sign goes into bit 3.
    """
    sign_bit = (val < 0).to(tl.uint8)
    absv = tl.abs(val)
    ord_ = (
        (absv >= 0.25).to(tl.uint8)
        + (absv >= 0.75).to(tl.uint8)
        + (absv >= 1.25).to(tl.uint8)
        + (absv >= 1.75).to(tl.uint8)
        + (absv >= 2.5).to(tl.uint8)
        + (absv >= 3.5).to(tl.uint8)
        + (absv >= 5.0).to(tl.uint8)
    )
    return (sign_bit << 3) | ord_


@triton.jit
def _cast_kernel(
    mxfp4_w_ptr,          # uint8, [R, C_pack]   (packed E2M1 nibbles, MXFP4 layout)
    mxfp4_s_ptr,          # uint8, [R, C_mxs]    (E8M0 exponents, byte-equivalent to uint8)
    nvfp4_w_ptr,          # uint8, [R, C_pack]   output, same packing as MXFP4
    nvfp4_s_ptr,          # uint8, [R, C_nvs]    output, E4M3 byte representation
    R, C_pack, C_mxs, C_nvs,
    m,                    # i32 scalar — the chosen 2^m for this tensor
    weight_scale_2,       # f32 scalar — equals 2^m, hoisted to avoid recompute
    BLOCK_NVS: tl.constexpr,  # NVFP4 blocks per program (>=2; pairs map to MXFP4 blocks)
):
    """Per-(row, NVFP4-block-tile) cast.

    Each program handles one row × ``BLOCK_NVS`` consecutive NVFP4 blocks. Two
    adjacent NVFP4 blocks share an MXFP4 ``k_j`` (since one MXFP4 block of 32
    spawns two NVFP4 blocks of 16). The kernel emits one E4M3 scale byte and
    eight packed weight bytes per NVFP4 block.

    Lossless branch (``-9 <= k_j - m <= 8``):
      * weight bytes copied verbatim from MXFP4 (bit-exact reconstruction)
      * scale byte = E4M3 representation of ``2^(k_j - m)``, built directly
        from bits — exponent ``(k_j - m + 7) << 3`` for normals,
        ``1 << (k_j - m + 9)`` for subnormals.

    OOR branch:
      * unpack 16 nibbles → FP32 magnitudes via the E2M1 grid
      * dequant via ``2^k_j``
      * data-derived per-block ``amax``, ``per_block_scale = clamp_E4M3(amax / (6 * 2^m))``
      * re-quantize each value to E2M1 nibble via boundary-search rounding
      * pack new nibbles, store new scale.
    """
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)

    nv_offsets = pid_b * BLOCK_NVS + tl.arange(0, BLOCK_NVS)  # (BLOCK_NVS,)
    nv_mask = nv_offsets < C_nvs

    # Adjacent NVFP4 blocks share an MXFP4 block (and its k_j).
    mx_idx = nv_offsets // 2  # (BLOCK_NVS,)
    e8m0 = tl.load(
        mxfp4_s_ptr + pid_r * C_mxs + mx_idx, mask=nv_mask, other=0
    ).to(tl.int32)
    k_j = e8m0 - 127
    delta = k_j - m
    in_range = (delta >= -9) & (delta <= 8) & nv_mask

    # Load 8 packed bytes per NVFP4 block (= 16 elements / 16 nibbles).
    bidx = tl.arange(0, 8)  # bytes within an NVFP4 block
    byte_offs = nv_offsets[:, None] * 8 + bidx[None, :]  # (BLOCK_NVS, 8)
    byte_mask = nv_mask[:, None] & (byte_offs < C_pack)
    bytes_2d = tl.load(
        mxfp4_w_ptr + pid_r * C_pack + byte_offs,
        mask=byte_mask, other=0,
    )  # (BLOCK_NVS, 8) uint8

    # ---------- Lossless branch: closed-form scale, copy weights ----------
    # E4M3 byte for 2^delta (normals: (delta+7)<<3; subnormals: 1<<(delta+9)).
    # Clamp shifts to non-negative so the OOR-branch evaluation here doesn't
    # invoke undefined shifts; the tl.where below keeps only in-range values.
    exp_bits = tl.maximum(delta + 7, 0)
    sub_shift = tl.maximum(delta + 9, 0)
    normal_byte = (exp_bits * 8).to(tl.uint8)
    subnormal_byte = (tl.cast(1, tl.int32) << sub_shift).to(tl.uint8)
    e4m3_lossless = tl.where(delta >= -6, normal_byte, subnormal_byte)

    # ---------- OOR branch: dequant 16 elements, re-quantize ----------
    low_nib = (bytes_2d & 0x0F).to(tl.int32)         # (BLOCK_NVS, 8) — even-positioned elems
    high_nib = ((bytes_2d >> 4) & 0x0F).to(tl.int32) # (BLOCK_NVS, 8) — odd-positioned elems

    low_v = _e2m1_decode(low_nib)
    high_v = _e2m1_decode(high_nib)

    # Apply MXFP4 per-block scale 2^k_j (broadcast per NVFP4 block).
    pow_k = tl.exp2(k_j.to(tl.float32))[:, None]
    low_dq = low_v * pow_k
    high_dq = high_v * pow_k

    # Per-block amax over 16 elements.
    amax = tl.maximum(
        tl.max(tl.abs(low_dq), axis=1),
        tl.max(tl.abs(high_dq), axis=1),
    )  # (BLOCK_NVS,)

    # Target per-block-scale = amax / (6 * 2^m). Cast to E4M3 (saturates above
    # 448; rounds to subnormal for very small targets, eventually clamping at 0).
    target_pbs = amax / (6.0 * weight_scale_2)
    target_pbs_clamped = tl.minimum(target_pbs, 448.0)
    pbs_e4m3 = tl.cast(target_pbs_clamped, tl.float8e4nv)
    e4m3_oor = pbs_e4m3.to(tl.uint8, bitcast=True)  # (BLOCK_NVS,) byte representation

    # Effective scale after E4M3 round, used to re-quantize into nibbles.
    eff_scale = pbs_e4m3.to(tl.float32) * weight_scale_2  # (BLOCK_NVS,)
    eff_scale = tl.where(eff_scale > 0, eff_scale, 1.0)

    inv_eff = 1.0 / eff_scale[:, None]  # (BLOCK_NVS, 1) — broadcast over the 8 byte slots
    low_new = _e2m1_encode(low_dq * inv_eff)   # (BLOCK_NVS, 8)
    high_new = _e2m1_encode(high_dq * inv_eff)
    bytes_oor = (high_new << 4) | low_new  # (BLOCK_NVS, 8)

    # ---------- Combine paths: use lossless when in_range, else OOR ----------
    out_bytes = tl.where(in_range[:, None], bytes_2d, bytes_oor)
    out_scale = tl.where(in_range, e4m3_lossless, e4m3_oor)

    tl.store(
        nvfp4_w_ptr + pid_r * C_pack + byte_offs,
        out_bytes,
        mask=byte_mask,
    )
    tl.store(
        nvfp4_s_ptr + pid_r * C_nvs + nv_offsets,
        out_scale,
        mask=nv_mask,
    )


def cast_layer_triton(
    mxfp4_weight: torch.Tensor,  # uint8 / int8, [R, in_features // 2]
    mxfp4_scale: torch.Tensor,   # uint8, [R, in_features // 32], E8M0
    m: int,
    block_nvs: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch ``_cast_kernel`` for a single MXFP4 weight tensor.

    Inputs must be CUDA tensors (the kernel is GPU-only). Returns
    ``(nvfp4_weight, nvfp4_scale, weight_scale_2)`` where
    ``nvfp4_scale.dtype == float8_e4m3fn`` and ``weight_scale_2`` is a F32
    scalar tensor on the same device.
    """
    if mxfp4_weight.dtype != torch.uint8:
        # MXFP4 source weights load as int8 from safetensors; reinterpret.
        mxfp4_weight = mxfp4_weight.view(torch.uint8)
    assert mxfp4_weight.is_cuda and mxfp4_scale.is_cuda, (
        "Triton cast requires CUDA tensors"
    )
    assert mxfp4_weight.is_contiguous() and mxfp4_scale.is_contiguous(), (
        "MXFP4 inputs must be contiguous"
    )

    R = mxfp4_weight.shape[0]
    C_pack = mxfp4_weight.shape[1]
    C_mxs = mxfp4_scale.shape[-1]
    C_nvs = C_mxs * 2  # two NVFP4 blocks per MXFP4 block
    assert C_pack == C_mxs * 16, (
        f"shape mismatch: weight {tuple(mxfp4_weight.shape)} vs scale {tuple(mxfp4_scale.shape)}"
    )

    nv_w = torch.empty_like(mxfp4_weight)
    # Allocate the per-block scale as uint8; ``.view(float8_e4m3fn)`` after the
    # kernel reinterprets the bytes for safetensors export with no copy.
    nv_s_u8 = torch.empty((R, C_nvs), dtype=torch.uint8, device=mxfp4_weight.device)

    weight_scale_2 = float(2.0**m)
    grid = (R, triton.cdiv(C_nvs, block_nvs))
    _cast_kernel[grid](
        mxfp4_weight, mxfp4_scale, nv_w, nv_s_u8,
        R, C_pack, C_mxs, C_nvs,
        m,
        weight_scale_2,
        BLOCK_NVS=block_nvs,
    )

    nv_s = nv_s_u8.view(torch.float8_e4m3fn)
    ws2 = torch.tensor(weight_scale_2, dtype=torch.float32, device=mxfp4_weight.device)
    return nv_w, nv_s, ws2


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """``packed [..., F]`` (uint8) -> nibbles ``[..., 2*F]`` (uint8 in [0, 15]).

    The convention matches both modelopt's NVFP4 packing
    (``(q[1] << 4) | q[0]``) and the MXFP4 source layout from gpt-oss /
    DeepSeek MXFP4 checkpoints — element ``2*i`` lives in the low nibble of
    byte ``i``, element ``2*i + 1`` in the high nibble.
    """
    out_shape = list(packed.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = (packed >> 4) & 0x0F
    return out


def _e2m1_dequant(nibbles: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """E2M1 nibbles (uint8 in [0, 15]) -> signed magnitude floats."""
    table = _E2M1_MAGNITUDE.to(device=nibbles.device, dtype=dtype)
    sign = ((nibbles >> 3) & 0x01).to(dtype) * -2 + 1  # 0 -> +1, 1 -> -1
    mag = table[(nibbles & 0x07).long()]
    return sign * mag


def mxfp4_dequant(
    weight_packed: torch.Tensor, scale_e8m0: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Reference MXFP4 dequant: ``nibble_value * 2^(scale_e8m0 - 127)`` per 32-elt block.

    Args:
        weight_packed: uint8 ``[..., in_features // 2]`` packed E2M1 nibbles.
        scale_e8m0:    uint8 ``[..., in_features // 32]`` E8M0 exponents.
    Returns:
        ``[..., in_features]`` floats.
    """
    nibbles = _unpack_nibbles(weight_packed)  # [..., in_features]
    values = _e2m1_dequant(nibbles, dtype=dtype)  # [..., in_features]
    in_features = values.shape[-1]
    # Reshape to (..., n_blocks, block_size)
    blocks = values.view(*values.shape[:-1], in_features // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
    k = scale_e8m0.to(torch.int32) - E8M0_BIAS  # (..., n_blocks)
    pow2 = torch.exp2(k.to(dtype))
    blocks = blocks * pow2.unsqueeze(-1)
    return blocks.view(*values.shape[:-1], in_features).to(dtype)


def nvfp4_dequant(
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    weight_scale_2: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference NVFP4 dequant:
    ``nibble_value * weight_scale_e4m3.float() * weight_scale_2`` per 16-elt block.
    """
    nibbles = _unpack_nibbles(weight_packed)
    values = _e2m1_dequant(nibbles, dtype=dtype)
    in_features = values.shape[-1]
    blocks = values.view(*values.shape[:-1], in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    per_block_scale = weight_scale_e4m3.to(dtype) * weight_scale_2.to(dtype)  # (..., n_blocks)
    blocks = blocks * per_block_scale.unsqueeze(-1)
    return blocks.view(*values.shape[:-1], in_features).to(dtype)


def compute_k_stats(mxfp4_scale: torch.Tensor) -> tuple[int, int]:
    """Per-tensor ``(k_min, k_max)`` over non-zero E8M0 entries.

    All-zero blocks (``e8m0 == 0``) are MXFP4's all-zero-weight sentinel and
    are excluded from ``k_max`` so they don't drag ``m`` toward ``-127``.
    """
    k = mxfp4_scale.to(torch.int32) - E8M0_BIAS
    nonzero = mxfp4_scale > 0
    if nonzero.any():
        k_nz = k[nonzero]
        return int(k_nz.min().item()), int(k_nz.max().item())
    return 0, 0


def compute_block_stats(mxfp4_scale: torch.Tensor, m: int) -> dict:
    """Per-tensor lossless stats for a chosen ``m``.

    A block is lossless under the cast iff its ``delta = k_j - m`` lands in
    E4M3's exact-power-of-2 window ``[-9, 8]``. All-zero blocks are trivially
    lossless (their reconstruction is 0 regardless of scale).

    Returns ``{m, k_min, k_max, n_total_blocks, n_lossless_blocks,
    pct_lossless, is_lossless}`` — sized in MXFP4 blocks (each MXFP4 block
    spawns two NVFP4 blocks sharing ``k_j``, so the per-NVFP4-block counts
    are 2x these).
    """
    k = mxfp4_scale.to(torch.int32) - E8M0_BIAS
    nonzero = mxfp4_scale > 0
    delta = k - m
    in_range = ((delta >= E4M3_KMIN) & (delta <= E4M3_KMAX)) | (~nonzero)
    n_total = int(in_range.numel())
    n_lossless = int(in_range.sum().item())
    if nonzero.any():
        k_nz = k[nonzero]
        k_min = int(k_nz.min().item())
        k_max = int(k_nz.max().item())
    else:
        k_min = k_max = 0
    return {
        "m": m,
        "k_min": k_min,
        "k_max": k_max,
        "n_total_blocks": n_total,
        "n_lossless_blocks": n_lossless,
        "pct_lossless": 100.0 * n_lossless / n_total if n_total else 100.0,
        "is_lossless": n_lossless == n_total,
    }


def prepass(
    mxfp4_src: Path,
    src_map: dict[str, str],
    quant_bases: list[str],
    src_readers: dict[str, "SafeShardReader"],
    device: torch.device,
) -> tuple[dict[str, int], dict[str, dict]]:
    """Single-pass scan over MXFP4 ``*.scale`` tensors.

    Computes:
      - ``m_per_base[base]``: chosen ``m = k_max - 8``, with w1/w3 paired
        tensors sharing ``m = max(k_max_w1, k_max_w3) - 8`` so SwiGLU fusion
        sees a common ``weight_scale_2``.
      - ``stats_per_base[base]``: ``compute_block_stats`` output (k_min/k_max,
        n_total_blocks, n_lossless_blocks, pct_lossless, is_lossless) under
        the chosen ``m``.

    ``src_readers`` is mutated in-place with cached ``SafeShardReader`` handles
    so the main pass can reuse them without reopening shards.
    """

    def _src_reader(name: str) -> "SafeShardReader":
        if name not in src_readers:
            src_readers[name] = SafeShardReader(mxfp4_src / name)
        return src_readers[name]

    # First scan: per-base ``k_max`` from the MXFP4 scale.
    k_max_per_base: dict[str, int] = {}
    for base in quant_bases:
        scale_key = f"{base}.scale"
        if scale_key not in src_map:
            continue
        scale = _src_reader(src_map[scale_key]).get_tensor(scale_key, device=device)
        k_max_per_base[base] = compute_k_stats(scale)[1]

    # Second scan: pair w1 with w3 for shared m.
    m_per_base: dict[str, int] = {}
    for base, k_max in k_max_per_base.items():
        if base in m_per_base:
            continue
        if base.endswith(".w1"):
            sibling = base[: -len(".w1")] + ".w3"
            if sibling in k_max_per_base:
                shared_m = max(k_max, k_max_per_base[sibling]) - E4M3_KMAX
                m_per_base[base] = shared_m
                m_per_base[sibling] = shared_m
                continue
        m_per_base[base] = k_max - E4M3_KMAX

    # Third scan: compute per-tensor lossless stats under the chosen m.
    stats_per_base: dict[str, dict] = {}
    for base, m in m_per_base.items():
        scale_key = f"{base}.scale"
        if scale_key not in src_map:
            continue
        scale = _src_reader(src_map[scale_key]).get_tensor(scale_key, device=device)
        stats_per_base[base] = compute_block_stats(scale, m)

    return m_per_base, stats_per_base


# --- Whole-checkpoint driver ------------------------------------------------
def _load_index(path: Path) -> dict:
    idx_file = path / "model.safetensors.index.json"
    with idx_file.open() as f:
        return json.load(f)


def _shard_groups(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Group keys by shard filename."""
    out: dict[str, list[str]] = defaultdict(list)
    for k, shard in weight_map.items():
        out[shard].append(k)
    return out


def _is_quantized_weight(base: str, all_keys: set[str]) -> bool:
    """``<base>.weight`` is NVFP4-quantized iff ``<base>.weight_scale_2`` exists."""
    return f"{base}.weight_scale_2" in all_keys


def cast_checkpoint(
    mxfp4_src: Path,
    nvfp4_src: Path,
    nvfp4_dst: Path,
    device: torch.device,
    verbose: bool = False,
    mse_sample: int = 32,
    block_nvs: int = 16,
) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "cast_checkpoint requires CUDA — the cast runs as a Triton kernel. "
            "Run on a CUDA host or use a smaller subset for CPU testing."
        )
    nvfp4_dst.mkdir(parents=True, exist_ok=True)

    src_idx = _load_index(mxfp4_src)
    v3_idx = _load_index(nvfp4_src)
    src_map = src_idx["weight_map"]
    v3_map = v3_idx["weight_map"]
    v3_all_keys = set(v3_map.keys())

    # Aggregate stats.
    total_layers = 0
    total_lossless_layers = 0
    total_blocks = 0
    total_lossless_blocks = 0
    # MSE accounting. v3 MSE is sampled (every Nth tensor); v3.5 MSE is only
    # computed for tensors that aren't fully lossless (since lossless ones are
    # bit-exact by construction). Each gets its own counter for averaging.
    mse_v3_sum = 0.0
    mse_v3_n = 0
    mse_v3_max = 0.0
    mse_v35_sum = 0.0
    mse_v35_n = 0
    mse_v35_max = 0.0
    mse_n = 0  # total tensors seen — used to drive the v3 sampling stride

    # MXFP4 source shard handles, cached by shard name (populated by the
    # pre-pass and reused by the main pass).
    src_readers: dict[str, SafeShardReader] = {}

    v3_groups = _shard_groups(v3_map)

    # Collect every quantized base across the whole checkpoint up front so
    # the pre-pass can compute m + lossless stats with a single sweep over
    # the MXFP4 ``*.scale`` tensors.
    all_quant_bases: list[str] = sorted(
        b[: -len(".weight")] for b in v3_all_keys
        if b.endswith(".weight") and _is_quantized_weight(b[: -len(".weight")], v3_all_keys)
    )
    print(
        f"[cast_mxfp4_to_nvfp4] pre-pass scanning {len(all_quant_bases)} quant tensors..."
    )
    t_pre = time.time()
    m_per_base, stats_per_base = prepass(
        mxfp4_src, src_map, all_quant_bases, src_readers, device
    )
    n_w1w3_paired = sum(
        1 for b in m_per_base if b.endswith(".w1") and b[: -3] + ".w3" in m_per_base
    )
    n_lossless = sum(1 for s in stats_per_base.values() if s["is_lossless"])
    print(
        f"[cast_mxfp4_to_nvfp4] pre-pass done in {time.time() - t_pre:.1f}s: "
        f"{n_w1w3_paired} w1/w3 pairs share m, "
        f"{n_lossless}/{len(stats_per_base)} tensors are 100% lossless"
    )

    log_path = nvfp4_dst / "cast_mxfp4_to_nvfp4.log"
    # ``buffering=1`` gives line-buffered text mode, so progress is visible in
    # the log without waiting for the process to exit / the buffer to fill.
    log_f = log_path.open("w", buffering=1)

    def _log(msg: str) -> None:
        log_f.write(msg + "\n")
        if verbose:
            print(msg)

    try:
        for shard_name in tqdm(sorted(v3_groups), desc="shards"):
            v3_shard_path = nvfp4_src / shard_name
            # Each entry: key -> (dtype_name, tensor) — tensors all the way down.
            new_state: dict[str, tuple[str, torch.Tensor]] = {}
            with SafeShardReader(v3_shard_path) as v3_f:
                shard_keys = set(v3_groups[shard_name])
                # First pass: identify quantized weight bases living in this shard.
                quant_bases: list[str] = []
                copy_keys: list[str] = []
                for k in shard_keys:
                    if k.endswith(".weight"):
                        base = k[: -len(".weight")]
                        if _is_quantized_weight(base, v3_all_keys):
                            quant_bases.append(base)
                            continue
                    copy_keys.append(k)

                # Second pass: recompute quantized weight tensors.
                for base in quant_bases:
                    src_w_key = f"{base}.weight"
                    src_s_key = f"{base}.scale"
                    if src_w_key not in src_map or src_s_key not in src_map:
                        # Source has no MXFP4 record for this layer — copy v3
                        # tensors verbatim (e.g. layers that aren't MXFP4 in
                        # the source). Should be rare; warn and continue.
                        _log(f"[warn] {base}: no MXFP4 source pair, copying v3 verbatim")
                        for suffix in [".weight", ".weight_scale", ".weight_scale_2", ".input_scale"]:
                            k = base + suffix
                            if k in v3_all_keys and v3_map[k] == shard_name:
                                new_state[k] = v3_f.get_named_tensor(k)
                        continue

                    src_reader = src_readers[src_map[src_w_key]]
                    mx_w = src_reader.get_tensor(src_w_key, device=device)
                    mx_s = src_readers[src_map[src_s_key]].get_tensor(src_s_key, device=device)
                    m = m_per_base[base]
                    info = stats_per_base[base]
                    new_w, new_scale, new_scale_2 = cast_layer_triton(
                        mx_w, mx_s, m=m, block_nvs=block_nvs
                    )

                    total_layers += 1
                    is_lossless = info["is_lossless"]
                    if is_lossless:
                        total_lossless_layers += 1
                    total_blocks += info["n_total_blocks"] * 2  # NVFP4 blocks = 2x MXFP4 blocks
                    total_lossless_blocks += info["n_lossless_blocks"] * 2

                    # MSE accounting. v3.5 is bit-exact for fully-lossless
                    # tensors by construction, so ``mse_v35 == 0`` without
                    # running the dequant. v3 MSE is sampled (see
                    # ``mse_sample`` flag); OOR tensors always get measured.
                    do_mse_v3 = mse_sample > 0 and (mse_n % mse_sample == 0)
                    do_mse_v35 = not is_lossless
                    mse_v3 = mse_v35 = 0.0
                    if do_mse_v3 or do_mse_v35:
                        mxfp4_deq = mxfp4_dequant(mx_w, mx_s, dtype=torch.float32)
                        if do_mse_v3:
                            v3_w_t = v3_f.get_tensor(base + ".weight", device=device)
                            v3_scale_t = v3_f.get_tensor(
                                base + ".weight_scale", device=device
                            )
                            v3_scale_2_t = v3_f.get_tensor(
                                base + ".weight_scale_2", device=device
                            )
                            v3_deq = nvfp4_dequant(
                                v3_w_t, v3_scale_t, v3_scale_2_t, dtype=torch.float32
                            )
                            mse_v3 = float(((mxfp4_deq - v3_deq) ** 2).mean().item())
                            mse_v3_sum += mse_v3
                            mse_v3_n += 1
                            mse_v3_max = max(mse_v3_max, mse_v3)
                            del v3_w_t, v3_scale_t, v3_scale_2_t, v3_deq
                        if do_mse_v35:
                            v35_deq = nvfp4_dequant(
                                new_w, new_scale, new_scale_2, dtype=torch.float32
                            )
                            mse_v35 = float(((mxfp4_deq - v35_deq) ** 2).mean().item())
                            mse_v35_sum += mse_v35
                            mse_v35_n += 1
                            mse_v35_max = max(mse_v35_max, mse_v35)
                            del v35_deq
                        del mxfp4_deq
                    mse_n += 1
                    shared_tag = ""
                    if base.endswith(".w1") and (base[: -3] + ".w3") in m_per_base:
                        shared_tag = " (w1/w3-shared)"
                    elif base.endswith(".w3") and (base[: -3] + ".w1") in m_per_base:
                        shared_tag = " (w1/w3-shared)"
                    if do_mse_v3 or do_mse_v35:
                        _log(
                            f"{base}: m={info['m']:+d}{shared_tag} "
                            f"lossless={info['pct_lossless']:.4f}% "
                            f"mse(v3)={mse_v3:.4e} mse(v3.5)={mse_v35:.4e}"
                        )

                    # Stage cast outputs as ``(safetensors_dtype_name, tensor)``.
                    new_state[base + ".weight"] = ("U8", new_w.cpu())
                    new_state[base + ".weight_scale"] = ("F8_E4M3", new_scale.cpu())
                    new_state[base + ".weight_scale_2"] = ("F32", new_scale_2.cpu())
                    # input_scale is unchanged from v3 (PTQ-derived activation amax).
                    in_scale_key = base + ".input_scale"
                    if in_scale_key in v3_all_keys and v3_map[in_scale_key] == shard_name:
                        new_state[in_scale_key] = v3_f.get_named_tensor(in_scale_key)

                    del mx_w, mx_s, new_w, new_scale, new_scale_2

                # Copy non-quantized keys verbatim. Skip ones already staged
                # via the input_scale path above.
                for k in copy_keys:
                    if k in new_state:
                        continue
                    new_state[k] = v3_f.get_named_tensor(k)

            save_safetensors(nvfp4_dst / shard_name, new_state)
            del new_state

    finally:
        log_f.close()
        for r in src_readers.values():
            r.close()

    # Copy non-tensor files (config.json, hf_quant_config.json, tokenizer, etc.)
    # from v3 verbatim.
    for child in nvfp4_src.iterdir():
        if child.is_file() and child.suffix not in {".safetensors"} and child.name != "cast_mxfp4_to_nvfp4.log":
            dst = nvfp4_dst / child.name
            if not dst.exists():
                shutil.copy2(child, dst)
        elif child.is_dir():
            dst = nvfp4_dst / child.name
            if not dst.exists():
                shutil.copytree(child, dst)
    # Copy the safetensors index too (the per-shard key sets are unchanged).
    idx_src = nvfp4_src / "model.safetensors.index.json"
    if idx_src.exists():
        shutil.copy2(idx_src, nvfp4_dst / "model.safetensors.index.json")

    # Final summary.
    pct_layers = 100.0 * total_lossless_layers / total_layers if total_layers else 100.0
    pct_blocks = 100.0 * total_lossless_blocks / total_blocks if total_blocks else 100.0
    # v3.5 MSE is only measured on non-lossless tensors. Lossless tensors
    # contribute MSE = 0 by construction; fold them into the average.
    n_v35_lossless = total_layers - mse_v35_n
    avg_mse_v3 = mse_v3_sum / mse_v3_n if mse_v3_n else 0.0
    avg_mse_v35 = (
        mse_v35_sum / (mse_v35_n + n_v35_lossless)
        if (mse_v35_n + n_v35_lossless) > 0
        else 0.0
    )

    summary = [
        "",
        "=" * 72,
        f"[cast_mxfp4_to_nvfp4] processed {total_layers} expert weight tensors "
        f"across {len(v3_groups)} shards",
        f"[cast_mxfp4_to_nvfp4] lossless layers: {total_lossless_layers}/{total_layers} ({pct_layers:.2f}%)",
        f"[cast_mxfp4_to_nvfp4] lossless blocks: {total_lossless_blocks}/{total_blocks} ({pct_blocks:.4f}%)",
        f"[cast_mxfp4_to_nvfp4] MSE vs MXFP4: v3 (sampled n={mse_v3_n}): mean={avg_mse_v3:.4e} max={mse_v3_max:.4e}",
        f"[cast_mxfp4_to_nvfp4] MSE vs MXFP4: v3.5 (n={mse_v35_n} OOR + {n_v35_lossless} lossless): mean={avg_mse_v35:.4e} max={mse_v35_max:.4e}",
        "=" * 72,
    ]
    for line in summary:
        print(line)
    with log_path.open("a") as f:
        for line in summary:
            f.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mxfp4_src",
        required=True,
        type=Path,
        help="Path to the original MXFP4 HF checkpoint (e.g. DeepSeek-V4-Flash).",
    )
    p.add_argument(
        "--nvfp4_src",
        required=True,
        type=Path,
        help="Path to the existing NVFP4 export to reuse non-quant tensors from "
        "(e.g. flash-nvfp4-experts-v3).",
    )
    p.add_argument(
        "--nvfp4_dst",
        required=True,
        type=Path,
        help="Path to write the new NVFP4 checkpoint (e.g. flash-nvfp4-experts-v3.5).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="CUDA device for the Triton cast (default: cuda). CPU is not supported "
        "— the cast is implemented as a Triton kernel.",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-tensor stats to stdout.")
    p.add_argument(
        "--mse_sample",
        type=int,
        default=32,
        help="Compute v3-vs-MXFP4 MSE on every Nth expert tensor (default 32). "
        "0 disables v3 MSE entirely. v3.5 MSE is always measured on tensors "
        "with any OOR blocks (fully-lossless tensors are bit-exact by "
        "construction so their MSE is 0).",
    )
    p.add_argument(
        "--block_nvs",
        type=int,
        default=16,
        help="NVFP4 blocks processed per Triton program (default 16). Must be "
        "even; tune for GPU occupancy.",
    )
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required: the cast runs as a Triton kernel. "
            "Run on a host with a CUDA GPU."
        )

    mxfp4_src = args.mxfp4_src.expanduser().resolve()
    nvfp4_src = args.nvfp4_src.expanduser().resolve()
    nvfp4_dst = args.nvfp4_dst.expanduser().resolve()
    if not mxfp4_src.is_dir():
        raise FileNotFoundError(f"--mxfp4_src not found: {mxfp4_src}")
    if not nvfp4_src.is_dir():
        raise FileNotFoundError(f"--nvfp4_src not found: {nvfp4_src}")

    device = torch.device(args.device)
    print(f"[cast_mxfp4_to_nvfp4] mxfp4_src = {mxfp4_src}")
    print(f"[cast_mxfp4_to_nvfp4] nvfp4_src = {nvfp4_src}")
    print(f"[cast_mxfp4_to_nvfp4] nvfp4_dst = {nvfp4_dst}")
    print(f"[cast_mxfp4_to_nvfp4] device    = {device}")
    t0 = time.time()
    cast_checkpoint(
        mxfp4_src,
        nvfp4_src,
        nvfp4_dst,
        device=device,
        verbose=args.verbose,
        mse_sample=args.mse_sample,
        block_nvs=args.block_nvs,
    )
    print(f"[cast_mxfp4_to_nvfp4] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
