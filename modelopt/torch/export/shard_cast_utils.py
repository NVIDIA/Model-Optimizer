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

"""Shared primitives for streaming checkpoint casts and export."""

import errno
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Collection
from pathlib import Path

import torch

from modelopt.torch.quantization.qtensor import FP8QTensor, MXFP4QTensor, NVFP4QTensor
from modelopt.torch.quantization.utils import reduce_block_amax, reduce_block_padding
from modelopt.torch.quantization.utils.numeric_utils import (
    E2M1_MAX,
    E4M3_KMAX,
    E4M3_KMIN,
    E4M3_MAX,
    E8M0_BIAS,
    mxfp4_to_nvfp4_global_amax,
    mxfp4_to_nvfp4_per_block_amax,
)
from modelopt.torch.utils.plugins.hf_checkpoint_utils import resolve_checkpoint_file

__all__ = [
    "build_w13_amax_overrides",
    "build_w13_kmax_overrides",
    "dequantize_fp8_ue8m0_to_bf16",
    "dequantize_mxfp4_to_bf16",
    "mxfp4_kmax",
    "quantize_bf16_to_fp8_ue8m0",
    "quantize_mxfp4_to_nvfp4",
    "quantize_mxfp4_to_nvfp4_lossless",
]

_MXFP4_BLOCK = 32
_MXFP4_BYTES_PER_BLOCK = 16
_NVFP4_BLOCK = 16


@torch.no_grad()
def dequantize_fp8_ue8m0_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor, block: int = 128, *, validate_scales: bool = True
) -> torch.Tensor:
    """Decode a native DSV4 FP8 matrix with square UE8M0 blocks into BF16.

    Uses square blocks of size ``block``, including partial trailing blocks.
    Scales are uint8/int8 exponent bytes or float8_e8m0fnu, with no NaN scale codes.
    Set ``validate_scales=False`` only for immutable scales already checked at load time,
    to avoid synchronizing CUDA on repeated calls. Shape and dtype checks always run.
    """
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn or not weight.numel():
        raise ValueError(
            f"Expected a nonempty FP8 E4M3 matrix, got shape {tuple(weight.shape)} ({weight.dtype})"
        )
    if not isinstance(block, int) or block <= 0:
        raise ValueError(f"FP8 block size must be a positive integer, got {block}")
    m, n = weight.shape
    expected = ((m + block - 1) // block, (n + block - 1) // block)
    if scale.shape != expected or scale.dtype not in (
        torch.uint8,
        torch.int8,
        torch.float8_e8m0fnu,
    ):
        raise ValueError(
            f"Expected UE8M0 scales with shape {expected}, got {tuple(scale.shape)} ({scale.dtype})"
        )
    scale = scale.to(device=weight.device).contiguous().view(torch.uint8)
    if validate_scales and (scale == 255).any():
        raise ValueError("Native checkpoint scales must not contain NaN codes")
    exp = scale.to(torch.int32) - E8M0_BIAS
    exp = exp.repeat_interleave(block, 0).repeat_interleave(block, 1)[:m, :n]
    # Some PyTorch CUDA ldexp kernels use the current device rather than the tensor's device.
    with torch.cuda.device_of(weight):
        return torch.ldexp(weight.to(torch.float32), exp).to(torch.bfloat16)


@torch.no_grad()
def quantize_bf16_to_fp8_ue8m0(
    weight: torch.Tensor, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a finite BF16 matrix as native DSV4 FP8 weights and uint8 UE8M0 scales.

    Uses square blocks of size ``block`` and recomputes scales for the trained weights.
    Rejects values whose FP8 rounding would decode to nonfinite BF16.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.numel():
        raise ValueError(
            f"Expected a nonempty BF16 matrix, got shape {tuple(weight.shape)} ({weight.dtype})"
        )
    if not isinstance(block, int) or block <= 0:
        raise ValueError(f"FP8 block size must be a positive integer, got {block}")
    blocks = {-2: block, -1: block}
    amax = reduce_block_amax(reduce_block_padding(weight, blocks), blocks).float()
    # At exponent 120, FP8 rounds 248 up to 256, which decodes to overflowing 2**128.
    if not (torch.isfinite(amax) & (amax < math.ldexp(248.0, 120))).all():
        raise ValueError("BF16 weights must be finite and remain finite after native FP8 rounding")
    # Choose the smallest power-of-two scale that fits E4M3, without log2 rounding.
    max_mantissa, max_exponent = math.frexp(E4M3_MAX)
    mantissa, exponent = torch.frexp(amax)
    exponent = (exponent - max_exponent + (mantissa > max_mantissa)).clamp(min=-E8M0_BIAS)
    exponent.masked_fill_(amax == 0, -E8M0_BIAS)
    with torch.cuda.device_of(weight):
        scales = torch.ldexp(torch.ones_like(amax), exponent)
    quantized, _ = FP8QTensor.quantize(weight.float(), scales=scales, block_sizes=blocks)
    return quantized._quantized_data.contiguous(), (exponent + E8M0_BIAS).to(torch.uint8)


def dequantize_mxfp4_to_bf16(
    mxfp4_weight: torch.Tensor, mxfp4_scale: torch.Tensor, device: str
) -> torch.Tensor:
    """Dequantize packed MXFP4 weights and E8M0 scales to BF16."""
    packed = mxfp4_weight.to(device).contiguous().view(torch.uint8)
    scale = mxfp4_scale.to(device).contiguous().view(torch.uint8)
    original_shape = torch.Size((*packed.shape[:-1], packed.shape[-1] * 2))
    assert packed.shape[:-1] == scale.shape[:-1] and (
        2 * packed.shape[-1] == scale.shape[-1] * _MXFP4_BLOCK
    ), f"Incompatible MXFP4 shapes: weight {tuple(packed.shape)} vs scale {tuple(scale.shape)}"
    return MXFP4QTensor(original_shape, torch.bfloat16, packed).dequantize(
        dtype=torch.bfloat16,
        scale=scale,
        block_sizes=[_MXFP4_BLOCK],
    )


def _w13_pairs(expert_bases: list[str]) -> list[tuple[str, str]]:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for base in expert_bases:
        prefix, proj = base.rsplit(".", 1)
        if proj in {"w1", "w3"}:
            groups[prefix][proj] = base

    pairs: list[tuple[str, str]] = []
    for prefix, paths in groups.items():
        if "w1" not in paths or "w3" not in paths:
            raise RuntimeError(
                "w1/w3 of one expert are split across shards, so they cannot share "
                f"scale_2 for the fused GEMM1: {prefix}"
            )
        pairs.append((paths["w1"], paths["w3"]))
    return pairs


def build_w13_kmax_overrides(
    expert_bases: list[str],
    get_scale: Callable[[str], torch.Tensor],
    device: str,
) -> dict[str, int]:
    """Return one shared E8M0 maximum exponent for each fused w1/w3 pair."""
    overrides: dict[str, int] = {}
    for w1, w3 in _w13_pairs(expert_bases):
        k1 = mxfp4_kmax(get_scale(w1), device)
        k3 = mxfp4_kmax(get_scale(w3), device)
        overrides[w1] = overrides[w3] = max(k1, k3)
    return overrides


def build_w13_amax_overrides(
    expert_bases: list[str],
    get_amax: Callable[[str], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return one shared weight amax for each fused w1/w3 pair."""
    overrides: dict[str, torch.Tensor] = {}
    for w1, w3 in _w13_pairs(expert_bases):
        shared = torch.maximum(get_amax(w1).reshape(()), get_amax(w3).reshape(()))
        overrides[w1] = overrides[w3] = shared
    return overrides


def mxfp4_kmax(mxfp4_scale: torch.Tensor, device: str = "cpu") -> int:
    """Return the largest non-zero unbiased exponent in an E8M0 scale tensor."""
    e8m0 = mxfp4_scale.to(device).contiguous().view(torch.uint8)
    return mxfp4_to_nvfp4_global_amax(e8m0)[1]["k_max"]


def quantize_mxfp4_to_nvfp4(
    mxfp4_weight: torch.Tensor,
    mxfp4_scale: torch.Tensor,
    weight_amax: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Dequantize MXFP4 and requantize it to NVFP4 using an optional global amax."""
    bf16 = dequantize_mxfp4_to_bf16(mxfp4_weight, mxfp4_scale, device)
    synthesized = weight_amax is None
    if weight_amax is None:
        weight_amax = bf16.abs().max()
    weight_scale_2 = (weight_amax.to(device).float() / (E2M1_MAX * E4M3_MAX)).reshape(())
    q_tensor, weight_scale, _ = NVFP4QTensor.quantize(
        bf16, _NVFP4_BLOCK, None, weight_scale_2, try_tensorrt=False
    )
    return q_tensor._quantized_data, weight_scale, weight_scale_2, synthesized


def quantize_mxfp4_to_nvfp4_lossless(
    mxfp4_weight: torch.Tensor,
    mxfp4_scale: torch.Tensor,
    k_max: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Closed-form MXFP4-to-NVFP4 cast with lossless-block accounting."""
    bf16 = dequantize_mxfp4_to_bf16(mxfp4_weight, mxfp4_scale, device)
    e8m0 = mxfp4_scale.to(bf16.device).contiguous().view(torch.uint8)
    packed = mxfp4_weight.to(bf16.device).contiguous().view(torch.uint8)
    blocks = packed.view(*packed.shape[:-1], e8m0.shape[-1], _MXFP4_BYTES_PER_BLOCK)
    per_block_amax = mxfp4_to_nvfp4_per_block_amax(blocks, e8m0)

    weight_scale_2 = torch.tensor(
        2.0 ** (k_max - E4M3_KMAX), dtype=torch.float32, device=bf16.device
    ).reshape(())
    per_block_scale = (
        (per_block_amax / (E2M1_MAX * weight_scale_2))
        .clamp(min=2**E4M3_KMIN, max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )

    k = e8m0.to(torch.int32) - E8M0_BIAS
    lossless = (k >= (k_max - (E4M3_KMAX - E4M3_KMIN))) | (e8m0 == 0)
    n_blocks = k.numel()
    n_lossless = int(lossless.sum().item())

    q_tensor, weight_scale, _ = NVFP4QTensor.quantize(
        bf16, _NVFP4_BLOCK, per_block_scale, weight_scale_2, try_tensorrt=False
    )
    return q_tensor._quantized_data, weight_scale, weight_scale_2, n_blocks, n_lossless


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link a file, copying when the filesystem cannot create the link."""
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"source must be a regular file, not a symlink: {src}")
    try:
        os.link(src, dst)
    except OSError as exc:
        copy_errnos = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            getattr(errno, "EOPNOTSUPP", errno.EXDEV),
            getattr(errno, "ENOTSUP", errno.EXDEV),
        }
        if exc.errno not in copy_errnos:
            raise
        shutil.copy2(src, dst)


def _collect_aux_files(
    src_dir: Path,
    *,
    skip_top_level: Collection[str] = (),
    skip_dir_names: Collection[str] = (),
    skip_file: Callable[[Path], bool] | None = None,
) -> tuple[list[Path], list[tuple[Path, Path]]]:
    destination_dirs: list[Path] = []
    sources: list[tuple[Path, Path]] = []
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        rel = root_path.relative_to(src_dir)
        at_top_level = rel == Path(".")
        dirs[:] = [
            name
            for name in dirs
            if name not in skip_dir_names and not (at_top_level and name in skip_top_level)
        ]
        for name in dirs:
            source_dir = root_path / name
            if source_dir.is_symlink() or not source_dir.is_dir():
                raise ValueError(f"source must be a regular directory: {source_dir}")
        destination_dirs.append(rel)
        for name in files:
            relative_path = rel / name
            if at_top_level and name in skip_top_level:
                continue
            if skip_file is not None and skip_file(relative_path):
                continue
            resolved_src = resolve_checkpoint_file(src_dir, relative_path, max_bytes=None)
            sources.append((relative_path, resolved_src))
    return destination_dirs, sources


def validate_aux_files(
    src_dir: Path,
    *,
    skip_top_level: Collection[str] = (),
    skip_dir_names: Collection[str] = (),
    skip_file: Callable[[Path], bool] | None = None,
) -> None:
    """Validate checkpoint sidecars without writing an output directory."""
    _collect_aux_files(
        src_dir,
        skip_top_level=skip_top_level,
        skip_dir_names=skip_dir_names,
        skip_file=skip_file,
    )


def link_aux_files(
    src_dir: Path,
    dst_dir: Path,
    *,
    skip_top_level: Collection[str] = (),
    skip_dir_names: Collection[str] = (),
    skip_file: Callable[[Path], bool] | None = None,
) -> None:
    """Recursively link checkpoint sidecars while applying model-specific skips."""
    destination_dirs, sources = _collect_aux_files(
        src_dir,
        skip_top_level=skip_top_level,
        skip_dir_names=skip_dir_names,
        skip_file=skip_file,
    )
    for relative_dir in destination_dirs:
        (dst_dir / relative_dir).mkdir(parents=True, exist_ok=True)
    for relative_path, src in sources:
        dst = dst_dir / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        link_or_copy(src, dst)


def log(message: str) -> None:
    """Print a checkpoint-conversion progress message immediately."""
    print(message, flush=True)


def validate_paths(source_ckpt: Path, output_ckpt: Path) -> None:
    """Reject overlapping source and output checkpoint directories."""
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


def prepare_output_dir(output_ckpt: Path, overwrite: bool) -> None:
    """Create an empty output directory, replacing its contents when allowed."""
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
