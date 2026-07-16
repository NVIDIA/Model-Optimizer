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

"""Run conditional-only PDD inference from an authenticated Qwen-Image export."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_FASTGEN_DIR = _THIS_DIR.parent
_REPO_ROOT = _FASTGEN_DIR.parents[2]
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--schedule", choices=("pdd-2", "pdd-4", "pdd-8"), default="pdd-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _dtype_from_name(name: Any) -> torch.dtype:
    if not isinstance(name, str):
        raise ValueError("PDD export model dtype must be a string.")
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"PDD inference does not support model dtype {name!r}.") from error


def _model_identity(descriptor: Any) -> Mapping[str, Any]:
    identity = descriptor.manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("PDD export has no identity mapping.")
    model = identity.get("model")
    if not isinstance(model, Mapping) or set(model) != {"id", "revision", "dtype"}:
        raise RuntimeError("PDD export model identity is malformed.")
    if not isinstance(model["id"], str) or not model["id"]:
        raise RuntimeError("PDD export model ID is invalid.")
    revision = model["revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("PDD export requires a pinned 40-character model revision.")
    try:
        int(revision, 16)
    except ValueError as error:
        raise RuntimeError("PDD export model revision must be hexadecimal.") from error
    return model


def _validate_qwen_projection(student: nn.Module, metadata: Any) -> nn.Linear:
    """Validate the ordinary Qwen projection before widening it for PDD."""
    try:
        base_projection = student.get_submodule("proj_out")
    except AttributeError as error:
        raise RuntimeError("reconstructed Qwen student has no proj_out linear layer.") from error
    in_channels = getattr(getattr(student, "config", None), "in_channels", None)
    if type(in_channels) is not int or in_channels <= 0 or in_channels % 4:
        raise RuntimeError("Qwen transformer in_channels must be a positive multiple of four.")
    if (
        not isinstance(base_projection, nn.Linear)
        or base_projection.in_features != metadata.projection_in_features
        or base_projection.out_features != metadata.projection_out_features
        or (base_projection.bias is not None) != metadata.projection_bias
    ):
        raise RuntimeError("reconstructed Qwen proj_out does not match authenticated metadata.")
    if base_projection.out_features != in_channels:
        raise RuntimeError(
            "Qwen proj_out width must equal transformer in_channels for 2x2 latent packing."
        )
    return base_projection


def build_pdd_student(export_dir: str | Path) -> tuple[nn.Module, Any, torch.dtype]:
    """Reconstruct and strictly load the converted Qwen student on CPU."""
    from diffusers import QwenImageTransformer2DModel

    from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
        adopt_qwen_image_mr210_forward,
        convert_qwen_image_to_pdd,
        require_qwen_image_pdd_forward_substrate,
    )
    from pdd.export import inspect_pdd_export, load_pdd_export_into_model, pdd_config_from_metadata

    descriptor = inspect_pdd_export(export_dir)
    model_identity = _model_identity(descriptor)
    require_qwen_image_pdd_forward_substrate(
        descriptor.manifest["identity"].get("forward_substrate")
    )
    dtype = _dtype_from_name(model_identity["dtype"])
    loaded_transformer = QwenImageTransformer2DModel.from_config(
        dict(descriptor.transformer_config)
    )
    student = adopt_qwen_image_mr210_forward(loaded_transformer)
    metadata = descriptor.metadata
    _validate_qwen_projection(student, metadata)
    config = pdd_config_from_metadata(metadata, blocks=metadata.inference_blocks)
    convert_qwen_image_to_pdd(student, config)
    descriptor = load_pdd_export_into_model(export_dir, student)
    student.to(dtype=dtype)
    return student, descriptor, dtype


def _normalize_prompt_condition(
    prompt_embeds: Any,
    prompt_mask: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the pinned Diffusers Qwen prompt-encoding contract for PDD."""
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 3:
        raise RuntimeError("Qwen prompt embeddings must have shape [B, S, D].")
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    expected_shape = prompt_embeds.shape[:2]
    if prompt_mask is None:
        prompt_mask = torch.ones(expected_shape, device=device, dtype=torch.long)
    elif not isinstance(prompt_mask, torch.Tensor) or prompt_mask.ndim != 2:
        raise RuntimeError("Qwen prompt mask must have shape [B, S] or be None.")
    elif tuple(prompt_mask.shape) != tuple(expected_shape):
        raise RuntimeError("Qwen prompt mask shape does not match prompt embeddings.")
    elif prompt_mask.dtype.is_floating_point or prompt_mask.dtype.is_complex:
        raise RuntimeError("Qwen prompt mask must use an integer or boolean dtype.")
    else:
        prompt_mask = prompt_mask.to(device=device, dtype=torch.long)
    return prompt_embeds, prompt_mask


def _latent_shape(pipe: Any, *, height: int, width: int) -> tuple[int, int, int, int]:
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers.")
    quantum = int(pipe.vae_scale_factor) * 2
    if height % quantum or width % quantum:
        raise ValueError(f"height and width must be divisible by {quantum}.")
    in_channels = getattr(pipe.transformer.config, "in_channels", None)
    if type(in_channels) is not int or in_channels <= 0 or in_channels % 4:
        raise RuntimeError("Qwen transformer in_channels must be a positive multiple of four.")
    latent_height = 2 * (height // quantum)
    latent_width = 2 * (width // quantum)
    return 1, in_channels // 4, latent_height, latent_width


def _decode_qwen_latents(pipe: Any, latents: torch.Tensor) -> list[Any]:
    if latents.ndim != 4:
        raise ValueError("PDD Qwen latents must have shape [B, C, H, W].")
    vae = pipe.vae
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    if mean.numel() != latents.shape[1] or std.numel() != latents.shape[1]:
        raise RuntimeError("Qwen VAE latent statistics do not match the student channels.")
    decoded_input = latents.unsqueeze(2) * std.view(1, -1, 1, 1, 1)
    decoded_input = decoded_input + mean.view(1, -1, 1, 1, 1)
    decoded = vae.decode(decoded_input, return_dict=False)[0]
    if decoded.ndim != 5 or decoded.shape[2] != 1:
        raise RuntimeError("Qwen VAE must return one-frame 5D image tensors.")
    return pipe.image_processor.postprocess(decoded[:, :, 0], output_type="pil")


def _save_png(path: Path, image: Any) -> None:
    if path.is_symlink():
        raise ValueError("PDD inference output cannot be a symlink.")
    path = path.resolve()
    if path.suffix.lower() != ".png":
        raise ValueError("PDD inference output must use a .png suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"PDD inference output already exists: {path}.")
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        image.save(staging, format="PNG")
        with staging.open("rb") as stream:
            os.fsync(stream.fileno())
        staging.rename(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        staging.unlink(missing_ok=True)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    from diffusers import QwenImagePipeline

    from modelopt.torch.fastgen import PDDPipeline
    from modelopt.torch.fastgen.plugins.qwen_image_pdd import QwenImagePDDAdapter
    from pdd.artifacts import sha256_file, write_canonical_json
    from pdd.export import pdd_config_from_metadata

    if args.output.is_symlink() or args.result_json.is_symlink():
        raise ValueError("PDD output and result JSON cannot be symlinks.")
    output = args.output.resolve()
    result_json = args.result_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"PDD inference output already exists: {output}.")
    if result_json.exists() or result_json.is_symlink():
        raise FileExistsError(f"PDD result JSON already exists: {result_json}.")
    try:
        output_reference = output.relative_to(result_json.parent).as_posix()
    except ValueError as error:
        raise ValueError("PDD output must be beneath the result JSON directory.") from error
    if not isinstance(args.prompt_id, str) or not args.prompt_id.strip():
        raise ValueError("prompt_id must be non-empty.")
    if args.seed < 0 or args.seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63).")
    if args.max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive.")
    student, descriptor, dtype = build_pdd_student(args.export_dir)
    model_identity = _model_identity(descriptor)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    student.to(device=device)
    pipe = QwenImagePipeline.from_pretrained(
        model_identity["id"],
        revision=model_identity["revision"],
        transformer=student,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    config = pdd_config_from_metadata(descriptor.metadata, schedule=args.schedule)
    adapter = QwenImagePDDAdapter(config, compute_dtype=dtype)
    sampler = PDDPipeline(student, nn.Identity(), config, adapter)
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    condition = _normalize_prompt_condition(
        prompt_embeds,
        prompt_mask,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = _latent_shape(pipe, height=args.height, width=args.width)
    noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    transformer_invocations = 0

    def count_invocation(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: Mapping[str, Any]
    ) -> None:
        nonlocal transformer_invocations
        transformer_invocations += 1

    hook = student.register_forward_pre_hook(count_invocation, with_kwargs=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        sampled = sampler.sample(noise, condition=condition)
    finally:
        hook.remove()
    images = _decode_qwen_latents(pipe, sampled.to(dtype))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    expected_invocations = len(config.inference_blocks)
    if transformer_invocations != expected_invocations:
        raise RuntimeError(
            f"PDD sampler made {transformer_invocations} transformer calls; "
            f"expected {expected_invocations}."
        )
    if len(images) != 1:
        raise RuntimeError(f"PDD single-prompt inference returned {len(images)} images.")
    if not math.isfinite(latency) or latency <= 0:
        raise RuntimeError("PDD inference latency measurement is invalid.")
    _save_png(output, images[0])

    result_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "record_type": "pdd_inference",
        "condition": args.schedule.replace("-", "_"),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "schedule": args.schedule,
        "blocks": list(config.inference_blocks),
        "height": args.height,
        "width": args.width,
        "forward_substrate_id": descriptor.manifest["identity"]["forward_substrate"]["id"],
        "export_manifest_sha256": sha256_file(descriptor.root / "manifest.json"),
        "output": {"path": output_reference, "sha256": sha256_file(output)},
        "scheduler_steps": expected_invocations,
        "actual_transformer_invocations": transformer_invocations,
        "batch_normalized_transformer_evaluations": transformer_invocations,
        "latency_seconds": latency,
    }
    write_canonical_json(result_json, result)
    print(result_json)


if __name__ == "__main__":
    main()
