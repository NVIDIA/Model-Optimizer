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

"""Shared authenticated Qwen-Image PDD inference runtime."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

from .artifacts import canonical_json_bytes
from .export import PDD_INFERENCE_SCHEDULES, pdd_config_from_metadata

_TENSOR_HASH_DOMAINS = {
    "raw_noise",
    "initial_state",
    "full_time_nodes",
    "boundary_time_nodes",
}


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
    from modelopt.torch.fastgen.plugins.qwen_image_pdd import QWEN_IMAGE_PDD_EXECUTION

    identity = descriptor.manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("PDD export has no identity mapping.")
    if identity.get("qwen_image") != {"execution": QWEN_IMAGE_PDD_EXECUTION}:
        raise RuntimeError("PDD export has an incompatible Qwen execution identity.")
    model = identity.get("model")
    if not isinstance(model, Mapping) or set(model) != {"id", "revision", "dtype"}:
        raise RuntimeError("PDD export model identity is malformed.")
    if not isinstance(model["id"], str) or not model["id"]:
        raise RuntimeError("PDD export model ID is invalid.")
    revision = model["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("PDD export model revision must be an exact lowercase commit.")
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
        raise RuntimeError("reconstructed Qwen proj_out does not match the export metadata.")
    if base_projection.out_features != in_channels:
        raise RuntimeError(
            "Qwen proj_out width must equal transformer in_channels for 2x2 latent packing."
        )
    return base_projection


def build_pdd_student(
    export_dir: str | Path, *, schedule: str = "pdd-4"
) -> tuple[nn.Module, Any, torch.dtype]:
    """Reconstruct and strictly load a converted Qwen student on CPU."""
    from diffusers import QwenImageTransformer2DModel

    from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
        adopt_qwen_image_mr210_forward,
        convert_qwen_image_to_pdd,
    )

    from .export import inspect_pdd_export, load_pdd_export_into_model

    if schedule not in PDD_INFERENCE_SCHEDULES:
        raise ValueError(
            f"Unknown PDD schedule {schedule!r}; expected {sorted(PDD_INFERENCE_SCHEDULES)}."
        )
    descriptor = inspect_pdd_export(export_dir)
    model_identity = _model_identity(descriptor)
    dtype = _dtype_from_name(model_identity["dtype"])
    student = QwenImageTransformer2DModel.from_config(dict(descriptor.transformer_config))
    _validate_qwen_projection(student, descriptor.metadata)
    student = adopt_qwen_image_mr210_forward(student)
    config = pdd_config_from_metadata(descriptor.metadata, schedule=schedule)
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
    return 1, in_channels // 4, 2 * (height // quantum), 2 * (width // quantum)


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


def pdd_tensor_sha256(tensor: torch.Tensor, domain: str) -> str:
    """Hash one exact FP32 tensor using the evaluation protocol."""
    if domain not in _TENSOR_HASH_DOMAINS:
        raise ValueError(f"unknown PDD tensor hash domain {domain!r}.")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise TypeError("PDD tensor hashing requires a float32 tensor.")
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError("PDD tensor hashing rejects non-finite values.")
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    if array.dtype.str != "<f4":
        raise RuntimeError("PDD tensor hash payload is not little-endian float32.")
    header = {
        "schema_version": 1,
        "domain": domain,
        "dtype": "float32",
        "shape": list(array.shape),
        "byte_order": "little",
        "order": "C",
    }
    return hashlib.sha256(
        canonical_json_bytes(header) + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def save_png(path: Path, image: Any) -> None:
    """Publish one PNG exclusively and durably."""
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


@dataclass(frozen=True)
class QwenPDDInferenceRuntime:
    """One loaded Qwen pipeline and one authenticated PDD sampler."""

    student: nn.Module
    scheduler: Any
    descriptor: Any
    model_identity: Mapping[str, Any]
    dtype: torch.dtype
    device: torch.device
    config: Any
    pipe: Any
    sampler: Any

    def encode_prompt(self, prompt: str, max_sequence_length: int) -> Any:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string.")
        if type(max_sequence_length) is not int or max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive.")
        prompt_embeds, prompt_mask = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        return _normalize_prompt_condition(
            prompt_embeds,
            prompt_mask,
            device=self.device,
            dtype=self.dtype,
        )

    def make_raw_noise(self, *, seed: int, height: int, width: int) -> torch.Tensor:
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63).")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = _latent_shape(self.pipe, height=height, width=width)
        return torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    def sample_decode(self, condition: Any, raw_noise: torch.Tensor) -> list[Any]:
        sampled = self.sampler.sample(raw_noise, condition=condition)
        return _decode_qwen_latents(self.pipe, sampled.to(self.dtype))

    def trajectory_identity(self, raw_noise: torch.Tensor) -> dict[str, Any]:
        full = self.sampler.time_grid(raw_noise.device).to(device="cpu", dtype=torch.float32)
        boundaries = [0]
        for block in self.config.inference_blocks:
            boundaries.append(boundaries[-1] + block)
        boundary = full[boundaries]
        initial = (raw_noise.to(torch.float64) * self.config.grid_max_t).to(torch.float32)
        return {
            "raw_noise_sha256": pdd_tensor_sha256(raw_noise, "raw_noise"),
            "initial_state_sha256": pdd_tensor_sha256(initial, "initial_state"),
            "full_time_nodes": full.tolist(),
            "full_time_nodes_sha256": pdd_tensor_sha256(full, "full_time_nodes"),
            "boundary_indices": boundaries,
            "boundary_time_nodes": boundary.tolist(),
            "boundary_time_nodes_sha256": pdd_tensor_sha256(boundary, "boundary_time_nodes"),
            "first_sigma": float(full[0].item()),
        }


def load_qwen_pdd_runtime(
    export_dir: str | Path, schedule: str, device: str | torch.device
) -> QwenPDDInferenceRuntime:
    """Load one authenticated Qwen PDD runtime for a source-owned schedule."""
    from diffusers import QwenImagePipeline

    from modelopt.torch.fastgen import PDDPipeline
    from modelopt.torch.fastgen.plugins.qwen_image_pdd import QwenImagePDDAdapter

    if schedule not in PDD_INFERENCE_SCHEDULES:
        raise ValueError(
            f"Unknown PDD schedule {schedule!r}; expected {sorted(PDD_INFERENCE_SCHEDULES)}."
        )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    student, descriptor, dtype = build_pdd_student(export_dir, schedule=schedule)
    model_identity = _model_identity(descriptor)
    student.to(device=resolved_device)
    pipe = QwenImagePipeline.from_pretrained(
        model_identity["id"],
        revision=model_identity["revision"],
        transformer=student,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if pipe.transformer is not student:
        raise RuntimeError("Qwen pipeline did not retain the adopted PDD transformer.")
    pipe.to(resolved_device)
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None or not callable(getattr(scheduler, "step", None)):
        raise RuntimeError("Qwen pipeline scheduler does not expose a callable step method.")
    config = pdd_config_from_metadata(descriptor.metadata, schedule=schedule)
    if tuple(config.inference_blocks) != PDD_INFERENCE_SCHEDULES[schedule]:
        raise RuntimeError("authenticated PDD schedule changed after validation.")
    sampler = PDDPipeline(
        student,
        nn.Identity(),
        config,
        QwenImagePDDAdapter(config, compute_dtype=dtype),
    )
    return QwenPDDInferenceRuntime(
        student=student,
        scheduler=scheduler,
        descriptor=descriptor,
        model_identity=model_identity,
        dtype=dtype,
        device=resolved_device,
        config=config,
        pipe=pipe,
        sampler=sampler,
    )
