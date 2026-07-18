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

"""Direct PDD updates and logical-ID-stable held-out validation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from modelopt.torch.fastgen import PDDConfig, PDDOutputProjection, PDDPipeline

_TRAINER_STATE_VERSION = 1
_VALIDATION_SCHEMA_VERSION = 1
_VALIDATION_ORDER_DOMAIN = b"modelopt-pdd-validation-order-v1\0"
_VALIDATION_PAIR_DOMAIN = b"modelopt-pdd-validation-pair-v1\0"
_VALIDATION_NOISE_DOMAIN = b"modelopt-pdd-validation-noise-v1\0"


@dataclass(frozen=True)
class PreparedPDDBatch:
    """Canonical PDD inputs extracted from one Qwen cache batch."""

    data: torch.Tensor
    condition: tuple[torch.Tensor, torch.Tensor]
    negative_condition: tuple[torch.Tensor, torch.Tensor] | None
    sample_ids: tuple[str, ...]
    valid_mask: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class PDDStepDiagnostics:
    """Host-side diagnostics for one completed direct student update."""

    completed_step: int
    loss: float
    grad_norm: float
    student_adamw_nominal_update_ratio: float | None
    pdd_projection_update_ratio: float | None
    learning_rate: float
    n: tuple[int, ...]
    k: tuple[int, ...]
    student_velocity_rms: float
    teacher_velocity_rms: float
    student_teacher_velocity_rms_ratio: float
    reconstructed_state_rms: float


@dataclass(frozen=True)
class PDDValidationAssignment:
    """One canonical held-out logical ID and its explicit PDD target indices."""

    ordinal: int
    sample_id: str
    n: int
    k: int


@dataclass(frozen=True)
class PDDValidationRecord:
    """Per-sample deterministic held-out result."""

    ordinal: int
    sample_id: str
    n: int
    k: int
    loss: float


@dataclass(frozen=True)
class PDDValidationResult:
    """Rank-invariant held-out records and canonical float64 aggregate."""

    records: tuple[PDDValidationRecord, ...]
    mean_loss: float
    ordered_id_sha256: str
    pair_count: int
    start_count: int
    head_count: int
    schema_version: int = _VALIDATION_SCHEMA_VERSION


class PDDCoverage:
    """Exact host-side n/k/pair coverage with deterministic coarse head bins."""

    def __init__(self, config: PDDConfig, *, bins: int = 8) -> None:
        if type(bins) is not int or bins <= 0:
            raise ValueError("bins must be a positive integer.")
        self.grid_size = config.grid_size
        self.block_size_min = config.block_size_min
        self.block_size_max = config.block_size_max
        self.bins = min(bins, config.grid_size)
        self.n_counts = torch.zeros(config.grid_size, dtype=torch.int64)
        self.k_counts = torch.zeros(config.grid_size, dtype=torch.int64)
        self.pair_counts = torch.zeros((config.grid_size, config.grid_size), dtype=torch.int64)
        self.bin_counts = torch.zeros(self.bins, dtype=torch.int64)
        self.n_loss_sums = torch.zeros(config.grid_size, dtype=torch.float64)
        self.k_loss_sums = torch.zeros(config.grid_size, dtype=torch.float64)
        self.pair_loss_sums = torch.zeros((config.grid_size, config.grid_size), dtype=torch.float64)
        self.bin_loss_sums = torch.zeros(self.bins, dtype=torch.float64)

    def update(self, n: torch.Tensor, k: torch.Tensor, losses: torch.Tensor) -> None:
        n_cpu = n.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        k_cpu = k.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        losses_cpu = losses.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if n_cpu.shape != k_cpu.shape or n_cpu.shape != losses_cpu.shape:
            raise ValueError("n, k, and loss coverage tensors must have identical shapes.")
        upper = torch.minimum(
            n_cpu + self.block_size_max,
            torch.full_like(n_cpu, self.grid_size),
        )
        valid = (
            (n_cpu >= 0)
            & (n_cpu < self.grid_size)
            & (n_cpu.remainder(self.block_size_min) == 0)
            & (k_cpu >= n_cpu)
            & (k_cpu < upper)
        )
        if not bool(valid.all()):
            raise RuntimeError("observed n/k lies outside the exact configured support.")
        device = n.device
        n_device = n.detach().to(device=device, dtype=torch.int64).reshape(-1)
        k_device = k.detach().to(device=device, dtype=torch.int64).reshape(-1)
        losses_device = losses.detach().to(device=device, dtype=torch.float64).reshape(-1)
        bins_device = torch.minimum(
            k_device * self.bins // self.grid_size,
            torch.full_like(k_device, self.bins - 1),
        )
        pair_indices = n_device * self.grid_size + k_device
        counts = (
            torch.bincount(n_device, minlength=self.grid_size),
            torch.bincount(k_device, minlength=self.grid_size),
            torch.bincount(pair_indices, minlength=self.grid_size * self.grid_size),
            torch.bincount(bins_device, minlength=self.bins),
        )
        loss_sums = []
        for indices, size in (
            (n_device, self.grid_size),
            (k_device, self.grid_size),
            (pair_indices, self.grid_size * self.grid_size),
            (bins_device, self.bins),
        ):
            sums = torch.zeros(size, dtype=torch.float64, device=device)
            sums.index_add_(0, indices, losses_device)
            loss_sums.append(sums)
        if dist.is_available() and dist.is_initialized():
            for tensor in (*counts, *loss_sums):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.n_counts += counts[0].cpu()
        self.k_counts += counts[1].cpu()
        self.pair_counts += counts[2].reshape(self.grid_size, self.grid_size).cpu()
        self.bin_counts += counts[3].cpu()
        self.n_loss_sums += loss_sums[0].cpu()
        self.k_loss_sums += loss_sums[1].cpu()
        self.pair_loss_sums += loss_sums[2].reshape(self.grid_size, self.grid_size).cpu()
        self.bin_loss_sums += loss_sums[3].cpu()

    def require_pairs(self, expected: Sequence[tuple[int, int]]) -> None:
        missing = [pair for pair in expected if int(self.pair_counts[pair]) == 0]
        if missing:
            raise RuntimeError(f"targeted PDD smoke did not cover required n/k pairs: {missing}.")

    def state_dict(self) -> dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "block_size_min": self.block_size_min,
            "block_size_max": self.block_size_max,
            "bins": self.bins,
            "n_counts": self.n_counts.clone(),
            "k_counts": self.k_counts.clone(),
            "pair_counts": self.pair_counts.clone(),
            "bin_counts": self.bin_counts.clone(),
            "n_loss_sums": self.n_loss_sums.clone(),
            "k_loss_sums": self.k_loss_sums.clone(),
            "pair_loss_sums": self.pair_loss_sums.clone(),
            "bin_loss_sums": self.bin_loss_sums.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "grid_size",
            "block_size_min",
            "block_size_max",
            "bins",
            "n_counts",
            "k_counts",
            "pair_counts",
            "bin_counts",
            "n_loss_sums",
            "k_loss_sums",
            "pair_loss_sums",
            "bin_loss_sums",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("PDD coverage state has incompatible keys.")
        identity = (
            state["grid_size"],
            state["block_size_min"],
            state["block_size_max"],
            state["bins"],
        )
        if identity != (
            self.grid_size,
            self.block_size_min,
            self.block_size_max,
            self.bins,
        ):
            raise ValueError("PDD coverage state does not match the current configuration.")
        for name in (
            "n_counts",
            "k_counts",
            "pair_counts",
            "bin_counts",
            "n_loss_sums",
            "k_loss_sums",
            "pair_loss_sums",
            "bin_loss_sums",
        ):
            saved = state[name]
            current = getattr(self, name)
            if not isinstance(saved, torch.Tensor) or saved.shape != current.shape:
                raise ValueError(f"PDD coverage {name} has an incompatible tensor shape.")
            current.copy_(saved.to(device="cpu", dtype=current.dtype))


def prepare_qwen_pdd_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    require_negative_condition: bool,
    expected_latent_channels: int,
    expected_condition_features: int,
) -> PreparedPDDBatch:
    """Move a portable Qwen cache batch into the PDD adapter contract."""
    if type(expected_latent_channels) is not int or expected_latent_channels <= 0:
        raise ValueError("expected_latent_channels must be a positive integer.")
    if type(expected_condition_features) is not int or expected_condition_features <= 0:
        raise ValueError("expected_condition_features must be a positive integer.")
    if not dtype.is_floating_point:
        raise TypeError("Qwen PDD model dtype must be floating point.")
    if not isinstance(batch, Mapping):
        raise TypeError(f"batch must be a mapping, got {type(batch).__name__}.")
    required = {"image_latents", "text_embeddings", "text_embeddings_mask", "metadata"}
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"Qwen PDD batch is missing required keys: {missing}.")
    data = batch["image_latents"]
    text = batch["text_embeddings"]
    mask = batch["text_embeddings_mask"]
    metadata = batch["metadata"]
    if not all(isinstance(value, torch.Tensor) for value in (data, text, mask)):
        raise TypeError("Qwen PDD latent, text embedding, and mask values must be tensors.")
    if data.ndim != 4:
        raise ValueError(f"Qwen PDD image_latents must be 4D, got {tuple(data.shape)}.")
    if not data.dtype.is_floating_point:
        raise TypeError("Qwen PDD image_latents must use a floating-point dtype.")
    if data.shape[0] <= 0 or data.shape[1] != expected_latent_channels:
        raise ValueError(
            "Qwen PDD image_latents must have a non-empty batch and exactly "
            f"{expected_latent_channels} channels, got {tuple(data.shape)}."
        )
    if data.shape[2] <= 0 or data.shape[3] <= 0 or data.shape[2] % 2 or data.shape[3] % 2:
        raise ValueError(
            "Qwen PDD image_latents must have positive even spatial dimensions, got "
            f"{tuple(data.shape[2:])}."
        )
    if not isinstance(metadata, Mapping):
        raise TypeError("Qwen PDD batch metadata must be a mapping.")
    sample_ids = metadata.get("logical_sample_ids", metadata.get("sample_ids"))
    if isinstance(sample_ids, torch.Tensor):
        sample_ids = tuple(str(value) for value in sample_ids.tolist())
    if isinstance(sample_ids, str) or not isinstance(sample_ids, Sequence):
        raise TypeError("Qwen PDD metadata.sample_ids must be a sequence of strings.")
    sample_ids = tuple(sample_ids)
    if len(sample_ids) != data.shape[0] or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids
    ):
        raise ValueError("Qwen PDD sample_ids must be non-empty strings matching batch size.")

    def prepare_condition(
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not embeddings.dtype.is_floating_point:
            raise TypeError(f"{name} embeddings must use a floating-point dtype.")
        if attention_mask.dtype.is_floating_point or attention_mask.dtype.is_complex:
            raise TypeError(f"{name} mask must use an integer or boolean dtype.")
        if embeddings.ndim not in (2, 3):
            raise ValueError(f"{name} embeddings must be 2D or 3D, got {embeddings.ndim}D.")
        if attention_mask.ndim not in (1, 2):
            raise ValueError(f"{name} mask must be 1D or 2D, got {attention_mask.ndim}D.")
        if embeddings.shape[-2] <= 0 or embeddings.shape[-1] <= 0:
            raise ValueError(f"{name} embeddings must have non-empty sequence and feature axes.")
        if embeddings.shape[-1] != expected_condition_features:
            raise ValueError(
                f"{name} embeddings must have exactly {expected_condition_features} features."
            )
        if attention_mask.shape[-1] != embeddings.shape[-2]:
            raise ValueError(f"{name} mask sequence length must match its embeddings.")
        if embeddings.ndim == 3 and embeddings.shape[0] != data.shape[0]:
            raise ValueError(f"{name} embedding batch size must match image_latents.")
        if attention_mask.ndim == 2 and attention_mask.shape[0] != data.shape[0]:
            raise ValueError(f"{name} mask batch size must match image_latents.")

        embeddings = embeddings.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0).expand(data.shape[0], -1, -1).contiguous()
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0).expand(data.shape[0], -1).contiguous()
        return embeddings, attention_mask

    data = data.to(device=device, dtype=dtype)
    condition = prepare_condition(text, mask, name="Qwen PDD condition")

    negative: tuple[torch.Tensor, torch.Tensor] | None = None
    negative_text = batch.get("negative_text_embeddings")
    negative_mask = batch.get("negative_text_embeddings_mask")
    if negative_text is not None or negative_mask is not None:
        if not isinstance(negative_text, torch.Tensor) or not isinstance(
            negative_mask, torch.Tensor
        ):
            raise TypeError("negative Qwen conditioning requires embedding and mask tensors.")
        negative = prepare_condition(
            negative_text,
            negative_mask,
            name="negative Qwen PDD condition",
        )
    if require_negative_condition and negative is None:
        raise ValueError("guided Qwen PDD training requires negative prompt conditioning.")
    return PreparedPDDBatch(data, condition, negative, sample_ids, (True,) * len(sample_ids))


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _replication_factor(value: torch.Tensor) -> int:
    placements = getattr(value, "placements", ())
    mesh = getattr(value, "device_mesh", None)
    factor = 1
    if mesh is not None:
        for dimension, placement in enumerate(placements):
            if type(placement).__name__ == "Replicate":
                factor *= int(mesh.size(dimension))
    return factor


def _global_squared_sum(values: Sequence[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=device)
    for value in values:
        local = _local_tensor(value.detach())
        contribution = local.float().square().sum(dtype=torch.float64)
        total += contribution / _replication_factor(value)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


def _global_any(flag: bool, *, device: torch.device) -> bool:
    tensor = torch.tensor(int(flag), dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def _all_parameters_finite(parameters: Sequence[torch.Tensor], *, device: torch.device) -> bool:
    local_finite = torch.ones((), dtype=torch.bool, device=device)
    for parameter in parameters:
        local_finite.logical_and_(torch.isfinite(_local_tensor(parameter.detach())).all())
    return not _global_any(not bool(local_finite.item()), device=device)


def _global_sample_mean(value: torch.Tensor) -> float:
    value = value.detach().reshape(-1)
    totals = torch.stack(
        (
            value.double().sum(),
            torch.tensor(float(value.numel()), dtype=torch.float64, device=value.device),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[1] <= 0:
        raise RuntimeError("cannot aggregate an empty PDD sample metric.")
    return float((totals[0] / totals[1]).item())


def _require_supported_adamw(optimizer: torch.optim.Optimizer) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("PDD v1 diagnostics require the stock torch.optim.AdamW optimizer.")
    rejected_truthy = ("amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused")
    for group_index, group in enumerate(optimizer.param_groups):
        for name in rejected_truthy:
            if group.get(name) is not False:
                raise ValueError(
                    f"PDD v1 requires AdamW param_groups[{group_index}][{name!r}]=False."
                )
        lr = group.get("lr")
        betas = group.get("betas")
        eps = group.get("eps")
        weight_decay = group.get("weight_decay")
        if not isinstance(lr, float) or not isinstance(weight_decay, float):
            raise TypeError("PDD v1 AdamW learning rate and weight decay must be scalar floats.")
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or any(not isinstance(beta, float) for beta in betas)
            or not isinstance(eps, float)
        ):
            raise TypeError("PDD v1 AdamW betas and epsilon must be scalar floats.")
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("PDD v1 AdamW learning rate/weight decay is invalid.")
        if any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("PDD v1 AdamW betas must be finite and in [0, 1).")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("PDD v1 AdamW epsilon must be finite and > 0.")


def _adamw_nominal_update_ratio(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    """Stream the public AdamW equation over local shards without cloning the model."""
    update_squared = torch.zeros((), dtype=torch.float64, device=device)
    parameter_squared = torch.zeros((), dtype=torch.float64, device=device)
    for group in optimizer.param_groups:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        decay = 1.0 - lr * group["weight_decay"]
        if decay <= 0.0:
            raise RuntimeError("AdamW decoupled weight decay factor must remain positive.")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state.get(parameter)
            if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
                continue
            step_value = state.get("step")
            if isinstance(step_value, torch.Tensor):
                step = float(step_value.item())
            else:
                step = float(step_value)
            if step <= 0:
                raise RuntimeError("AdamW state has a non-positive step after optimizer.step().")
            parameter_local = _local_tensor(parameter.detach()).float()
            exp_avg = _local_tensor(state["exp_avg"].detach()).float()
            exp_avg_sq = _local_tensor(state["exp_avg_sq"].detach()).float()
            bias_correction1 = 1.0 - beta1**step
            bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
            denominator = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
            direction = exp_avg.div(denominator).div_(bias_correction1)
            parameter_before = (parameter_local + lr * direction) / decay
            nominal_delta = parameter_local - parameter_before
            factor = _replication_factor(parameter)
            update_squared += nominal_delta.square().sum(dtype=torch.float64) / factor
            parameter_squared += parameter_before.square().sum(dtype=torch.float64) / factor
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(update_squared, op=dist.ReduceOp.SUM)
        dist.all_reduce(parameter_squared, op=dist.ReduceOp.SUM)
    if not bool(torch.isfinite(update_squared) & torch.isfinite(parameter_squared)):
        raise RuntimeError("PDD AdamW nominal update diagnostics became non-finite.")
    return float((update_squared.sqrt() / parameter_squared.sqrt().clamp_min(1e-30)).item())


class PDDTrainer:
    """Own direct student updates while leaving algorithm and checkpoint state separate."""

    def __init__(
        self,
        pipeline: PDDPipeline,
        optimizer: torch.optim.Optimizer,
        *,
        projection: PDDOutputProjection,
        max_grad_norm: float,
        warmup_steps: int = 0,
    ) -> None:
        if not isinstance(pipeline, PDDPipeline):
            raise TypeError(f"pipeline must be PDDPipeline, got {type(pipeline).__name__}.")
        if not isinstance(projection, PDDOutputProjection):
            raise TypeError("projection must be PDDOutputProjection.")
        if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, int | float):
            raise TypeError("max_grad_norm must be a real number.")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and > 0.")
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be an integer >= 0.")
        _require_supported_adamw(optimizer)
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.projection = projection
        self.max_grad_norm = float(max_grad_norm)
        self.warmup_steps = warmup_steps
        self.completed_steps = 0
        self.consecutive_zero_grad_steps = 0
        self.coverage = PDDCoverage(pipeline.config)

    def _projection_snapshot(self) -> list[torch.Tensor]:
        parameters = [self.projection.weight]
        if self.projection.bias is not None:
            parameters.append(self.projection.bias)
        return [_local_tensor(parameter.detach()).clone() for parameter in parameters]

    def _projection_update_ratio(self, before: Sequence[torch.Tensor]) -> float:
        parameters = [self.projection.weight]
        if self.projection.bias is not None:
            parameters.append(self.projection.bias)
        update_squared = torch.zeros((), dtype=torch.float64, device=self.pipeline.device)
        parameter_squared = torch.zeros((), dtype=torch.float64, device=self.pipeline.device)
        for parameter, saved in zip(parameters, before):
            local = _local_tensor(parameter.detach()).float()
            factor = _replication_factor(parameter)
            update_squared += (local - saved.float()).square().sum(dtype=torch.float64) / factor
            parameter_squared += local.square().sum(dtype=torch.float64) / factor
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(update_squared, op=dist.ReduceOp.SUM)
            dist.all_reduce(parameter_squared, op=dist.ReduceOp.SUM)
        if not bool(torch.isfinite(update_squared) & torch.isfinite(parameter_squared)):
            raise RuntimeError("PDD projection update diagnostics became non-finite.")
        return float((update_squared.sqrt() / parameter_squared.sqrt().clamp_min(1e-30)).item())

    def train_step(
        self,
        batch: PreparedPDDBatch,
        *,
        noise: torch.Tensor | None = None,
        n: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        measure_updates: bool = True,
    ) -> PDDStepDiagnostics:
        """Run one direct PDD student update and enforce all immediate hard aborts."""
        if not isinstance(batch, PreparedPDDBatch):
            raise TypeError("batch must be PreparedPDDBatch.")
        self.pipeline.student.train()
        self.pipeline.teacher.eval()
        self.optimizer.zero_grad(set_to_none=True)
        before_projection = self._projection_snapshot() if measure_updates else None
        loss, metrics = self.pipeline.compute_loss(
            batch.data,
            noise=noise,
            condition=batch.condition,
            negative_condition=batch.negative_condition,
            n=n,
            k=k,
            generator=generator,
        )
        finite_metrics = (
            "all_student_heads_finite",
            "student_target_finite",
            "teacher_target_finite",
            "reconstructed_state_finite",
            "loss_finite",
        )
        local_nonfinite = not bool(torch.isfinite(loss)) or any(
            not bool(metrics[name].all()) for name in finite_metrics
        )
        if _global_any(local_nonfinite, device=batch.data.device):
            raise FloatingPointError(
                "PDD loss, prediction, target, or reconstruction is non-finite."
            )
        self.coverage.update(metrics["n"], metrics["k"], metrics["student_target_mse"])
        loss.backward()

        teacher_gradient = any(
            parameter.grad is not None for parameter in self.pipeline.teacher.parameters()
        )
        if _global_any(teacher_gradient, device=batch.data.device):
            raise RuntimeError("PDD frozen teacher received a gradient.")
        trainable = [
            parameter for parameter in self.pipeline.student.parameters() if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients:
            grad_norm = 0.0
        else:
            grad_squared = _global_squared_sum(gradients, device=batch.data.device)
            if not bool(torch.isfinite(grad_squared)):
                raise FloatingPointError("PDD student gradient became non-finite.")
            grad_norm = float(grad_squared.sqrt().item())
        if self.completed_steps >= self.warmup_steps and grad_norm == 0.0:
            self.consecutive_zero_grad_steps += 1
            if self.consecutive_zero_grad_steps >= 2:
                raise RuntimeError("PDD student gradient was zero for two consecutive updates.")
        else:
            self.consecutive_zero_grad_steps = 0
        clip_coefficient = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))
        if clip_coefficient < 1.0:
            for gradient in gradients:
                gradient.mul_(clip_coefficient)

        self.optimizer.step()
        if not _all_parameters_finite(trainable, device=batch.data.device):
            raise FloatingPointError("PDD student parameter update became non-finite.")
        nominal_ratio = (
            _adamw_nominal_update_ratio(
                self.optimizer,
                device=batch.data.device,
            )
            if measure_updates
            else None
        )
        projection_ratio = (
            self._projection_update_ratio(before_projection)
            if before_projection is not None
            else None
        )
        if nominal_ratio == 0.0 and grad_norm > 0.0:
            raise RuntimeError(
                "PDD optimizer produced a zero nominal update from a nonzero gradient."
            )
        if projection_ratio == 0.0 and grad_norm > 0.0:
            raise RuntimeError(
                "PDD optimizer produced a zero actual projection update from a nonzero gradient."
            )
        self.completed_steps += 1
        student_velocity_rms = _global_sample_mean(metrics["student_velocity_rms"])
        teacher_velocity_rms = _global_sample_mean(metrics["teacher_velocity_rms"])

        return PDDStepDiagnostics(
            completed_step=self.completed_steps,
            loss=_global_sample_mean(metrics["student_target_mse"]),
            grad_norm=grad_norm,
            student_adamw_nominal_update_ratio=nominal_ratio,
            pdd_projection_update_ratio=projection_ratio,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            n=tuple(int(value) for value in metrics["n"].detach().cpu().tolist()),
            k=tuple(int(value) for value in metrics["k"].detach().cpu().tolist()),
            student_velocity_rms=student_velocity_rms,
            teacher_velocity_rms=teacher_velocity_rms,
            student_teacher_velocity_rms_ratio=student_velocity_rms
            / max(teacher_velocity_rms, 1e-30),
            reconstructed_state_rms=_global_sample_mean(metrics["reconstructed_state_rms"]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _TRAINER_STATE_VERSION,
            "completed_steps": self.completed_steps,
            "consecutive_zero_grad_steps": self.consecutive_zero_grad_steps,
            "coverage": self.coverage.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "completed_steps",
            "consecutive_zero_grad_steps",
            "coverage",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("PDD trainer state has incompatible keys.")
        if state["schema_version"] != _TRAINER_STATE_VERSION:
            raise ValueError(f"unsupported PDD trainer schema {state['schema_version']!r}.")
        completed = state["completed_steps"]
        zero_steps = state["consecutive_zero_grad_steps"]
        if type(completed) is not int or completed < 0:
            raise ValueError("saved completed_steps must be an integer >= 0.")
        if type(zero_steps) is not int or zero_steps < 0:
            raise ValueError("saved consecutive_zero_grad_steps must be an integer >= 0.")
        self.completed_steps = completed
        self.consecutive_zero_grad_steps = zero_steps
        self.coverage.load_state_dict(state["coverage"])


def _stable_digest(domain: bytes, validation_seed: int, payload: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(validation_seed).encode())
    digest.update(b"\0")
    digest.update(payload.encode())
    return digest.digest()


def pdd_validation_support(config: PDDConfig) -> tuple[tuple[int, int], ...]:
    """Return the exact lexicographic support of explicit PDD validation pairs."""
    return tuple(
        (n, k)
        for n in range(0, config.grid_size, config.block_size_min)
        for k in range(n, min(n + config.block_size_max, config.grid_size))
    )


def build_pdd_validation_assignments(
    sample_ids: Sequence[str],
    config: PDDConfig,
    *,
    validation_seed: int,
    require_full_coverage: bool = True,
) -> tuple[PDDValidationAssignment, ...]:
    """Assign every logical ID a rank/batch-order-independent explicit n/k pair."""
    if isinstance(sample_ids, str) or not isinstance(sample_ids, Sequence):
        raise TypeError("sample_ids must be a sequence of strings.")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("sample_ids must contain non-empty strings.")
    if not sample_ids:
        raise ValueError("sample_ids must contain at least one logical ID.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("held-out validation sample_ids must be unique.")
    if type(validation_seed) is not int or validation_seed < 0:
        raise ValueError("validation_seed must be an integer >= 0.")
    support = pdd_validation_support(config)
    if require_full_coverage and len(sample_ids) < len(support):
        raise ValueError(
            f"full PDD validation coverage requires at least {len(support)} logical IDs, "
            f"found {len(sample_ids)}."
        )
    ordered_ids = sorted(
        sample_ids,
        key=lambda sample_id: (
            _stable_digest(_VALIDATION_ORDER_DOMAIN, validation_seed, sample_id),
            sample_id,
        ),
    )
    permuted_support = sorted(
        support,
        key=lambda pair: (
            _stable_digest(_VALIDATION_PAIR_DOMAIN, validation_seed, f"{pair[0]}:{pair[1]}"),
            pair,
        ),
    )
    return tuple(
        PDDValidationAssignment(ordinal, sample_id, *permuted_support[ordinal % len(support)])
        for ordinal, sample_id in enumerate(ordered_ids)
    )


def pdd_validation_noise(
    sample_id: str,
    shape: Sequence[int],
    *,
    validation_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate per-ID CPU float32 noise without touching the global RNG."""
    digest = _stable_digest(_VALIDATION_NOISE_DOMAIN, validation_seed, sample_id)
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)


def _ordered_id_digest(records: Sequence[PDDValidationRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(b"modelopt-pdd-ordered-validation-ids-v1\0")
    for record in records:
        digest.update(record.sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _raise_collective_validation_error(error: BaseException | None, *, context: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        if error is not None:
            raise error
        return
    local = None if error is None else f"{type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError(f"distributed PDD validation {context} failed; " + "; ".join(failures))


def _reshard_fsdp2_modules(model: torch.nn.Module) -> None:
    from torch.distributed.fsdp import FSDPModule

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def run_pdd_validation(
    pipeline: PDDPipeline,
    batches: Iterable[PreparedPDDBatch],
    assignments: Sequence[PDDValidationAssignment],
    *,
    validation_seed: int,
) -> PDDValidationResult:
    """Evaluate explicit per-ID targets and aggregate identically across rank partitions."""
    if not isinstance(pipeline, PDDPipeline):
        raise TypeError("pipeline must be PDDPipeline.")
    distributed = dist.is_available() and dist.is_initialized()
    assignment_by_id = {assignment.sample_id: assignment for assignment in assignments}
    assignment_error: BaseException | None = None
    if len(assignment_by_id) != len(assignments):
        assignment_error = ValueError("validation assignments contain duplicate logical IDs.")
    elif not assignments:
        assignment_error = ValueError("validation assignments cannot be empty.")
    _raise_collective_validation_error(assignment_error, context="assignment preflight")
    if distributed:
        assignment_identity = tuple(
            (item.ordinal, item.sample_id, item.n, item.k) for item in assignments
        )
        assignment_identities: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(assignment_identities, assignment_identity)
        if any(identity != assignment_identities[0] for identity in assignment_identities[1:]):
            raise RuntimeError("distributed PDD validation assignments differ across ranks.")
    student_was_training = pipeline.student.training
    teacher_was_training = pipeline.teacher.training
    pipeline.student.eval()
    pipeline.teacher.eval()
    local_records: list[PDDValidationRecord] = []
    try:
        with torch.no_grad():
            iterator = iter(batches)
            batch_index = 0
            while True:
                batch = None
                next_error: BaseException | None = None
                exhausted = False
                try:
                    batch = next(iterator)
                except StopIteration:
                    exhausted = True
                except BaseException as error:
                    next_error = error
                if distributed:
                    status = "error" if next_error is not None else "end" if exhausted else "batch"
                    statuses: list[str] = [""] * dist.get_world_size()
                    dist.all_gather_object(statuses, status)
                    if "error" in statuses:
                        _raise_collective_validation_error(next_error, context="iteration")
                    if all(item == "end" for item in statuses):
                        break
                    if any(item != "batch" for item in statuses):
                        raise RuntimeError(
                            "distributed PDD validation ranks produced different batch counts."
                        )
                else:
                    if next_error is not None:
                        raise next_error
                    if exhausted:
                        break

                local_error: BaseException | None = None
                selected: list[PDDValidationAssignment] = []
                valid_mask: tuple[bool, ...] = ()
                try:
                    if not isinstance(batch, PreparedPDDBatch):
                        raise TypeError("validation batches must contain PreparedPDDBatch values.")
                    valid_mask = (
                        (True,) * len(batch.sample_ids)
                        if batch.valid_mask is None
                        else batch.valid_mask
                    )
                    if len(valid_mask) != len(batch.sample_ids) or any(
                        type(valid) is not bool for valid in valid_mask
                    ):
                        raise ValueError(
                            "validation valid_mask must contain one bool per sample ID."
                        )
                    for position, (sample_id, valid) in enumerate(
                        zip(batch.sample_ids, valid_mask)
                    ):
                        if valid and sample_id not in assignment_by_id:
                            raise ValueError(
                                f"validation batch contains unassigned sample ID {sample_id!r}."
                            )
                        if valid:
                            selected.append(assignment_by_id[sample_id])
                        else:
                            template = assignments[(batch_index + position) % len(assignments)]
                            selected.append(
                                PDDValidationAssignment(
                                    template.ordinal,
                                    f"__pdd_dummy__:{batch_index}:{position}",
                                    template.n,
                                    template.k,
                                )
                            )
                except BaseException as error:
                    local_error = error
                _raise_collective_validation_error(local_error, context="batch preflight")
                assert isinstance(batch, PreparedPDDBatch)
                # Inputs remain rank-local. Equal batch counts above preserve collective
                # ordering, while prompt sequence padding may legitimately differ by rank.
                noise = torch.stack(
                    [
                        pdd_validation_noise(
                            assignment.sample_id,
                            batch.data.shape[1:],
                            validation_seed=validation_seed,
                            device=batch.data.device,
                        )
                        for assignment in selected
                    ]
                )
                n = torch.tensor(
                    [assignment.n for assignment in selected],
                    dtype=torch.long,
                    device=batch.data.device,
                )
                k = torch.tensor(
                    [assignment.k for assignment in selected],
                    dtype=torch.long,
                    device=batch.data.device,
                )
                _, metrics = pipeline.compute_loss(
                    batch.data,
                    noise=noise,
                    condition=batch.condition,
                    negative_condition=batch.negative_condition,
                    n=n,
                    k=k,
                )
                finite_metrics = (
                    "all_student_heads_finite",
                    "student_target_finite",
                    "teacher_target_finite",
                    "reconstructed_state_finite",
                    "loss_finite",
                )
                local_nonfinite = any(not bool(metrics[name].all()) for name in finite_metrics)
                losses = metrics["student_target_mse"].double().cpu().tolist()
                local_nonfinite = local_nonfinite or any(not math.isfinite(loss) for loss in losses)
                if _global_any(local_nonfinite, device=batch.data.device):
                    raise FloatingPointError(
                        "deterministic PDD validation produced a non-finite prediction, target, "
                        "or loss."
                    )
                local_records.extend(
                    PDDValidationRecord(
                        assignment.ordinal,
                        assignment.sample_id,
                        assignment.n,
                        assignment.k,
                        float(loss),
                    )
                    for assignment, loss, valid in zip(selected, losses, valid_mask)
                    if valid
                )
                batch_index += 1
    finally:
        pipeline.student.train(student_was_training)
        pipeline.teacher.train(teacher_was_training)
        _reshard_fsdp2_modules(pipeline.student)
        _reshard_fsdp2_modules(pipeline.teacher)

    gathered: list[list[PDDValidationRecord]]
    if distributed:
        gathered = [[] for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]
    records = sorted(
        (record for rank_records in gathered for record in rank_records),
        key=lambda record: record.ordinal,
    )
    if len(records) != len(assignments):
        raise RuntimeError(
            f"validation contributed {len(records)} records for {len(assignments)} assignments."
        )
    if [record.ordinal for record in records] != list(range(len(assignments))):
        raise RuntimeError("validation ordinals are duplicated or missing across ranks.")
    for record, assignment in zip(records, assignments):
        if (record.sample_id, record.n, record.k) != (
            assignment.sample_id,
            assignment.n,
            assignment.k,
        ):
            raise RuntimeError("validation record does not match its canonical assignment.")
    pairs = {(record.n, record.k) for record in records}
    starts = {record.n for record in records}
    heads = {record.k for record in records}
    return PDDValidationResult(
        records=tuple(records),
        mean_loss=math.fsum(record.loss for record in records) / len(records),
        ordered_id_sha256=_ordered_id_digest(records),
        pair_count=len(pairs),
        start_count=len(starts),
        head_count=len(heads),
    )
