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

"""GPTQ-initialized Gumbel-Softmax refinement of native IQ candidate bundles."""

import hashlib
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from modelopt.torch.utils import print_rank_0

from .common import pin_packed_weight
from .gsq_candidates import build_candidates
from .registry import IQ_FORMAT_REGISTRY

__all__ = ["IQGSQResult", "quantize_iq_gsq"]

_GSQ_CAPTURE: ContextVar[Any] = ContextVar("iq_gsq_capture", default=None)
_SETTINGS: dict[str, Any] = {
    "logit_lr": 1e-3,
    "relative_scale_lr": 1e-3,
    "temperature": (2.0, 0.05),
    "logit_multiplier": (100.0, 500.0),
    "logit_prior": 0.06,
    "logit_noise_std": 0.01,
    "logit_weight_decay": 1.0,
    "lion_betas": (0.9, 0.95),
    "lr_floor": 0.1,
    "checkpoint_interval": 25,
    "rows_per_chunk": 4096,
}


@dataclass
class IQGSQResult:
    """Native GPTQ and GSQ states that can be activated without requantization."""

    payloads: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    formats: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @torch.no_grad()
    def activate(self, model: torch.nn.Module, num_steps: int) -> None:
        """Activate an evaluated budget, or the exact GPTQ baseline with ``num_steps=0``."""
        targets = []
        for name, states in self.payloads.items():
            module = model.get_submodule(name)
            quantizer = module.weight_quantizer
            format_name = self.formats[name]
            if quantizer.backend != "ggml" or quantizer.num_bits != format_name:
                raise ValueError(f"IQ GSQ format does not match module {name!r}.")
            if num_steps not in states:
                raise ValueError(f"No IQ GSQ checkpoint for {num_steps} updates at {name!r}.")
            payload = states[num_steps]
            fmt = IQ_FORMAT_REGISTRY[format_name]
            expected = (*module.weight.shape[:-1], module.weight.shape[-1] // 256, fmt.block_bytes)
            if payload.dtype != torch.uint8 or tuple(payload.shape) != expected:
                raise ValueError(f"IQ GSQ payload shape does not match module {name!r}.")
            targets.append((module, quantizer, fmt, payload))
        for module, quantizer, fmt, payload in targets:
            payload = payload.to(module.weight.device)
            decoded = fmt.dequantize(
                payload, torch.tensor(module.weight.shape), dtype=module.weight.dtype
            )
            module.weight.copy_(decoded)
            pin_packed_weight(quantizer, fmt.name, payload, source=module.weight)

    def save(self, path: str | Path) -> None:
        """Save only native payloads and experiment metrics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {"payloads": self.payloads, "formats": self.formats, "metrics": self.metrics}, temporary
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "IQGSQResult":
        """Load native states on the CPU, without optimizer or executable model objects."""
        return cls(**torch.load(path, map_location="cpu", weights_only=True))


def quantize_iq_gsq(
    model: torch.nn.Module,
    quant_cfg: dict,
    forward_loop,
    *,
    num_steps: tuple[int, ...] = (200,),
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> IQGSQResult:
    """Run paired IQ GPTQ and GSQ, leaving the model at its exact GPTQ baseline.

    Every GSQ budget uses the same original weights, undamped covariance and fixed candidates.
    Calibration propagates the GPTQ prefix; activate a GSQ budget only after calibration ends.
    """
    # Deferred to avoid the model_quant -> GGML helper -> GSQ -> model_quant import cycle.
    from ..model_quant import quantize

    budgets = tuple(sorted(set(num_steps)))
    if not budgets or any(not isinstance(n, int) or isinstance(n, bool) or n < 2 for n in budgets):
        raise ValueError("IQ GSQ requires positive update budgets of at least two steps.")
    algorithms = quant_cfg.get("algorithm")
    algorithms = algorithms if isinstance(algorithms, list) else [algorithms]
    methods = [a.get("method") if isinstance(a, dict) else a for a in algorithms]
    if methods.count("gptq") != 1 or any(m not in ("gptq", "imatrix") for m in methods):
        raise ValueError(
            "IQ GSQ supports only GPTQ initialization, optionally followed by imatrix."
        )
    for rule in quant_cfg.get("quant_cfg", []):
        cfg = rule.get("cfg", rule)
        if rule.get("enable", True) and "num_bits" in cfg:
            if cfg["num_bits"] not in IQ_FORMAT_REGISTRY or cfg.get("backend") != "ggml":
                raise ValueError("IQ GSQ supports only the native GGML IQ weight formats.")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise ValueError("IQ GSQ currently supports a single process.")
    if _GSQ_CAPTURE.get() is not None:
        raise ValueError("IQ GSQ calibration cannot be nested.")
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if (
            hasattr(module, "num_experts")
            or hasattr(module, "num_gemms")
            or (
                isinstance(weight, torch.Tensor)
                and weight.ndim > 2
                and not isinstance(module, torch.nn.modules.conv._ConvNd)
            )
        ):
            raise ValueError(
                f"IQ GSQ supports only dense 2-D weights; fused or grouped experts at {name!r} "
                "are unsupported."
            )
    names = {id(module): name for name, module in model.named_modules()}
    result = IQGSQResult()
    directory = None if checkpoint_dir is None else Path(checkpoint_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)

    def capture(helper, teacher, covariance, payload):
        if helper._weight_quantizer is not None or teacher.ndim != 2:
            raise ValueError("IQ GSQ does not support per-expert GPTQ helpers.")
        name = names[id(helper.module)]
        format_name = helper.weight_quantizer.num_bits
        if format_name not in IQ_FORMAT_REGISTRY:
            raise ValueError(f"IQ GSQ cannot refine {format_name!r} at {name!r}.")
        path = (
            None
            if directory is None
            else directory / f"{hashlib.sha256(name.encode()).hexdigest()}.pt"
        )
        signature = None
        if path is not None:
            signature = (budgets, seed, _SETTINGS, _fingerprint(teacher), _fingerprint(covariance))
        if path is not None and path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["signature"] != signature or not torch.equal(
                cached["states"][0], payload.cpu()
            ):
                raise ValueError(f"IQ GSQ resume inputs changed at {name!r}.")
            states, metrics = cached["states"], cached["metrics"]
        else:
            print_rank_0(f"[{name}] IQ GSQ budgets {budgets}")
            started = time.monotonic()
            states, metrics = _refine_linear(
                teacher, covariance, payload, format_name, budgets, seed
            )
            metrics["seconds"] = time.monotonic() - started
            if path is not None:
                temporary = path.with_suffix(".tmp")
                torch.save(
                    {"signature": signature, "states": states, "metrics": metrics}, temporary
                )
                temporary.replace(path)
        result.payloads[name] = states
        result.formats[name] = format_name
        result.metrics[name] = metrics

    token = _GSQ_CAPTURE.set(capture)
    try:
        quantize(model, quant_cfg, forward_loop=forward_loop)
    finally:
        _GSQ_CAPTURE.reset(token)
    if not result.payloads:
        raise ValueError("IQ GSQ calibration did not reach an eligible IQ linear layer.")
    return result


def _fingerprint(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for chunk in tensor.detach().split(64):
        digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _error(weight: torch.Tensor, teacher: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    delta = weight.float() - teacher.float()
    return (delta.mm(covariance) * delta).sum() / delta.shape[0]


def _refine_linear(teacher, hessian, payload, format_name, budgets, seed):
    states: dict[int, list[torch.Tensor]] = {0: [payload.detach().cpu().clone()]}
    states.update({n: [] for n in budgets})
    errors: dict[int, float] = dict.fromkeys((0, *budgets), 0.0)
    changed = dict.fromkeys(budgets, 0)
    devices = [teacher.device.index] if teacher.is_cuda else []
    with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=devices):
        covariance = hessian.detach().to(teacher.device).clone() / 2
        if not torch.isfinite(covariance).all():
            raise ValueError("IQ GSQ requires finite calibration covariance.")
        importance = covariance.diagonal().clamp_min(0).sqrt()
        # Output rows have independent objectives, so their best native states can be combined.
        rows = _SETTINGS["rows_per_chunk"]
        for start in range(0, teacher.shape[0], rows):
            reference = teacher[start : start + rows].detach().clone()
            native = payload[start : start + rows].detach().clone()
            candidates = build_candidates(format_name, reference, native, importance=importance)
            for n in budgets:
                torch.manual_seed(seed + start)
                packed, before, after, changed_bundles = _optimize(
                    reference, covariance, candidates, native, format_name, n
                )
                states[n].append(packed.detach().cpu())
                errors[n] += after * reference.shape[0] / teacher.shape[0]
                changed[n] += changed_bundles
            errors[0] += before * reference.shape[0] / teacher.shape[0]
    joined = {n: torch.cat(chunks, dim=0) for n, chunks in states.items()}
    return joined, {
        "reconstruction_error": errors,
        "selected_noninitial_bundles": changed,
        "budgets": list(budgets),
        "seed": seed,
        "optimizer": _SETTINGS.copy(),
    }


def _optimize(teacher, covariance, candidates, native, format_name, num_steps):
    fmt = IQ_FORMAT_REGISTRY[format_name]
    coefficients = candidates.coefficients.detach().clone()
    valid = candidates.valid_mask
    base_scale = candidates.base_scale.detach().float().clone()
    shape = torch.tensor(teacher.shape)

    def hard_error(packed):
        with torch.no_grad():
            weight = fmt.dequantize(packed, shape, dtype=teacher.dtype)
            return float(_error(weight, teacher, covariance))

    best_payload = native.clone()
    initial_error = best_error = hard_error(native)
    logits = (
        torch.randn(coefficients.shape[:-1], device=teacher.device) * _SETTINGS["logit_noise_std"]
    )
    logits[..., 0] += _SETTINGS["logit_prior"]
    logits.requires_grad_()
    relative_scale = torch.zeros_like(base_scale, requires_grad=True)
    moments = [torch.zeros_like(p) for p in (logits, relative_scale)]
    best_changed = 0
    for step in range(num_steps):
        progress = step / (num_steps - 1)
        first, last = _SETTINGS["temperature"]
        temperature = first + (last - first) * progress
        first, last = _SETTINGS["logit_multiplier"]
        multiplier = first + (last - first) * progress
        probabilities = F.gumbel_softmax(
            (logits * multiplier).masked_fill(~valid, -torch.inf), tau=temperature, dim=-1
        )
        scale = base_scale * relative_scale.clamp(-3, 3).exp()
        weight = (probabilities.unsqueeze(-1) * coefficients).sum(dim=-2)
        weight = (weight * scale[..., None, None]).reshape_as(teacher)
        loss = _error(weight, teacher, covariance)
        loss.backward()
        floor = _SETTINGS["lr_floor"]
        decay = floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
        with torch.no_grad():
            for parameter, moment, lr, wd in (
                (logits, moments[0], _SETTINGS["logit_lr"], _SETTINGS["logit_weight_decay"]),
                (relative_scale, moments[1], _SETTINGS["relative_scale_lr"], 0.0),
            ):
                gradient = parameter.grad
                beta1, beta2 = _SETTINGS["lion_betas"]
                update = moment.mul(beta1).add(gradient, alpha=1 - beta1).sign_()
                parameter.mul_(1 - lr * decay * wd).add_(update, alpha=-lr * decay)
                moment.mul_(beta2).add_(gradient, alpha=1 - beta2)
                parameter.grad = None
            if (step + 1) % _SETTINGS["checkpoint_interval"] == 0 or step + 1 == num_steps:
                if not torch.isfinite(loss):
                    raise FloatingPointError("IQ GSQ optimization produced a nonfinite loss.")
                hard_scale = (base_scale * relative_scale.clamp(-3, 3).exp()).half().float()
                choice = logits.masked_fill(~valid, -torch.inf).argmax(dim=-1)
                packed = candidates.pack(choice, hard_scale)
                error = hard_error(packed)
                if math.isfinite(error) and error < best_error:
                    best_payload, best_error = packed.clone(), error
                    best_changed = int(choice.count_nonzero())
    return best_payload, initial_error, best_error, best_changed
