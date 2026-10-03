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

"""Validation-selected output-head shifts (https://arxiv.org/abs/2609.31291)."""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

__all__ = ["DEFAULT_COEFFICIENTS", "SearchResult", "head_kl", "search_head"]

DEFAULT_COEFFICIENTS = (-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)


@dataclass
class SearchResult:
    """Selected ModelOpt head container and token-averaged validation KL for every candidate."""

    model: nn.Sequential
    coefficient: float
    validation_kl: tuple[tuple[float, float], ...]


def _check_states(states: torch.Tensor, width: int) -> None:
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != width:
        raise ValueError("States must be a nonempty [tokens, hidden_width] tensor.")
    if not states.is_floating_point() or not torch.isfinite(states).all():
        raise ValueError("States must contain finite floating-point values.")


@torch.no_grad()
def head_kl(
    source: nn.Linear,
    candidate: nn.Module,
    states: torch.Tensor,
    batch_size: int = 128,
) -> float:
    """Mean forward KL from source to candidate on final hidden states.

    Both readouts use the source weight dtype; log-softmax and KL use FP32.
    States may reside on CPU and are moved one batch at a time. Use disjoint
    test states here after freezing the coefficient selected by ``search_head``.
    """
    _check_states(states, source.in_features)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    total = torch.zeros((), device=source.weight.device, dtype=torch.float64)
    for batch in states.split(batch_size):
        batch = batch.to(device=source.weight.device, dtype=source.weight.dtype)
        log_p = source(batch).float().log_softmax(-1)
        log_q = candidate(batch).float().log_softmax(-1)
        total += (log_p.exp() * (log_p - log_q)).sum(-1).double().sum()
    value = (total / states.shape[0]).item()
    if not math.isfinite(value):
        raise ValueError("Non-finite KL; check the head, quantizer, and hidden states.")
    return value


@torch.no_grad()
def search_head(
    head: nn.Linear,
    fit_states: torch.Tensor,
    validation_states: torch.Tensor,
    quant_config: dict[str, Any],
    coefficients: Sequence[float] = DEFAULT_COEFFICIENTS,
    batch_size: int = 128,
) -> SearchResult:
    """Fit shifted copies of a linear-softmax head and select by validation KL.

    Each candidate subtracts ``t * head.weight.float().mean(0)`` from every
    vocabulary row before independently fitting the same ModelOpt quantizer.
    The original head (including a shared embedding parameter) is never mutated.

    Args:
        head: Plain, unquantized ``nn.Linear`` immediately preceding softmax.
            Nonlinear logit paths and distributed/sharded heads are not supported.
        fit_states: Final hidden states used exclusively for quantizer fitting.
        validation_states: States from disjoint articles used only for selection.
            The caller is responsible for enforcing article-level disjointness.
        quant_config: ModelOpt weight-only fake-quant config for ``max`` or ``gptq``.
            Patterns address a one-layer Sequential (e.g. ``0.weight_quantizer``).
        coefficients: Finite, unique ordered candidates including zero; ties keep
            the first candidate. Including one provides a fixed-centering control.
        batch_size: Maximum number of token states per readout.

    Returns:
        The selected calibrated Sequential, coefficient, and validation curve.
        Keep the container when using ModelOpt save/restore. Selection guarantees
        no worse measured validation KL than t=0, not improvement on unseen data.
    """
    if type(head) is not nn.Linear:
        raise TypeError("head must be a plain, unquantized nn.Linear.")
    if head.weight.device.type not in ("cpu", "cuda"):
        raise ValueError("Only materialized CPU or CUDA heads are supported.")
    if head.weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Use FP32, FP16, or BF16 head weights.")
    values = tuple(float(t) for t in coefficients)
    if not values or not all(math.isfinite(t) for t in values):
        raise ValueError("coefficients must be nonempty and finite.")
    if 0.0 not in values or len(set(values)) != len(values):
        raise ValueError("coefficients must be unique and include zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    _check_states(fit_states, head.in_features)
    _check_states(validation_states, head.in_features)
    algorithm = quant_config.get("algorithm", "max")
    method = algorithm.get("method") if isinstance(algorithm, dict) else algorithm
    if method not in ("max", "gptq"):
        raise ValueError("This prototype supports max and gptq calibration only.")

    def fit(model):
        for batch in fit_states.split(batch_size):
            model(batch.to(device=head.weight.device, dtype=head.weight.dtype))

    mean = head.weight.float().mean(0)
    best_model, best_t, best_kl = None, 0.0, math.inf
    curve = []
    for coefficient in values:
        candidate = nn.Sequential(copy.deepcopy(head)).eval().requires_grad_(False)
        if coefficient != 0:
            candidate[0].weight.copy_(head.weight.float() - coefficient * mean)
        mtq.quantize(candidate, copy.deepcopy(quant_config), forward_loop=fit)
        quantizer = candidate[0].weight_quantizer
        if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
            raise ValueError("Enable a single weight quantizer for the head.")
        if not quantizer.fake_quant:
            raise ValueError("Search requires fake quantization; pack only after selection.")
        if candidate[0].input_quantizer.is_enabled or candidate[0].output_quantizer.is_enabled:
            raise ValueError("Search supports weight-only quantization.")
        # Avoid requantizing the full vocabulary matrix for each validation batch.
        with mtq.temporarily_fold_weights(candidate):
            kl = head_kl(head, candidate, validation_states, batch_size)
        curve.append((coefficient, kl))
        if kl < best_kl:
            best_model, best_t, best_kl = candidate, coefficient, kl
    assert best_model is not None
    return SearchResult(best_model, best_t, tuple(curve))
