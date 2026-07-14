# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learning-rate schedules shared by native local-distillation recipes."""

from __future__ import annotations

import math

from omegaconf import DictConfig

__all__ = ["get_learning_rate"]


def get_learning_rate(cfg: DictConfig, step: int) -> float:
    """Cosine decay with linear warmup, preserving the legacy bypass schedule."""
    training = cfg.bypass.training
    warmup_steps = int(training.warmup_steps)
    decay_steps = int(training.lr_decay_steps)
    if decay_steps <= warmup_steps:
        return float(training.learning_rate)
    if step <= warmup_steps:
        if warmup_steps == 0:
            return float(training.learning_rate)
        return float(training.learning_rate) * step / warmup_steps
    if step > decay_steps:
        return float(training.min_lr)
    decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
    if not 0 <= decay_ratio <= 1:
        raise ValueError(f"invalid decay ratio {decay_ratio} at step {step}")
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return float(training.min_lr) + coefficient * (
        float(training.learning_rate) - float(training.min_lr)
    )

