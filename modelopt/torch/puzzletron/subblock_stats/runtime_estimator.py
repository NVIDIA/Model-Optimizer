# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout and arithmetic primitives for repeated-candidate runtime estimates."""

from statistics import median

from ..block_config import BlockConfig
from .runtime_vllm import RuntimeMeasurement


def effective_repeat_count(configured: int, pp_size: int) -> int:
    """Round a candidate repeat count up to a balanced PP multiple."""

    if configured < 1 or pp_size < 1:
        raise ValueError("repeat count and PP size must be positive")
    return ((configured + pp_size - 1) // pp_size) * pp_size


def homogeneous_layout(
    candidate: BlockConfig, repeat_count: int
) -> tuple[BlockConfig, ...]:
    """Return a candidate-only benchmark layout."""

    return (candidate,) * repeat_count


def scaffolded_layout(
    candidate: BlockConfig,
    scaffold: BlockConfig,
    repeat_count: int,
    pp_size: int,
) -> tuple[BlockConfig, ...]:
    """Insert one fixed scaffold before each PP stage's candidate chunk."""

    if repeat_count % pp_size:
        raise ValueError("candidate repeats must be divisible by PP size")
    per_stage = repeat_count // pp_size
    return tuple(
        block
        for _ in range(pp_size)
        for block in (scaffold, *((candidate,) * per_stage))
    )


def candidate_slope(
    short: RuntimeMeasurement, long: RuntimeMeasurement, repeat_count: int
) -> RuntimeMeasurement:
    """Recover one candidate's marginal from N-versus-2N measurements."""

    return (long - short) / repeat_count


def fixed_intercept(
    short: RuntimeMeasurement, long: RuntimeMeasurement
) -> RuntimeMeasurement:
    """Recover workload overhead that does not scale with candidate count."""

    return short + short - long


def median_measurement(values: list[RuntimeMeasurement]) -> RuntimeMeasurement:
    """Return a component-wise robust center for repeated overhead estimates."""

    if not values:
        raise ValueError("at least one fixed-overhead estimate is required")
    return RuntimeMeasurement(
        total_ms=median(value.total_ms for value in values),
        prefill_ms=median(value.prefill_ms for value in values),
    )
