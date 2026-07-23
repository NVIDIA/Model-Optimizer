# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

from .mip_with_multi_layer_replacements import run_mip as _run_pulp_mip

__all__ = ["cuopt_available", "run_mip_with_backend"]


def _backend_name(value: Any) -> str:
    if value is None:
        return "pulp"
    if isinstance(value, bool):
        return "cuopt" if value else "pulp"
    value = str(value).strip().lower()
    if value in ("cbc", "pulp", "local", "cpu", "false", "0", "none"):
        return "pulp"
    if value in ("cuopt", "gpu", "true", "1"):
        return "cuopt"
    if value in ("auto",):
        return "auto"
    raise ValueError(f"Unknown Puzzletron MIP solver backend {value!r}")


def cuopt_available() -> bool:
    try:
        from .mip_with_multi_layer_replacements_cuopt import run_mip as _unused  # noqa: F401

        return True
    except Exception:
        return False


def run_mip_with_backend(
    *,
    replacements,
    objective,
    constraints,
    bigger_is_better,
    max_seconds_per_solution=None,
    num_solutions=1,
    min_hamming_distance=1,
    solver_backend: Any = None,
):
    backend = _backend_name(solver_backend)
    if backend == "auto":
        backend = "cuopt" if cuopt_available() else "pulp"
    if backend == "cuopt":
        try:
            from .mip_with_multi_layer_replacements_cuopt import run_mip as _run_cuopt_mip
        except Exception as exc:
            raise RuntimeError(
                "Puzzletron MIP solver_backend='cuopt' was requested, but cuOpt is not "
                "available in this environment. Use solver_backend='pulp' or install/"
                "start the cuOpt backend."
            ) from exc
        return _run_cuopt_mip(
            replacements=replacements,
            objective=objective,
            constraints=constraints,
            bigger_is_better=bigger_is_better,
            max_seconds_per_solution=max_seconds_per_solution,
            num_solutions=num_solutions,
            min_hamming_distance=min_hamming_distance,
        )

    if solver_backend is not None and _backend_name(solver_backend) == "auto":
        warnings.warn("cuOpt is unavailable; falling back to the local PuLP/CBC solver.")
    return _run_pulp_mip(
        replacements=replacements,
        objective=objective,
        constraints=constraints,
        bigger_is_better=bigger_is_better,
        max_seconds_per_solution=max_seconds_per_solution,
        num_solutions=num_solutions,
        min_hamming_distance=min_hamming_distance,
    )
