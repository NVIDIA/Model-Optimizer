# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light setup helpers for Puzzletron pruning campaigns."""

__all__ = [
    "WORKER_REPOSITORY_PLACEHOLDER",
    "WORKER_VENV_PLACEHOLDER",
    "SetupError",
    "validate_worker_path",
]

WORKER_REPOSITORY_PLACEHOLDER = "REPLACE_WITH_WORKER_VISIBLE_MODELOPT_CHECKOUT"
WORKER_VENV_PLACEHOLDER = "REPLACE_WITH_WORKER_VISIBLE_MODELOPT_VENV"


def validate_worker_path(value: str) -> bool | str:
    """Require a concrete path rather than an unchanged example placeholder."""
    if not value.strip():
        return "Enter a path visible on every worker."
    if value.strip().startswith("REPLACE_WITH_"):
        return "Replace the placeholder with a path visible on every worker."
    return True


class SetupError(RuntimeError):
    """An actionable setup failure that should be shown without a traceback."""
