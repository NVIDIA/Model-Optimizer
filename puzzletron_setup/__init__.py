# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light setup helpers for Puzzletron pruning campaigns."""

__all__ = ["SetupError"]


class SetupError(RuntimeError):
    """An actionable setup failure that should be shown without a traceback."""
