#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI and compatibility re-exports for Puzzletron campaign artifact inventory."""

from modelopt.torch.puzzletron import artifact_inventory as _artifact_inventory
from modelopt.torch.puzzletron.artifact_inventory import *  # noqa: F403

__all__ = _artifact_inventory.__all__


if __name__ == "__main__":
    raise SystemExit(main())  # noqa: F405
