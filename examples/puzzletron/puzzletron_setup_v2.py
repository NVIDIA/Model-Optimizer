#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a fully configurable Puzzletron campaign without launching it."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from puzzletron_setup.v2.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
