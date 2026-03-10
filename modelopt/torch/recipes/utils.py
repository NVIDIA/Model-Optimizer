# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utilities for the recipe system."""

from __future__ import annotations

from typing import Any


def make_serializable(obj: Any) -> Any:
    """Convert tuples and other non-JSON-safe types for serialization/display.

    Recursively converts dicts (with any key type), lists, and tuples into
    JSON-compatible structures. Dict keys are stringified.
    """
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def try_import_load_config():
    """Try to import PR #1000's load_config function.

    Returns the function if available, None otherwise. This is the forward-compatible
    import point for load_config() from modelopt.torch.opt.config.
    """
    try:
        from modelopt.torch.opt.config import load_config  # type: ignore[attr-defined]

        return load_config
    except (ImportError, ModuleNotFoundError):
        return None
