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

"""Validation helpers for security-sensitive Puzzletron configuration."""

from typing import Any

__all__ = ["require_boolean_policy"]


def require_boolean_policy(
    value: Any,
    *,
    path: str,
    default: bool | None = None,
) -> bool:
    """Return a policy boolean, resolving ``None`` only to an explicit default."""
    if default is not None and not isinstance(default, bool):
        raise ValueError(f"{path} default must be a boolean")
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value
