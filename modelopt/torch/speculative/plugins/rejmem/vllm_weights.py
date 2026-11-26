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

"""C3 draft weight updates: worker-side hooks (weight_version counter, refresh of cached weight copies)."""

__all__ = ["current_weight_version", "install"]

_weight_version = 0


def current_weight_version() -> int:
    """Completed draft weight updates in this worker process; 0 means the launch checkpoint."""
    return _weight_version


def install() -> None:
    """Patch vLLM when enabled by the environment; no-op otherwise. Not implemented yet."""
