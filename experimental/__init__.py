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

"""Experimental optimization techniques for Model Optimizer.

This package contains experimental and research-stage optimization algorithms
that are under active development. APIs may change without notice.

Warning:
    Code in this package is experimental and not covered by semantic versioning.
    Use at your own risk in production environments.
"""

import warnings

warnings.warn(
    "The 'experimental' package contains unstable APIs that may change. "
    "Use at your own risk in production environments.",
    FutureWarning,
    stacklevel=2,
)

__all__ = []
