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

"""Context passed from the export engine into per-model hooks.

The engine fills this with the builder callables a hook may need and passes it in, so
hooks never import the engine modules directly (keeping ``modeling/`` free of cycles).
"""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["ExportContext"]


@dataclass
class ExportContext:
    """Builders and metadata available to ``ModelHooks`` methods."""

    build_layernorm_config: Callable
