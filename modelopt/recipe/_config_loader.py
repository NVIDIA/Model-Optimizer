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

"""Re-export config loading utilities from ``modelopt.torch.opt.config_loader``."""

from modelopt.torch.opt.config_loader import (
    BUILTIN_CONFIG_ROOT,
    _load_raw_config,
    _resolve_imports,
    load_config,
)

BUILTIN_RECIPES_LIB = BUILTIN_CONFIG_ROOT

__all__ = [
    "BUILTIN_CONFIG_ROOT",
    "BUILTIN_RECIPES_LIB",
    "_load_raw_config",
    "_resolve_imports",
    "load_config",
]
