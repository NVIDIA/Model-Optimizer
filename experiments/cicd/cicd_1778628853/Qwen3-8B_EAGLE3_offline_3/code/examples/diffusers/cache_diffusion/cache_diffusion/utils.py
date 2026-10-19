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

import re

SDXL_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: "up_blocks.2" not in name,
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]

PIXART_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: not re.search(
            r"transformer_blocks\.(2[1-7])\.", name
        ),
        "select_cache_step_func": lambda step: (step % 3) != 0,
    }
]

SVD_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: "up_blocks.3" not in name,
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]

SD3_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: re.search(
            r"^((?!transformer_blocks\.(1[6-9]|2[0-3])).)*$", name
        ),
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]


def replace_module(parent, name_path, new_module):
    path_parts = name_path.split(".")
    for part in path_parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path_parts[-1], new_module)
