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

"""Small general helpers shared across the HF export modules.

Home for leaf utilities used by more than one export module (e.g. moe_utils and distribute)
that belong to neither -- keeping them here avoids a cross-module dependency between siblings. This
module must not import other export modules, so it stays a safe common dependency for all of them.
"""

import torch


def _size_to_bytes(size: "str | int") -> int:
    """Parse an HF-style shard-size string (``"5GB"``, ``"500MB"``, ``"1GiB"``) to bytes.

    Matches transformers' decimal convention (GB == 10**9). Bare ints pass through.
    """
    if isinstance(size, int):
        return size
    s = str(size).strip().upper()
    units = {
        "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40,
        "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
    }  # fmt: skip
    for unit in ("KIB", "MIB", "GIB", "TIB", "KB", "MB", "GB", "TB"):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * units[unit])
    return int(float(s))


