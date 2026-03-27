# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Serialization utilities for secure checkpoint loading."""

import os
from io import BytesIO
from typing import Any, BinaryIO

import torch


def safe_load(f: str | os.PathLike | BinaryIO | bytes, **kwargs) -> Any:
    \"\"\"Load a checkpoint securely using weights_only=True by default.\"\"\"
    kwargs.setdefault(\"weights_only\", True)

    if isinstance(f, (bytes, bytearray)):
        f = BytesIO(f)

    return torch.load(f, **kwargs)  # nosec B614


def add_modelopt_safe_globals(classes: list[type]) -> None:
    \"\"\"Register ModelOpt classes as safe globals for torch serialization.

    This is required when weights_only=True is used in torch.load().
    \"\"\"
    if hasattr(torch.serialization, \"add_safe_globals\"):
        torch.serialization.add_safe_globals(classes)

