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

"""Serialization utilities for secure checkpoint saving and loading."""

import os
from collections import Counter, OrderedDict
from io import BytesIO
from typing import Any, BinaryIO

import torch

_SAFE_DICT_TYPES = (dict, OrderedDict, Counter)


def _sanitize_for_save(obj: Any) -> Any:
    """Recursively convert container subclasses to types accepted by ``weights_only=True``.

    * ``dict`` subclasses not in {dict, OrderedDict, Counter} e.g. defaultdict → plain ``dict``
    * ``list`` subclasses → plain ``list``
    * Recurses into dict values, list/tuple elements.
    * Leaves tensors, scalars, strings, bytes, etc. untouched.
    """
    if isinstance(obj, dict):
        sanitized = {k: _sanitize_for_save(v) for k, v in obj.items()}
        if type(obj) in _SAFE_DICT_TYPES:
            return type(obj)(sanitized)
        return sanitized
    if isinstance(obj, list):
        sanitized_list = [_sanitize_for_save(v) for v in obj]
        if type(obj) is list:
            return sanitized_list
        return sanitized_list
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_save(v) for v in obj)
    return obj


def safe_save(obj: Any, f: str | os.PathLike | BinaryIO, **kwargs) -> None:
    """Save a checkpoint after sanitizing known types for ``weights_only=True`` compatibility."""
    torch.save(_sanitize_for_save(obj), f, **kwargs)


def safe_load(f: str | os.PathLike | BinaryIO | bytes, **kwargs) -> Any:
    """Load a checkpoint securely using weights_only=True by default."""
    kwargs.setdefault("weights_only", True)

    if isinstance(f, (bytes, bytearray)):
        f = BytesIO(f)

    return torch.load(f, **kwargs)


# Add safe globals for serialization
torch.serialization.add_safe_globals([slice])
