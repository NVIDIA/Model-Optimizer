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

"""Resolve native class-module aliases without importing AutoModel."""

__all__ = ["native_family", "register"]

_FAMILIES = {}


def register(adapter):
    """Register one native implementation, shared by its release aliases."""
    if adapter.name in _FAMILIES:
        raise ValueError(f"Duplicate native repair family {adapter.name!r}")
    _FAMILIES[adapter.name] = adapter


def native_family(model):
    """Resolve the original native class through distributed/quantized wrappers."""
    prefix = "nemo_automodel.components.models."
    for base in type(model).__mro__:
        if base.__module__.startswith(prefix):
            return _FAMILIES.get(base.__module__[len(prefix) :].split(".", 1)[0])
    return None
