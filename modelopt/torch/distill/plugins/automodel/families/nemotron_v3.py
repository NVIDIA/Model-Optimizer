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

"""Nemotron hybrid blocks identify attention/Mamba roles through block_type."""

from .base import NativeFamilyAdapter
from .registry import register

__all__ = []


class _NemotronFamily(NativeFamilyAdapter):
    def unit_attributes(self, block, selection, attributes):
        if getattr(block, "block_type", None) in selection.unit_types:
            return [*attributes, "mixer"]
        return attributes


register(_NemotronFamily("nemotron_v3"))
