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

"""Muse Spark keeps compressed native experts and selects its feed_forward boundary."""

from .base import NativeFamilyAdapter
from .registry import register

__all__ = []


class _MuseSparkFamily(NativeFamilyAdapter):
    def unit_attributes(self, block, selection, attributes):
        if {"mlp", "moe"}.intersection(selection.unit_types) and "feed_forward" not in attributes:
            return [*attributes, "feed_forward"]
        return attributes


register(_MuseSparkFamily("muse_spark", explicit_padding_mask=True))
