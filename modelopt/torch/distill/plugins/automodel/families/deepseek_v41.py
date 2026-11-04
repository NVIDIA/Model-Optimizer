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

"""Native deepseek_v41 repair descriptor."""

from ..runners import native_fast_replay
from .base import NativeFamilyAdapter
from .registry import register

__all__ = []

register(
    NativeFamilyAdapter(
        "deepseek_v41", packed_document_key="packed_seq_ids", unit_runner=native_fast_replay
    )
)
