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

"""Kimi-K2 specs (trust-remote-code model type ``kimi_k2``).

A DeepSeek-V3 architecture under its own model type, with its modeling code shipped in the
checkpoint. Only what the PTQ flow needs is recorded: no MoE or export section, so those lookups
find none, as they did before this type had a spec.
"""

from ..deepseek.specs import PREVIEW_UNSUPPORTED
from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="kimi_k2",
        modeling_source="remote_code",
        ptq_spec=PTQSpec(preview_unsupported=PREVIEW_UNSUPPORTED),
    )
)
