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

"""kimi_k25 specs (HF model type ``kimi_k25``).

Kimi-K2.7: the compressed-expert sibling of ``kimi_vl`` -- same swap to a native
deepseek_v3 language model, plus on-the-fly decompression of its packed experts at load.
"""

from ..specs import EPSpec, ModelSpec, register
from .ep import prepare

__all__: list[str] = []

register(
    ModelSpec(
        model_type="kimi_k25",
        modeling_source="remote_code",
        ep_spec=EPSpec(prepare=prepare),
    )
)
