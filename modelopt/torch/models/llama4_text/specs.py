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

"""Llama-4 text-only specs (HF model type ``llama4_text``).

The same architecture as ``llama4`` reached through ``Llama4ForCausalLM``, with its own
``base_model_ep_plan`` on the root config, so it is refused expert parallelism the same way.
"""

from ..llama4.specs import EP_UNSUPPORTED
from ..specs import EPSpec, ModelSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="llama4_text",
        min_transformers_version="5.5",
        ep_spec=EPSpec(unsupported=EP_UNSUPPORTED),
    )
)
