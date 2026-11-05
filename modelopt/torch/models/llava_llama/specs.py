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

"""VILA specs (trust-remote-code model type ``llava_llama``).

The model type of VILA, NVILA and NVILA-Lite, which VILA's own builder loads. LLaVA-repo fine-tunes
share it and are covered too: the registry holds one spec per model type, not one per variant.
"""

from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="llava_llama",
        modeling_source="remote_code",
        ptq_spec=PTQSpec(
            fsdp2_unsupported="VILA / llava_llama (custom builder + non-standard layer layout)"
        ),
    )
)
