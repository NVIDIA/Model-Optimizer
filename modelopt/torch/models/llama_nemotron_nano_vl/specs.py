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

"""Llama-Nemotron Nano VL specs (trust-remote-code model type ``Llama_Nemotron_Nano_VL``).

An InternVL-style wrapper around a Llama language model; the package name lowercases the model
type, which a module name may not carry in capitals.
"""

from ..nemotron_h_nano_vl_v2.specs import NO_AUTO_DEVICE_MAP
from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="Llama_Nemotron_Nano_VL",
        modeling_source="remote_code",
        ptq_spec=PTQSpec(vlm_wrapper=True, no_auto_device_map=NO_AUTO_DEVICE_MAP),
    )
)
