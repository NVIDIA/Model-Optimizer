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

"""Nemotron Nano VL v2 specs (trust-remote-code model type ``NemotronH_Nano_VL_V2``).

``nemotron_h_omni`` takes its PTQ facts from here, and ``calibration.py`` is the calibration support
of both. The package name lowercases the model type.
"""

from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

NO_AUTO_DEVICE_MAP = "the wrapper is not compatible with automatic device mapping"

register(
    ModelSpec(
        model_type="NemotronH_Nano_VL_V2",
        modeling_source="remote_code",
        ptq_spec=PTQSpec(vlm_wrapper=True, no_auto_device_map=NO_AUTO_DEVICE_MAP),
    )
)
