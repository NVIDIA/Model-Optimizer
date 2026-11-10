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

"""Nemotron-3 Nano Omni specs (trust-remote-code model type ``NemotronH_Nano_Omni_Reasoning_V3``).

The published Nemotron-3-Nano-Omni checkpoints declare this type and load only through their own
code: transformers' port of the model (from 5.18) registers ``nemotron_h_omni`` and refuses these
checkpoints without ``trust_remote_code=True``. Both types take the same PTQ facts and calibration
hook. The package has a snake_case name because the model type is not a lowercase module name (see
the models README).
"""

from ..nemotron_h_nano_vl_v2.specs import NO_AUTO_DEVICE_MAP
from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="NemotronH_Nano_Omni_Reasoning_V3",
        modeling_source="remote_code",
        ptq_spec=PTQSpec(vlm_wrapper=True, no_auto_device_map=NO_AUTO_DEVICE_MAP),
    )
)
