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

"""Step-3.7 specs (HF model type ``step3p7``).

Shares Step-3.5's architecture; its PTQ modeling (in the HF quantization plugin)
matches every Step revision.
"""

from ..specs import ModelSpec, register

__all__: list[str] = []

# No MoESpec, for the same reason as step3p5 (see step3p5/specs.py).
register(ModelSpec(model_type="step3p7", modeling_source="remote_code"))
