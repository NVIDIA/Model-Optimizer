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

"""Step-3.7 PTQ modeling (HF model type ``step3p7``).

Step-3.7 shares Step-3.5's expert-indexed ``MoELinear`` layout, so its PTQ support is the
Step-family code in ``step3p5/modeling_ptq.py``, which matches every Step revision. This
module imports it, so that loading the PTQ modeling for model type ``step3p7`` by name
registers that support too.
"""

from ..step3p5 import modeling_ptq as _step3p5_modeling_ptq  # noqa: F401

__all__: list[str] = []
