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

"""PTQ modeling for Falcon's original remote-code checkpoints (HF model type ``RefinedWebModel``).

Their ``RWForCausalLM`` is handled by the Falcon PTQ code in ``falcon/modeling_ptq.py``. This
module imports it, so that loading the PTQ modeling for model type ``RefinedWebModel`` by name
registers that support too.
"""

from ..falcon import modeling_ptq as _falcon_modeling_ptq  # noqa: F401

__all__: list[str] = []
