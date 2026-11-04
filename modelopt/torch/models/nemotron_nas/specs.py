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

"""Nemotron-NAS specs (trust-remote-code model type ``nemotron-nas``).

The DeciLM architecture under another model type; see ``deci``. The package name drops the hyphen
the model type carries, which a module name cannot.
"""

from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="nemotron-nas",
        modeling_source="remote_code",
        ptq_spec=PTQSpec(dtype_kwarg="torch_dtype"),
    )
)
