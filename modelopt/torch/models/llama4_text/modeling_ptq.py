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

"""Llama-4 text-only PTQ modeling (HF model type ``llama4_text``).

``Llama4ForCausalLM`` holds the same ``Llama4TextExperts`` as the multimodal Llama-4, whose PTQ
support is in ``llama4/modeling_ptq.py``. This module imports it, so that loading the PTQ
modeling for model type ``llama4_text`` by name registers that support too.
"""

from ..llama4 import modeling_ptq as _llama4_modeling_ptq  # noqa: F401

__all__: list[str] = []
