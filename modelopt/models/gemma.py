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

"""Gemma 1/2/3 specs (HF model types ``gemma``/``gemma2``/``gemma3``); Gemma4 is in ``gemma4.py``."""

from .registry import register
from .specs import ModelSpec

# Gemma RMSNorms store weight - 1 (the effective scale is weight + 1).
# Gemma4RMSNorm is intentionally absent until the +1 handling is validated on Gemma4.
register(ModelSpec(model_type="gemma", weight_plus_one_norm_names=("GemmaRMSNorm",)))
register(ModelSpec(model_type="gemma2", weight_plus_one_norm_names=("Gemma2RMSNorm",)))
register(ModelSpec(model_type="gemma3", weight_plus_one_norm_names=("Gemma3RMSNorm",)))
