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


"""Gemma 3 specs (HF model type ``gemma3``)."""

from ..registry import register
from ..specs import ModelSpec

# Gemma 3 RMSNorm stores weight - 1 (the effective scale is weight + 1).
register(ModelSpec(model_type="gemma3", weight_plus_one_norm_names=("Gemma3RMSNorm",)))
