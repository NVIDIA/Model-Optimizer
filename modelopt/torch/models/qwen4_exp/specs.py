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

"""Qwen3.8 specs (HF model type ``qwen4_exp``)."""

from ..specs import ModelSpec, register

__all__: list[str] = []

# Only the model type is recorded, so the dispatcher in ``hf.py`` can find this package's
# calibration helper. The fused MoE layout is left undeclared: export and quantization handle
# these experts generically, and declaring it would change their behavior for this model.
register(ModelSpec(model_type="qwen4_exp", min_transformers_version="5.16"))
