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

"""Bypass facts shared by the dense Qwen3.5 root and native text model."""

from ..specs import BypassSpec, ModelSpec, register

__all__: list[str] = []

for _model_type in ("qwen3_5", "qwen3_5_text"):
    register(
        ModelSpec(
            model_type=_model_type,
            min_transformers_version="5.5",
            bypass_spec=BypassSpec(
                decoder_paths=(
                    "model.language_model.layers",
                    "language_model.layers",
                    "model.layers",
                    "layers",
                ),
                batch_packed_documents=True,
                functional_children=("linear_attn.conv1d",),
            ),
        )
    )
