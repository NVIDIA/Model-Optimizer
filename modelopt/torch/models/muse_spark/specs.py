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


"""muse_spark task-owned output transformation."""

from ..specs import BypassSpec, ModelSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="muse_spark",
        min_transformers_version="5.23",
        bypass_spec=BypassSpec(
            logits_transform_path="modelopt.torch.distill.plugins.transformers.output_head.softcapped_logits",
            logits_transform_slices=True,
        ),
    )
)
