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


"""Step 5 owns its shared expert beside the routed MoE boundary."""

from .base import TransformersFamilyAdapter
from .registry import register_transformers_family

__all__ = []


class _Step5Family(TransformersFamilyAdapter):
    def unit_attributes(self, block, unit_types):
        result = super().unit_attributes(block, unit_types)
        if "mlp" in unit_types and block._modules.get("share_expert") is not None:
            result.append("share_expert")
        return list(dict.fromkeys(result))


register_transformers_family(_Step5Family("step5", ("model.layers",)))
