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

"""Nemotron's mixer operation changes with each registered block type."""

from .base import TransformersFamilyAdapter
from .registry import register_transformers_family

__all__ = []


class _NemotronFamily(TransformersFamilyAdapter):
    def unit_attributes(self, block, unit_types):
        operations = {
            "mlp": "mlp",
            "moe": "moe",
            "full_attention": "attention",
            "linear_attention": "mamba",
        }
        operation = operations.get(getattr(block, "block_type", None))
        if operation in unit_types:
            return ["mixer"]
        return super().unit_attributes(block, unit_types)


register_transformers_family(
    _NemotronFamily("nemotron_h", isolated_documents=True, batch_packed_documents=True)
)

register_transformers_family(
    _NemotronFamily("nemotron_h_omni", ("language_model.model.layers", "model.layers", "layers"))
)
