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

"""Native Mistral Large 4 text-decoder boundaries and model-owned MoE math."""

from .base import NativeFamilyAdapter
from .registry import register

__all__ = []


class _MistralLarge4Family(NativeFamilyAdapter):
    def packed_batch(self, batch):
        documents = batch.get("_packed_seq_ids")
        if documents is None:
            return batch
        # Large 4 consumes compact document IDs directly and has an explicit
        # forward signature; the loader's legacy metadata key must stay outside.
        return {
            **{key: value for key, value in batch.items() if key != "_packed_seq_ids"},
            "attention_mask": documents,
        }


register(_MistralLarge4Family("mistral_large4"))
