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

"""Flash-Next owns QSA/GDN/PLE packed boundaries and CP metadata."""

import torch

from .base import NativeFamilyAdapter
from .registry import register

__all__ = []


class _FlashNextFamily(NativeFamilyAdapter):
    def packed_batch(self, batch):
        documents = batch.get("_packed_seq_ids")
        if documents is not None:
            # Native QSA owns packed routing and GDN/PLE resets. A generic
            # quadratic SDPA mask is not its input contract.
            from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
                packed_boundaries_from_seq_lens,
            )

            if documents.shape[0] != 1:
                raise ValueError("Flash Next packed execution requires micro_batch_size=1.")
            valid = documents[0] > 0
            count = int(valid.sum())
            if not count or not torch.equal(
                valid, torch.arange(valid.numel(), device=valid.device) < count
            ):
                raise ValueError("Flash Next expects nonempty, right-padded packed rows.")
            starts = torch.cat(
                (
                    documents.new_zeros(1),
                    torch.where((documents[0, 1:] != documents[0, :-1]) & valid[1:])[0] + 1,
                )
            )
            widths = torch.diff(torch.cat((starts, starts.new_tensor([count]))))
            boundaries = packed_boundaries_from_seq_lens(widths, total_tokens=documents.shape[1])
            batch = {
                **batch,
                "attention_mask": documents,
                "cu_seqlens": boundaries,
                "qkv_format": "thd",
            }
            batch.pop("_packed_seq_ids")
        return super().packed_batch(batch)

    def packed_cp_batch(self, batch, *, cp_size=1):
        if batch.get("cu_seqlens") is not None:
            return {
                key: value
                for key, value in batch.items()
                if key not in ("attention_mask", "padding_mask")
            }
        return batch


register(_FlashNextFamily("qwen3_8_flash_next"))
