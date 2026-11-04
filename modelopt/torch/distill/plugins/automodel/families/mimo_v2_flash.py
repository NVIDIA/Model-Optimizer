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

"""MiMo native THD padding and attention-sink-compatible CP transport."""

import torch
import torch.nn.functional as F

from ..runners import native_fast_replay
from .base import NativeFamilyAdapter
from .registry import register

__all__ = []


class _MiMoFamily(NativeFamilyAdapter):
    def packed_cp_batch(self, batch, *, cp_size=1):
        """Use native boundary-only CP inputs, retaining validity outside the model."""
        if batch.get("qkv_format") != "thd":
            documents = batch.get("_packed_seq_ids", batch.get("attention_mask"))
            if documents is None or documents.ndim != 2 or documents.shape[0] != 1:
                raise ValueError("MiMo packed CP requires one document-indexed row per microbatch.")
            starts = torch.cat(
                (documents.new_zeros(1), torch.where(documents[0, 1:] != documents[0, :-1])[0] + 1)
            )
            ends = torch.cat((starts[1:], starts.new_tensor([documents.shape[1]])))
            intervals = [
                (int(start), int(end))
                for start, end in zip(starts.tolist(), ends.tolist())
                if int(documents[0, start]) > 0
            ]
            real_lengths = [end - start for start, end in intervals]
            padded_lengths = [length + (-length) % (2 * cp_size) for length in real_lengths]
            result = {
                key: value
                for key, value in batch.items()
                if key not in ("attention_mask", "padding_mask", "_packed_seq_ids", "document_ids")
            }
            for key, fill in (
                ("input_ids", 0),
                ("position_ids", 0),
                ("labels", -100),
                ("valid_mask", False),
            ):
                if key not in batch:
                    continue
                result[key] = torch.cat(
                    [
                        F.pad(batch[key][:, start:end], (0, padded - (end - start)), value=fill)
                        for (start, end), padded in zip(intervals, padded_lengths)
                    ],
                    dim=1,
                )
            result.update(
                qkv_format="thd",
                seq_lens=documents.new_tensor([real_lengths]),
                seq_lens_padded=documents.new_tensor([padded_lengths]),
            )
            if "valid_mask" in result:
                # Native CP also uses this mask to exclude padding from MoE
                # routing/calibration; token values need not identify padding.
                result["padding_mask"] = ~result["valid_mask"].bool()
            return result
        return batch

    def cp_loss_mask(self, valid):
        return None

    def configure_context_parallel(self, model, mesh_context):
        from nemo_automodel.components.models.mimo_v2_flash.parallelization import (
            setup_mimo_te_context_parallel,
        )

        # Generic TE attachment selects all_gather, which cannot handle sinks.
        setup_mimo_te_context_parallel(model, mesh_context.device_mesh["cp"])


register(_MiMoFamily("mimo_v2_flash", unit_runner=native_fast_replay))
