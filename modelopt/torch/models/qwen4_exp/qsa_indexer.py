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

"""Calibration support for Qwen3.8's QSA indexer."""

import types

import torch

__all__ = ["prepare_for_calibration"]


def _forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Transformers' indexer, short-circuited when every visible key fits the token budget.

    The reference loops in Python over every (sample, query) pair: about 340 s per forward of
    1,024 x 512 calibration tokens, on each full-attention layer. A query keeps its
    ``block_topk`` best complete key blocks plus the incomplete tail, so when the keys hold no
    more blocks than that, it keeps every visible token: the selection is the visibility mask the
    indexer was given, which this returns directly.
    """
    if attention_mask.shape[-1] // self.compress_ratio > self.block_topk:
        return self._modelopt_reference_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values
        )
    if past_key_values is not None:
        # The reference appends this step's indexer keys to the cache before selecting.
        batch_size, seq_length, _ = hidden_states.shape
        token_k = self.index_qk_proj(hidden_states)[..., self.index_n_heads * self.index_head_dim :]
        past_key_values.update_indexer(
            token_k.reshape(batch_size, seq_length, -1, self.index_head_dim).squeeze(2),
            self.layer_idx,
        )
    # The reference reads head slot 0 of the 4-D mask and returns one head slot.
    mask = attention_mask[:, :1]
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        return torch.where(mask == 0, mask.new_zeros(()), min_dtype)
    return mask.clone()


def prepare_for_calibration(model) -> bool:
    """Give each QSA indexer of ``model`` the fast path; return whether ``model`` has one.

    Calibration sequences rarely exceed the indexer's token budget, so on Qwen3.8-Flash-Next
    (2,048 tokens) this skips the loop for every ``--calib_seq`` up to 2,048, with bit-identical
    results, and runs the reference otherwise.
    """
    found = False
    for module in model.modules():
        if type(module).__name__ != "Qwen4ExpTextQSAIndexer":
            continue
        found = True
        if not hasattr(module, "_modelopt_reference_forward"):
            module._modelopt_reference_forward = module.forward
            module.forward = types.MethodType(_forward, module)
    return found
