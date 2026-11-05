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

"""Tests for cached EAGLE training-time attention masks."""

from copy import deepcopy

import torch
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG


def test_ttt_attention_mask_cache_with_variable_lengths():
    model = get_tiny_llama(dtype=torch.float32)
    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        draft_vocab_size=model.config.vocab_size,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.update(eagle_ttt_steps=3, eagle_use_torch_compile=False)
    mtsp.convert(model, mode=[("eagle", config)])
    model.train()

    for batch_size, seq_length in ((1, 8), (2, 4), (1, 8)):
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids, labels=input_ids)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        model.zero_grad(set_to_none=True)
        assert len(model._cached_attn_blk_masks) == 2
        for step in (1, 2):
            mask = model._get_ttt_attention_mask(batch_size, seq_length, step)
            assert mask.shape == (1, 1, seq_length, seq_length * (step + 1))
            assert model._get_ttt_attention_mask(batch_size, seq_length, step) is mask
