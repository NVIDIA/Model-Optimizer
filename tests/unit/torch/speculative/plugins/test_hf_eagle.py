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

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from torch.nn.attention.flex_attention import BlockMask

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG


@pytest.mark.parametrize("context_change", ["dtype", "device"])
def test_ttt_attention_mask_cache_with_variable_lengths(tmp_path, context_change):
    """Reuse masks across batches and checkpoint saves, rebuilding when context changes."""
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

    projected_logits = []
    handle = model._base_model_lm_head.register_forward_hook(
        lambda _module, _inputs, logits: projected_logits.append(logits.detach().clone())
    )
    previous_masks = None
    previous_length = None
    for batch_size, seq_length in ((1, 8), (2, 8), (2, 4), (1, 8)):
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids, labels=input_ids)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        if previous_masks is None:
            masks_before_save = dict(model._cached_attn_blk_masks)
            expected_logits = list(projected_logits)
            projected_logits.clear()
            expected_gradients = {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            model.zero_grad(set_to_none=True)
            model.save_pretrained(tmp_path)
            restored_output = model(input_ids=input_ids, labels=input_ids)
            assert all(
                model._cached_attn_blk_masks[key] is mask for key, mask in masks_before_save.items()
            )
            torch.testing.assert_close(projected_logits, expected_logits)
            torch.testing.assert_close(restored_output.loss, output.loss)
            restored_output.loss.backward()
            for name, parameter in model.named_parameters():
                if name in expected_gradients:
                    torch.testing.assert_close(parameter.grad, expected_gradients[name])
        model.zero_grad(set_to_none=True)
        assert len(model._cached_attn_blk_masks) == 2
        for step in (1, 2):
            mask = model._get_ttt_attention_mask(
                batch_size, seq_length, step, device=input_ids.device, dtype=torch.float32
            )
            assert mask.shape == (1, 1, seq_length, seq_length * (step + 1))
            assert (
                model._get_ttt_attention_mask(
                    batch_size, seq_length, step, device=input_ids.device, dtype=torch.float32
                )
                is mask
            )
            if previous_length == seq_length:
                assert mask is previous_masks[step]
        previous_masks = {
            step: model._get_ttt_attention_mask(
                batch_size, seq_length, step, device=input_ids.device, dtype=torch.float32
            )
            for step in (1, 2)
        }
        previous_length = seq_length
        projected_logits.clear()

    handle.remove()
    if context_change == "dtype":
        model.bfloat16()
        model.save_pretrained(tmp_path)
        model.float()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.isfinite(model(input_ids=input_ids, labels=input_ids).loss)
        masks_before_cast = dict(model._cached_attn_blk_masks)
        assert all(mask.dtype == torch.bfloat16 for mask in masks_before_cast.values())
        assert torch.isfinite(model(input_ids=input_ids, labels=input_ids).loss)
        assert all(
            mask.dtype == torch.float32 and mask is not masks_before_cast[key]
            for key, mask in model._cached_attn_blk_masks.items()
        )
    else:
        # A split base must not place draft masks on its first parameter's device.
        model.get_input_embeddings().to("meta")
    model.eagle_config._attn_implementation = "flex_attention"
    block_mask = model._get_ttt_attention_mask(
        1, 8, 1, device=input_ids.device, dtype=torch.float32
    )
    assert isinstance(block_mask, BlockMask)
    assert block_mask.kv_num_blocks.device == model.eagle_module.fc.weight.device

    first_bucket_masks = {}
    for seq_length in (8, 4, 8, 12):
        for step in (1, 2):
            mask = model._get_ttt_attention_mask(
                2, seq_length, step, device=input_ids.device, dtype=torch.float32
            )
            if seq_length == 8:
                if step in first_bucket_masks:
                    assert mask is first_bucket_masks[step]
                else:
                    first_bucket_masks[step] = mask
            fresh_mask = model._compute_ttt_attention_mask(
                1, seq_length, step, device=input_ids.device, dtype=torch.float32
            )
            query = torch.arange(seq_length).view(-1, 1)
            key = torch.arange(seq_length * (step + 1)).view(1, -1)
            torch.testing.assert_close(
                mask.mask_mod(None, None, query, key),
                fresh_mask.mask_mod(None, None, query, key),
            )
        assert len(model._cached_attn_blk_masks) <= 4
    assert {length for length, _step in model._cached_attn_blk_masks} == {4, 12}
