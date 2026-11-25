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

"""Tests for EAGLE training losses and attention boundaries."""

from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from torch.nn.attention.flex_attention import BlockMask

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG
from modelopt.torch.speculative.losses import (
    _masked_soft_target_cross_entropy,
    masked_lk_lambda_loss,
)


@pytest.mark.parametrize("loss_type", ["soft_target_ce", "lk_lambda"])
@pytest.mark.parametrize("vocabulary", ["default", "full", "reduced"])
def test_eagle_loss_vocabulary_compatibility(loss_type, vocabulary):
    model = get_tiny_llama(dtype=torch.float32)
    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.update(
        eagle_ttt_steps=1,
        eagle_use_torch_compile=False,
        eagle_loss_type=loss_type,
        eagle_lk_eta=1.7,
    )
    if vocabulary != "default":
        config["eagle_architecture_config"]["draft_vocab_size"] = model.config.vocab_size // (
            2 if vocabulary == "reduced" else 1
        )
    if loss_type == "lk_lambda" and vocabulary == "reduced":
        original_type = type(model)
        parameters = dict(model.named_parameters())
        with pytest.raises(ValueError):
            mtsp.convert(model, mode=[("eagle", config)])
        assert type(model) is original_type
        assert dict(model.named_parameters()).keys() == parameters.keys()
        assert all(
            p is parameters[name] and p.requires_grad for name, p in model.named_parameters()
        )
        return
    mtsp.convert(model, mode=[("eagle", config)])
    indices = None
    if vocabulary == "reduced":
        draft_vocab_size = model.eagle_config.draft_vocab_size
        indices = 2 * torch.arange(draft_vocab_size) + 1
        model.eagle_module.d2t.copy_(indices - torch.arange(draft_vocab_size))
    draft_logits = []
    head = getattr(model.eagle_module, "lm_head", model._base_model_lm_head)
    handle = head.register_forward_hook(
        lambda _module, _inputs, output: draft_logits.append(output)
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    loss_mask = torch.ones_like(input_ids)
    loss_mask[:, :3] = 0
    output = model(input_ids=input_ids, loss_mask=loss_mask)
    handle.remove()
    teacher = output.logits[:, 1:]
    if loss_type == "lk_lambda":
        expected = masked_lk_lambda_loss(
            teacher, draft_logits[-1][:, :-1], loss_mask[:, 1:], eta=1.7
        )
    else:
        if indices is not None:
            teacher = teacher[..., indices]
        expected = _masked_soft_target_cross_entropy(
            teacher.softmax(-1), draft_logits[-1][:, :-1], loss_mask[:, 1:]
        )
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert any(p.grad is not None for p in model.eagle_module.parameters())
    assert model._base_model_lm_head.weight.grad is None


@pytest.mark.parametrize("mix_mode", ["uniform_layer_sample", "sparse_replace", None])
def test_rounds_do_not_expose_future_tokens(mix_mode):
    model = get_tiny_llama(dtype=torch.float32)
    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        intermediate_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8
    )
    config.update(
        eagle_ttt_steps=5,
        eagle_use_torch_compile=False,
        eagle_mix_hidden_states=mix_mode is not None,
        eagle_mix_hidden_states_mode=mix_mode or "uniform_layer_sample",
    )
    mtsp.convert(model, mode=[("eagle", config)])
    model.train()
    rounds = []
    handle = model.eagle_module.register_forward_hook(
        lambda _module, _inputs, outputs: rounds.append(outputs[0].detach().clone())
    )
    tokens = torch.arange(12).unsqueeze(0) % model.config.vocab_size
    torch.manual_seed(0)
    model(input_ids=tokens)
    expected = rounds[:]
    rounds.clear()
    tokens[:, -2:] = (tokens[:, -2:] + 1) % model.config.vocab_size
    torch.manual_seed(0)
    model(input_ids=tokens)
    handle.remove()
    # The changed suffix lies beyond every compared query's input context.
    for depth, (actual, reference) in enumerate(zip(rounds, expected)):
        torch.testing.assert_close(actual[:, depth:-3], reference[:, depth:-3], rtol=0, atol=0)


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
                model._cached_attn_blk_masks[step] is mask
                for step, mask in masks_before_save.items()
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
        previous_masks = dict(model._cached_attn_blk_masks)
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
            mask.dtype == torch.float32 and mask is not masks_before_cast[step]
            for step, mask in model._cached_attn_blk_masks.items()
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
