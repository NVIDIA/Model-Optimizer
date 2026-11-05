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

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from modelopt.torch.speculative.plugins.modeling_eagle import EagleModule


def tiny_eagle(use_aux=True, layers=1):
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=32,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
    )
    config.use_aux_hidden_state = use_aux
    config.eagle_aux_hidden_state_layer_ids = [0, 1, 2]
    config.use_last_layernorm = True
    config.draft_vocab_size = 32
    config.has_lm_head = True
    config.eagle_decoder_type = "llama"
    config._attn_implementation = "eager"
    model = EagleModule(config, LlamaDecoderLayer).eval()
    model._maybe_init_rope(device="cpu")
    with torch.no_grad():
        for layer in model.layers:
            layer.input_layernorm.weight.copy_(torch.linspace(0.5, 1.7, 8))
        if use_aux:
            model.layers[0].hidden_norm.weight.copy_(torch.linspace(0.8, 1.2, 8))
    return model


def inputs(use_aux=True):
    features = torch.randn(1, 3, 24 if use_aux else 8, requires_grad=True)
    embeds = torch.randn(1, 3, 8, requires_grad=True)
    positions = torch.arange(3).unsqueeze(0)
    mask = torch.triu(torch.full((1, 1, 3, 3), torch.finfo(torch.float32).min), diagonal=1)
    return features, embeds, positions, mask


def forward(model, features, embeds, positions, mask):
    hidden = model.fc(features) if model.config.use_aux_hidden_state else features
    return model(hidden, embeds, mask, position_ids=positions, use_cache=False)[:2]


def reference(model, features, embeds, positions, mask):
    """EAGLE-3 equation with native attention/MLP, bypassing the attention hook."""
    hidden = model.fc(features) if model.config.use_aux_hidden_state else features
    rotary = model.rotary_emb(hidden, positions)
    for i, layer in enumerate(model.layers):
        if i == 0 and model.config.use_aux_hidden_state:
            attention_input = torch.cat(
                (layer.input_layernorm(embeds), layer.hidden_norm(hidden)), dim=-1
            )
        else:
            attention_input = layer.input_layernorm(hidden)
        # Calling forward directly avoids EagleModule's pre-hook while using
        # the actual Transformers attention implementation and learned weights.
        attention_output = layer.self_attn.forward(
            hidden_states=attention_input, attention_mask=mask, position_embeddings=rotary
        )[0]
        hidden = hidden + attention_output
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return model.norm(hidden), hidden


@pytest.mark.parametrize(("use_aux", "layers"), [(True, 1), (True, 2), (False, 1)])
def test_forward_and_gradients_match_reference(use_aux, layers):
    torch.manual_seed(17)
    model = tiny_eagle(use_aux, layers)
    expected_model = deepcopy(model)
    features, embeds, positions, mask = inputs(use_aux)
    expected_features = features.detach().clone().requires_grad_()
    expected_embeds = embeds.detach().clone().requires_grad_()
    actual = forward(model, features, embeds, positions, mask)
    expected = reference(expected_model, expected_features, expected_embeds, positions, mask)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    labels = torch.arange(3)
    F.cross_entropy(model.lm_head(actual[0]).flatten(0, 1), labels).backward()
    F.cross_entropy(expected_model.lm_head(expected[0]).flatten(0, 1), labels).backward()
    for actual_input, expected_input in [(features, expected_features), (embeds, expected_embeds)]:
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
    expected_parameters = dict(expected_model.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_parameters[name].grad)


@pytest.mark.parametrize("failure_at", ["decoder", "attention"])
def test_failed_forward_releases_inputs(failure_at):
    model = tiny_eagle()
    args = inputs()
    target = model.layers[0] if failure_at == "decoder" else model.layers[0].self_attn

    def fail(*_):
        raise RuntimeError()

    handle = target.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError):
            forward(model, *args)
    finally:
        handle.remove()
    assert model._input_embeds is None and model._input_hidden_states is None
    torch.testing.assert_close(forward(model, *args), reference(model, *args))


def test_nested_forward_restores_outer_inputs():
    model = tiny_eagle()
    outer, inner = inputs(), inputs()
    entered, nested_outputs = False, []

    def nested(*_):
        nonlocal entered
        if not entered:
            entered = True
            nested_outputs.append(forward(model, *inner))

    handle = model.layers[0].input_layernorm.register_forward_pre_hook(nested)
    try:
        actual = forward(model, *outer)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, reference(model, *outer))
    torch.testing.assert_close(nested_outputs[0], reference(model, *inner))
    assert model._input_embeds is None and model._input_hidden_states is None
