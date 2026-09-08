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

"""Small forward coverage for the checkpoint-native MTP loss contract."""

from types import SimpleNamespace

import torch
from torch import nn

from modelopt.torch.speculative.eagle.utils import masked_soft_target_cross_entropy
from modelopt.torch.speculative.mtp.adapter import load_native_mtp_boost_checkpoint
from modelopt.torch.speculative.mtp.deepseek_v4 import NativeMTPBoostModel


class _Output(dict):
    """Tiny attribute-accessible model output for the dependency-free toy model."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class _Rotary(nn.Module):
    def forward(self, hidden_states, **_):
        return hidden_states, hidden_states


class _IdentityDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.past_key_values = []

    def forward(self, hidden_states, *, past_key_values, **_):
        self.past_key_values.append(past_key_values)
        return hidden_states


class _FirstHCStream(nn.Module):
    def forward(self, hidden_states):
        return hidden_states[..., 0, :]


def _toy_mtp_model() -> NativeMTPBoostModel:
    """Build just enough of the native module to exercise its offline forward."""
    model = NativeMTPBoostModel.__new__(NativeMTPBoostModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_mult=2, hidden_size=4, sliding_window=128)
    model.mtp_layer_index = 0
    model.rollout_steps = 1
    model.hsm_mode = "off"
    model._model_output_type = _Output
    model.embedding = nn.Embedding(4, 4)
    model.lm_head = nn.Linear(4, 4, bias=False)
    model.mtp = nn.Module()
    model.mtp.e_proj = nn.Linear(4, 4, bias=False)
    model.mtp.h_proj = nn.Linear(4, 4, bias=False)
    model.mtp.enorm = nn.Identity()
    model.mtp.hnorm = nn.Identity()
    model.mtp.rotary_emb = _Rotary()
    model.mtp.decoder = _IdentityDecoder()
    model.mtp.hc_head = _FirstHCStream()
    model.mtp.norm = nn.Identity()
    with torch.no_grad():
        eye = torch.eye(4)
        model.embedding.weight.copy_(eye)
        model.lm_head.weight.copy_(eye)
        model.mtp.e_proj.weight.copy_(eye)
        model.mtp.h_proj.weight.zero_()
    model.freeze_target_endpoints()
    return model


def test_native_mtp_offline_forward_shifts_masks_and_freezes_target_endpoints():
    """MTP token t uses token t+1 and trains only MTP masters with EAGLE loss."""
    model = _toy_mtp_model()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    target_mtp_hidden_states = torch.zeros(1, 4, 2, 4, requires_grad=True)
    target_lm_head_hidden_states = torch.flip(torch.eye(4), dims=[0]).unsqueeze(0).requires_grad_()
    loss_mask = torch.tensor([[0, 1, 0, 1]])

    output = model(
        input_ids=input_ids,
        loss_mask=loss_mask,
        mtp_boost_inputs={
            "target_mtp_hidden_states": target_mtp_hidden_states,
            "target_lm_head_hidden_states": target_lm_head_hidden_states,
        },
    )

    assert torch.isfinite(output.loss)
    assert torch.equal(output.hidden_states[0, :-1], torch.eye(4)[1:])
    expected_loss = masked_soft_target_cross_entropy(
        torch.softmax(model.lm_head(target_lm_head_hidden_states[:, 1:]), dim=-1),
        output.logits[:, :-1],
        loss_mask[:, 1:],
    )
    assert torch.allclose(output.loss, expected_loss)

    output.loss.backward()
    assert model.mtp.e_proj.weight.grad is not None
    assert torch.isfinite(model.mtp.e_proj.weight.grad).all()
    assert torch.count_nonzero(model.mtp.e_proj.weight.grad) > 0
    assert model.embedding.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert target_mtp_hidden_states.grad is None
    assert target_lm_head_hidden_states.grad is None
    assert all(
        not parameter.requires_grad for parameter in (model.embedding.weight, model.lm_head.weight)
    )


def test_native_mtp_online_features_match_offline_forward():
    """Online target features feed the same MTP loss path as cached features."""
    model = _toy_mtp_model()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    raw_hidden_states = torch.zeros(1, 4, 2, 4)
    lm_head_hidden_states = torch.flip(torch.eye(4), dims=[0]).unsqueeze(0)
    model.set_online_target_feature_provider(
        lambda _: (raw_hidden_states, lm_head_hidden_states),
        model_parallel=False,
    )

    offline = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        loss_mask=torch.tensor([[0, 1, 0, 1]]),
        mtp_boost_inputs={
            "target_mtp_hidden_states": raw_hidden_states,
            "target_lm_head_hidden_states": lm_head_hidden_states,
        },
    )
    online = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=torch.tensor([[-100, 1, -100, 3]]),
    )

    assert torch.equal(online.logits, offline.logits)
    assert torch.equal(online.loss, offline.loss)


def test_native_mtp_rollout_uses_normal_causal_hsm_without_a_kv_cache():
    """Multiple MTP rounds reuse raw states and retain gradients only on the MTP."""
    model = _toy_mtp_model()
    model.rollout_steps = 2
    model.hsm_mode = "uniform_layer_sample"
    input_ids = torch.tensor([[0, 1, 2, 3]])
    target_mtp_hidden_states = torch.zeros(1, 4, 2, 4)
    target_lm_head_hidden_states = torch.flip(torch.eye(4), dims=[0]).unsqueeze(0)

    output = model(
        input_ids=input_ids,
        loss_mask=torch.ones_like(input_ids),
        mtp_boost_inputs={
            "target_mtp_hidden_states": target_mtp_hidden_states,
            "target_lm_head_hidden_states": target_lm_head_hidden_states,
        },
    )

    assert torch.isfinite(output.loss)
    assert len(output.train_acc[0]) == 2
    assert model.mtp.decoder.past_key_values == [None, None]
    output.loss.backward()
    assert model.mtp.e_proj.weight.grad is not None


def test_native_mtp_training_checkpoint_is_mtp_only_and_resumable(tmp_path):
    """Trainer saves masters independently of frozen target endpoints and restores them."""
    model = _toy_mtp_model()
    with torch.no_grad():
        model.mtp.e_proj.weight.fill_(0.25)
    model.save_pretrained(tmp_path)

    saved_state = torch.load(tmp_path / "mtp_boost.pt", weights_only=True)
    assert saved_state
    assert all(name.startswith("mtp.") for name in saved_state)
    assert all(
        value.dtype == torch.bfloat16 for value in saved_state.values() if value.is_floating_point()
    )
    assert (tmp_path / "pytorch_model.bin").is_file()

    restored = _toy_mtp_model()
    load_native_mtp_boost_checkpoint(restored, tmp_path)
    assert torch.equal(restored.mtp.e_proj.weight, model.mtp.e_proj.weight)
    assert torch.equal(restored.embedding.weight, torch.eye(4))
