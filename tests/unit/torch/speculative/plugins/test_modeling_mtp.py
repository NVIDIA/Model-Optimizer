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

"""Native MTP training and HF checkpoints through a small real adapter."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, PretrainedConfig, Trainer, TrainingArguments

from modelopt.torch.speculative.eagle.utils import (
    EagleOfflineDataCollator,
    OfflineSupervisedDataset,
)
from modelopt.torch.speculative.losses import (
    masked_lk_lambda_loss,
    masked_soft_target_cross_entropy,
)
from modelopt.torch.speculative.mtp import (
    NativeMTPAdapter,
    NativeMTPCheckpointError,
    create_native_mtp_boost_model,
    get_native_mtp_adapter,
    register_native_mtp_adapter,
)
from modelopt.torch.speculative.mtp import adapter as adapter_registry
from modelopt.torch.speculative.plugins.modeling_mtp import (
    MTPModule,
    MTPModuleOutput,
    create_mtp_model,
)


class TinyMTPModule(MTPModule):
    def __init__(self, config):
        state_shape = (
            (config.streams, config.hidden_size) if config.streams > 1 else (config.hidden_size,)
        )
        super().__init__(config, state_shape=state_shape)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer("route_bias", torch.zeros(config.hidden_size))

    def _forward(
        self,
        hidden_states,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        input_ids,
    ):
        projected = self.proj(hidden_states)
        query = projected.mean(2) if self.config.streams > 1 else projected
        query = query.unsqueeze(1)
        # Include tokens and positions so missing/misaligned adapter inputs change the result.
        key = query + input_ids[:, None, :, None].to(query.dtype) / self.config.vocab_size
        value = query + position_ids[:, None, :, None].to(query.dtype) / 100
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, 0)
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        inputs_embeds = inputs_embeds + context.squeeze(1)
        if self.config.streams > 1:
            inputs_embeds = inputs_embeds.unsqueeze(2)
        state = torch.tanh(projected + inputs_embeds + self.route_bias)
        hidden = state.mean(2) if self.config.streams > 1 else state
        return MTPModuleOutput(hidden, state, past_key_values)


class TinyMTPAdapter(NativeMTPAdapter):
    name = "tiny_native_mtp"

    @classmethod
    def supports(cls, model_path):
        path = Path(model_path) / "config.json"
        return path.is_file() and json.loads(path.read_text()).get("native_mtp") is True

    @classmethod
    def create(cls, model_path, *, dtype, device, config):
        architecture = PretrainedConfig(
            **json.loads((Path(model_path) / "config.json").read_text())
        )
        model = _make_model(architecture, config)
        model.load_state_dict(torch.load(Path(model_path) / "base.pt", weights_only=True))
        return model.to(device=device, dtype=dtype)


def _make_model(architecture, config=None):
    embedding = nn.Embedding(architecture.vocab_size, architecture.hidden_size)
    head = nn.Linear(architecture.hidden_size, architecture.vocab_size, bias=False)
    head.weight = embedding.weight
    return create_mtp_model(TinyMTPModule(architecture), embedding, head, config)


@pytest.fixture(autouse=True)
def _register_tiny_adapter(monkeypatch):
    monkeypatch.setattr(adapter_registry, "_ADAPTERS", {})
    register_native_mtp_adapter(TinyMTPAdapter)


def _write_base(path, streams=1):
    torch.manual_seed(7)
    architecture = PretrainedConfig(
        vocab_size=11, hidden_size=4, streams=streams, native_mtp=True, num_hidden_layers=1
    )
    (path / "config.json").write_text(architecture.to_json_string())
    torch.save(_make_model(architecture).state_dict(), path / "base.pt")
    return path


def _batch(streams, length=7):
    shape = (2, length, 4) if streams == 1 else (2, length, streams, 4)
    mask = torch.ones(2, length)
    mask[:, ::3] = 0
    return {
        "input_ids": torch.arange(2 * length).reshape(2, length) % 11,
        "loss_mask": mask,
        "target_mtp_hidden_states": torch.randn(shape, requires_grad=True),
        "target_lm_head_hidden_states": torch.randn(2, length, 4, requires_grad=True),
    }


def _reference(model, batch):
    """Compute EAGLE-aligned losses with explicit history sampling and native attention."""
    cfg = model.mtp_config
    tokens = batch["input_ids"]
    teacher = model.lm_head(batch["base_model_outputs"]["base_model_hidden_states"]).detach()
    state = batch["base_model_outputs"]["draft_hidden_states"].detach()
    shifted = tokens.roll(-1, 1)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    mask = torch.full(
        (tokens.shape[1], tokens.shape[1]), torch.finfo(state.dtype).min, dtype=state.dtype
    ).triu(1)[None, None]
    if "attention_mask" in batch:
        invalid = ~batch["attention_mask"].roll(-1, 1).bool()
        mask = mask.expand(tokens.shape[0], -1, -1, -1).clone()
        mask.masked_fill_(invalid[:, None, None], torch.finfo(state.dtype).min)
    history, losses = [state], []
    cache = None if cfg.mtp_mix_hidden_states else DynamicCache(config=model.eagle_module.config)
    steps = cfg.mtp_ttt_steps if model.training else 1
    for depth in range(steps):
        attn = (
            mask
            if depth == 0 or cache is None
            else model._get_ttt_attention_mask(
                tokens.shape[0], tokens.shape[1], depth, device=state.device, dtype=state.dtype
            )
        )
        # Bypass activation checkpointing to compare its gradients as well.
        hidden, state, cache = model.eagle_module._forward(
            state, model.embedding(shifted), attn, positions, cache, True, shifted
        )
        logits = model.lm_head(hidden)
        student = logits[:, depth:-1]
        loss_mask = batch["loss_mask"][:, depth + 1 :]
        if cfg.mtp_loss_type == "lk_lambda":
            loss = masked_lk_lambda_loss(
                teacher[:, depth + 1 :], student, loss_mask, eta=cfg.mtp_lk_eta
            )
        else:
            loss = masked_soft_target_cross_entropy(
                teacher.softmax(-1)[:, depth + 1 :], student, loss_mask
            )
        losses.append(loss * cfg.mtp_loss_decay_factor**depth)
        state = state.roll(1, 1)
        if cfg.mtp_mix_hidden_states:
            history.append(state)
            choices = torch.randint(len(history), tokens.shape)
            state = torch.stack(history)[
                choices, torch.arange(tokens.shape[0])[:, None], torch.arange(tokens.shape[1])
            ]
    return sum(losses), teacher


@pytest.mark.parametrize("streams", [1, 3])
@pytest.mark.parametrize(
    ("loss_type", "steps", "mix_hidden_states", "length", "dtype"),
    [
        ("lk_lambda", 1, False, 7, torch.float32),
        ("lk_lambda", 3, False, 7, torch.float32),
        ("lk_lambda", 3, True, 7, torch.float32),
        ("soft_target_ce", 3, True, 7, torch.float32),
        ("lk_lambda", 3, True, 257, torch.float32),
        ("lk_lambda", 3, True, 7, torch.bfloat16),
    ],
)
def test_native_rounds_share_eagle_alignment_and_gradients(
    tmp_path, streams, loss_type, steps, mix_hidden_states, length, dtype
):
    model = create_native_mtp_boost_model(
        _write_base(tmp_path, streams),
        dtype=dtype,
        mtp_ttt_steps=steps,
        mtp_mix_hidden_states=mix_hidden_states,
        mtp_mix_hidden_states_mode="uniform_layer_sample",
        mtp_loss_type=loss_type,
        mtp_loss_decay_factor=0.7,
    )
    reference = deepcopy(model)
    batch = _batch(streams, length)
    for key in ("target_mtp_hidden_states", "target_lm_head_hidden_states"):
        batch[key] = batch[key].detach().to(dtype).requires_grad_()
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    batch["attention_mask"][0, -2:] = 0
    batch["loss_mask"] *= batch["attention_mask"]
    batch["base_model_outputs"] = {
        "draft_hidden_states": batch.pop("target_mtp_hidden_states"),
        "base_model_hidden_states": batch.pop("target_lm_head_hidden_states"),
    }
    rng = torch.get_rng_state()
    expected_loss, expected_logits = _reference(reference, batch)
    expected_rng = torch.get_rng_state()
    expected_loss.backward()
    torch.set_rng_state(rng)
    outputs = model(**batch)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(outputs.loss, expected_loss)
    torch.testing.assert_close(outputs.logits, expected_logits)
    outputs.loss.backward()
    for actual, expected in zip(
        model.eagle_module.parameters(), reference.eagle_module.parameters()
    ):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert batch["base_model_outputs"]["draft_hidden_states"].grad is None
    assert batch["base_model_outputs"]["base_model_hidden_states"].grad is None
    assert model.embedding.weight is model.lm_head.weight
    assert model.embedding.weight.grad is None
    assert not model.embedding.weight.requires_grad
    assert len(outputs.train_acc) == 1 and len(outputs.train_acc[0]) == steps
    model.eval()
    reference.eval()
    with torch.no_grad():
        expected_eval, _ = _reference(reference, batch)
        evaluated = model(**batch)
    torch.testing.assert_close(evaluated.loss, expected_eval)
    assert len(evaluated.train_acc[0]) == 1


def test_offline_batch_and_empty_assistant_mask(tmp_path):
    model = create_native_mtp_boost_model(
        _write_base(tmp_path, 3), dtype=torch.float32, mtp_ttt_steps=2
    )
    batch = _batch(3)
    features = ("target_mtp_hidden_states", "target_lm_head_hidden_states")
    paths = []
    for index in range(2):
        path = tmp_path / f"sample{index}.pt"
        torch.save({key: value[index].detach() for key, value in batch.items()}, path)
        paths.append(path)
    dataset = OfflineSupervisedDataset(
        paths,
        answer_only_loss=True,
        feature_key_mapping=dict(
            zip(features, ("draft_hidden_states", "base_model_hidden_states"))
        ),
    )
    collator = EagleOfflineDataCollator(
        7,
        batch_keys=("input_ids", "attention_mask", "loss_mask"),
        feature_keys=("draft_hidden_states", "base_model_hidden_states"),
    )
    collated = collator([dataset[0], dataset[1]])
    batch["base_model_outputs"] = {
        "draft_hidden_states": batch.pop("target_mtp_hidden_states"),
        "base_model_hidden_states": batch.pop("target_lm_head_hidden_states"),
    }
    torch.testing.assert_close(model(**collated).loss, model(**batch).loss)
    collated["loss_mask"].zero_()
    output = model(**collated)
    assert output.loss.item() == 0
    output.loss.backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.eagle_module.parameters())
    del collated["base_model_outputs"]["draft_hidden_states"]
    with pytest.raises(ValueError):
        model(**collated)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sharded", [False, True])
def test_trainer_checkpoint_roundtrip_preserves_dtype_and_frozen_endpoints(
    tmp_path, dtype, sharded
):
    base = _write_base(tmp_path)
    model = create_native_mtp_boost_model(base, dtype=dtype)
    with torch.no_grad():
        model.eagle_module.proj.weight.add_(1)
    destination = tmp_path / "checkpoint"
    args = TrainingArguments(output_dir=str(destination), use_cpu=True, report_to=[])
    trainer = Trainer(model=model, args=args)
    if sharded:
        model.save_pretrained(destination, max_shard_size=32)
        assert (destination / "model.safetensors.index.json").is_file()
    else:
        trainer.save_model()
    state = {
        key: value
        for shard in destination.glob("*.safetensors")
        for key, value in load_file(shard).items()
    }
    assert all(k.startswith("eagle_module.") and v.dtype == dtype for k, v in state.items())
    metadata = json.loads((destination / "config.json").read_text())["mtp_boost"]
    assert metadata["mtp_adapter"] == TinyMTPAdapter.name
    assert metadata["mtp_loss_type"] == "lk_lambda"
    restored = create_native_mtp_boost_model(base, dtype=dtype, **metadata)
    frozen = restored.embedding.weight.detach().clone()
    Trainer(model=restored, args=args)._load_from_checkpoint(str(destination))
    torch.testing.assert_close(
        restored.eagle_module.state_dict(), model.eagle_module.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(restored.embedding.weight, frozen, rtol=0, atol=0)
    assert restored.embedding.weight is restored.lm_head.weight


@pytest.mark.parametrize(
    "options",
    [
        {"mtp_ttt_steps": 0},
        {"mtp_ttt_steps": True},
        {"mtp_mix_hidden_states": True},
        {"mtp_mix_hidden_states_mode": "unknown"},
        {"mtp_lk_eta": float("nan")},
        {"mtp_loss_decay_factor": 0},
        {"mtp_loss_type": "unknown"},
    ],
)
def test_invalid_training_configuration(tmp_path, options):
    with pytest.raises(ValidationError):
        create_native_mtp_boost_model(tmp_path, **options)


def test_adapter_selection_rejects_unsupported_and_ambiguous_checkpoints(tmp_path):
    with pytest.raises(NativeMTPCheckpointError):
        get_native_mtp_adapter(tmp_path)
    base = _write_base(tmp_path)
    with pytest.raises(ValueError):
        register_native_mtp_adapter(TinyMTPAdapter)

    class OtherAdapter(TinyMTPAdapter):
        name = "other_native_mtp"

    register_native_mtp_adapter(OtherAdapter)
    with pytest.raises(NativeMTPCheckpointError):
        get_native_mtp_adapter(base)
    assert get_native_mtp_adapter(base, TinyMTPAdapter.name) is TinyMTPAdapter
    with pytest.raises(NativeMTPCheckpointError):
        get_native_mtp_adapter(base, "unknown")
