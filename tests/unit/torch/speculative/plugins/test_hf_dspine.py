# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CPU unit tests for the DSpine speculative decoding plugin."""

import json
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins import hf_dspine
from modelopt.torch.speculative.plugins.hf_dspine import HFDSpineModel
from modelopt.torch.speculative.plugins.modeling_dflash import DFlashModule
from modelopt.torch.speculative.plugins.modeling_dspine import DSpineCodeBridge

BLOCK_SIZE = 4
SEQ_LEN = 16

# From the released tencent/Qwen3-8B-DSpine-b16 checkpoint, which SGLang's DSPINE loader reads.
RELEASED_BRIDGE_TENSORS = {
    "codebook",
    "early_read_weights",
    "feature_write_weights",
    "gate.bias",
    "gate.weight",
    "independent_write_version",
    "out.weight",
    "predecessor_gate.weight",
    "prev.weight",
    "read_weight",
    "terminal_disabled_version",
}
RELEASED_DFLASH_CONFIG_KEYS = {
    "draft_block_causal",
    "mask_token_id",
    "selected_prefix_code_bridge",
    "selected_prefix_decode_mode",
    "selected_prefix_feature_gt_training",
    "selected_prefix_feature_layers",
    "selected_prefix_feature_teacher_layers",
    "selected_prefix_feature_teacher_schedule",
    "selected_prefix_feature_write_mode",
    "selected_prefix_feature_writes",
    "selected_prefix_last_write_refine",
    "selected_prefix_terminal",
    "target_layer_ids",
}


def _dspine_config(arch=None, **overrides):
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config.update(
        dflash_block_size=BLOCK_SIZE,
        dflash_mask_token_id=0,
        dflash_draft_attention="causal",
        dflash_loss_objective="decay",
        dflash_loss_decay_factor=7.0,
    )
    config["dflash_architecture_config"] = {
        "num_hidden_layers": 2,
        "projector_type": "dspine",
        "code_size": 8,
        "code_rank": 8,
        "refine_top_k": 4,
        "teacher_forcing_layers": 1,
        **(arch or {}),
    }
    config.update(overrides)
    return config


def _dspine_model(arch=None, **overrides):
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, [("dflash", _dspine_config(arch, **overrides))])
    return model


def _randomize_writes(model):
    """Move the bridge off its identity start, as training does."""
    bridge = model.dflash_module.selected_prefix_bridge
    with torch.no_grad():
        for weight in (bridge.feature_write_weights, bridge.predecessor_gate.weight):
            weight.normal_(std=0.5)


def _draft_inputs(model, n_blocks=2, bsz=2, seed=0):
    """Module inputs for ``n_blocks`` blocks anchored inside a random context."""
    gen = torch.Generator().manual_seed(seed)
    hidden = model.dflash_config.hidden_size
    dtype = model.dflash_module.fc.weight.dtype
    anchors = torch.tensor([[2, 8], [4, 10]])[:bsz, :n_blocks]
    keep = torch.ones_like(anchors, dtype=torch.bool)
    input_ids = torch.randint(1, 32, (bsz, SEQ_LEN), generator=gen)
    block_ids = hf_dspine._block_tokens(input_ids, anchors, BLOCK_SIZE, 0)
    inputs = {
        "noise_embedding": torch.randn(bsz, n_blocks * BLOCK_SIZE, hidden, generator=gen).to(dtype),
        "target_hidden": torch.randn(bsz, SEQ_LEN, 2 * hidden, generator=gen).to(dtype),
        "position_ids": model._build_position_ids(SEQ_LEN, anchors, "cpu"),
        "attention_mask": model._build_draft_attention_mask(
            SEQ_LEN, anchors, keep, n_blocks, dtype, "cpu"
        ),
    }
    return inputs, block_ids


def test_train_step_and_generate():
    model = _dspine_model()
    assert isinstance(model, HFDSpineModel)
    model.train()
    model.set_training_progress(step=0, total_steps=10, lr_warmup_steps=2)
    input_ids = torch.randint(1, 32, (2, SEQ_LEN))
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)

    assert torch.isfinite(out.loss)
    assert out.train_acc == [[out.dspine_metrics["accuracy"]]]
    assert out.dspine_metrics["teacher_forcing"] == 0.5
    out.loss.backward()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert any("selected_prefix_bridge" in n for n in trainable)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())

    model.eval()
    base_token, draft_tokens = model.pseudo_speculative_generate(input_ids[:1, :8], steps=3)
    assert base_token.shape == (1, 1)
    assert draft_tokens.shape == (1, 3)


def test_untrained_draft_is_a_causal_dflash():
    """Zero-initialized writes make the injection exactly the identity."""
    model = _dspine_model()
    inputs, block_ids = _draft_inputs(model)
    dspine = model.dflash_module(**inputs, block_token_ids=block_ids).hidden_states
    dflash = DFlashModule._forward_body(model.dflash_module, *inputs.values())
    assert torch.equal(dspine, dflash)


def test_injection_is_causal_within_block():
    model = _dspine_model()
    _randomize_writes(model)
    inputs, block_ids = _draft_inputs(model, n_blocks=1, bsz=1)
    before = model.dflash_module(**inputs, block_token_ids=block_ids).hidden_states
    inputs["noise_embedding"] = inputs["noise_embedding"].clone()
    inputs["noise_embedding"][:, 2] += 1.0
    after = model.dflash_module(**inputs, block_token_ids=block_ids).hidden_states
    assert torch.equal(before[:, :2], after[:, :2])
    assert not torch.equal(before[:, 2:], after[:, 2:])


def test_refine_reproduces_teacher_forced_last_write():
    """The exact-sender refinement is the last layer's write given the true previous token."""
    model = _dspine_model(arch={"teacher_forcing_layers": 2})
    _randomize_writes(model)
    inputs, block_ids = _draft_inputs(model)
    replace_mask = torch.ones(block_ids.shape[:2], dtype=torch.bool)
    out = model.dflash_module(**inputs, block_token_ids=block_ids, replace_mask=replace_mask)

    shape = (*block_ids.shape, -1)
    refined = model.dflash_module.refine(
        out.pre_injection.view(shape)[:, :, 1:], block_ids[..., :-1]
    )
    torch.testing.assert_close(refined, out.hidden_states.view(shape)[:, :, 1:])


def test_objective_is_chunk_invariant(monkeypatch):
    # float32, so the per-chunk gradient sums agree to rounding rather than to bf16.
    model = _dspine_model().float()
    _randomize_writes(model)
    model.train()
    model.set_training_progress(step=3, total_steps=10, lr_warmup_steps=4)
    input_ids = torch.randint(1, 32, (2, SEQ_LEN))
    results = []
    for chunk_rows in (1, 1024):
        monkeypatch.setattr(hf_dspine, "_OBJECTIVE_CHUNK_ROWS", chunk_rows)
        model.zero_grad()
        torch.manual_seed(0)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
        results.append((out.loss.detach(), grads, out.dspine_metrics))
    (loss_a, grads_a, metrics_a), (loss_b, grads_b, metrics_b) = results
    torch.testing.assert_close(loss_a, loss_b)
    for grad_a, grad_b in zip(grads_a, grads_b):
        torch.testing.assert_close(grad_a, grad_b)
    assert metrics_a == pytest.approx(metrics_b)


def _fsdp2_train_step_job(rank, size):
    input_ids = torch.randint(1, 32, (2, SEQ_LEN), generator=torch.Generator().manual_seed(0))
    results = []
    for sharded in (False, True):
        torch.manual_seed(0)
        model = _dspine_model().float()
        _randomize_writes(model)
        model.train()
        model.set_training_progress(step=0, total_steps=10, lr_warmup_steps=2)
        if sharded:
            # The draft as its own unit, as accelerate wraps it when listed in the wrap policy.
            mesh = init_device_mesh("cpu", (size,))
            fully_shard(model.dflash_module, mesh=mesh)
            fully_shard(model, mesh=mesh)
        torch.manual_seed(1)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        grads = {
            name: p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        results.append((out.loss.detach(), grads))
    (loss, grads), (sharded_loss, sharded_grads) = results
    torch.testing.assert_close(sharded_loss, loss)
    assert sharded_grads.keys() == grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(sharded_grads[name], grad, msg=name)


def test_fsdp2_train_step_matches_unsharded(cpu_dist_workers):
    """FSDP2 gathers a unit's weights only around its forward, which must cover the refinement."""
    cpu_dist_workers.run(_fsdp2_train_step_job)


def test_schedule():
    model = _dspine_model()
    with pytest.raises(RuntimeError, match="DSpineScheduleCallback"):
        model._schedule()
    # 60 steps: teacher forcing held at 0.5 through step 10, zero from step 20.
    for step, beta, teacher_forcing in [
        (0, 0.0, 0.5),
        (5, 0.25, 0.5),
        (15, 0.5, 0.25),
        (30, 0.5, 0),
    ]:
        model.set_training_progress(step, total_steps=60, lr_warmup_steps=10)
        assert model._schedule() == pytest.approx((beta, teacher_forcing))
    model.dflash_dspine_alignment_warmup_steps = 0
    model.set_training_progress(0, total_steps=60, lr_warmup_steps=10)
    assert model._schedule()[0] == 0.5


@pytest.mark.parametrize(
    ("arch", "overrides", "match"),
    [
        (None, {"dflash_draft_attention": "bidirectional"}, "causal"),
        (None, {"dflash_loss_objective": "dpace"}, "decay"),
        (None, {"dflash_attention_sink": True}, "sink"),
        ({"refine_top_k": 3}, {}, "power of two"),
        ({"code_size": 40}, {}, "code_size"),
    ],
)
def test_rejects_settings_dspine_cannot_train(arch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _dspine_model(arch, **overrides)


@pytest.mark.parametrize("swa_window", [None, 8])
def test_export_matches_released_layout_and_warm_starts(tmp_path, swa_window):
    model = _dspine_model(dflash_swa_window_size=swa_window)
    _randomize_writes(model)
    model.get_exporter().export(tmp_path, dtype=torch.bfloat16)
    state_dict = load_file(tmp_path / "model.safetensors")
    config = json.loads((tmp_path / "config.json").read_text())

    bridge = {k.removeprefix("selected_prefix_bridge."): v for k, v in state_dict.items()}
    assert {k for k in state_dict if k.startswith("selected_prefix_bridge.")} == {
        f"selected_prefix_bridge.{name}" for name in RELEASED_BRIDGE_TENSORS
    }
    for name in (
        "draft_block_causal_version",
        "last_write_refine_version",
        "selected_prefix_bridge.independent_write_version",
        "selected_prefix_bridge.terminal_disabled_version",
    ):
        assert state_dict[name].dtype == torch.int64 and state_dict[name].item() == 1
    assert not bridge["out.weight"].any()

    assert config["architectures"] == ["DSpineDraftModel"] and config["is_causal"] is True
    dflash_config = config["dflash_config"]
    assert dflash_config.keys() >= RELEASED_DFLASH_CONFIG_KEYS
    # The DSPINE loader rejects any projector_type and any extra refinement key.
    assert "projector_type" not in dflash_config
    assert dflash_config["selected_prefix_last_write_refine"].keys() == {
        "kind",
        "version",
        "head_topk",
        "weight",
    }
    if swa_window is None:
        assert config["layer_types"] == ["full_attention"] * 2
    else:
        assert config["layer_types"] == ["sliding_attention"] * 2
        assert config["use_sliding_window"] is True and config["sliding_window"] == swa_window

    warm = _dspine_model(dflash_init_checkpoint=str(tmp_path), dflash_swa_window_size=swa_window)
    warm_bridge = warm.dflash_module.selected_prefix_bridge
    assert torch.equal(warm_bridge.codebook, bridge["codebook"])
    assert torch.equal(warm_bridge.feature_write_weights, bridge["feature_write_weights"])


def test_transfer_space_is_whitened_principal_directions():
    bridge = DSpineCodeBridge(hidden_size=32, num_layers=3, vocab_size=64, code_size=8, rank=4)
    bridge.initialize_transfer_space(torch.randn(64, 32))
    readout = bridge.read_weight
    torch.testing.assert_close(readout @ readout.T, torch.eye(8), atol=1e-5, rtol=0)
    torch.testing.assert_close(bridge.codebook.norm(dim=-1), torch.ones(64))
    assert torch.equal(bridge.early_read_weights, readout.expand(2, -1, -1))
    # Eigenvector signs are fixed so every rank derives the same space.
    assert (readout.gather(1, readout.abs().argmax(dim=1, keepdim=True)) > 0).all()

    with pytest.raises(ValueError, match="principal directions"):
        bridge.initialize_transfer_space(torch.randn(64, 4) @ torch.randn(4, 32))
