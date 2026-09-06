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

"""CPU unit tests for DFlash speculative decoding plugin.

GPU-dependent tests (training forward, module forward) are in tests/gpu/.
"""

import json
import logging
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import (
    get_tiny_llama,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
import modelopt.torch.speculative.plugins.hf_dflash as hf_dflash
from modelopt.torch.speculative.plugins.hf_dflash import (
    DFlashAttention,
    DFlashModule,
    HFDFlashModel,
    _dpace_position_weights,
    build_target_layer_ids,
)
from modelopt.torch.speculative.utils import AcceptanceRateValidation
from modelopt.torch.utils.plugins.transformers_dataset import LanguageDataCollator

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


class TestDFlashConvert:
    """Test DFlash model conversion."""

    def test_convert_creates_dflash_model(self):
        """Test that convert produces an HFDFlashModel."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)

    def test_convert_creates_dflash_module(self):
        """Test that convert attaches a DFlashModule."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "dflash_module")
        assert isinstance(model.dflash_module, DFlashModule)

    def test_convert_freezes_base_model(self):
        """Test that base model parameters are frozen after convert."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        for name, param in model.named_parameters():
            if "dflash_module" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_convert_dflash_module_trainable(self):
        """Test that DFlash module parameters are trainable after convert."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        dflash_params = [(n, p) for n, p in model.named_parameters() if "dflash_module" in n]
        assert len(dflash_params) > 0
        for name, param in dflash_params:
            assert param.requires_grad, f"DFlash param {name} should be trainable"

    def test_convert_sets_target_layer_ids(self):
        """Test that target layer IDs are set correctly."""
        model = get_tiny_llama(num_hidden_layers=8)
        config = get_dflash_config(num_layers=3)
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "target_layer_ids")
        assert len(model.target_layer_ids) == 3
        for lid in model.target_layer_ids:
            assert 0 <= lid < 8

    def test_convert_sets_mask_token_id(self):
        """Test that mask_token_id is set from config."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "mask_token_id")
        assert model.mask_token_id == 0


def test_qwen3_vl_transformers_530_position_ids_expand_video_grid(monkeypatch):
    """Only mRoPE receives a per-frame video grid on Transformers 5.3.0."""
    original_grid = torch.tensor([[3, 4, 5], [2, 6, 7]])
    expected_position_ids = torch.ones(3, 1, 12, dtype=torch.long)
    get_rope_index = MagicMock(return_value=(expected_position_ids, torch.zeros(1, 1)))
    compute_position_ids = MagicMock()
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl"),
        model=SimpleNamespace(
            get_rope_index=get_rope_index,
            compute_3d_position_ids=compute_position_ids,
        ),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.3.0")

    position_ids = HFDFlashModel._qwen3_vl_position_ids(
        fake_model,
        input_ids=torch.ones(1, 12, dtype=torch.long),
        attention_mask=torch.ones(1, 12, dtype=torch.long),
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        model_kwargs={
            "video_grid_thw": original_grid,
            "mm_token_type_ids": torch.tensor([[2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0]]),
        },
    )

    assert position_ids is expected_position_ids
    assert not compute_position_ids.called
    assert torch.equal(original_grid, torch.tensor([[3, 4, 5], [2, 6, 7]]))
    assert torch.equal(
        get_rope_index.call_args.kwargs["video_grid_thw"],
        torch.tensor([[1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 6, 7], [1, 6, 7]]),
    )


def test_qwen3_vl_moe_transformers_530_position_ids_expand_video_grid(monkeypatch):
    """Qwen3-VL family variants use the same 5.3.0 mRoPE workaround."""
    expected_position_ids = torch.ones(3, 1, 4, dtype=torch.long)
    get_rope_index = MagicMock(return_value=(expected_position_ids, torch.zeros(1, 1)))
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl_moe"),
        model=SimpleNamespace(get_rope_index=get_rope_index),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.3.0")

    position_ids = HFDFlashModel._qwen3_vl_position_ids(
        fake_model,
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        model_kwargs={
            "video_grid_thw": torch.tensor([[2, 4, 4]]),
            "mm_token_type_ids": torch.tensor([[2, 0, 2, 0]]),
        },
    )

    assert position_ids is expected_position_ids
    assert torch.equal(
        get_rope_index.call_args.kwargs["video_grid_thw"],
        torch.tensor([[1, 4, 4], [1, 4, 4]]),
    )


def test_qwen3_vl_transformers_53_patch_release_raises(monkeypatch):
    """Avoid double expansion when a 5.3 patch backports the upstream fix."""
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl"),
        model=SimpleNamespace(get_rope_index=MagicMock()),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.3.1")

    with pytest.raises(RuntimeError, match=r"5\.3\.0 or >=5\.4\.0"):
        HFDFlashModel._qwen3_vl_position_ids(
            fake_model,
            input_ids=torch.ones(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            model_kwargs={
                "video_grid_thw": torch.tensor([[1, 4, 4]]),
                "mm_token_type_ids": torch.tensor([[2, 0, 0, 0]]),
            },
        )


def test_qwen3_vl_transformers_54_uses_native_position_ids(monkeypatch):
    """Transformers 5.4+ performs the grid expansion inside get_rope_index."""
    get_rope_index = MagicMock()
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl"),
        model=SimpleNamespace(get_rope_index=get_rope_index),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.4.0")

    position_ids = HFDFlashModel._qwen3_vl_position_ids(
        fake_model,
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        model_kwargs={
            "video_grid_thw": torch.tensor([[1, 4, 4]]),
            "mm_token_type_ids": torch.tensor([[2, 0, 0, 0]]),
        },
    )

    assert position_ids is None
    assert not get_rope_index.called


def test_qwen3_vl_transformers_530_rejects_bad_video_frame_groups(monkeypatch):
    """Fail before mRoPE construction when processor and video-grid contracts differ."""
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl"),
        model=SimpleNamespace(get_rope_index=MagicMock()),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.3.0")

    with pytest.raises(ValueError, match="video frame groups"):
        HFDFlashModel._qwen3_vl_position_ids(
            fake_model,
            input_ids=torch.ones(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            model_kwargs={
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
                "mm_token_type_ids": torch.tensor([[2, 2, 0, 0]]),
            },
        )


def test_multimodal_forward_kwargs_exclude_non_model_inputs():
    """Do not forward Trainer or collator-only fields to Hugging Face models."""
    pixel_values = torch.ones(1)
    mm_token_type_ids = torch.zeros(1, 4, dtype=torch.long)

    forwarded = hf_dflash._multimodal_forward_kwargs(
        {
            "pixel_values": pixel_values,
            "mm_token_type_ids": mm_token_type_ids,
            "assistant_masks": torch.ones(1, 4),
            "loss_mask": torch.ones(1, 4),
            "num_items_in_batch": 4,
            "unexpected_dataset_column": "drop me",
        }
    )

    assert set(forwarded) == {"pixel_values", "mm_token_type_ids"}
    assert forwarded["pixel_values"] is pixel_values
    assert forwarded["mm_token_type_ids"] is mm_token_type_ids


def test_eval_does_not_precompute_qwen3_vl_position_ids(monkeypatch):
    """Evaluation delegates mRoPE construction to the base model and its cache."""
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, [("dflash", get_dflash_config())])
    precompute_position_ids = MagicMock()
    monkeypatch.setattr(model, "_qwen3_vl_position_ids", precompute_position_ids)

    model.eval()
    model(input_ids=torch.tensor([[1, 2, 3, 4]]))

    precompute_position_ids.assert_not_called()


def test_qwen3_vl_transformers_53_position_ids_require_mm_token_types(monkeypatch):
    """Never silently fall back to one-dimensional positions for a visual batch."""
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl"),
        model=SimpleNamespace(get_rope_index=MagicMock()),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.3.0")

    with pytest.raises(ValueError, match="mm_token_type_ids"):
        HFDFlashModel._qwen3_vl_position_ids(
            fake_model,
            input_ids=torch.ones(1, 12, dtype=torch.long),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            model_kwargs={"image_grid_thw": torch.tensor([[1, 4, 4]])},
        )


def test_qwen3_vl_transformers_53_position_ids_reject_bad_mm_token_shape(monkeypatch):
    """Keep processor-produced modality ids aligned with the padded text sequence."""
    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl"),
        model=SimpleNamespace(get_rope_index=MagicMock()),
    )
    monkeypatch.setattr(hf_dflash.transformers, "__version__", "5.3.0")

    with pytest.raises(ValueError, match="same shape as input_ids"):
        HFDFlashModel._qwen3_vl_position_ids(
            fake_model,
            input_ids=torch.ones(1, 12, dtype=torch.long),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            model_kwargs={
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
                "mm_token_type_ids": torch.zeros(1, 11, dtype=torch.long),
            },
        )


class TestDPaceWeights:
    """Test the D-PACE position-weighting objective (arXiv:2605.18810)."""

    @staticmethod
    def _reference_weights(conf, alpha):
        """Paper closed form computed by explicit summation (Eq.7-8).

        q~_i = (1-a)q_i + a; C_m = prod_{i<=m} q~_i; w_j = sum_{m>=j} C_m.
        Deliberately a plain double loop so it is an independent oracle for the
        vectorized implementation under test.
        """
        smoothed = alpha + (1.0 - alpha) * conf
        length = smoothed.shape[-1]
        cum = torch.ones_like(smoothed)
        running = torch.ones(smoothed.shape[:-1])
        for m in range(length):
            running = running * smoothed[..., m]
            cum[..., m] = running
        expected = torch.zeros_like(smoothed)
        for j in range(length):
            expected[..., j] = cum[..., j:].sum(dim=-1)
        return expected

    def test_weights_match_paper_formula(self):
        """Eq.7-8, pinned both to a hand-worked value and an independent loop oracle.

        conf=[0.8, 0.5], alpha=0.5 -> q~=[0.9, 0.75] -> prefix=[0.9, 0.675]
        -> w=[0.9+0.675, 0.675]=[1.575, 0.675].
        """
        hand = _dpace_position_weights(torch.tensor([[0.8, 0.5]]), alpha=0.5)
        assert torch.allclose(hand, torch.tensor([[1.575, 0.675]]), atol=1e-6)
        conf = torch.tensor([[0.9, 0.6, 0.3, 0.8]])
        assert torch.allclose(
            _dpace_position_weights(conf, 0.5), self._reference_weights(conf, 0.5), atol=1e-6
        )

    def test_mask_makes_invalid_positions_noops(self):
        """Invalid positions neither shrink the prefix product nor add to the sum."""
        alpha = 0.5
        conf = torch.tensor([[0.9, 0.2, 0.3, 0.8]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        masked = _dpace_position_weights(conf, alpha, valid_mask=mask)
        # Dropping the invalid slot entirely must give the same weights at the kept slots.
        kept = _dpace_position_weights(conf[:, [0, 2, 3]], alpha)
        assert torch.allclose(masked[:, [0, 2, 3]], kept, atol=1e-6)

    def test_weights_are_detached(self):
        """Weights must carry no gradient (paper Eq.9 detaches them)."""
        conf = torch.rand(2, 3, 5, requires_grad=True)
        weights = _dpace_position_weights(conf, 0.5)
        assert not weights.requires_grad

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="dflash_dpace_alpha"):
            _dpace_position_weights(torch.rand(1, 4), alpha=1.5)

    def test_default_objective_is_dpace(self):
        """D-PACE is the default (alpha=0.5); an explicit alpha override is wired through."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", get_dflash_config())])
        assert model.dflash_loss_objective == "dpace"
        assert model.dflash_dpace_alpha == 0.5

        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        config["dflash_dpace_alpha"] = 0.3
        mtsp.convert(model, [("dflash", config)])
        assert model.dflash_dpace_alpha == 0.3

    def test_convert_rejects_bad_objective(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        config["dflash_loss_objective"] = "nope"
        with pytest.raises(ValueError, match="dflash_loss_objective"):
            mtsp.convert(model, [("dflash", config)])

    def test_convert_rejects_degenerate_alpha(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        config["dflash_loss_objective"] = "dpace"
        config["dflash_dpace_alpha"] = 0.0
        with pytest.raises(ValueError, match="dflash_dpace_alpha"):
            mtsp.convert(model, [("dflash", config)])

    def test_convert_dpace_with_decay_factor_warns(self, caplog):
        """dpace + a non-zero decay factor converts but warns that decay is ignored."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        config["dflash_loss_objective"] = "dpace"
        config["dflash_loss_decay_factor"] = 4.0
        with caplog.at_level(logging.WARNING):
            mtsp.convert(model, [("dflash", config)])
        assert any("dflash_loss_decay_factor" in r.message for r in caplog.records)


class TestDPaceLossIntegration:
    """Exercise the _compute_loss block-weighting branches on CPU."""

    @staticmethod
    def _make_inputs(vocab=32, seq_len=SEQ_LEN, n_blocks=2):
        """Synthetic CPU inputs for _compute_loss (no model forward needed)."""
        bsz = 1
        logits = torch.randn(bsz, n_blocks * BLOCK_SIZE, vocab)
        input_ids = torch.randint(0, vocab, (bsz, seq_len))
        anchor_positions = torch.tensor([[0, BLOCK_SIZE]])[:, :n_blocks]
        block_keep_mask = torch.ones(bsz, n_blocks)
        loss_mask = torch.ones(bsz, seq_len)
        return logits, input_ids, anchor_positions, block_keep_mask, loss_mask

    def _converted_model(self, objective, **overrides):
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        config["dflash_loss_objective"] = objective
        config.update(overrides)
        mtsp.convert(model, [("dflash", config)])
        return model

    def test_compute_loss_dpace_branch(self):
        """Default dpace objective produces a finite loss and valid accuracy."""
        model = self._converted_model("dpace")
        loss, acc = model._compute_loss(*self._make_inputs())
        assert torch.isfinite(loss).item() and loss.item() > 0
        assert 0.0 <= acc <= 1.0

    def test_compute_loss_decay_branch(self):
        """The static-decay objective path also produces a finite loss."""
        model = self._converted_model("decay", dflash_loss_decay_factor=4.0)
        loss, acc = model._compute_loss(*self._make_inputs())
        assert torch.isfinite(loss).item() and loss.item() > 0
        assert 0.0 <= acc <= 1.0

    def test_compute_loss_dpace_kd_branch(self):
        """dpace + KD (base_logits given): confidences use a dedicated no_grad CE pass."""
        vocab = 32
        model = self._converted_model("dpace")
        inputs = self._make_inputs(vocab=vocab)
        base_logits = torch.randn(1, SEQ_LEN, vocab)
        loss, acc = model._compute_loss(*inputs, base_logits=base_logits)
        assert torch.isfinite(loss).item()
        assert 0.0 <= acc <= 1.0


class TestDFlashSaveRestore:
    """Test DFlash model save and restore."""

    def test_save_and_restore(self, tmp_path):
        """Test round-trip save/load preserves modelopt state and outputs."""
        mto.enable_huggingface_checkpointing()
        model_ref = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model_ref, [("dflash", config)])

        model_ref.save_pretrained(tmp_path / "modelopt_model")
        assert os.path.exists(tmp_path / "modelopt_model/modelopt_state.pth")

        model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
        assert isinstance(model_test, HFDFlashModel)
        tf_modelopt_state_and_output_tester(model_ref, model_test)


class TestDFlashLazyRotaryEmb:
    """Rotary embedding creation: eager on a real device, still lazy on meta.

    The laziness exists for one reason — ``__init__`` runs on the meta device during
    ``from_pretrained`` restore, and building ``inv_freq`` there produces a meta buffer.
    That reason only applies on meta, and deferring everywhere else costs a correctness
    property under DDP: ``inv_freq`` is non-persistent, so a rank whose batch has no
    valid anchor returns before running the draft and ends the step one buffer short.
    ``broadcast_buffers`` then coalesces mismatched buffer lists across ranks and hangs
    rather than raising, which single-node runs never reproduce.

    So the buffer is built during ``modify()`` when the base model is on a real device,
    and still deferred when it is on meta.
    """

    def test_rotary_emb_created_during_convert_on_a_real_device(self):
        """On a real device the buffer exists after convert, before any forward."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model.dflash_module, "rotary_emb")
        assert not any(b.is_meta for b in model.dflash_module.rotary_emb.buffers())

    def test_rotary_emb_deferred_on_meta(self):
        """On meta the buffer is still deferred, which is what the laziness is for."""
        model = get_tiny_llama(num_hidden_layers=4).to("meta")
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert not hasattr(model.dflash_module, "rotary_emb")

    def test_rotary_emb_init_is_idempotent(self):
        """A later _maybe_init_rotary_emb must not replace an existing buffer."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        dflash_mod = model.dflash_module
        first = dflash_mod.rotary_emb
        dflash_mod._maybe_init_rotary_emb(device="cpu")
        assert dflash_mod.rotary_emb is first
        assert not any(b.is_meta for b in dflash_mod.rotary_emb.buffers())


class TestBuildTargetLayerIds:
    """Test target layer selection."""

    def test_single_draft_layer(self):
        ids = build_target_layer_ids(32, 1)
        assert len(ids) == 1
        assert ids[0] == 16

    def test_multiple_draft_layers(self):
        ids = build_target_layer_ids(36, 5)
        assert len(ids) == 5
        assert ids == sorted(ids)
        assert all(1 <= lid <= 33 for lid in ids)

    def test_layer_ids_no_duplicates(self):
        ids = build_target_layer_ids(32, 5)
        assert len(set(ids)) == 5

    def test_layer_ids_match_zlab(self):
        """Test layer IDs match z-lab reference for Qwen3-8B (36 layers, 5 draft)."""
        ids = build_target_layer_ids(36, 5)
        assert ids == [1, 9, 17, 25, 33]


class TestDFlashSlidingWindow:
    """Test sliding window attention support."""

    def test_sliding_window_from_config(self):
        """Test DFlashAttention reads sliding_window from config.layer_types."""
        from transformers import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=256,
            _attn_implementation="sdpa",
        )
        attn_full = DFlashAttention(config, layer_idx=0)
        attn_sliding = DFlashAttention(config, layer_idx=1)
        assert attn_full.sliding_window is None
        assert attn_sliding.sliding_window == 256

    def test_no_sliding_window_without_config(self):
        """Test DFlashAttention defaults to no sliding window."""
        from transformers import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            _attn_implementation="sdpa",
        )
        attn = DFlashAttention(config, layer_idx=0)
        assert attn.sliding_window is None


class TestDFlashSwaMask:
    """Test all-layer non-causal sliding-window attention mask (MiMo-style)."""

    def test_window_masks_context_beyond_window(self):
        """Context beyond the window (relative to each query's real position) is masked out."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config(block_size=4)
        window = 6
        config["dflash_swa_window_size"] = window
        mtsp.convert(model, [("dflash", config)])

        seq_len = 16
        block_size = 4
        # One block anchored at position 10 → query real positions [10, 11, 12, 13].
        anchor_positions = torch.tensor([[10]], dtype=torch.long)
        block_keep_mask = torch.tensor([[True]])
        dtype = torch.float32
        device = torch.device("cpu")

        mask = model._build_draft_attention_mask(
            seq_len, anchor_positions, block_keep_mask, 1, dtype, device, window=window
        )
        neg = torch.finfo(dtype).min
        attend = mask > neg / 2  # True where a position is attended (additive mask == 0)

        # Context kv are positions [0, seq_len). For query k (real pos 10 + k) only context
        # positions in (10 + k - window, 10) are visible.
        for k in range(block_size):
            q_real = 10 + k
            for c in range(seq_len):
                visible = attend[0, 0, k, c].item()
                if c < 10:  # context strictly before the anchor
                    assert visible == (c > q_real - window), (
                        f"query k={k} (pos {q_real}), context c={c}: "
                        f"expected visible={c > q_real - window}, got {visible}"
                    )

    def test_window_is_subset_of_full(self):
        """The windowed mask attends to a subset of what the full-attention mask attends to."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config(block_size=4)
        config["dflash_swa_window_size"] = 6
        mtsp.convert(model, [("dflash", config)])

        args = (
            16,
            torch.tensor([[10]]),
            torch.tensor([[True]]),
            1,
            torch.float32,
            torch.device("cpu"),
        )
        full = model._build_draft_attention_mask(*args, window=None)
        windowed = model._build_draft_attention_mask(*args, window=6)
        neg = torch.finfo(torch.float32).min
        # Everything masked by full attention must also be masked by the windowed mask.
        assert ((full <= neg / 2) <= (windowed <= neg / 2)).all()
        # The window strictly removes some connections (it is not a no-op here).
        assert (windowed <= neg / 2).sum() > (full <= neg / 2).sum()

    def test_window_smaller_than_block_rejected(self):
        """A window smaller than the block size is rejected at config validation."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config(block_size=4)
        config["dflash_swa_window_size"] = 2  # < block_size
        with pytest.raises(ValueError, match="dflash_swa_window_size"):
            mtsp.convert(model, [("dflash", config)])


class TestValidateOnline:
    """Test validate_online acceptance counting logic."""

    def test_all_accepted(self):
        """When all draft tokens match posterior, AR = 1 + steps."""
        validator = AcceptanceRateValidation.__new__(AcceptanceRateValidation)
        validator.check_data_consistency_across_ranks = lambda x: x

        # Mock model: draft produces [10, 20], posterior agrees
        mock_model = MagicMock()
        # pseudo_speculative_generate returns (base_token, draft_tokens)
        call_count = [0]

        def mock_psg(input_ids, steps=1):
            call_count[0] += 1
            base = torch.tensor([[100]])
            drafts = torch.tensor([[10, 20]])
            return base, drafts

        mock_model.pseudo_speculative_generate = mock_psg

        # Base model returns logits where argmax matches draft tokens
        def mock_base_model(candidate):
            # posterior at pos i = candidate[i+1] (perfect prediction)
            return SimpleNamespace(last_hidden_state=candidate.unsqueeze(-1).float())

        def mock_lm_head(hidden):
            # One-hot logits: argmax = candidate token at next position
            bsz, seq_len, _ = hidden.shape
            logits = torch.zeros(bsz, seq_len, 200)
            for i in range(seq_len - 1):
                token_id = int(hidden[0, i + 1, 0].item())
                if 0 <= token_id < 200:
                    logits[0, i, token_id] = 1.0
            return logits

        mock_model._base_model = mock_base_model
        mock_model._base_model_lm_head = mock_lm_head
        validator.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        # osl=3: need 3 new tokens. Step 1: base(1) + draft(2) = 3 tokens → done in 1 step
        result_ids, ar = validator.validate_online(osl=3, input_ids=input_ids, steps=2)
        assert ar == 3.0  # 1 step, 3 tokens accepted (1 base + 2 drafts)

    def test_all_rejected(self):
        """When no draft tokens match, AR = 1 (base token only + correction)."""
        validator = AcceptanceRateValidation.__new__(AcceptanceRateValidation)
        validator.check_data_consistency_across_ranks = lambda x: x

        mock_model = MagicMock()

        def mock_psg(input_ids, steps=1):
            base = torch.tensor([[100]])
            drafts = torch.tensor([[10, 20]])
            return base, drafts

        mock_model.pseudo_speculative_generate = mock_psg

        # Base model disagrees with all drafts
        def mock_base_model(candidate):
            return SimpleNamespace(last_hidden_state=candidate.unsqueeze(-1).float())

        def mock_lm_head(hidden):
            bsz, seq_len, _ = hidden.shape
            # All predictions are token 99 (won't match drafts 10, 20)
            logits = torch.zeros(bsz, seq_len, 200)
            logits[:, :, 99] = 1.0
            return logits

        mock_model._base_model = mock_base_model
        mock_model._base_model_lm_head = mock_lm_head
        validator.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        # osl=4: each step produces base(1) + correction(1) = 2 tokens, AR = 1
        # (correction token not counted as accepted)
        result_ids, ar = validator.validate_online(osl=4, input_ids=input_ids, steps=2)
        # 2 steps, each producing 1 base token + 0 accepted drafts = 2 total accepted
        # But correction token still advances sequence, so 2 steps produce 4 tokens
        assert ar == 1.0  # total_accepted=2, cnt=2


class TestDFlashExporter:
    """Test DFlash checkpoint export."""

    def test_export_creates_files(self, tmp_path):
        """Test that export produces model.safetensors and config.json."""

        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        exporter = model.get_exporter()
        export_dir = tmp_path / "exported"
        exporter.export(export_dir)

        assert (export_dir / "model.safetensors").exists()
        assert (export_dir / "config.json").exists()

    def test_export_state_dict_has_no_prefix(self, tmp_path):
        """Exported weights should not have dflash_module. or model. prefix."""
        from safetensors.torch import load_file

        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        exporter = model.get_exporter()
        export_dir = tmp_path / "exported"
        exporter.export(export_dir)

        sd = load_file(str(export_dir / "model.safetensors"))
        for key in sd:
            assert "dflash_module." not in key, f"Key has prefix: {key}"
            assert "model." not in key, f"Key has prefix: {key}"
            assert "rotary_emb" not in key, f"Rotary buffer should be excluded: {key}"

    def test_export_config_fields(self, tmp_path):
        """Exported config.json should have required DFlash fields."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        exporter = model.get_exporter()
        export_dir = tmp_path / "exported"
        exporter.export(export_dir)

        with open(export_dir / "config.json") as f:
            cfg = json.load(f)

        assert cfg["architectures"] == ["DFlashDraftModel"]
        assert cfg["block_size"] == BLOCK_SIZE
        assert "dflash_config" in cfg
        assert "mask_token_id" in cfg["dflash_config"]
        assert "target_layer_ids" in cfg["dflash_config"]
        assert cfg["num_hidden_layers"] == NUM_DRAFT_LAYERS
        assert "hidden_size" in cfg
        assert "vocab_size" in cfg
        assert "layer_types" in cfg
        assert len(cfg["layer_types"]) == NUM_DRAFT_LAYERS
        # Without SWA configured, no sliding-window fields are emitted.
        assert "sliding_window" not in cfg
        assert "use_swa" not in cfg["dflash_config"]

    def test_export_swa_fields(self, tmp_path):
        """With dflash_swa_window_size set, exported config carries vLLM's SWA fields."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        config["dflash_swa_window_size"] = 256
        mtsp.convert(model, [("dflash", config)])

        exporter = model.get_exporter()
        export_dir = tmp_path / "exported"
        exporter.export(export_dir)

        with open(export_dir / "config.json") as f:
            cfg = json.load(f)

        # vLLM _resolve_layer_attention reads these; all-full layer_types + use_swa=True
        # → non-causal sliding window on every draft layer.
        assert cfg["sliding_window"] == 256
        assert cfg["dflash_config"]["use_swa"] is True
        assert cfg["dflash_config"]["swa_window_size"] == 256
        assert cfg["dflash_config"]["causal"] is False
        # The pre-existing dflash_config keys must survive the update.
        assert "mask_token_id" in cfg["dflash_config"]
        assert "target_layer_ids" in cfg["dflash_config"]

    def test_export_tensor_count(self, tmp_path):
        """Exported model should have the right number of tensors."""
        from safetensors.torch import load_file

        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        exporter = model.get_exporter()
        export_dir = tmp_path / "exported"
        exporter.export(export_dir)

        sd = load_file(str(export_dir / "model.safetensors"))
        # Should have: fc, hidden_norm, norm, + per-layer (q/k/v/o_proj, q/k_norm,
        # gate/up/down_proj, input_layernorm, post_attention_layernorm)
        assert len(sd) > 0
        assert any("fc.weight" in k for k in sd)
        assert any("norm.weight" in k for k in sd)
        assert any("layers.0" in k for k in sd)


class TestEnsureGenerationTags:
    """Test _ensure_generation_tags masking with the local tiny tokenizer.

    Masking is driven by the injected ``{% generation %}`` blocks and
    ``return_assistant_tokens_mask``, so the local tiny tokenizer suffices — no
    HF Hub download needed.
    """

    def test_chatml_think_template_produces_assistant_mask(self, tiny_tokenizer):
        """Verify generation-tagged chat template produces correct assistant masks."""
        collator = LanguageDataCollator(
            tokenizer=tiny_tokenizer,
            train_len=128,
            return_labels=True,
            answer_only_loss=True,
        )

        # The tiny tokenizer ships a generation-tagged chat template
        assert "generation" in collator.tokenizer.chat_template

        # Tokenize a sample conversation
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4."},
                ]
            }
        ]
        result = collator(samples)

        labels = result["labels"]
        input_ids = result["input_ids"]

        # Labels should NOT be all-masked (assistant content should have real labels)
        assert (labels != -100).any(), "All labels masked — generation tags not working"

        # Labels should have SOME masked positions (user/system tokens)
        assert (labels == -100).any(), "No labels masked — answer_only_loss not working"

        # Decode the non-masked positions to verify they're assistant content
        non_masked = input_ids[labels != -100]
        decoded = tiny_tokenizer.decode(non_masked)
        assert "The answer is 4" in decoded

    def test_multi_turn_masks_only_assistant(self, tiny_tokenizer):
        """Verify multi-turn: only assistant turns are unmasked."""
        collator = LanguageDataCollator(
            tokenizer=tiny_tokenizer,
            train_len=256,
            return_labels=True,
            answer_only_loss=True,
        )

        samples = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I am fine."},
                ]
            }
        ]
        result = collator(samples)
        labels = result["labels"]
        input_ids = result["input_ids"]

        non_masked = input_ids[labels != -100]
        decoded = tiny_tokenizer.decode(non_masked)
        # Both assistant responses should appear in unmasked tokens
        assert "Hi there" in decoded
        assert "I am fine" in decoded
        # User/system content should NOT appear in unmasked tokens
        assert "You are helpful" not in decoded
        assert "How are you?" not in decoded


def _dflash_batch(vocab_size, bsz=2, seq_len=SEQ_LEN):
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (bsz, seq_len))
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _converted(fp32_master_weights=None, num_hidden_layers=4):
    model = get_tiny_llama(num_hidden_layers=num_hidden_layers)
    config = get_dflash_config()
    if fp32_master_weights is not None:
        config["dflash_fp32_master_weights"] = fp32_master_weights
    mtsp.convert(model, [("dflash", config)])
    return model


class TestDFlashFp32MasterWeights:
    """``dflash_fp32_master_weights``: what is promoted, and what the optimizer inherits.

    The parameter dtype is the visible half; the OPTIMIZER's dtype is the point of the
    change, and it is not decided until AdamW allocates its moments with ``zeros_like(p)``
    inside the first step. A test that looked only at parameters would pass while the
    feature was broken.
    """

    def test_flag_off_leaves_the_draft_in_the_base_dtype(self):
        model = _converted(fp32_master_weights=False)
        assert model._base_model.dtype == torch.bfloat16
        assert {p.dtype for p in model.dflash_module.parameters()} == {torch.bfloat16}

    def test_flag_on_promotes_only_the_draft(self):
        model = _converted(fp32_master_weights=True)
        assert {p.dtype for p in model.dflash_module.parameters()} == {torch.float32}
        # The frozen base is deliberately left alone: no trainable parameters, no
        # optimizer state, and promoting it would change the hidden states the draft
        # is trained against.
        assert model._base_model.dtype == torch.bfloat16

    def test_adam_moments_follow_the_parameters(self):
        """The half that actually matters, and the one a parameter check would miss."""
        moments = {}
        for flag, expected in ((False, torch.bfloat16), (True, torch.float32)):
            model = _converted(fp32_master_weights=flag)
            model.train()
            trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            out = model(**_dflash_batch(model.dflash_config.vocab_size))
            out.loss.backward()
            optimizer.step()

            seen = {
                state[key].dtype
                for state in optimizer.state.values()
                for key in ("exp_avg", "exp_avg_sq")
            }
            assert seen == {expected}, f"flag={flag}: moment dtypes {seen}"
            moments[flag] = seen
        assert moments[False] != moments[True]

    def test_promotion_keeps_the_full_precision_draw(self):
        """Promoted and unpromoted runs share a draw; only the stored precision differs.

        The draft is drawn in fp32 by ``_init_weights``. An unpromoted run rounds that draw
        to the base model's dtype; a promoted one keeps it. So the two arms of a
        bf16-vs-fp32 comparison start from the same initialization, each at the precision it
        trains in, rather than from two different draws.
        """
        torch.manual_seed(1234)
        bf16_model = _converted(fp32_master_weights=False)
        torch.manual_seed(1234)
        fp32_model = _converted(fp32_master_weights=True)

        bf16_params = dict(bf16_model.dflash_module.named_parameters())
        fp32_params = dict(fp32_model.dflash_module.named_parameters())
        assert bf16_params.keys() == fp32_params.keys()
        keeps_finer_bits = False
        for name, bf16_param in bf16_params.items():
            promoted = fp32_params[name]
            assert promoted.dtype == torch.float32
            # Same draw: rounding the promoted copy down recovers the unpromoted arm exactly.
            assert torch.equal(promoted.to(torch.bfloat16), bf16_param), name
            keeps_finer_bits |= not torch.equal(promoted, bf16_param.float())
        assert keeps_finer_bits, "the promoted draft should hold bits bf16 cannot represent"

    def test_promotion_survives_a_checkpoint_restore(self):
        """A resumed run must not quietly drop back to the base dtype.

        ``modify()`` runs with the base on meta during ``from_pretrained``, so it cannot
        place the draft at all. If nothing re-applies it, the draft resumes at the loaded
        dtype and AdamW allocates its moments to match, which switches the feature off for
        the whole remainder of a long run.
        """
        mto.enable_huggingface_checkpointing()
        model = _converted(fp32_master_weights=True)
        reference = {n: p.detach().clone() for n, p in model.dflash_module.named_parameters()}

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored = AutoModelForCausalLM.from_pretrained(tmp)

            # What the restore leaves behind on its own, which is the bug this guards.
            assert {p.dtype for p in restored.dflash_module.parameters()} == {torch.bfloat16}

            restored.restore_draft_precision(tmp)

        assert {p.dtype for p in restored.dflash_module.parameters()} == {torch.float32}
        assert hasattr(restored.dflash_module, "rotary_emb")
        # The checkpoint stores the draft in fp32; reloading at the stored dtype is what
        # keeps a resume from costing the run a rounding of its master weights.
        for name, param in restored.dflash_module.named_parameters():
            assert torch.equal(param.detach(), reference[name]), name

        restored.train()
        optimizer = torch.optim.AdamW(
            [p for p in restored.dflash_module.parameters() if p.requires_grad], lr=1e-4
        )
        out = restored(**_dflash_batch(restored.dflash_config.vocab_size))
        out.loss.backward()
        optimizer.step()
        assert {
            state[key].dtype
            for state in optimizer.state.values()
            for key in ("exp_avg", "exp_avg_sq")
        } == {torch.float32}

    def test_forward_needs_no_autocast_from_the_caller(self):
        """A promoted draft is fed bf16 hidden states by the frozen bf16 target.

        The model supplies the autocast that reconciles them, so this works without the
        caller wrapping anything -- which is what evaluation, generation and any direct
        ``convert()`` + forward rely on.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        out = model(**_dflash_batch(model.dflash_config.vocab_size))
        assert torch.isfinite(out.loss)

    def test_supplied_autocast_matches_the_trainers(self):
        """Nesting inside HF Trainer's own autocast changes nothing.

        The published training path runs under ``TrainingArguments.bf16``, so the model's
        context must be inert there rather than merely harmless.
        """
        batch = _dflash_batch(_converted().dflash_config.vocab_size)

        torch.manual_seed(7)
        model = _converted(fp32_master_weights=True)
        model.train()
        torch.manual_seed(7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            wrapped = model(**batch).loss

        torch.manual_seed(7)
        model = _converted(fp32_master_weights=True)
        model.train()
        torch.manual_seed(7)
        bare = model(**batch).loss

        assert torch.equal(wrapped.detach(), bare.detach())

    def test_generation_needs_no_autocast_either(self):
        """``pseudo_speculative_generate`` is called directly, not through ``__call__``."""
        model = _converted(fp32_master_weights=True)
        model.eval()
        input_ids = _dflash_batch(model.dflash_config.vocab_size, bsz=1)["input_ids"]
        _, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=2)
        assert draft_tokens.shape[0] == 1


class TestDFlashDraftActivationCheckpointing:
    """``training.gradient_checkpointing`` has to reach the draft, and be inert when it does.

    The draft is the only trainable part of a DFlash setup, so it is the only part where
    checkpointing saves anything: the frozen target runs under ``no_grad`` and stores no
    activations, and a flag that landed only there would report the feature as enabled
    while saving nothing.
    """

    def test_the_flag_reaches_the_draft_layers(self):
        model = _converted()
        layers = list(model.dflash_module.layers)
        assert layers
        # Inheriting the supported base class is what makes HF's own recursion find them.
        assert all(isinstance(layer, GradientCheckpointingLayer) for layer in layers)
        assert all(layer.gradient_checkpointing is False for layer in layers)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        assert all(layer.gradient_checkpointing for layer in layers)

    def test_gradients_are_bit_identical_with_and_without(self):
        """Recompute is mathematically neutral; it trades step time for memory only."""
        grads = {}
        for enabled in (False, True):
            torch.manual_seed(99)
            model = _converted()
            model.train()
            if enabled:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            torch.manual_seed(99)
            out = model(**_dflash_batch(model.dflash_config.vocab_size))
            out.loss.backward()
            grads[enabled] = {
                name: param.grad.detach().clone()
                for name, param in model.dflash_module.named_parameters()
                if param.grad is not None
            }

        assert grads[False] and grads[False].keys() == grads[True].keys()
        for name, grad in grads[False].items():
            assert torch.equal(grad, grads[True][name]), name
