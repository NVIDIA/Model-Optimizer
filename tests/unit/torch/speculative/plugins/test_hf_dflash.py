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

import copy
import json
import logging
import os
import warnings
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import (
    get_tiny_llama,
    get_tiny_qwen3vl,
    tf_modelopt_state_and_output_tester,
)
from torch.nn.attention.flex_attention import create_mask
from transformers import (
    AutoModelForCausalLM,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    GraniteConfig,
    GraniteForCausalLM,
)
from transformers.modeling_layers import GradientCheckpointingLayer

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
import modelopt.torch.speculative.plugins.dflash_flex_attention as dflash_flex_attention
import modelopt.torch.speculative.plugins.hf_dflash as hf_dflash
from modelopt.torch.speculative.config import DFlashConfig, DFlashLossTerm
from modelopt.torch.speculative.plugins.dflash_losses import (
    DIVERGENCES,
    DraftLogits,
    _block_position_weights,
    _block_targets,
    _dpace_position_weights,
    _tvd_per_token,
    objective_term,
)
from modelopt.torch.speculative.plugins.hf_dflash import (
    DFlashAttention,
    DFlashBaseModelOutput,
    DFlashModule,
    DraftBatch,
    HFDFlashModel,
    build_target_layer_ids,
)
from modelopt.torch.speculative.plugins.hf_dflash2 import HFDFlash2Model
from modelopt.torch.speculative.plugins.hf_domino import HFDominoModel
from modelopt.torch.speculative.plugins.hf_dspark import HFDSparkModel
from modelopt.torch.speculative.plugins.hf_lilicorr import HFLiLiCorrModel
from modelopt.torch.speculative.plugins.master_weight_adamw import (
    MasterWeightAdamW,
    VerifyMasterWeightsCallback,
)
from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel
from modelopt.torch.speculative.utils import AcceptanceRateValidation, load_vlm_or_llm
from modelopt.torch.utils.plugins.transformers_dataset import LanguageDataCollator

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


def _compute_loss(model, logits, input_ids, anchors, keep, loss_mask, base_outputs=None):
    """``model._compute_loss`` on given draft logits [B, N * bs, V]: (loss, accuracy)."""
    batch = DraftBatch(input_ids, loss_mask, anchors, keep, base_outputs, hidden=None)
    block_logits = logits.view(*anchors.shape, BLOCK_SIZE, -1)
    with patch.object(model, "_draft_logits", return_value=DraftLogits(block_logits, block_logits)):
        out = model._compute_loss(model._loss_context(batch))
    return out.loss, out.accuracy


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

    def test_convert_nope_target_keeps_the_draft_rope_theta(self):
        """A target config that declares rope_theta but leaves it unset is not inherited."""
        model = get_tiny_llama(num_hidden_layers=4)
        model.config.rope_theta = None
        if isinstance(getattr(model.config, "rope_parameters", None), dict):
            model.config.rope_parameters.pop("rope_theta", None)
        mtsp.convert(model, [("dflash", get_dflash_config())])
        draft = model.dflash_config
        draft_rope_params = getattr(draft, "rope_parameters", None) or {}
        assert draft_rope_params.get("rope_theta", getattr(draft, "rope_theta", None)) is not None
        model.train()
        assert torch.isfinite(model(**_dflash_batch(model.config.vocab_size)).loss)


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


def _legacy_term_loss(name, flat_logits, flat_targets, flat_teacher, flat_weights):
    """Verbatim copy of the divergences' pre-refactor reductions, kept as the semantic reference."""
    valid_count = flat_weights.sum() + 1e-6
    if name == "kd":
        target_soft = torch.softmax(flat_teacher, dim=-1)
        draft_logsoft = torch.log_softmax(flat_logits, dim=-1)
        per_token = -(target_soft * draft_logsoft).sum(dim=-1)
    elif name == "tvd":
        per_token = _tvd_per_token(flat_logits, flat_teacher)
    else:
        per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        if name == "tv":
            per_token = 1.0 - torch.exp(-per_token)
    return (per_token * flat_weights).sum() / valid_count


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
        model = self._converted_model("dpace", dflash_self_logit_distillation=False)
        loss, acc = _compute_loss(model, *self._make_inputs())
        assert torch.isfinite(loss).item() and loss.item() > 0
        assert 0.0 <= acc <= 1.0

    def test_compute_loss_decay_branch(self):
        """The static-decay objective path also produces a finite loss."""
        model = self._converted_model(
            "decay", dflash_loss_decay_factor=4.0, dflash_self_logit_distillation=False
        )
        loss, acc = _compute_loss(model, *self._make_inputs())
        assert torch.isfinite(loss).item() and loss.item() > 0
        assert 0.0 <= acc <= 1.0

    def test_compute_loss_dpace_kd_branch(self):
        """dpace + KD: confidences use a dedicated no_grad CE pass."""
        vocab = 32
        model = self._converted_model("dpace")
        inputs = self._make_inputs(vocab=vocab)
        teacher = DFlashBaseModelOutput(None, logits=torch.randn(1, SEQ_LEN, vocab))
        loss, acc = _compute_loss(model, *inputs, base_outputs=teacher)
        assert torch.isfinite(loss).item()
        assert 0.0 <= acc <= 1.0

    def test_kd_without_a_teacher_raises(self):
        """KD is decided by the config, so a missing teacher cannot fall back to CE."""
        model = self._converted_model("dpace")
        with pytest.raises(ValueError, match="base model's distribution"):
            _compute_loss(model, *self._make_inputs(), base_outputs=DFlashBaseModelOutput(None))

    @pytest.mark.parametrize("name", ["ce", "kd", "tvd", "tv"])
    def test_each_term_reduces_as_before(self, name):
        """Under non-uniform position weights, each divergence is bitwise its legacy reduction."""
        vocab = 32
        model = self._converted_model("decay", dflash_loss_decay_factor=4.0)
        model._objective = [objective_term(name, 1.0)]
        logits, input_ids, anchors, keep, loss_mask = self._make_inputs(vocab=vocab)
        base_outputs = DFlashBaseModelOutput(None, logits=torch.randn(1, SEQ_LEN, vocab))
        loss, _ = _compute_loss(model, logits, input_ids, anchors, keep, loss_mask, base_outputs)

        targets = _block_targets(input_ids, anchors, keep, loss_mask, BLOCK_SIZE, 0)
        weights = _block_position_weights(targets.mask, "decay", 1, 4.0, model.dflash_dpace_alpha)
        teacher = model._teacher_logits(base_outputs, targets.positions)
        legacy = _legacy_term_loss(
            name,
            logits.view(-1, vocab),
            targets.token_ids.view(-1),
            teacher.reshape(-1, vocab),
            weights.view(-1),
        )
        assert torch.equal(loss, legacy)


# What each variant's conversion requires beyond get_dflash_config(), sized for the tiny
# model: its dflash_architecture_config fields, then its other config fields.
_VARIANTS = {
    "dflash": ({}, {}),
    "dflash2": (
        {"conv_kernel_size": 2, "conv_group_size": 4, "selector_rank": 8, "selector_top_k": 5},
        {},
    ),
    "lilicorr": (
        {
            "lilicorr_candidate_topk": 4,
            "lilicorr_num_layers": 1,
            "lilicorr_num_heads": 4,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 8,
            "lilicorr_vector_eps": 1e-4,
            "lilicorr_logit_scale": 8.0,
        },
        {
            "dflash_lilicorr_w_ce": 0.25,
            "dflash_lilicorr_w_margin": 0.0,
            "dflash_lilicorr_w_pen": 0.25,
        },
    ),
    "dspark": ({"markov_rank": 16}, {}),
    # lambda_base below 1, so that Domino's corrected logits train too.
    "domino": ({"emb_dim": 16, "gru_hidden_dim": 32}, {"dflash_lambda_base_start": 0.5}),
}


def _variant_model(projector_type, offline=False, arch=None, **overrides):
    """A tiny model converted to ``projector_type``, with ``arch`` and ``overrides`` applied."""
    arch_fields, fields = _VARIANTS[projector_type]
    config = get_dflash_config(offline=offline)
    config["dflash_architecture_config"].update(
        projector_type=projector_type, **arch_fields, **(arch or {})
    )
    config.update(fields, **overrides)
    model = get_tiny_llama(num_hidden_layers=4)
    model.config.num_orig_hidden_layers = 4
    mtsp.convert(model, [("dflash", config)])
    return model


def _objective_spec(model):
    """The objective as data: each term's name, weight, class and bound parameters."""
    return [(name, weight, type(term), vars(term)) for name, weight, term in model._objective]


class TestLossTerms:
    """``dflash_loss_terms`` composes the block objective of any variant from the registry."""

    def test_every_configurable_term_is_implemented(self):
        assert set(get_args(DFlashLossTerm)) == set(DIVERGENCES)

    @pytest.mark.parametrize(
        ("terms", "match"),
        [
            ({"ce": 1.0, "l1": 1.0}, "'ce', 'kd', 'tvd', 'tv' or 'lambda'"),
            ({"ce": -0.5}, "greater than or equal to 0"),
            ({"ce": 0.0}, "weight > 0"),
        ],
        ids=["unknown", "negative", "all-zero"],
    )
    def test_bad_terms_are_rejected(self, terms, match):
        with pytest.raises(ValueError, match=match):
            DFlashConfig(dflash_loss_terms=terms)

    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("dflash_self_logit_distillation", False),
            ("dflash_lk_loss_type", "tv"),
            ("dflash_ce_loss_alpha", 0.5),
            ("dflash_l1_loss_alpha", 0.5),
        ],
    )
    def test_terms_and_a_legacy_selector_are_rejected(self, selector, value):
        """One source for the objective."""
        with pytest.raises(ValueError, match=selector):
            DFlashConfig(dflash_loss_terms={"ce": 1.0}, **{selector: value})

    @pytest.mark.parametrize(
        ("projector_type", "legacy", "terms"),
        [
            ("dflash", {}, {"kd": 1.0}),
            ("dflash", {"dflash_self_logit_distillation": False}, {"ce": 1.0}),
            (
                "dflash2",
                {"dflash_self_logit_distillation": False, "dflash_lk_loss_type": "tv"},
                {"tv": 1.0},
            ),
            (
                "dflash2",
                {"dflash_self_logit_distillation": False, "dflash_lk_loss_type": "lambda"},
                {"lambda": 1.0},
            ),
            ("lilicorr", {}, {"kd": 1.0}),
            ("dspark", {"dflash_ce_loss_alpha": 0.3}, {"ce": 0.3, "tvd": 0.9}),
            ("domino", {}, {"ce": 1.0}),
        ],
        ids=[
            "dflash-kd",
            "dflash-ce",
            "dflash2-tv",
            "dflash2-lambda",
            "lilicorr",
            "dspark",
            "domino",
        ],
    )
    def test_unset_terms_keep_each_variants_objective(self, projector_type, legacy, terms):
        """Unset, each variant builds its own objective, which the same terms reproduce."""
        blend = {"dflash_lk_ce_scale": 0.8, "dflash_lk_ce_decay": 2.0}
        model = _variant_model(projector_type, **legacy, **blend)
        assert [(name, weight) for name, weight, _ in model._objective] == list(terms.items())
        explicit = _variant_model(projector_type, dflash_loss_terms=terms, **blend)
        assert _objective_spec(explicit) == _objective_spec(model)

    def test_terms_survive_the_config_round_trips(self):
        """main.py converts from a model_dump() and a restore reads one back: all fields set."""
        config = DFlashConfig(**get_dflash_config(), dflash_loss_terms={"tvd": 0.5, "ce": 0.5})
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", config.model_dump())])
        restored = mto.restore_from_modelopt_state(
            get_tiny_llama(num_hidden_layers=4), mto.modelopt_state(model)
        )
        assert [name for name, _, _ in restored._objective] == ["tvd", "ce"]
        assert _objective_spec(restored) == _objective_spec(model)

    @pytest.mark.parametrize(
        ("projector_type", "terms"),
        [
            ("dflash", {"ce": 0.5, "tvd": 0.5}),
            ("dflash2", {"tv": 0.5, "kd": 0.5}),
            ("lilicorr", {"tvd": 1.0}),
            ("dspark", {"kd": 1.0}),
            ("domino", {"kd": 0.5, "lambda": 0.5, "tvd": 0.0}),
        ],
        ids=["dflash", "dflash2", "lilicorr", "dspark", "domino"],
    )
    def test_a_composed_objective_trains_every_variant(self, projector_type, terms):
        """Online, the teacher every variant now has; every draft parameter gets a gradient.

        Every term is reported, one of weight 0 too.
        """
        model = _variant_model(projector_type, dflash_loss_terms=terms).float()
        model.train()
        out = model(**_dflash_batch(model.config.vocab_size))
        assert torch.isfinite(out.loss)
        assert {f"loss_{name}" for name in terms} <= out.draft_metrics.keys()
        out.loss.backward()
        ungraded = [
            name
            for name, p in model.dflash_module.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        assert not ungraded, ungraded

    @pytest.mark.parametrize(
        ("projector_type", "terms", "confidence", "needs"),
        [
            ("dspark", {"ce": 1.0}, 0.0, False),
            ("dspark", {"ce": 1.0}, 0.5, True),
            ("domino", None, 0.0, False),
            ("domino", {"kd": 1.0}, 0.0, True),
        ],
        ids=["dspark-ce", "dspark-ce-confidence", "domino", "domino-kd"],
    )
    def test_offline_needs_the_teacher_only_for_terms_that_read_it(
        self, projector_type, terms, confidence, needs
    ):
        """Objective terms and variant terms declare it; nothing else does."""
        model = _variant_model(
            projector_type,
            offline=True,
            arch={"use_confidence_head": bool(confidence)},
            dflash_loss_terms=terms,
            dflash_confidence_head_alpha=confidence,
        )
        assert model._needs_base_logits is needs


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


class TestFlexAttentionConflicts:
    """FlexAttention's conflicts are rejected when the model is built, resume included."""

    @staticmethod
    def _config(**overrides):
        config = get_dflash_config()
        config.update(overrides)
        return config

    def test_attention_sink_is_rejected_at_convert(self):
        config = self._config(dflash_use_flex_attention=True, dflash_attention_sink=True)
        with pytest.raises(ValueError, match="dflash_attention_sink"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])

    def test_attention_dropout_is_rejected_at_convert(self):
        config = self._config(dflash_use_flex_attention=True)
        config["dflash_architecture_config"] = {
            **config["dflash_architecture_config"],
            "attention_dropout": 0.1,
        }
        with pytest.raises(ValueError, match="attention_dropout"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])

    def test_flex_and_compile_survive_a_from_pretrained_resume(self, tmp_path):
        """Resume re-runs modify() with the base on meta; the checks and the compile hold."""
        mto.enable_huggingface_checkpointing()
        model = get_tiny_llama(num_hidden_layers=4)
        config = self._config(dflash_use_flex_attention=True, dflash_use_torch_compile=True)
        mtsp.convert(model, [("dflash", config)])
        model.save_pretrained(tmp_path / "ckpt")

        restored = AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt")
        restored.restore_draft_precision()
        assert restored.dflash_use_flex_attention
        restored.train()
        assert restored.dflash_module._body() != restored.dflash_module._forward_body


class TestFlexTiles:
    """One BlockMask serves every layer, whatever each layer's head_dim."""

    @staticmethod
    def _block_size(monkeypatch, head_dims):
        seen = {}

        def create_block_mask(*_args, BLOCK_SIZE, **_kwargs):  # noqa: N803
            seen["block"] = BLOCK_SIZE

        monkeypatch.setattr(dflash_flex_attention, "_flex_ops", lambda: (None, create_block_mask))
        dflash_flex_attention.build_draft_block_mask(None, 1, 64, 128, "cpu", head_dims)
        return seen["block"]

    @pytest.mark.parametrize("head_dims", [{256}, {512}, {256, 512}])
    def test_every_layer_tile_divides_the_mask_block(self, monkeypatch, head_dims):
        block = self._block_size(monkeypatch, head_dims)
        for head_dim in head_dims:
            _, options = dflash_flex_attention._tiles(head_dim)
            largest_tile = 128 if options is None else max(options["BLOCK_M"], options["BLOCK_N"])
            assert block % largest_tile == 0, (head_dims, block)

    def test_large_head_dims_alone_keep_the_finer_block(self, monkeypatch):
        assert self._block_size(monkeypatch, {512}) == 64


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


class TestDraftMaskRule:
    """The dense mask and the FlexAttention BlockMask evaluate one rule, ``_draft_mask_mod``."""

    SEQ_LEN, BLOCK_SIZE = 16, 4

    def _model(self, attention="bidirectional"):
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config(block_size=self.BLOCK_SIZE)
        config["dflash_draft_attention"] = attention
        mtsp.convert(model, [("dflash", config)])
        return model

    def _visible(self, model, anchors, keep, window=None):
        mask = model._build_draft_attention_mask(
            self.SEQ_LEN, anchors, keep, anchors.shape[1], torch.float32, "cpu", window=window
        )
        return mask > torch.finfo(torch.float32).min / 2

    @pytest.mark.parametrize("attention", ["bidirectional", "causal"])
    def test_matches_a_hand_written_mask(self, attention):
        """Context strictly before each anchor, then the query's own block, nothing else."""
        anchors = (6, 9)
        visible = self._visible(
            self._model(attention), torch.tensor([anchors]), torch.tensor([[True, True]])
        )
        n_q = len(anchors) * self.BLOCK_SIZE
        expected = torch.zeros(n_q, self.SEQ_LEN + n_q, dtype=torch.bool)
        for block, anchor in enumerate(anchors):
            for i in range(self.BLOCK_SIZE):
                row = block * self.BLOCK_SIZE + i
                expected[row, :anchor] = True
                start = self.SEQ_LEN + block * self.BLOCK_SIZE
                expected[
                    row, start : start + (i + 1 if attention == "causal" else self.BLOCK_SIZE)
                ] = True
        assert torch.equal(visible[0, 0], expected)

    def test_dropped_block_sees_nothing(self):
        visible = self._visible(
            self._model(), torch.tensor([[6, 9]]), torch.tensor([[True, False]])
        )
        assert visible[0, 0, : self.BLOCK_SIZE].any(dim=-1).all()
        assert not visible[0, 0, self.BLOCK_SIZE :].any()

    @pytest.mark.parametrize("attention", ["bidirectional", "causal"])
    @pytest.mark.parametrize("window", [None, 6])
    def test_dense_mask_matches_the_vmapped_rule(self, attention, window):
        """Broadcast evaluation (dense) and vmap evaluation (flex) of the rule agree."""
        anchors = torch.tensor([[2, 7, 11], [5, 0, 9]])
        keep = torch.tensor([[True, True, False], [True, False, True]])
        mask_mod = hf_dflash._draft_mask_mod(
            self.SEQ_LEN, anchors, keep, self.BLOCK_SIZE, window, causal=attention == "causal"
        )
        q_len = anchors.shape[1] * self.BLOCK_SIZE
        vmapped = create_mask(mask_mod, 2, 1, q_len, self.SEQ_LEN + q_len, device="cpu")
        dense = self._visible(self._model(attention), anchors, keep, window)
        assert torch.equal(dense, vmapped)

    def test_padded_query_rows_index_in_bounds(self):
        """torch 2.8's flex backward also evaluates the rule on padded rows, q_idx >= Q_LEN."""
        anchors = torch.tensor([[2, 7, 11]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        mask_mod = hf_dflash._draft_mask_mod(
            self.SEQ_LEN, anchors, keep, self.BLOCK_SIZE, None, causal=False
        )
        q_len = anchors.shape[1] * self.BLOCK_SIZE
        padded_q = torch.arange(q_len, q_len + 2 * self.BLOCK_SIZE).view(-1, 1)
        kv = torch.arange(self.SEQ_LEN + q_len).view(1, -1)
        visible = mask_mod(torch.tensor(0), None, padded_q, kv)
        assert visible.shape == (padded_q.numel(), kv.numel())


def _legacy_generate_swa_mask(model, ctx_len, bsz, dtype, device):
    """Verbatim copy of the hand-written generation mask, kept as the semantic reference."""
    window = model.dflash_swa_window_size
    causal = model.dflash_draft_attention == "causal"
    if window is None and not causal:
        return None
    block_size = model.dflash_block_size
    kv_len = ctx_len + block_size
    kv_idx = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
    q_pos_in_block = torch.arange(block_size, device=device).view(1, 1, -1, 1)
    q_real_pos = ctx_len + q_pos_in_block
    is_ctx = kv_idx < ctx_len
    keep_ctx = is_ctx if window is None else (is_ctx & (kv_idx > q_real_pos - window))
    keep_block = ~is_ctx
    if causal:
        keep_block = keep_block & ((kv_idx - ctx_len) <= q_pos_in_block)
    keep = keep_ctx | keep_block
    attn_mask = torch.zeros(bsz, 1, block_size, kv_len, device=device, dtype=dtype)
    attn_mask.masked_fill_(~keep, torch.finfo(dtype).min)
    return attn_mask


class TestGenerateMaskRule:
    """Generation evaluates the training rule for its one block, anchored at ``ctx_len``."""

    @pytest.mark.parametrize("ctx_len", [1, 5, 16])
    @pytest.mark.parametrize("window", [None, BLOCK_SIZE, 2 * BLOCK_SIZE])
    @pytest.mark.parametrize("attention", ["bidirectional", "causal"])
    def test_matches_the_hand_written_mask(self, ctx_len, window, attention):
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config(block_size=BLOCK_SIZE)
        config["dflash_swa_window_size"] = window
        config["dflash_draft_attention"] = attention
        mtsp.convert(model, [("dflash", config)])
        args = (ctx_len, 2, torch.float32, torch.device("cpu"))
        got = model._build_generate_swa_mask(*args)
        want = _legacy_generate_swa_mask(model, *args)
        if want is None:
            assert got is None
        else:
            assert torch.equal(got, want)


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
    """``dflash_fp32_master_weights``: the extra precision lives in the OPTIMIZER.

    The model is untouched -- draft and frozen base share a dtype either way -- so what has
    to be pinned is that the optimizer holds an fp32 master copy and fp32 moments, and that
    a resume does not quietly round them back down.
    """

    def test_the_draft_matches_the_base_dtype_either_way(self):
        """The flag must not move the model, or something has to reconcile dtypes again."""
        for flag in (False, True):
            model = _converted(fp32_master_weights=flag)
            assert model._base_model.dtype == torch.bfloat16
            assert {p.dtype for p in model.dflash_module.parameters()} == {torch.bfloat16}

    def test_forward_and_generate_need_no_autocast(self):
        """Nothing in the model holds a dtype the rest of it does not, so nothing wraps."""
        model = _converted(fp32_master_weights=True)
        model.train()
        assert torch.isfinite(model(**_dflash_batch(model.dflash_config.vocab_size)).loss)
        model.eval()
        input_ids = _dflash_batch(model.dflash_config.vocab_size, bsz=1)["input_ids"]
        _, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=2)
        assert draft_tokens.shape[0] == 1

    def test_master_and_moments_are_fp32_for_a_bf16_draft(self):
        """The half that matters: AdamW's moments follow the master, not the parameter."""
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        optimizer = MasterWeightAdamW(trainable, lr=1e-4)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        optimizer.step()

        assert {p.dtype for p in trainable} == {torch.bfloat16}
        for key in ("master", "exp_avg", "exp_avg_sq"):
            seen = {state[key].dtype for state in optimizer.state.values()}
            assert seen == {torch.float32}, f"{key}: {seen}"

    def test_plain_adamw_leaves_the_moments_in_bf16(self):
        """The control, and the reason the flag exists at all."""
        model = _converted(fp32_master_weights=False)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        optimizer.step()
        assert {
            state[key].dtype
            for state in optimizer.state.values()
            for key in ("exp_avg", "exp_avg_sq")
        } == {torch.bfloat16}

    def test_resume_does_not_round_the_master_back_down(self):
        """``Optimizer.load_state_dict`` casts float state to the parameter's dtype.

        For a bf16 model that silently rounds the master copy and both moments on every
        resume, which is precisely the loss this optimizer exists to avoid and produces no
        error at all. ``MasterWeightAdamW`` puts them back from the incoming state dict.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        optimizer = MasterWeightAdamW(trainable, lr=1e-2)
        for _ in range(2):
            model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
        saved = optimizer.state_dict()
        reference = {
            i: {k: v.clone() for k, v in state.items() if isinstance(v, torch.Tensor)}
            for i, state in saved["state"].items()
        }

        restored = MasterWeightAdamW(trainable, lr=1e-2)
        restored.load_state_dict(saved)

        params = [p for group in restored.param_groups for p in group["params"]]
        for i, state in reference.items():
            got = restored.state[params[i]]
            for key in ("master", "exp_avg", "exp_avg_sq"):
                assert got[key].dtype == torch.float32, f"{key} came back as {got[key].dtype}"
                assert torch.equal(got[key], state[key]), key

    def test_resume_from_plain_adamw_upcasts_instead_of_crashing(self):
        """The upgrade path every in-flight job takes, and the one that used to crash.

        A checkpoint written before this flag defaulted to ``True`` holds moments but no
        master, so an all-or-nothing state guard skipped the master and ``step()`` died on a
        bare ``KeyError``. Those moments are also bf16, so restoring them verbatim would trade
        the crash for a run where the feature is silently absent.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]

        plain = torch.optim.AdamW(trainable, lr=1e-2)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        plain.step()
        model.zero_grad(set_to_none=True)
        saved = plain.state_dict()
        assert {
            state[key].dtype for state in plain.state.values() for key in ("exp_avg", "exp_avg_sq")
        } == {torch.bfloat16}

        resumed = MasterWeightAdamW(trainable, lr=1e-2)
        resumed.load_state_dict(saved)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        resumed.step()

        for key in ("master", "exp_avg", "exp_avg_sq"):
            seen = {state[key].dtype for state in resumed.state.values()}
            assert seen == {torch.float32}, f"{key}: {seen}"
        assert {p.dtype for p in trainable} == {torch.bfloat16}

    def test_resume_keyed_by_name_still_restores_fp32(self):
        """FSDP2 restores through torch's distributed checkpoint, which keys state by FQN.

        It hands ``load_state_dict`` a dict whose ids are strings rather than the integers a
        plain ``torch.save`` round trip produces, and does not convert them back. Anything
        that only recognises integer ids silently restores nothing and leaves the base class's
        downcast to the parameter dtype standing.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        optimizer = MasterWeightAdamW(trainable, lr=1e-2)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        optimizer.step()

        saved = optimizer.state_dict()
        names = {i: f"dflash_module.p{i}" for i in range(len(trainable))}
        saved["state"] = {names[i]: state for i, state in saved["state"].items()}
        for group in saved["param_groups"]:
            group["params"] = [names[i] for i in group["params"]]

        restored = MasterWeightAdamW(trainable, lr=1e-2)
        restored.load_state_dict(saved)
        for key in ("master", "exp_avg", "exp_avg_sq"):
            seen = {state[key].dtype for state in restored.state.values()}
            assert seen == {torch.float32}, f"{key}: {seen}"

    def test_resume_does_not_restore_the_fused_kernel(self):
        """``load_state_dict`` takes every hyperparameter from the checkpoint, not this run.

        ``TrainingArguments.optim`` defaults to ``adamw_torch_fused``, so a checkpoint written
        before the flag was on carries ``fused``; restoring it undoes the switch to foreach
        that building this optimizer performs, and ``step()`` refuses fused before it reaches
        the master. The resume would fail on the most ordinary checkpoint there is.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        plain = torch.optim.AdamW(trainable, lr=1e-2)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        plain.step()
        model.zero_grad(set_to_none=True)
        saved = plain.state_dict()
        for group in saved["param_groups"]:
            group["fused"] = True

        resumed = MasterWeightAdamW(trainable, lr=1e-2, foreach=True)
        resumed.load_state_dict(saved)
        assert not any(group["fused"] for group in resumed.param_groups)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        resumed.step()

    def test_resume_drops_a_master_the_parameter_no_longer_needs(self):
        """A master left on an fp32 parameter goes stale and later rolls the weights back.

        ``step()`` updates an fp32 parameter directly, so a master restored onto one never
        advances again -- but it is still written to the next checkpoint, and the resume after
        that copies it over a parameter thousands of steps newer.
        """
        bf16 = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
        source = MasterWeightAdamW(bf16.parameters(), lr=1e-2)
        bf16(torch.randn(4, 16, dtype=torch.bfloat16)).pow(2).mean().backward()
        source.step()
        assert all("master" in state for state in source.state.values())

        fp32 = torch.nn.Linear(16, 16, bias=False)
        resumed = MasterWeightAdamW(fp32.parameters(), lr=1e-2)
        resumed.load_state_dict(source.state_dict())
        assert all("master" not in state for state in resumed.state.values())

    def test_a_restore_that_skipped_load_state_dict_heals_the_master_too(self):
        """Not every restore reaches ``load_state_dict`` -- DeepSpeed never calls it.

        On those paths the base class's cast to the parameter's dtype stands, and a bf16
        master reads as present, so it survives the per-key creation and is then updated in
        bf16. That is the loss this optimizer exists to prevent, arrived at from the other
        side, and the functional update promotes rather than raising, so nothing says so.
        """
        bf16 = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
        optimizer = MasterWeightAdamW(bf16.parameters(), lr=1e-2)
        bf16(torch.randn(4, 16, dtype=torch.bfloat16)).pow(2).mean().backward()
        optimizer.step()
        for state in optimizer.state.values():
            for key in ("master", "exp_avg", "exp_avg_sq"):
                state[key] = state[key].to(torch.bfloat16)

        bf16.zero_grad(set_to_none=True)
        bf16(torch.randn(4, 16, dtype=torch.bfloat16)).pow(2).mean().backward()
        optimizer.step()
        for key in ("master", "exp_avg", "exp_avg_sq"):
            seen = {state[key].dtype for state in optimizer.state.values()}
            assert seen == {torch.float32}, f"{key}: {seen}"

    def test_a_closure_may_call_backward(self):
        """``step(closure)`` is documented to re-evaluate the loss, which means ``backward()``."""
        model = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
        optimizer = MasterWeightAdamW(model.parameters(), lr=1e-2)

        def closure():
            optimizer.zero_grad()
            loss = model(torch.randn(4, 16, dtype=torch.bfloat16)).pow(2).mean()
            loss.backward()
            return loss

        assert optimizer.step(closure) is not None

    def test_differentiable_is_refused_rather_than_ignored(self):
        """The update runs on a master and is copied back, so a graph would stop at the copy."""
        model = torch.nn.Linear(16, 16, bias=False)
        optimizer = MasterWeightAdamW(model.parameters(), lr=1e-2, differentiable=True)
        model(torch.randn(4, 16)).pow(2).mean().backward()
        with pytest.raises(ValueError, match="differentiable"):
            optimizer.step()

    def test_the_callback_stays_armed_when_no_step_landed(self):
        """An empty optimizer state means nothing happened, not that everything is fine.

        A ``GradScaler`` can skip the first step on a non-finite gradient; consuming the
        tripwire there would pass the whole run on a check that examined nothing.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        plain = torch.optim.AdamW(trainable, lr=1e-4)

        callback = VerifyMasterWeightsCallback()
        callback.on_train_begin(None, SimpleNamespace(global_step=0), None)
        callback.on_step_end(None, SimpleNamespace(global_step=1), None, optimizer=plain)
        assert not callback._checked

        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        plain.step()
        with pytest.raises(RuntimeError, match="dflash_fp32_master_weights"):
            callback.on_step_end(None, SimpleNamespace(global_step=2), None, optimizer=plain)

    def test_an_fp32_model_is_plain_adamw(self):
        """No master copy where there is no precision to recover, and identical arithmetic."""
        torch.manual_seed(3)
        a = torch.nn.Linear(16, 16, bias=False)
        torch.manual_seed(3)
        b = torch.nn.Linear(16, 16, bias=False)
        master_opt = MasterWeightAdamW(a.parameters(), lr=1e-2)
        plain_opt = torch.optim.AdamW(b.parameters(), lr=1e-2)
        for step in range(4):
            x = torch.randn(4, 16, generator=torch.Generator().manual_seed(step))
            a(x).pow(2).mean().backward()
            b(x).pow(2).mean().backward()
            master_opt.step()
            plain_opt.step()
            a.weight.grad = b.weight.grad = None
        assert torch.equal(a.weight, b.weight)
        assert "master" not in next(iter(master_opt.state.values()))

    def test_the_callback_refuses_a_loop_that_forgot_the_optimizer(self):
        """Wiring the optimizer is the caller's job, so a missed wiring has to be loud."""
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        state = SimpleNamespace(global_step=1)

        plain = torch.optim.AdamW(trainable, lr=1e-4)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        plain.step()
        with pytest.raises(RuntimeError, match="dflash_fp32_master_weights"):
            VerifyMasterWeightsCallback().on_step_end(None, state, None, optimizer=plain)

        model.zero_grad(set_to_none=True)
        correct = MasterWeightAdamW(trainable, lr=1e-4)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        correct.step()
        VerifyMasterWeightsCallback().on_step_end(None, state, None, optimizer=correct)

    def test_the_callback_still_checks_after_a_resume(self):
        """A resume restores ``global_step``, so an exact step number never matches again.

        That would leave the check dead on the one path where the fp32 state can be lost
        without anything else noticing -- a restore that does not go through
        ``MasterWeightAdamW.load_state_dict`` comes back at the parameter's dtype.
        """
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        plain = torch.optim.AdamW(trainable, lr=1e-4)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        plain.step()

        callback = VerifyMasterWeightsCallback()
        callback.on_train_begin(None, SimpleNamespace(global_step=4210), None)
        with pytest.raises(RuntimeError, match="dflash_fp32_master_weights"):
            callback.on_step_end(None, SimpleNamespace(global_step=4211), None, optimizer=plain)

    def test_the_callback_checks_once_and_then_stays_out_of_the_way(self):
        """It runs per ``train()`` call, not per step -- the state cannot change mid-run."""
        model = _converted(fp32_master_weights=True)
        model.train()
        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        correct = MasterWeightAdamW(trainable, lr=1e-4)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        correct.step()

        callback = VerifyMasterWeightsCallback()
        callback.on_train_begin(None, SimpleNamespace(global_step=0), None)
        callback.on_step_end(None, SimpleNamespace(global_step=1), None, optimizer=correct)
        assert callback._checked
        # a second step must not re-check, so a bf16 optimizer handed over later is ignored
        plain = torch.optim.AdamW(trainable, lr=1e-4)
        model.zero_grad(set_to_none=True)
        model(**_dflash_batch(model.dflash_config.vocab_size)).loss.backward()
        plain.step()
        callback.on_step_end(None, SimpleNamespace(global_step=2), None, optimizer=plain)


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


def _legacy_sample_anchor_positions(model, seq_len, loss_mask, device):
    """Verbatim copy of the pre-static implementation, kept as the semantic reference."""
    bs = model.dflash_block_size
    bsz = loss_mask.shape[0]
    max_anchor = max(seq_len - bs, 0)
    num_anchors = getattr(model, "_num_anchors", 512)

    valid = loss_mask[:, : max_anchor + 1] > 0.5
    valid_counts = valid.sum(dim=1)
    max_n = min(num_anchors, int(valid_counts.max().item()) - 1)

    if max_n <= 0:
        return (
            torch.zeros(bsz, 1, dtype=torch.long, device=device),
            torch.zeros(bsz, 1, dtype=torch.bool, device=device),
        )

    indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
    masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))
    random_vals = torch.rand(bsz, max_anchor + 1, device=device)
    random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    anchors = gathered[:, :max_n].sort(dim=1).values
    keep = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(
        max=max_n
    )
    anchors = torch.where(keep, anchors, torch.tensor(0, dtype=torch.long, device=device))
    return anchors, keep


class TestAnchorSamplingStaticShape:
    """With static shapes, n_blocks is fixed by the config; the anchors are the legacy ones."""

    DEVICE = torch.device("cpu")

    @staticmethod
    def _model(num_anchors, static=False):
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config(block_size=BLOCK_SIZE)
        config["dflash_num_anchors"] = num_anchors
        config["dflash_self_logit_distillation"] = False  # the padding test calls the CE loss
        mtsp.convert(model, [("dflash", config)])
        # After conversion only the sampler reads it, so nothing gets compiled.
        model.dflash_use_torch_compile = static
        return model

    @staticmethod
    def _loss_mask(lengths, seq_len=SEQ_LEN):
        mask = torch.zeros(len(lengths), seq_len)
        for row, n in enumerate(lengths):
            mask[row, :n] = 1.0
        return mask

    # Each case puts the legacy bound somewhere different: under the cap, ragged across
    # rows, at 1, and in the two degenerate cases the old code special-cased.
    @pytest.mark.parametrize("static", [False, True], ids=["eager", "static"])
    @pytest.mark.parametrize(
        "lengths",
        [(13, 13), (13, 5), (9, 9), (5, 3), (2, 1), (1, 1), (0, 0), (13, 0)],
    )
    def test_matches_legacy_sampling(self, lengths, static):
        """Same seed in, same anchors and same keep mask out -- bitwise."""
        model = self._model(num_anchors=8, static=static)
        loss_mask = self._loss_mask(lengths)

        torch.manual_seed(1234)
        legacy_anchors, legacy_keep = _legacy_sample_anchor_positions(
            model, SEQ_LEN, loss_mask, self.DEVICE
        )
        torch.manual_seed(1234)
        anchors, keep = model._sample_anchor_positions(SEQ_LEN, loss_mask, self.DEVICE)

        n_old = legacy_keep.shape[1]
        assert keep.shape[1] >= n_old if static else keep.shape[1] == n_old
        assert torch.equal(keep[:, :n_old], legacy_keep)
        assert torch.equal(anchors[:, :n_old], legacy_anchors)
        # Everything past the legacy bound is inert padding.
        assert not keep[:, n_old:].any()
        assert not anchors[:, n_old:].any()

    def test_sort_is_truncated_before_it_widens(self):
        """The surplus columns must not pull unsampled anchors in among the kept ones."""
        model = self._model(num_anchors=8, static=True)
        # One long row, so the legacy bound (valid_counts.max() - 1) sits below the static
        # width and there really are surplus columns to get wrong.
        loss_mask = self._loss_mask((6, 6))
        torch.manual_seed(7)
        legacy_anchors, legacy_keep = _legacy_sample_anchor_positions(
            model, SEQ_LEN, loss_mask, self.DEVICE
        )
        torch.manual_seed(7)
        anchors, keep = model._sample_anchor_positions(SEQ_LEN, loss_mask, self.DEVICE)
        assert legacy_keep.shape[1] < keep.shape[1], "fixture no longer exercises truncation"
        kept = anchors[keep]
        assert torch.equal(kept, legacy_anchors[legacy_keep])

    def test_shape_is_independent_of_batch_contents(self):
        """The whole point: one shape, therefore one compile."""
        model = self._model(num_anchors=8, static=True)
        shapes = {
            model._sample_anchor_positions(SEQ_LEN, self._loss_mask(lengths), self.DEVICE)[0].shape
            for lengths in [(13, 13), (13, 5), (9, 9), (5, 3), (2, 1), (1, 1), (0, 0)]
        }
        assert len(shapes) == 1, f"n_blocks still varies with the batch: {shapes}"

    def test_shape_is_min_of_num_anchors_and_sequence(self):
        anchors, _ = self._model(num_anchors=4, static=True)._sample_anchor_positions(
            SEQ_LEN, self._loss_mask((13, 13)), self.DEVICE
        )
        assert anchors.shape[1] == 4, "num_anchors should bind here"
        anchors, _ = self._model(num_anchors=512, static=True)._sample_anchor_positions(
            SEQ_LEN, self._loss_mask((13, 13)), self.DEVICE
        )
        assert anchors.shape[1] == SEQ_LEN - BLOCK_SIZE + 1, "the sequence should bind here"

    def test_static_sampling_does_not_sync_on_the_batch(self, monkeypatch):
        """With static shapes, anchor sampling must not read the batch back to the host."""
        model = self._model(num_anchors=8, static=True)

        def host_read(*_args, **_kwargs):
            raise AssertionError("anchor sampling read a tensor back to the host")

        # Implicit reads too: int(t) and `if t:` sync as surely as t.item() does.
        for name in ("item", "tolist", "__int__", "__float__", "__bool__", "__index__"):
            monkeypatch.setattr(torch.Tensor, name, host_read)
        model._sample_anchor_positions(SEQ_LEN, self._loss_mask((13, 5)), self.DEVICE)

    def test_trailing_padding_blocks_do_not_change_the_loss(self):
        """The padding the static shape introduces is weightless in the loss and accuracy.

        Exact in math, but the reduction runs over more elements, so compare to tolerance.
        """
        model = self._model(num_anchors=8)
        vocab, bsz, n_blocks, pad = 32, 1, 2, 3
        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab, (bsz, SEQ_LEN))
        loss_mask = torch.ones(bsz, SEQ_LEN)
        logits = torch.randn(bsz, n_blocks * BLOCK_SIZE, vocab)
        anchors = torch.tensor([[0, BLOCK_SIZE]])[:, :n_blocks]
        keep = torch.ones(bsz, n_blocks)

        base_loss, base_acc = _compute_loss(model, logits, input_ids, anchors, keep, loss_mask)
        padded_loss, padded_acc = _compute_loss(
            model,
            torch.cat([logits, torch.randn(bsz, pad * BLOCK_SIZE, vocab)], dim=1),
            input_ids,
            torch.cat([anchors, torch.zeros(bsz, pad, dtype=anchors.dtype)], dim=1),
            torch.cat([keep, torch.zeros(bsz, pad)], dim=1),
            loss_mask,
        )
        torch.testing.assert_close(padded_loss, base_loss)
        assert padded_acc == pytest.approx(base_acc)


# Anchors 1 and 4 of an 8-token sequence whose loss_mask is 0 only at position 3:
# (label positions, supervision mask) per block position.
_LABELS_AT_ANCHOR = ([[1, 2, 3, 4], [4, 5, 6, 7]], [[0, 1, 0, 1], [0, 1, 1, 1]])
_LABELS_AFTER_ANCHOR = ([[2, 3, 4, 5], [5, 6, 7, 7]], [[1, 0, 1, 1], [1, 1, 1, 0]])


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (HFDFlashModel, _LABELS_AT_ANCHOR),
        (HFDFlash2Model, _LABELS_AT_ANCHOR),
        (HFLiLiCorrModel, _LABELS_AT_ANCHOR),
        (HFDSparkModel, _LABELS_AFTER_ANCHOR),
        (HFDominoModel, _LABELS_AFTER_ANCHOR),
    ],
    ids=["dflash", "dflash2", "lilicorr", "dspark", "domino"],
)
def test_block_targets_follow_the_label_shift(model_cls, expected):
    """Each block position's label and supervision; a wrong shift trains silently worse."""
    positions, mask = expected
    input_ids = 100 + torch.arange(8).unsqueeze(0)  # token id = 100 + its position
    loss_mask = torch.ones(1, 8)
    loss_mask[0, 3] = 0.0
    targets = _block_targets(
        input_ids,
        torch.tensor([[1, 4]]),
        torch.tensor([[True, True]]),
        loss_mask,
        4,
        model_cls._label_shift,
    )
    assert torch.equal(targets.positions, torch.tensor([positions]))
    assert torch.equal(targets.token_ids, 100 + torch.tensor([positions]))
    assert torch.equal(targets.mask, torch.tensor([mask], dtype=torch.float32))


@pytest.mark.parametrize(
    "head_fields",
    [
        {"projector_type": "dspark", "markov_rank": 4},
        {"projector_type": "domino", "emb_dim": 8, "gru_hidden_dim": 8},
    ],
    ids=["dspark", "domino"],
)
def test_next_token_variants_refuse_shift_label_false_at_convert(head_fields):
    """Left unset, as the recipes do, shift_label converts; set to false, it is refused.

    Restore does not refuse it: DSpark used to accept the field, ignore it in training and
    save it into the checkpoint's modelopt_state.
    """
    config = get_dflash_config()
    config["dflash_architecture_config"].update(head_fields)
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, [("dflash", config)])

    state = mto.modelopt_state(model)
    saved_config = state["modelopt_state_dict"][0][1]["config"]
    saved_config["dflash_architecture_config"]["shift_label"] = False
    restored = mto.restore_from_modelopt_state(get_tiny_llama(num_hidden_layers=4), state)
    assert restored._label_shift == 1

    config["dflash_architecture_config"]["shift_label"] = False
    with pytest.raises(ValueError, match="only next-token alignment"):
        mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])


class TestTeacherLogits:
    """The base distribution is projected only at the rows, and vocab entries, asked for.

    It is the next-token distribution, so label ``i`` reads row ``i - 1``.
    """

    @staticmethod
    def _model(softcap=None):
        """A converted tiny model; ``softcap`` sets the base's cap and turns the switch on."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        if softcap is not None:
            model.config.final_logit_softcapping = softcap
            config["dflash_teacher_softcap"] = True
        mtsp.convert(model, [("dflash", config)])
        return model

    @staticmethod
    def _labels(bsz=2):
        torch.manual_seed(0)
        return torch.randint(1, SEQ_LEN, (bsz, 3, BLOCK_SIZE))

    @staticmethod
    def _rows(full, positions):
        return full[torch.arange(positions.shape[0]).view(-1, 1, 1), positions]

    @staticmethod
    def _hidden(model, bsz=2):
        dtype = model._base_model_lm_head.weight.dtype
        return torch.randn(bsz, SEQ_LEN, model.config.hidden_size, dtype=dtype)

    def test_matches_the_full_sequence_projection(self):
        model = self._model()
        hidden = self._hidden(model)
        labels = self._labels()
        got = model._teacher_logits(DFlashBaseModelOutput(None, base_hidden=hidden), labels)
        want = self._rows(model._base_model_lm_head(hidden), labels - 1)
        torch.testing.assert_close(got, want)

    def test_prenorm_hidden_gets_the_base_final_norm(self):
        model = self._model()
        hidden = self._hidden(model)
        labels = self._labels()
        outputs = DFlashBaseModelOutput(None, base_hidden=hidden, base_hidden_prenorm=True)
        want = self._rows(model._base_model_lm_head(model._base_model_norm(hidden)), labels - 1)
        torch.testing.assert_close(model._teacher_logits(outputs, labels), want)

    def test_token_ids_pick_entries_of_the_full_rows(self):
        model = self._model()
        outputs = DFlashBaseModelOutput(None, base_hidden=self._hidden(model))
        labels = self._labels()
        token_ids = torch.randint(0, model.config.vocab_size, (*labels.shape, 5))
        rows = model._teacher_logits(outputs, labels)
        torch.testing.assert_close(
            model._teacher_logits(outputs, labels, token_ids), rows.gather(-1, token_ids)
        )

    @pytest.mark.parametrize("narrow", [False, True], ids=["full", "token_ids"])
    def test_softcap_caps_the_projected_teacher(self, narrow):
        cap = 0.5
        model = self._model(softcap=cap).float()
        hidden = 10 * self._hidden(model)
        labels = self._labels()
        rows = self._rows(model._base_model_lm_head(hidden), labels - 1)
        assert rows.abs().max() > cap, "fixture does not reach the cap"
        want = cap * torch.tanh(rows / cap)
        token_ids = None
        if narrow:
            token_ids = torch.randint(0, model.config.vocab_size, (*labels.shape, 5))
            want = want.gather(-1, token_ids)
        got = model._teacher_logits(
            DFlashBaseModelOutput(None, base_hidden=hidden), labels, token_ids
        )
        torch.testing.assert_close(got, want)

    @pytest.mark.parametrize("softcap", [None, 0.5], ids=["uncapped", "capped"])
    def test_handed_over_logits_are_gathered_not_recomputed(self, softcap):
        """Handed-over logits are the model's own: gathered as they are, never capped again."""
        model = self._model(softcap)
        logits = torch.randn(2, SEQ_LEN, model.config.vocab_size)
        outputs = DFlashBaseModelOutput(None, logits=logits)
        labels = self._labels()
        token_ids = torch.randint(0, model.config.vocab_size, (*labels.shape, 5))
        want = self._rows(logits, labels - 1)
        assert torch.equal(model._teacher_logits(outputs, labels), want)
        assert torch.equal(
            model._teacher_logits(outputs, labels, token_ids), want.gather(-1, token_ids)
        )

    def test_label_zero_does_not_wrap_to_the_last_row(self):
        model = self._model()
        logits = torch.randn(1, SEQ_LEN, model.config.vocab_size)
        outputs = DFlashBaseModelOutput(None, logits=logits)
        got = model._teacher_logits(outputs, torch.tensor([[0]]))
        assert torch.equal(got[0, 0], logits[0, 0])

    def test_missing_base_distribution_raises(self):
        model = self._model()
        with pytest.raises(ValueError, match="base_model_hidden_states"):
            model._teacher_logits(DFlashBaseModelOutput(None), self._labels())

    def test_offline_dict_keeps_the_hidden_unprojected(self):
        d = {
            "aux_hidden_states": torch.randn(1, SEQ_LEN, 8),
            "base_model_hidden_states": torch.randn(1, SEQ_LEN, 4),
            "base_hidden_prenorm": True,
        }
        outputs = DFlashBaseModelOutput.from_offline_dict(d)
        assert outputs.logits is None
        assert outputs.base_hidden is d["base_model_hidden_states"]
        assert outputs.base_hidden_prenorm

    def test_kd_loss_matches_full_logits(self):
        """KD read from the base hidden equals KD from the same logits built in full."""
        model = self._model()
        hidden = self._hidden(model, bsz=1)
        inputs = TestDPaceLossIntegration._make_inputs(vocab=model.config.vocab_size)
        full = DFlashBaseModelOutput(None, logits=model._base_model_lm_head(hidden))
        loss_full, _ = _compute_loss(model, *inputs, base_outputs=full)
        loss, _ = _compute_loss(
            model, *inputs, base_outputs=DFlashBaseModelOutput(None, base_hidden=hidden)
        )
        torch.testing.assert_close(loss, loss_full)

    @pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
    def test_step_never_projects_the_full_sequence(self, offline):
        """With KD on, lm_head only ever sees draft rows and teacher rows."""
        model = get_tiny_llama(num_hidden_layers=4)
        model.config.num_orig_hidden_layers = 4
        mtsp.convert(model, [("dflash", get_dflash_config(offline=offline))])
        assert model.dflash_self_logit_distillation
        model.train()
        seen = []
        model._base_model_lm_head.register_forward_hook(
            lambda _module, args, _out: seen.append(tuple(args[0].shape))
        )
        bsz, hidden = 2, model.config.hidden_size
        dtype = next(model.dflash_module.parameters()).dtype
        kwargs = {}
        if offline:
            kwargs["base_model_outputs"] = {
                "aux_hidden_states": torch.randn(
                    bsz, SEQ_LEN, len(model.target_layer_ids) * hidden, dtype=dtype
                ),
                "base_model_hidden_states": torch.randn(bsz, SEQ_LEN, hidden, dtype=dtype),
            }
        input_ids = torch.randint(1, model.config.vocab_size, (bsz, SEQ_LEN))
        model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs
        ).loss.backward()
        assert len(seen) == 2, seen  # the draft's logits, then the KD teacher rows
        assert (bsz, SEQ_LEN, hidden) not in seen, seen

    def test_online_teacher_is_the_causal_lm_distribution(self):
        """The rows projected from the inner model's final hidden are the CausalLM's logits."""
        model = self._model().float()
        labels = self._labels()
        input_ids = torch.randint(1, model.config.vocab_size, (2, SEQ_LEN))
        attention_mask = torch.ones_like(input_ids)
        outputs = model._online_base_outputs(input_ids, attention_mask)
        model.eval()
        with torch.no_grad():
            full = model(input_ids=input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(
            model._teacher_logits(outputs, labels), self._rows(full, labels - 1)
        )


# Qwen3-VL's vision placeholder ids, moved inside get_tiny_qwen3vl's 32-token vocab.
_VISION_START, _IMAGE = 28, 30


def _tiny_vlm(arch=None, softcap=None):
    """A tiny Qwen3-VL converted online, in fp32; ``arch`` updates the draft's architecture.

    ``softcap`` names a cap in its text config, which Qwen3-VL's own forward never applies, and
    turns ``dflash_teacher_softcap`` on.
    """
    model = get_tiny_qwen3vl(num_hidden_layers=4)
    model.config.vision_start_token_id, model.config.image_token_id = _VISION_START, _IMAGE
    config = get_dflash_config()
    config["dflash_architecture_config"].update(arch or {})
    if softcap is not None:
        model.config.text_config.final_logit_softcapping = softcap
        config["dflash_teacher_softcap"] = True
    mtsp.convert(model, [("dflash", config)])
    return model.float()


def _image_batch():
    """Two sequences, each holding a 2x2-patch image at positions 4-7, and its pixels."""
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, _VISION_START, (2, SEQ_LEN), generator=g)
    input_ids[:, 3] = _VISION_START
    input_ids[:, 4:8] = _IMAGE
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        # One row per 4x4 patch of 3 channels: get_tiny_qwen3vl merges no patches.
        "pixel_values": torch.randn(2 * 4, 3 * 4 * 4, generator=g),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "mm_token_type_ids": (input_ids == _IMAGE).long(),
    }


def _tiny_gemma4_and_batch(modality):
    """A tiny Gemma 4 VLM converted online, in fp32, and a batch with an image or a video at 3-6."""
    torch.manual_seed(0)
    image_token, video_token = 62, 63
    text_config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "vocab_size_per_layer_input": 64,
        "hidden_size_per_layer_input": 8,
    }
    vision_config = {
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "patch_size": 4,
        "position_embedding_size": 64,
    }
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token,
            # The other special tokens, inside the tiny vocabulary too.
            boi_token_id=60,
            eoi_token_id=61,
            video_token_id=video_token,
            audio_token_id=57,
            boa_token_id=59,
            eoa_token_index=58,
        )
    )
    mtsp.convert(model, [("dflash", get_dflash_config())])
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(1, 56, (2, SEQ_LEN), generator=g)
    # 6x6 patches of 4x4 pixels, pooled to 2x2 tokens; a video has one such frame.
    input_ids[:, 3:7] = image_token if modality == "image" else video_token
    ys, xs = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    pixels = torch.randn(2, 6 * 6, 3 * 4 * 4, generator=g)
    positions = torch.stack([xs.flatten(), ys.flatten()], -1).expand(2, -1, -1)
    if modality == "image":
        visual = {"pixel_values": pixels, "image_position_ids": positions}
    else:
        visual = {"pixel_values_videos": pixels[:, None], "video_position_ids": positions[:, None]}
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        **visual,
        # Gemma 4's processor marks image tokens 1 and video tokens 2.
        "mm_token_type_ids": (input_ids == image_token).long() + 2 * (input_ids == video_token),
    }
    return model.float(), batch


class TestMultimodalTeacher:
    """An image batch gets a text batch's teacher: lm_head projecting the inner model's output.

    The top-level forward inserts the image, and a hook reads the inner model's output during it.
    """

    @staticmethod
    def _labels():
        torch.manual_seed(0)
        return torch.randint(1, SEQ_LEN, (2, 3, BLOCK_SIZE))

    @staticmethod
    def _own_rows(model, batch, positions):
        """Rows of the model's own logits, which for Qwen3-VL are lm_head's output as it is."""
        model.eval()
        with torch.no_grad():
            logits = model(**batch).logits
        return logits[torch.arange(positions.shape[0]).view(-1, 1, 1), positions]

    def test_teacher_projects_the_final_hidden_state_over_the_whole_sequence(self):
        model = _tiny_vlm()
        batch = _image_batch()
        lm_head_inputs = []
        model._base_model_lm_head.register_forward_pre_hook(
            lambda _module, args: lm_head_inputs.append(args[0])
        )
        outputs = model._online_base_outputs(**batch)
        assert outputs.logits is None
        # Qwen3-VL feeds the inner model's output to lm_head as it is.
        assert torch.equal(outputs.base_hidden, lm_head_inputs[0])
        labels = self._labels()
        torch.testing.assert_close(
            model._teacher_logits(outputs, labels), self._own_rows(model, batch, labels - 1)
        )

    def test_text_and_image_batches_read_the_same_final_hidden_state(self):
        """Up to the image, also when the CausalLM rescales lm_head's input, as Inkling does."""
        model = _tiny_vlm()
        model._base_model_lm_head.register_forward_pre_hook(lambda _module, args: (args[0] / 24,))
        batch = _image_batch()
        image = model._online_base_outputs(**batch).base_hidden
        text = model._online_base_outputs(batch["input_ids"], batch["attention_mask"]).base_hidden
        torch.testing.assert_close(image[:, :4], text[:, :4])

    def test_softcap_caps_the_image_batch_teacher(self):
        cap = 0.1
        model = _tiny_vlm(softcap=cap)
        batch = _image_batch()
        outputs = model._online_base_outputs(**batch)
        labels = self._labels()
        rows = self._own_rows(model, batch, labels - 1)
        assert rows.abs().max() > cap, "fixture does not reach the cap"
        torch.testing.assert_close(
            model._teacher_logits(outputs, labels), cap * torch.tanh(rows / cap)
        )

    @pytest.mark.parametrize(
        "arch",
        [
            {},
            {"projector_type": "dspark", "markov_rank": 16},
            {"projector_type": "domino", "emb_dim": 16, "gru_hidden_dim": 32},
        ],
        ids=["dflash", "dspark", "domino"],
    )
    def test_every_variant_trains_on_the_image(self, arch):
        """Another image in the same text changes the base's rows from the image on, not before."""
        model = _tiny_vlm(arch)
        batch = _image_batch()
        other = {**batch, "pixel_values": torch.randn_like(batch["pixel_values"])}
        first, second = (model._online_base_outputs(**b) for b in (batch, other))
        for name in ("target_hidden", "base_hidden"):
            a, b = getattr(first, name), getattr(second, name)
            torch.testing.assert_close(a[:, :4], b[:, :4], msg=name)
            assert not torch.allclose(a[:, 4:], b[:, 4:]), name
        model.train()
        out = model(**batch)
        assert torch.isfinite(out.loss)
        out.loss.backward()

    @pytest.mark.parametrize("modality", ["image", "video"])
    def test_a_gemma4_image_or_video_reaches_the_base(self, modality):
        """Gemma 4's vision tower reads the patch positions its processor sends with the pixels."""
        model, batch = _tiny_gemma4_and_batch(modality)
        pixels = "pixel_values" if modality == "image" else "pixel_values_videos"
        other = {**batch, pixels: torch.randn_like(batch[pixels])}
        first, second = (model._online_base_outputs(**b).base_hidden for b in (batch, other))
        torch.testing.assert_close(first[:, :3], second[:, :3])
        assert not torch.allclose(first[:, 3:], second[:, 3:])

    def test_the_inner_model_is_hooked_only_inside_an_image_batch_forward(self):
        """Never for a text batch; an image batch's hook goes even when its forward fails."""
        model = _tiny_vlm()
        base = model._base_model
        hooks_seen = []
        base.register_forward_pre_hook(
            lambda module, _args: hooks_seen.append(len(module._forward_hooks))
        )
        batch = _image_batch()
        model._online_base_outputs(batch["input_ids"], batch["attention_mask"])
        model._online_base_outputs(**batch)
        assert hooks_seen == [0, 1]
        assert not base._forward_hooks
        with pytest.raises(RuntimeError):  # pixels that do not fit the vision tower
            model._online_base_outputs(**{**batch, "pixel_values": torch.randn(3, 5)})
        assert not base._forward_hooks

    def test_logits_to_keep_does_not_narrow_the_teacher(self, monkeypatch):
        """It narrows lm_head's input, not the inner model's output that the teacher reads."""
        model = _tiny_vlm()
        batch = _image_batch()
        want = model._online_base_outputs(**batch).base_hidden
        monkeypatch.setattr(
            hf_dflash,
            "_MULTIMODAL_FORWARD_KWARGS",
            hf_dflash._MULTIMODAL_FORWARD_KWARGS | {"logits_to_keep"},
        )
        assert torch.equal(model._online_base_outputs(**batch, logits_to_keep=1).base_hidden, want)

    @pytest.mark.parametrize("run", ["twice", "short"])
    def test_the_inner_model_must_run_once_over_the_whole_sequence(self, run):
        """Otherwise the forward gives the teacher no one final hidden state for every row."""
        model = _tiny_vlm()
        batch = _image_batch()
        base = model._base_model

        def run_it_again(_module, _args):
            base(input_ids=batch["input_ids"])

        def drop_the_last_row(_module, _args, output):
            output.last_hidden_state = output.last_hidden_state[:, :-1]

        if run == "twice":
            model._base_model_lm_head.register_forward_pre_hook(run_it_again)
        else:
            base.register_forward_hook(drop_the_last_row)
        hooks = dict(base._forward_hooks)
        with pytest.raises(RuntimeError, match="once over the whole sequence"):
            model._online_base_outputs(**batch)
        assert base._forward_hooks == hooks


class TestTeacherSoftcapAtConvert:
    """``dflash_teacher_softcap`` alone decides the cap, checked against the base at build time."""

    @staticmethod
    def _base(multimodal=True, cap=30.0):
        """A tiny base capping its logits at ``cap``; a nested text config marks a multimodal one."""
        model = get_tiny_llama(num_hidden_layers=4)
        model.config.num_orig_hidden_layers = 4
        text_config = model.config
        if multimodal:
            text_config = model.config.text_config = copy.deepcopy(model.config)
        text_config.final_logit_softcapping = cap
        return model

    def test_a_base_without_a_cap_is_rejected(self):
        config = get_dflash_config()
        config["dflash_teacher_softcap"] = True
        with pytest.raises(ValueError, match="final_logit_softcapping"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])

    @pytest.mark.parametrize("switch", [False, True], ids=["switch-off", "switch-on"])
    @pytest.mark.parametrize("multimodal", [False, True], ids=["text-only", "multimodal"])
    def test_a_capping_base_converts_with_the_cap_the_switch_sets(self, multimodal, switch):
        """Online, with KD reading the teacher: a multimodal base gets the same rule."""
        model = self._base(multimodal)
        config = get_dflash_config()
        config["dflash_teacher_softcap"] = switch
        mtsp.convert(model, [("dflash", config)])
        assert model.dflash_self_logit_distillation
        assert model._teacher_softcap == (30.0 if switch else None)

    @pytest.mark.parametrize(
        ("saved", "cap"),
        [(True, 30.0), (False, None), (None, None)],
        ids=["on", "off", "saved-before-the-switch"],
    )
    def test_restore_reads_the_switch_from_the_saved_config(self, saved, cap):
        """A checkpoint saved before the switch existed restores with it off."""
        model = self._base()
        config = get_dflash_config()
        config["dflash_teacher_softcap"] = bool(saved)
        mtsp.convert(model, [("dflash", config)])
        state = mto.modelopt_state(model)
        if saved is None:
            del state["modelopt_state_dict"][0][1]["config"]["dflash_teacher_softcap"]
        restored = mto.restore_from_modelopt_state(self._base(), state)
        assert restored._teacher_softcap == cap


class TestLogitScalingWarning:
    """Conversion names the logit scaling that the teacher, lm_head's raw projection, leaves out.

    Once, on rank 0, and only when a loss term reads the teacher.
    """

    @staticmethod
    def _granite(logits_scaling):
        """A tiny Granite, whose CausalLM divides its logits by ``logits_scaling``."""
        config = GraniteConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            logits_scaling=logits_scaling,
        )
        return GraniteForCausalLM(config)

    @staticmethod
    def _warnings(model, kd=True, offline=None):
        config = get_dflash_config(offline=offline)
        config["dflash_self_logit_distillation"] = kd
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mtsp.convert(model, [("dflash", config)])
        return [str(w.message) for w in caught if "teacher does not apply" in str(w.message)]

    def test_a_scaling_base_warns_once(self):
        (message,) = self._warnings(self._granite(8.0))
        assert "logits_scaling=8.0" in message

    def test_an_offline_fake_base_warns_too(self, tmp_path):
        """Offline VLMs always train on a fake base, which carries the scaling fields over."""
        self._granite(8.0).save_pretrained(tmp_path)
        model = load_vlm_or_llm(str(tmp_path), use_fake_base=True, use_offline_training=True)
        assert isinstance(model, FakeBaseModel)
        (message,) = self._warnings(model, offline=True)
        assert "logits_scaling=8.0" in message

    @pytest.mark.parametrize(
        ("base", "kd"),
        [("llama", True), ("granite-unscaled", True), ("granite", False)],
        ids=["llama", "identity-scaling", "teacher-unread"],
    )
    def test_no_warning_otherwise(self, base, kd):
        model = {
            "llama": lambda: get_tiny_llama(num_hidden_layers=4),
            "granite-unscaled": lambda: self._granite(1.0),
            "granite": lambda: self._granite(8.0),
        }[base]()
        assert not self._warnings(model, kd)

    def test_only_rank_0_warns(self, monkeypatch):
        monkeypatch.setattr("modelopt.torch.utils.distributed.is_master", lambda group=None: False)
        assert not self._warnings(self._granite(8.0))
