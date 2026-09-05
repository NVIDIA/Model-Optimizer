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

"""CPU unit tests for the LiLiCorr speculative decoding plugin.

LiLiCorr reuses the DFlash mode/pipeline and adds a reranker over the candidate
lattice the parallel backbone produces. These tests cover conversion routing, the
required-field validation that keeps a mis-specified head from loading silently,
the three-term objective and its absolute weights, gradient reach into both the head
and the drafter body (the "co-trained" claim), and the export contract the serving
loader rebuilds the head from.
"""

import json
import math
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_dflash import HFDFlashModel
from modelopt.torch.speculative.plugins.hf_lilicorr import HFLiLiCorrModel
from modelopt.torch.speculative.plugins.modeling_dflash import DFlashModule
from modelopt.torch.speculative.plugins.modeling_lilicorr import LiLiCorrModule

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be a multiple of BLOCK_SIZE
CANDIDATE_TOPK = 4
VOCAB_SIZE = 32  # get_tiny_llama's default; a lattice this wide always covers the label
HEAD_HIDDEN = 16
FACTOR_DIM = 16
HEAD_LAYERS = 2
HEAD_ATTENTION_HEADS = 4

# The head's own parameters, i.e. everything the lattice terms alone must reach.
HEAD_PARAM_NAMES = (
    "out_head.weight",
    "in_head.weight",
    "anchor_out_head.weight",
    "context_proj.weight",
    "factor_input_proj.weight",
    "pass_hidden_proj.weight",
    "token_proj.weight",
    "slot_embedding",
    "rank_embedding",
    "relative_slot_bias",
    "same_slot_bias",
)

# The two published compositions. Both hold the head's total weight at 0.50.
VARIANTS = {
    "base": {"w_ce": 0.25, "w_margin": 0.0, "w_pen": 0.25},
    "margin": {"w_ce": 0.125, "w_margin": 0.125, "w_pen": 0.25},
}


def _get_lilicorr_config(
    variant="base", head_hidden=HEAD_HIDDEN, margin=2.0, topk=CANDIDATE_TOPK, **overrides
):
    """Create a LiLiCorr config (dflash mode + projector_type=lilicorr)."""
    weights = VARIANTS[variant]
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = BLOCK_SIZE
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0  # token 0 as mask for the tiny model
    config["dflash_self_logit_distillation"] = False
    config["dflash_loss_objective"] = "decay"
    config["dflash_loss_decay_factor"] = 7.0
    config["dflash_lilicorr_w_ce"] = weights["w_ce"]
    config["dflash_lilicorr_w_margin"] = weights["w_margin"]
    config["dflash_lilicorr_w_pen"] = weights["w_pen"]
    config["dflash_lilicorr_margin"] = margin
    config["dflash_architecture_config"] = {
        "num_hidden_layers": NUM_DRAFT_LAYERS,
        "projector_type": "lilicorr",
        "lilicorr_candidate_topk": topk,
        "lilicorr_hidden_size": head_hidden,
        "lilicorr_factor_dim": FACTOR_DIM,
        "lilicorr_num_layers": HEAD_LAYERS,
        "lilicorr_num_heads": HEAD_ATTENTION_HEADS,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_logit_scale": 8.0,
        "lilicorr_vector_eps": 1.0e-4,
    }
    config.update(overrides)
    return config


def _converted(variant="base", num_hidden_layers=4, **kwargs):
    model = get_tiny_llama(num_hidden_layers=num_hidden_layers)
    mtsp.convert(model, [("dflash", _get_lilicorr_config(variant=variant, **kwargs))])
    return model


def _make_batch(vocab_size, bsz=2):
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (bsz, SEQ_LEN))
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


class TestLiLiCorrConvert:
    """Conversion routing and head construction."""

    def test_convert_creates_lilicorr_model(self):
        """projector_type=lilicorr routes to HFLiLiCorrModel (a HFDFlashModel subclass)."""
        model = _converted()
        assert isinstance(model, HFLiLiCorrModel)
        assert isinstance(model, HFDFlashModel)
        assert isinstance(model.dflash_module, LiLiCorrModule)

    def test_head_geometry_comes_from_the_config(self):
        """Every geometry field lands on the head as configured."""
        head = _converted().dflash_module.lilicorr
        assert head.candidate_topk == CANDIDATE_TOPK
        assert head.hidden_size == HEAD_HIDDEN
        assert head.factor_dim == FACTOR_DIM
        assert head.num_heads == HEAD_ATTENTION_HEADS
        assert head.logit_scale == 8.0
        assert head.vector_eps == 1.0e-4
        assert len(head.layers) == HEAD_LAYERS
        # One scoring slot per predicted block position; position 0 is the anchor.
        assert head.num_candidate_slots == BLOCK_SIZE - 1
        assert head.rank_embedding.shape[-2] == CANDIDATE_TOPK
        assert head.out_head.out_features == FACTOR_DIM

    def test_token_proj_is_identity_at_draft_width(self):
        """A head as wide as the draft skips the redundant projection."""
        wide = _converted(head_hidden=32).dflash_module.lilicorr
        narrow = _converted(head_hidden=HEAD_HIDDEN).dflash_module.lilicorr
        assert isinstance(wide.token_proj, torch.nn.Identity)
        assert isinstance(narrow.token_proj, torch.nn.Linear)

    def test_head_params_trainable(self):
        head_params = [(n, p) for n, p in _converted().named_parameters() if ".lilicorr." in n]
        assert head_params
        assert all(p.requires_grad for _, p in head_params)

    def test_dflash_mode_still_creates_plain_dflash(self):
        """Without projector_type=lilicorr, conversion still yields a plain DFlash model."""
        config = deepcopy(DFLASH_DEFAULT_CFG["config"])
        config["dflash_mask_token_id"] = 0
        config["dflash_architecture_config"] = {"num_hidden_layers": NUM_DRAFT_LAYERS}
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)
        assert not isinstance(model, HFLiLiCorrModel)
        assert type(model.dflash_module) is DFlashModule

    def test_unknown_projector_type_still_rejected(self):
        config = _get_lilicorr_config()
        config["dflash_architecture_config"]["projector_type"] = "not_a_head"
        with pytest.raises(ValueError, match="projector_type"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])


class TestLiLiCorrValidation:
    """A mis-specified LiLiCorr config must fail loudly, not score differently.

    ``logit_scale`` and ``vector_eps`` change the score without changing any tensor
    shape, so a defaulted value would build a head that loads cleanly and computes a
    different function — a small believable acceptance delta rather than an error.
    """

    @pytest.mark.parametrize(
        "field",
        [
            "lilicorr_candidate_topk",
            "lilicorr_factor_dim",
            "lilicorr_num_layers",
            "lilicorr_num_heads",
            "lilicorr_mlp_ratio",
            "lilicorr_logit_scale",
            "lilicorr_vector_eps",
        ],
    )
    def test_missing_geometry_field_raises(self, field):
        config = _get_lilicorr_config()
        del config["dflash_architecture_config"][field]
        with pytest.raises(ValueError, match=field):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])

    def test_partial_objective_weights_raise(self):
        """The three weights are all-or-nothing: no inherited composition."""
        config = _get_lilicorr_config()
        del config["dflash_lilicorr_w_margin"]
        with pytest.raises(ValueError, match="all-or-nothing"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])

    def test_all_zero_weights_raise(self):
        """A head that never receives a gradient would export randomly initialized."""
        config = _get_lilicorr_config()
        config["dflash_lilicorr_w_ce"] = 0.0
        config["dflash_lilicorr_w_margin"] = 0.0
        config["dflash_lilicorr_w_pen"] = 0.0
        with pytest.raises(ValueError, match="never receive a gradient"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])

    def test_hinge_without_margin_raises(self):
        with pytest.raises(ValueError, match="dflash_lilicorr_margin"):
            _converted(variant="margin", margin=0.0)

    def test_offline_rejected(self):
        """The distractor penalty needs a target model, which offline mode does not have."""
        with pytest.raises(ValueError, match="requires online training"):
            _converted(dflash_offline=True)

    def test_topk_above_vocab_rejected(self):
        """The lattice cannot be wider than the vocabulary."""
        model = _converted()
        vocab = model.dflash_config.vocab_size
        model.dflash_module.lilicorr.candidate_topk = vocab + 1
        model.train()
        with pytest.raises(ValueError, match="exceeds the vocabulary size"):
            model(**_make_batch(vocab))

    def test_indivisible_head_width_rejected(self):
        """The lattice attention needs hidden_size divisible by num_heads."""
        config = _get_lilicorr_config()
        config["dflash_architecture_config"]["lilicorr_num_heads"] = 5
        with pytest.raises(ValueError, match="divisible"):
            mtsp.convert(get_tiny_llama(num_hidden_layers=4), [("dflash", config)])


class TestLiLiCorrForward:
    """The training forward, its objective bookkeeping, and where its gradient goes."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_forward_loss_and_metrics(self, variant):
        model = _converted(variant=variant)
        model.train()
        out = model(**_make_batch(model.dflash_config.vocab_size))

        assert out.loss.requires_grad
        assert out.loss.dim() == 0
        assert torch.isfinite(out.loss)

        metrics = out.lilicorr_metrics
        for key in (
            "origin_loss",
            "lilicorr_loss",
            "lilicorr_ce",
            "lilicorr_margin",
            "lilicorr_penalty",
            "lilicorr_selected_prefix",
            "lilicorr_top1_prefix",
            "lilicorr_oracle_prefix",
            "lilicorr_logfactor_spread",
        ):
            assert key in metrics, key
        for name, expected in VARIANTS[variant].items():
            assert metrics[f"lilicorr_{name}"] == expected

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_loss_is_exactly_origin_plus_lilicorr(self, variant):
        """The parity check the training recipe relies on: no outer multiplier."""
        model = _converted(variant=variant)
        model.train()
        out = model(**_make_batch(model.dflash_config.vocab_size))
        metrics = out.lilicorr_metrics
        assert out.loss.item() == pytest.approx(
            metrics["origin_loss"] + metrics["lilicorr_loss"], abs=1e-5
        )

    def test_hinge_only_active_in_the_margin_variant(self):
        """`base` and `margin` are one code path; only the composition differs."""
        base = _converted(variant="base")
        base.train()
        assert (
            base(**_make_batch(base.dflash_config.vocab_size)).lilicorr_metrics["lilicorr_margin"]
            == 0.0
        )

        margin = _converted(variant="margin")
        margin.train()
        metrics = margin(**_make_batch(margin.dflash_config.vocab_size)).lilicorr_metrics
        # The hinge is relu(margin - gap) on an untrained head, so it is active.
        assert metrics["lilicorr_margin"] > 0.0

    def test_penalty_term_is_active(self):
        model = _converted()
        model.train()
        metrics = model(**_make_batch(model.dflash_config.vocab_size)).lilicorr_metrics
        assert metrics["lilicorr_penalty"] > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "degenerate"),
        [
            ({"dflash_block_size": 2}, "one slot, so no transitions to spread over"),
            ({"topk": 1}, "one candidate per slot, so no per-row spread"),
        ],
        ids=["block_size_2", "topk_1"],
    )
    def test_degenerate_lattice_reports_finite_metrics(self, kwargs, degenerate):
        """Every metric stays finite at the edges of the legal geometry.

        Both configurations leave a factor statistic mathematically undefined —
        a mean over an empty tensor, or a std over a single element. NaN in a
        training log reads as a diverging run rather than as a flat lattice, so
        these report 0 instead.
        """
        model = _converted(**kwargs)
        model.train()
        out = model(**_make_batch(model.dflash_config.vocab_size))

        assert torch.isfinite(out.loss), degenerate
        for name, value in out.lilicorr_metrics.items():
            assert math.isfinite(value), f"{name} is {value} ({degenerate})"

    def test_gradient_reaches_head_and_drafter_body(self):
        """Co-training: the lattice terms must reach the backbone, not only the head.

        The candidate ids are the hard top-k, so the only path back into the drafter is
        the soft-value re-attach — recomputing the candidate log-probs from the live
        logits. If that were detached, ``fc`` would still get a gradient from the block
        cross-entropy, so the head-only loss is checked in isolation below.

        Run with a vocabulary-wide lattice. The objective supervises the oracle prefix
        only, and an untrained tiny model puts the ground truth in a 4-wide lattice at
        ~6% of slots, so the transition parameters would correctly see no supervision at
        all — which would make this check about the sampling, not about the wiring.
        """
        model = _converted(topk=VOCAB_SIZE)
        model.train()
        out = model(**_make_batch(model.dflash_config.vocab_size))
        out.loss.backward()

        head = model.dflash_module.lilicorr
        head_params = dict(head.named_parameters())
        for name in HEAD_PARAM_NAMES:
            grad = head_params[name].grad
            assert grad is not None, name
            assert torch.isfinite(grad).all(), name
            assert grad.abs().sum() > 0, name

        body_grad = model.dflash_module.fc.weight.grad
        assert body_grad is not None and torch.isfinite(body_grad).all()
        assert body_grad.abs().sum() > 0

    def test_supervision_follows_lattice_coverage(self):
        """Only the oracle prefix is supervised, so coverage bounds what is trained.

        Past the first slot whose label is missing from the lattice nothing is accepted
        at inference, so those slots carry no acceptance signal and must carry no loss.
        A vocabulary-wide lattice covers every label; a narrow one on an untrained model
        covers almost none, and the metrics have to say so.
        """
        wide = _converted(topk=VOCAB_SIZE)
        wide.train()
        metrics = wide(**_make_batch(wide.dflash_config.vocab_size)).lilicorr_metrics
        assert metrics["lilicorr_slot_gt_in_lattice"] == 1.0
        assert metrics["lilicorr_oracle_prefix"] == float(BLOCK_SIZE - 1)
        assert metrics["lilicorr_zero_prefix_rate"] == 0.0

        narrow = _converted(topk=CANDIDATE_TOPK)
        narrow.train()
        metrics = narrow(**_make_batch(narrow.dflash_config.vocab_size)).lilicorr_metrics
        assert metrics["lilicorr_slot_gt_in_lattice"] < 1.0
        assert metrics["lilicorr_zero_prefix_rate"] > 0.0

    def test_lattice_terms_alone_reach_the_drafter_body(self):
        """Zero out the block cross-entropy: the reranker still trains the backbone."""
        model = _converted()
        model.train()
        batch = _make_batch(model.dflash_config.vocab_size)

        # Call the loss directly so the DFlash term can be dropped from the total.
        captured = {}
        original = model._compute_lilicorr_loss

        def capture(**kwargs):
            loss, accuracy, metrics = original(**kwargs)
            captured["loss"] = loss
            return loss, accuracy, metrics

        model._compute_lilicorr_loss = capture
        model(**batch)
        captured["loss"].backward()

        body_grad = model.dflash_module.fc.weight.grad
        assert body_grad is not None and body_grad.abs().sum() > 0

    def test_eval_forward_delegates_to_the_base_model(self):
        """Nothing LiLiCorr-specific runs outside training."""
        model = _converted()
        model.eval()
        batch = _make_batch(model.dflash_config.vocab_size)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        assert not hasattr(out, "lilicorr_metrics") or out.get("lilicorr_metrics") is None

    def test_pseudo_speculative_generate_still_runs(self):
        """AR validation falls back to the backbone; it must not raise."""
        model = _converted()
        model.eval()
        input_ids = _make_batch(model.dflash_config.vocab_size, bsz=1)["input_ids"]
        base_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=3)
        assert base_token.shape == (1, 1)
        assert draft_tokens.shape == (1, 3)


class TestLiLiCorrOptimization:
    """The objective is trainable: a fixed batch is driven down.

    Run in float32 — the tiny model is created in bfloat16, whose optimizer steps are
    too coarse to show a trend over a handful of iterations.
    """

    def test_single_batch_overfit(self):
        model = _converted().float()
        model.train()
        batch = _make_batch(model.dflash_config.vocab_size)

        trainable = [p for p in model.dflash_module.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            losses.append(out.lilicorr_metrics["lilicorr_loss"])

        assert all(math.isfinite(loss) for loss in losses), losses
        assert losses[-1] < losses[0], losses


class TestLiLiCorrExporter:
    """The export contract the serving loader rebuilds the head from."""

    def _export(self, tmp_path, **kwargs):
        model = _converted(**kwargs)
        export_dir = tmp_path / "exported"
        model.get_exporter().export(export_dir)
        return export_dir

    def test_export_weight_keys(self, tmp_path):
        """Head tensors ship under `lilicorr.*`, with no training-time prefix."""
        state_dict = load_file(str(self._export(tmp_path) / "model.safetensors"))
        for key in state_dict:
            assert "dflash_module." not in key
            assert "rotary_emb" not in key
        for name in HEAD_PARAM_NAMES:
            assert f"lilicorr.{name}" in state_dict, name
        # The backbone is exported unchanged alongside it.
        assert "fc.weight" in state_dict
        assert "norm.weight" in state_dict

    def test_export_config_declares_the_lilicorr_architecture(self, tmp_path):
        """`architectures` is the serving router.

        A checkpoint that declares DFlashDraftModel loads as plain DFlash and silently
        ignores the head, which reads as a small acceptance delta rather than an error.
        """
        with open(self._export(tmp_path) / "config.json") as f:
            config = json.load(f)
        assert config["architectures"] == ["LiLiCorrDraftModel"]

        dflash_config = config["dflash_config"]
        assert dflash_config["projector_type"] == "lilicorr"
        assert dflash_config["lilicorr_enabled"] is True
        assert dflash_config["lilicorr_candidate_topk"] == CANDIDATE_TOPK
        assert dflash_config["lilicorr_hidden_size"] == HEAD_HIDDEN
        assert dflash_config["lilicorr_factor_dim"] == FACTOR_DIM
        assert dflash_config["lilicorr_num_layers"] == HEAD_LAYERS
        assert dflash_config["lilicorr_num_heads"] == HEAD_ATTENTION_HEADS
        assert dflash_config["lilicorr_mlp_ratio"] == 2.0
        assert dflash_config["lilicorr_logit_scale"] == 8.0
        assert dflash_config["lilicorr_vector_eps"] == 1.0e-4
        # The DFlash fields the backbone still needs.
        assert "mask_token_id" in dflash_config
        assert "target_layer_ids" in dflash_config

    def test_exported_geometry_matches_the_exported_weights(self, tmp_path):
        """Geometry is read off the built head, so it cannot drift from the tensors."""
        export_dir = self._export(tmp_path)
        state_dict = load_file(str(export_dir / "model.safetensors"))
        with open(export_dir / "config.json") as f:
            dflash_config = json.load(f)["dflash_config"]
        out_head = state_dict["lilicorr.out_head.weight"]
        assert out_head.shape == (
            dflash_config["lilicorr_factor_dim"],
            dflash_config["lilicorr_hidden_size"],
        )
        rank_embedding = state_dict["lilicorr.rank_embedding"]
        assert rank_embedding.shape[-2] == dflash_config["lilicorr_candidate_topk"]


class TestLiLiCorrRoundTrip:
    """The head survives a modelopt save/restore and an exported warm start.

    Both go through ``modify`` again, so they also check that the objective fields
    round-trip through ``modelopt_state`` — a restore that lost them would raise on the
    all-or-nothing validation rather than train a different composition.
    """

    def test_save_and_restore(self, tmp_path):
        mto.enable_huggingface_checkpointing()
        reference = _converted()
        reference.save_pretrained(tmp_path / "modelopt_model")

        restored = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
        assert isinstance(restored, HFLiLiCorrModel)
        assert restored.dflash_lilicorr_w_ce == VARIANTS["base"]["w_ce"]
        assert restored.dflash_lilicorr_w_pen == VARIANTS["base"]["w_pen"]

        reference_params = dict(reference.dflash_module.named_parameters())
        for name, param in restored.dflash_module.named_parameters():
            # Cast to the loaded dtype: transformers <5 ignores the config's dtype and
            # loads fp32 regardless, and allclose raises on mismatched dtypes.
            assert torch.allclose(param, reference_params[name].to(param.dtype)), name

    def test_warm_start_from_an_exported_drafter(self, tmp_path):
        """An exported checkpoint reloads through the same key names it was written with."""
        source = _converted()
        with torch.no_grad():
            for param in source.dflash_module.parameters():
                param.fill_(0.125)
        export_dir = tmp_path / "drafter"
        source.get_exporter().export(export_dir)

        warm = _converted(dflash_init_checkpoint=str(export_dir))
        for name, param in warm.dflash_module.named_parameters():
            assert torch.allclose(param, torch.full_like(param, 0.125)), name

    def test_head_state_dict_round_trips_strictly(self):
        """The head's own key layout is complete: no missing or unexpected entries.

        There is one spelling of the head, ``lilicorr.*``, and it is the exported one.
        """
        module = _converted().dflash_module
        canonical = dict(module.state_dict())
        key = "lilicorr.out_head.weight"
        canonical[key] = torch.full_like(canonical[key], 0.25)
        result = module.load_state_dict(canonical, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        weight = module.lilicorr.out_head.weight
        assert torch.allclose(weight, torch.full_like(weight, 0.25))
