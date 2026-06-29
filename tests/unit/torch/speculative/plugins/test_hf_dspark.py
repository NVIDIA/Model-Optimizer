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

"""CPU unit tests for the DSpark speculative decoding plugin.

DSpark reuses the DFlash mode/pipeline and adds a lightweight sequential (Markov)
head plus an optional confidence head. These tests cover conversion routing for
the three head variants, the three-term training forward (CE + TVD + confidence
BCE), and the export format (head weights + config) against the z-lab-compatible
layout (``markov_w1.*`` / ``markov_w2.*`` / ``gate_proj.*`` / ``joint_proj.*`` /
``confidence_proj.*``).
"""

import json
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_dflash import HFDFlashModel
from modelopt.torch.speculative.plugins.hf_dspark import HFDSparkModel
from modelopt.torch.speculative.plugins.modeling_dflash import DFlashModule
from modelopt.torch.speculative.plugins.modeling_dspark import DSparkModule

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be a multiple of BLOCK_SIZE
MARKOV_RANK = 16

HEAD_TYPES = ["vanilla", "gated", "rnn"]


def _get_dspark_config(
    head_type="vanilla",
    use_confidence_head=False,
    confidence_alpha=0.0,
    block_size=BLOCK_SIZE,
    num_layers=NUM_DRAFT_LAYERS,
):
    """Create a DSpark config for testing (dflash mode + projector_type=dspark)."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0  # token 0 as mask for the tiny model
    config["dflash_self_logit_distillation"] = False
    config["dflash_confidence_head_alpha"] = confidence_alpha
    config["dflash_architecture_config"] = {
        "num_hidden_layers": num_layers,
        "projector_type": "dspark",
        "markov_rank": MARKOV_RANK,
        "markov_head_type": head_type,
        "use_confidence_head": use_confidence_head,
        "pure_draft_prefix_len": 1,
        "shift_label": True,
    }
    return config


class TestDSparkConvert:
    """Test DSpark model conversion routing and head construction."""

    def test_convert_creates_dspark_model(self):
        """projector_type=dspark routes to HFDSparkModel (a HFDFlashModel subclass)."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dspark_config())])
        assert isinstance(model, HFDSparkModel)
        assert isinstance(model, HFDFlashModel)
        assert isinstance(model.dflash_module, DSparkModule)

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_head_modules_per_type(self, head_type):
        """The Markov head builds the right submodules for each variant."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dspark_config(head_type=head_type))])
        head = model.dflash_module
        vocab = model.dflash_config.vocab_size

        # Low-rank transition shared by all variants; markov_w2 has no bias.
        assert isinstance(head.markov_w1, torch.nn.Embedding)
        assert head.markov_w1.embedding_dim == MARKOV_RANK
        assert head.markov_w2.in_features == MARKOV_RANK
        assert head.markov_w2.out_features == vocab
        assert head.markov_w2.bias is None

        # Variant-specific projections.
        assert hasattr(head, "gate_proj") == (head_type == "gated")
        assert hasattr(head, "joint_proj") == (head_type == "rnn")

    def test_confidence_head_built_when_enabled(self):
        """use_confidence_head=true attaches a confidence_proj; otherwise absent."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dspark_config(use_confidence_head=True))])
        assert hasattr(model.dflash_module, "confidence_proj")
        assert model.dflash_module.confidence_proj.out_features == 1

        model2 = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model2, [("dflash", _get_dspark_config(use_confidence_head=False))])
        assert not hasattr(model2.dflash_module, "confidence_proj")

    def test_head_params_trainable(self):
        """The Markov head parameters are trainable."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dspark_config())])
        head = [(n, p) for n, p in model.named_parameters() if "markov_w" in n]
        assert len(head) >= 2  # markov_w1.weight, markov_w2.weight
        assert all(p.requires_grad for _, p in head)

    def test_missing_markov_rank_raises(self):
        """projector_type=dspark without markov_rank is a configuration error."""
        config = _get_dspark_config()
        del config["dflash_architecture_config"]["markov_rank"]
        model = get_tiny_llama(num_hidden_layers=4)
        with pytest.raises(ValueError, match="markov_rank"):
            mtsp.convert(model, [("dflash", config)])

    def test_dflash_mode_still_creates_plain_dflash(self):
        """Without projector_type=dspark, conversion still yields a plain DFlash model."""
        config = deepcopy(DFLASH_DEFAULT_CFG["config"])
        config["dflash_mask_token_id"] = 0
        config["dflash_architecture_config"] = {"num_hidden_layers": NUM_DRAFT_LAYERS}
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)
        assert not isinstance(model, HFDSparkModel)
        assert type(model.dflash_module) is DFlashModule


class TestDSparkForward:
    """Test the DSpark training forward (online path on CPU)."""

    def _make_batch(self, vocab_size):
        torch.manual_seed(0)
        input_ids = torch.randint(1, vocab_size, (2, SEQ_LEN))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return input_ids, attention_mask, labels

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_forward_loss_metrics_and_grads(self, head_type):
        """Forward returns a scalar loss + metrics; backward fills head + backbone grads."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dspark_config(head_type=head_type))])
        model.train()

        input_ids, attention_mask, labels = self._make_batch(model.dflash_config.vocab_size)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert out.loss.requires_grad
        assert out.loss.dim() == 0
        # three-term loss bookkeeping
        for key in ("ce_loss", "l1_loss", "confidence_loss", "base_accuracy"):
            assert key in out.dspark_metrics
        assert out.dspark_metrics["confidence_loss"] == 0.0  # no confidence head here

        out.loss.backward()
        head_grad = model.dflash_module.markov_w2.weight.grad
        backbone_grad = model.dflash_module.fc.weight.grad
        assert head_grad is not None and torch.isfinite(head_grad).all()
        assert head_grad.abs().sum() > 0  # head actually participates in the loss
        assert backbone_grad is not None and torch.isfinite(backbone_grad).all()

    def test_confidence_head_contributes_grads(self):
        """With the confidence head + alpha>0, confidence_proj receives gradients."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(
            model,
            [("dflash", _get_dspark_config(use_confidence_head=True, confidence_alpha=1.0))],
        )
        model.train()

        input_ids, attention_mask, labels = self._make_batch(model.dflash_config.vocab_size)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out.dspark_metrics["confidence_loss"] > 0.0

        out.loss.backward()
        conf_grad = model.dflash_module.confidence_proj.weight.grad
        assert conf_grad is not None and torch.isfinite(conf_grad).all()
        assert conf_grad.abs().sum() > 0

    def test_confidence_alpha_without_head_raises(self):
        """confidence_head_alpha>0 but no confidence head is a configuration error."""
        model = get_tiny_llama(num_hidden_layers=4)
        with pytest.raises(ValueError, match="confidence"):
            mtsp.convert(
                model,
                [("dflash", _get_dspark_config(use_confidence_head=False, confidence_alpha=1.0))],
            )


class TestDSparkExporter:
    """Test the DSpark checkpoint export format (z-lab-compatible layout)."""

    def _export(self, tmp_path, head_type="vanilla", use_confidence_head=False):
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(
            model,
            [
                (
                    "dflash",
                    _get_dspark_config(
                        head_type=head_type, use_confidence_head=use_confidence_head
                    ),
                )
            ],
        )
        export_dir = tmp_path / "exported"
        model.get_exporter().export(export_dir)
        return export_dir

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_export_weight_keys_match_reference(self, tmp_path, head_type):
        """Exported weights carry the head tensors under reference names, no prefix."""
        sd = load_file(str(self._export(tmp_path, head_type=head_type) / "model.safetensors"))
        for key in sd:
            assert "dflash_module." not in key
            assert "rotary_emb" not in key
        assert "markov_w1.weight" in sd
        assert "markov_w2.weight" in sd
        assert ("gate_proj.weight" in sd) == (head_type == "gated")
        assert ("joint_proj.weight" in sd) == (head_type == "rnn")

    def test_export_includes_confidence_weights(self, tmp_path):
        """The confidence head weights are exported when enabled."""
        sd = load_file(str(self._export(tmp_path, use_confidence_head=True) / "model.safetensors"))
        assert "confidence_proj.weight" in sd

    def test_export_config_has_dspark_fields(self, tmp_path):
        """config.json carries the dflash_config DSpark head fields."""
        export_dir = self._export(tmp_path, head_type="gated")
        with open(export_dir / "config.json") as f:
            cfg = json.load(f)

        assert cfg["architectures"] == ["DFlashDraftModel"]
        dc = cfg["dflash_config"]
        assert dc["projector_type"] == "dspark"
        assert dc["markov_rank"] == MARKOV_RANK
        assert dc["markov_head_type"] == "gated"
        assert dc["use_confidence_head"] is False
        assert dc["shift_label"] is True
        assert "mask_token_id" in dc
        assert "target_layer_ids" in dc
