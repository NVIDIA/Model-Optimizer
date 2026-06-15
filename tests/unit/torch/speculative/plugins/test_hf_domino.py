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

"""CPU unit tests for the Domino speculative decoding plugin.

Domino reuses the DFlash mode/pipeline and adds a GRU-based causal correction
head. These tests cover conversion routing, the training forward (base + final
dual loss), and the export format (weights + config) against the z-lab reference
layout (``prefix_gru.*`` / ``embed_proj.*``).
"""

import json
from copy import deepcopy

import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_dflash import HFDFlashModel
from modelopt.torch.speculative.plugins.hf_domino import HFDominoModel, compute_lambda_base
from modelopt.torch.speculative.plugins.modeling_dflash import DFlashModule
from modelopt.torch.speculative.plugins.modeling_domino import DominoModule

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be a multiple of BLOCK_SIZE
GRU_HIDDEN_DIM = 32
EMB_DIM = 16


def _get_domino_config(block_size=BLOCK_SIZE, num_layers=NUM_DRAFT_LAYERS):
    """Create a Domino config for testing (dflash mode + projector_type=domino)."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0  # token 0 as mask for the tiny model
    config["dflash_self_logit_distillation"] = False
    config["dflash_loss_decay_factor"] = 4.0
    config["dflash_lambda_base_start"] = 1.0
    config["dflash_lambda_base_decay_ratio"] = 1.0
    config["dflash_architecture_config"] = {
        "num_hidden_layers": num_layers,
        "projector_type": "domino",
        "gru_hidden_dim": GRU_HIDDEN_DIM,
        "emb_dim": EMB_DIM,
        "pure_draft_prefix_len": 1,
        "shift_label": True,
    }
    return config


class TestDominoConvert:
    """Test Domino model conversion routing."""

    def test_convert_creates_domino_model(self):
        """projector_type=domino routes to HFDominoModel (a HFDFlashModel subclass)."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_domino_config())])
        assert isinstance(model, HFDominoModel)
        assert isinstance(model, HFDFlashModel)

    def test_convert_attaches_domino_module_with_head(self):
        """The draft module is a DominoModule with prefix_gru + embed_proj."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_domino_config())])
        assert isinstance(model.dflash_module, DominoModule)
        assert isinstance(model.dflash_module.prefix_gru, torch.nn.GRU)
        assert model.dflash_module.prefix_gru.bias is False
        # embed_proj: Linear(H+gru -> emb) -> SiLU -> Linear(emb -> vocab)
        assert model.dflash_module.embed_proj[0].in_features == (
            model.dflash_config.hidden_size + GRU_HIDDEN_DIM
        )
        assert model.dflash_module.embed_proj[2].out_features == model.dflash_config.vocab_size

    def test_head_params_trainable(self):
        """The GRU + projection head parameters are trainable."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_domino_config())])
        head = [
            (n, p) for n, p in model.named_parameters() if "prefix_gru" in n or "embed_proj" in n
        ]
        assert len(head) >= 3  # weight_ih_l0, weight_hh_l0, 2x embed_proj
        assert all(p.requires_grad for _, p in head)

    def test_dflash_mode_still_creates_plain_dflash(self):
        """Without projector_type=domino, conversion still yields a plain DFlash model."""
        config = deepcopy(DFLASH_DEFAULT_CFG["config"])
        config["dflash_mask_token_id"] = 0
        config["dflash_architecture_config"] = {"num_hidden_layers": NUM_DRAFT_LAYERS}
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)
        assert not isinstance(model, HFDominoModel)
        assert type(model.dflash_module) is DFlashModule


class TestDominoForward:
    """Test the Domino training forward (online path on CPU)."""

    def _make_batch(self, vocab_size):
        torch.manual_seed(0)
        input_ids = torch.randint(1, vocab_size, (2, SEQ_LEN))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return input_ids, attention_mask, labels

    def test_forward_produces_dual_loss_and_grads(self):
        """Forward returns a scalar loss; backward populates head + backbone grads."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_domino_config())])
        model.train()

        vocab = model.dflash_config.vocab_size
        input_ids, attention_mask, labels = self._make_batch(vocab)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out.loss.requires_grad
        assert out.loss.dim() == 0
        # dual-loss bookkeeping
        assert "base_loss" in out.domino_metrics
        assert "final_loss" in out.domino_metrics
        assert "base_accuracy" in out.domino_metrics
        assert out.domino_metrics["lambda_base"] == 1.0  # default before any callback

        out.loss.backward()
        gru_grad = model.dflash_module.prefix_gru.weight_ih_l0.grad
        proj_grad = model.dflash_module.embed_proj[2].weight.grad
        backbone_grad = model.dflash_module.fc.weight.grad
        assert gru_grad is not None and torch.isfinite(gru_grad).all()
        assert proj_grad is not None and torch.isfinite(proj_grad).all()
        assert backbone_grad is not None and torch.isfinite(backbone_grad).all()

    def test_lambda_zero_uses_final_only(self):
        """With lambda_base=0 the loss equals final_loss (correction head only)."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_domino_config())])
        model.train()
        model._lambda_base = 0.0

        vocab = model.dflash_config.vocab_size
        input_ids, attention_mask, labels = self._make_batch(vocab)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert abs(out.loss.item() - out.domino_metrics["final_loss"]) < 1e-4


class TestLambdaSchedule:
    """Test the lambda_base curriculum schedule."""

    def test_linear_decay(self):
        assert compute_lambda_base(0, 100, 1.0, 1.0) == 1.0
        assert abs(compute_lambda_base(50, 100, 1.0, 1.0) - 0.5) < 1e-6
        assert compute_lambda_base(100, 100, 1.0, 1.0) == 0.0
        # decay_ratio=0.5 → fully decayed at the halfway point
        assert compute_lambda_base(50, 100, 1.0, 0.5) == 0.0


class TestDominoExporter:
    """Test the Domino checkpoint export format (z-lab reference layout)."""

    def _export(self, tmp_path):
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_domino_config())])
        exporter = model.get_exporter()
        export_dir = tmp_path / "exported"
        exporter.export(export_dir)
        return export_dir

    def test_export_weight_keys_match_reference(self, tmp_path):
        """Exported weights include head tensors under the reference names, no prefix."""
        export_dir = self._export(tmp_path)
        sd = load_file(str(export_dir / "model.safetensors"))
        for key in sd:
            assert "dflash_module." not in key
            assert "rotary_emb" not in key
        assert "prefix_gru.weight_ih_l0" in sd
        assert "prefix_gru.weight_hh_l0" in sd
        assert "embed_proj.0.weight" in sd
        assert "embed_proj.2.weight" in sd
        # GRU stores no bias (bias=False)
        assert "prefix_gru.bias_ih_l0" not in sd

    def test_export_config_has_domino_fields(self, tmp_path):
        """config.json carries the dflash_config domino fields + top-level emb_dim."""
        export_dir = self._export(tmp_path)
        with open(export_dir / "config.json") as f:
            cfg = json.load(f)

        assert cfg["architectures"] == ["DFlashDraftModel"]
        assert cfg["emb_dim"] == EMB_DIM
        dc = cfg["dflash_config"]
        assert dc["projector_type"] == "domino"
        assert dc["shift_label"] is True
        assert dc["pure_draft_prefix_len"] == 1
        assert dc["gru_hidden_dim"] == GRU_HIDDEN_DIM
        assert dc["emb_dim"] == EMB_DIM
        assert "mask_token_id" in dc
        assert "target_layer_ids" in dc
