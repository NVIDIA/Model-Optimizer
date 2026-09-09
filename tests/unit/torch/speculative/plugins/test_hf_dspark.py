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
import shutil
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_dflash import HFDFlashModel
from modelopt.torch.speculative.plugins.hf_dspark import HFDSparkModel
from modelopt.torch.speculative.plugins.modeling_dflash import (
    DFlashModule,
    build_target_layer_ids,
    repeat_kv,
)
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


class TestDSparkSwa:
    """DSpark honors dflash_swa_window_size (regression: the window was silently ignored)."""

    def _make_swa_model(self, window=6):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dspark_config()
        config["dflash_swa_window_size"] = window
        mtsp.convert(model, [("dflash", config)])
        return model

    def test_forward_passes_window_to_mask(self, monkeypatch):
        """The training forward builds the draft mask with the configured window."""
        model = self._make_swa_model(window=6)
        model.train()
        torch.manual_seed(0)
        input_ids = torch.randint(1, model.dflash_config.vocab_size, (2, SEQ_LEN))

        windows = []
        orig = model._build_draft_attention_mask

        def spy(*args, **kwargs):
            windows.append(kwargs.get("window"))
            return orig(*args, **kwargs)

        monkeypatch.setattr(model, "_build_draft_attention_mask", spy)
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        )
        assert out.loss.dim() == 0
        assert windows == [6]

    def test_generate_applies_windowed_mask(self, monkeypatch):
        """pseudo_speculative_generate builds (and runs with) the generation-time SWA mask."""
        model = self._make_swa_model(window=6)
        model.eval()
        torch.manual_seed(0)
        input_ids = torch.randint(1, model.dflash_config.vocab_size, (1, SEQ_LEN))

        masks = []
        orig = model._build_generate_swa_mask

        def spy(*args, **kwargs):
            mask = orig(*args, **kwargs)
            masks.append(mask)
            return mask

        monkeypatch.setattr(model, "_build_generate_swa_mask", spy)
        base_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=3)
        assert len(masks) == 1 and masks[0] is not None
        assert masks[0].shape == (1, 1, BLOCK_SIZE, SEQ_LEN + BLOCK_SIZE)
        assert base_token.shape == (1, 1)
        assert draft_tokens.shape == (1, 3)


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


class TestDraftAttentionPattern:
    """dflash_draft_attention selects the block-internal attention pattern."""

    def _make_model(self, draft_attention, window=None, attention_sink=False):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dspark_config()
        config["dflash_draft_attention"] = draft_attention
        config["dflash_attention_sink"] = attention_sink
        if window is not None:
            config["dflash_swa_window_size"] = window
        mtsp.convert(model, [("dflash", config)])
        return model

    def _draft_block(self, model, seq_len=SEQ_LEN, n_blocks=2):
        """Return the [block_size, block_size] draft-vs-draft visibility of block 0."""
        anchors = torch.tensor([[5, 9]])[:, :n_blocks]
        keep = torch.ones(1, n_blocks, dtype=torch.bool)
        mask = model._build_draft_attention_mask(
            seq_len,
            anchors,
            keep,
            n_blocks,
            torch.float32,
            torch.device("cpu"),
            window=model.dflash_swa_window_size,
        )
        # additive mask: 0 == visible, -inf == masked
        visible = mask[0, 0] == 0
        return visible[:BLOCK_SIZE, seq_len : seq_len + BLOCK_SIZE]

    def test_default_is_bidirectional(self):
        model = self._make_model("bidirectional")
        assert model.dflash_draft_attention == "bidirectional"
        assert self._draft_block(model).all(), "every query should see the whole block"

    def test_causal_block_is_lower_triangular(self):
        model = self._make_model("causal")
        block = self._draft_block(model)
        expected = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bool))
        assert torch.equal(block, expected), f"expected lower-triangular, got {block}"

    def test_causal_does_not_change_context_visibility(self):
        """Only draft-vs-draft visibility changes; context masking is untouched."""
        anchors = torch.tensor([[5, 9]])
        keep = torch.ones(1, 2, dtype=torch.bool)
        args = (SEQ_LEN, anchors, keep, 2, torch.float32, torch.device("cpu"))
        bi = self._make_model("bidirectional")._build_draft_attention_mask(*args)
        ca = self._make_model("causal")._build_draft_attention_mask(*args)
        assert torch.equal(bi[..., :SEQ_LEN], ca[..., :SEQ_LEN])

    def test_causal_generate_mask_is_lower_triangular(self):
        """The generation-time mask matches training even without SWA."""
        model = self._make_model("causal")
        mask = model._build_generate_swa_mask(SEQ_LEN, 1, torch.float32, torch.device("cpu"))
        assert mask is not None, "causal blocks need a mask even with full context attention"
        visible = mask[0, 0] == 0
        assert visible[:, :SEQ_LEN].all(), "full attention over context"
        expected = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bool))
        assert torch.equal(visible[:, SEQ_LEN:], expected)

    def test_bidirectional_full_attention_needs_no_mask(self):
        """Legacy behaviour: nothing to mask means no mask is built."""
        model = self._make_model("bidirectional")
        assert (
            model._build_generate_swa_mask(SEQ_LEN, 1, torch.float32, torch.device("cpu")) is None
        )

    def test_causal_swa_trains_and_generates(self):
        """End-to-end: causal + SWA produces a finite loss and drafts tokens."""
        model = self._make_model("causal", window=6)
        torch.manual_seed(0)
        input_ids = torch.randint(1, model.dflash_config.vocab_size, (2, SEQ_LEN))
        model.train()
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        )
        assert torch.isfinite(out.loss)
        model.eval()
        base_token, draft_tokens = model.pseudo_speculative_generate(input_ids[:1], steps=3)
        assert base_token.shape == (1, 1)
        assert draft_tokens.shape == (1, 3)

    def test_export_records_draft_attention(self, tmp_path):
        """dflash_config.causal is exported for both settings, with and without SWA."""
        for mode, expected in (("bidirectional", False), ("causal", True)):
            model = self._make_model(mode)
            export_dir = tmp_path / f"exp_{mode}"
            model.get_exporter().export(export_dir)
            with open(export_dir / "config.json") as f:
                cfg = json.load(f)
            assert cfg["dflash_config"]["causal"] is expected


class TestAttentionSink:
    """dflash_attention_sink adds a learnable per-head sink to every draft layer."""

    def _make_model(self, attention_sink=True, draft_attention="causal"):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dspark_config()
        config["dflash_attention_sink"] = attention_sink
        config["dflash_draft_attention"] = draft_attention
        mtsp.convert(model, [("dflash", config)])
        return model

    def test_sink_parameter_created_per_layer(self):
        model = self._make_model()
        heads = model.dflash_config.num_attention_heads
        for layer in model.dflash_module.layers:
            assert layer.self_attn.attention_sink_bias is not None
            assert layer.self_attn.attention_sink_bias.shape == (heads,)
            assert layer.self_attn.attention_sink_bias.requires_grad

    def test_absent_by_default(self):
        model = self._make_model(attention_sink=False)
        for layer in model.dflash_module.layers:
            assert layer.self_attn.attention_sink_bias is None

    def test_sink_receives_gradient(self):
        model = self._make_model()
        torch.manual_seed(0)
        input_ids = torch.randint(1, model.dflash_config.vocab_size, (2, SEQ_LEN))
        model.train()
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
        )
        assert torch.isfinite(out.loss)
        out.loss.backward()
        grads = [layer.self_attn.attention_sink_bias.grad for layer in model.dflash_module.layers]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads), "sink should receive a non-zero gradient"

    def test_very_negative_sink_matches_no_sink(self):
        """A sink at -inf carries no mass, so the layer must match plain attention.

        Compared at the attention-layer level (not end-to-end loss) so the check isolates
        the sink math from the rest of the pipeline.
        """
        torch.manual_seed(0)
        model = self._make_model()
        attn = model.dflash_module.layers[0].self_attn
        heads, head_dim = attn.num_heads, attn.head_dim
        kv_heads = attn.num_kv_heads

        q = torch.randn(2, heads, BLOCK_SIZE, head_dim)
        k = torch.randn(2, kv_heads, SEQ_LEN, head_dim)
        v = torch.randn(2, kv_heads, SEQ_LEN, head_dim)

        # A sink at -inf contributes no probability mass, so dropping its column leaves
        # exactly the plain softmax attention distribution.
        torch.nn.init.constant_(attn.attention_sink_bias, float("-inf"))
        with torch.no_grad():
            got = attn._sink_attention(q, k, v, None)

            k_rep = repeat_kv(k, attn.num_key_value_groups)
            v_rep = repeat_kv(v, attn.num_key_value_groups)
            weights = torch.matmul(q, k_rep.transpose(2, 3)) * attn.scaling
            expected = (
                torch.matmul(torch.softmax(weights, dim=-1), v_rep).transpose(1, 2).contiguous()
            )
        assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()

    def test_sink_absorbs_probability_mass(self):
        """A finite sink strictly reduces the mass left for real tokens."""
        torch.manual_seed(0)
        model = self._make_model()
        attn = model.dflash_module.layers[0].self_attn
        q = torch.randn(1, attn.num_heads, BLOCK_SIZE, attn.head_dim)
        k = torch.randn(1, attn.num_kv_heads, SEQ_LEN, attn.head_dim)
        v = torch.randn(1, attn.num_kv_heads, SEQ_LEN, attn.head_dim)

        outs = []
        for value in (-8.0, 0.0, 8.0):
            torch.nn.init.constant_(attn.attention_sink_bias, value)
            with torch.no_grad():
                outs.append(attn._sink_attention(q, k, v, None).abs().sum().item())
        # More sink mass -> less mass on real tokens -> smaller output magnitude.
        assert outs[0] > outs[1] > outs[2], outs

    def test_export_includes_sink_weights_and_flag(self, tmp_path):
        model = self._make_model()
        export_dir = tmp_path / "exported"
        model.get_exporter().export(export_dir)

        sd = load_file(str(export_dir / "model.safetensors"))
        heads = model.dflash_config.num_attention_heads
        for i in range(model.dflash_config.num_hidden_layers):
            key = f"layers.{i}.self_attn.attention_sink_bias"
            assert key in sd, f"missing {key}"
            assert sd[key].shape == (heads,)

        with open(export_dir / "config.json") as f:
            cfg = json.load(f)
        assert cfg["dflash_config"]["attention_sink_bias"] is True
        assert cfg["attention_sink_bias"] is True

    def test_export_omits_sink_when_disabled(self, tmp_path):
        model = self._make_model(attention_sink=False)
        export_dir = tmp_path / "exported_nosink"
        model.get_exporter().export(export_dir)
        sd = load_file(str(export_dir / "model.safetensors"))
        assert not any("attention_sink_bias" in k for k in sd)
        with open(export_dir / "config.json") as f:
            cfg = json.load(f)
        assert "attention_sink_bias" not in cfg["dflash_config"]


class TestMarkovHeadKeyRemap:
    """Head weights load from either the flat or the nested `markov_head.` layout.

    ModelOpt (and the upstream DeepSpec reference) keeps the head tensors flat on the
    module and exports them that way. Some released drafters — notably
    nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-DSpark — nest them under a
    `markov_head.` parent, so loading accepts both.
    """

    def _module(self):
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dspark_config())])
        return model.dflash_module

    def test_nested_keys_load(self):
        module = self._module()
        flat = module.state_dict()
        nested = {
            (
                f"markov_head.{k}" if k.startswith(("markov_w1", "markov_w2")) else k
            ): torch.full_like(v, 0.5) if k.startswith(("markov_w1", "markov_w2")) else v
            for k, v in flat.items()
        }
        res = module.load_state_dict(nested, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        assert torch.allclose(
            module.markov_w1.weight, torch.full_like(module.markov_w1.weight, 0.5)
        )

    def test_flat_keys_still_load(self):
        """The exported (flat) layout keeps working unchanged."""
        module = self._module()
        flat = dict(module.state_dict())
        flat["markov_w1.weight"] = torch.full_like(flat["markov_w1.weight"], 0.25)
        res = module.load_state_dict(flat, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        assert torch.allclose(
            module.markov_w1.weight, torch.full_like(module.markov_w1.weight, 0.25)
        )

    def test_flat_key_wins_over_nested(self):
        """An explicit flat key is never clobbered by a nested duplicate."""
        module = self._module()
        sd = dict(module.state_dict())
        sd["markov_w1.weight"] = torch.full_like(sd["markov_w1.weight"], 1.0)
        sd["markov_head.markov_w1.weight"] = torch.full_like(sd["markov_w1.weight"], 9.0)
        module.load_state_dict(sd, strict=True)
        assert torch.allclose(
            module.markov_w1.weight, torch.full_like(module.markov_w1.weight, 1.0)
        )


class TestInitCheckpoint:
    """dflash_init_checkpoint warm-starts the draft module from exported weights."""

    def _make_model(self, init_checkpoint=None, **overrides):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dspark_config()
        config.update(overrides)
        if init_checkpoint is not None:
            config["dflash_init_checkpoint"] = str(init_checkpoint)
        mtsp.convert(model, [("dflash", config)])
        return model

    def _export(self, tmp_path, **overrides):
        """Train-free export of a converted model, to be reloaded as a warm start."""
        model = self._make_model(**overrides)
        # Make the weights distinctive so a silent re-init would be visible.
        with torch.no_grad():
            for p in model.dflash_module.parameters():
                p.fill_(0.125)
        export_dir = tmp_path / "drafter"
        model.get_exporter().export(export_dir)
        return export_dir

    def test_warm_start_loads_exported_weights(self, tmp_path):
        """Every draft parameter comes from the checkpoint, not a fresh init."""
        export_dir = self._export(tmp_path)
        model = self._make_model(init_checkpoint=export_dir)
        for name, p in model.dflash_module.named_parameters():
            assert torch.allclose(p, torch.full_like(p, 0.125)), f"{name} was not warm-started"

    def test_accepts_safetensors_file_path(self, tmp_path):
        """The file itself works, not just its directory."""
        export_dir = self._export(tmp_path)
        model = self._make_model(init_checkpoint=export_dir / "model.safetensors")
        assert torch.allclose(
            model.dflash_module.fc.weight, torch.full_like(model.dflash_module.fc.weight, 0.125)
        )

    def test_round_trip_with_sink_and_causal(self, tmp_path):
        """Warm start carries the sink weights of a causal + sink drafter."""
        export_dir = self._export(
            tmp_path, dflash_attention_sink=True, dflash_draft_attention="causal"
        )
        model = self._make_model(
            init_checkpoint=export_dir,
            dflash_attention_sink=True,
            dflash_draft_attention="causal",
        )
        for layer in model.dflash_module.layers:
            sink = layer.self_attn.attention_sink_bias
            assert sink is not None
            assert torch.allclose(sink, torch.full_like(sink, 0.125))

    def test_default_is_random_init(self, tmp_path):
        """Without the option nothing is loaded (regression guard for the default path)."""
        self._export(tmp_path)
        model = self._make_model()
        assert not torch.allclose(
            model.dflash_module.fc.weight, torch.full_like(model.dflash_module.fc.weight, 0.125)
        )

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no draft weights"):
            self._make_model(init_checkpoint=tmp_path / "does_not_exist")

    def test_architecture_mismatch_raises(self, tmp_path):
        """A checkpoint for a different draft depth must not partially load.

        Depth feeds `fc`'s input width (one target hidden per draft layer), so this trips
        the shape check; either failure mode is acceptable as long as it raises.
        """
        export_dir = self._export(tmp_path)
        with pytest.raises(ValueError, match=r"shape mismatch|does not match the configured draft"):
            self._make_model(
                init_checkpoint=export_dir,
                dflash_architecture_config={
                    **_get_dspark_config()["dflash_architecture_config"],
                    "num_hidden_layers": NUM_DRAFT_LAYERS + 1,
                },
            )

    def test_sink_mismatch_raises(self, tmp_path):
        """Exported-without-sink cannot warm-start a sink-enabled draft."""
        export_dir = self._export(tmp_path, dflash_attention_sink=False)
        with pytest.raises(ValueError, match="does not match the configured draft"):
            self._make_model(init_checkpoint=export_dir, dflash_attention_sink=True)

    def test_restore_does_not_replay_warm_start(self, tmp_path):
        """Restoring a warm-started checkpoint must not re-read the warm-start source.

        ``dflash_init_checkpoint`` is serialized into modelopt_state and restore goes
        through ``convert_to_dflash_model`` → ``modify``, so without an explicit opt-out the
        warm start runs again at load time. The saved checkpoint already carries the trained
        weights, so re-reading a path that may be gone (another machine, cleaned-up source)
        would fail the restore for weights that are immediately overwritten anyway.
        """
        mto.enable_huggingface_checkpointing()
        export_dir = self._export(tmp_path)
        model_ref = self._make_model(init_checkpoint=export_dir)
        model_ref.save_pretrained(tmp_path / "modelopt_model")

        # The warm-start source is gone by the time the checkpoint is loaded.
        shutil.rmtree(export_dir)

        model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
        assert isinstance(model_test, HFDSparkModel)
        for name, p in model_test.dflash_module.named_parameters():
            ref = dict(model_ref.dflash_module.named_parameters())[name]
            # Cast to the loaded dtype: transformers <5 ignores the config's ``dtype`` and
            # loads fp32 regardless, and ``allclose`` raises on mismatched dtypes rather
            # than returning False. What matters here is the values, not the load dtype.
            assert torch.allclose(p, ref.to(p.dtype)), f"{name} differs after restore"

    def test_nested_head_shape_mismatch_reported(self, tmp_path):
        """A wrong-shaped tensor is caught even under the nested `markov_head.` layout.

        The shape check has to resolve the module's load hooks first; otherwise a remapped
        key skips it and fails later with a far less obvious error.
        """
        export_dir = self._export(tmp_path)
        path = export_dir / "model.safetensors"
        sd = load_file(str(path))
        sd["markov_head.markov_w1.weight"] = torch.zeros(3, MARKOV_RANK)
        del sd["markov_w1.weight"]
        save_file(sd, str(path))
        with pytest.raises(ValueError, match="shape mismatch"):
            self._make_model(init_checkpoint=export_dir)


class TestExplicitTargetLayerIds:
    """dflash_architecture_config.target_layer_ids overrides the uniform default.

    A published draft is trained against specific capture points; recomputing the default
    would feed it features from layers it never saw (and mis-shape ``fc``).
    """

    def _make_model(self, target_layer_ids=None, num_layers=NUM_DRAFT_LAYERS):
        model = get_tiny_llama(num_hidden_layers=8)
        config = _get_dspark_config(num_layers=num_layers)
        if target_layer_ids is not None:
            config["dflash_architecture_config"]["target_layer_ids"] = target_layer_ids
        mtsp.convert(model, [("dflash", config)])
        return model

    def test_explicit_ids_are_used(self):
        model = self._make_model(target_layer_ids=[0, 7])
        assert model.target_layer_ids == [0, 7]
        assert model.dflash_config.target_layer_ids == [0, 7]

    def test_default_when_unset(self):
        """Without an override the uniform default is still derived."""
        model = self._make_model()
        assert len(model.target_layer_ids) == NUM_DRAFT_LAYERS
        assert model.target_layer_ids == build_target_layer_ids(8, NUM_DRAFT_LAYERS)

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError, match="one target layer per draft layer"):
            self._make_model(target_layer_ids=[0, 3, 7])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="beyond the base model"):
            self._make_model(target_layer_ids=[0, 99])

    def test_duplicate_ids_raise(self):
        """Duplicates pass the count and range checks but silently corrupt the features.

        ``fc`` still gets its expected input width, so training proceeds with one layer fed
        twice and a capture point missing; under streaming the producer also yields fewer
        planes than ``fc`` expects.
        """
        with pytest.raises(ValueError, match="duplicates"):
            self._make_model(target_layer_ids=[3, 3])

    def test_negative_id_raises(self):
        """A negative id would wrap to a valid layer in ``hidden_states[lid + 1]``.

        Without an explicit lower bound it passes validation and silently feeds the draft
        features from the wrong layer instead of failing.
        """
        with pytest.raises(ValueError, match="non-negative"):
            self._make_model(target_layer_ids=[-2, 7])

    def test_export_round_trips_explicit_ids(self, tmp_path):
        model = self._make_model(target_layer_ids=[0, 7])
        model.get_exporter().export(tmp_path / "exp")
        with open(tmp_path / "exp" / "config.json") as f:
            cfg = json.load(f)
        assert cfg["dflash_config"]["target_layer_ids"] == [0, 7]
