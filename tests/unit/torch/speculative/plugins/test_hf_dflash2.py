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

"""CPU unit tests for the DFlash2 speculative decoding plugin.

DFlash2 reuses the DFlash mode/pipeline and adds grouped dynamic convolutions
around every attention/MLP sublayer plus a low-rank candidate selector. These
tests cover conversion routing, the convolution's two structural invariants
(identity at initialization, no leakage across the block boundary), the selector
training objective, and the export format against the SGLang/vLLM
``DFlash2DraftModel`` layout (``attention_conv.*`` / ``mlp_conv.*`` /
``candidate_selector.*``).
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
from modelopt.torch.speculative.plugins.hf_dflash2 import HFDFlash2Model
from modelopt.torch.speculative.plugins.modeling_dflash import (
    DFlashModule,
    _IdentitySublayerWrapper,
)
from modelopt.torch.speculative.plugins.modeling_dflash2 import DFlash2Module, DFlashGroupedConv

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be a multiple of BLOCK_SIZE
CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 4
SELECTOR_RANK = 8
SELECTOR_TOP_K = 5

ARCH_FIELDS = ["conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"]


def _get_dflash2_config(selector_loss_alpha=1.0, block_size=BLOCK_SIZE, **arch_overrides):
    """Create a DFlash2 config for testing (dflash mode + projector_type=dflash2)."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0  # token 0 as mask for the tiny model
    config["dflash_self_logit_distillation"] = False
    config["dflash_selector_loss_alpha"] = selector_loss_alpha
    config["dflash_architecture_config"] = {
        "num_hidden_layers": NUM_DRAFT_LAYERS,
        "projector_type": "dflash2",
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "conv_group_size": CONV_GROUP_SIZE,
        "selector_rank": SELECTOR_RANK,
        "selector_top_k": SELECTOR_TOP_K,
        **arch_overrides,
    }
    return config


def _make_batch(vocab_size):
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (2, SEQ_LEN))
    return input_ids, torch.ones_like(input_ids), input_ids.clone()


class TestDFlash2Convert:
    """Test DFlash2 conversion routing and module construction."""

    def test_convert_creates_dflash2_model(self):
        """projector_type=dflash2 routes to HFDFlash2Model (a HFDFlashModel subclass)."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        assert isinstance(model, HFDFlash2Model)
        assert isinstance(model, HFDFlashModel)
        assert isinstance(model.dflash_module, DFlash2Module)

    def test_every_sublayer_wrapped_in_a_convolution(self):
        """Both sublayer wrappers on every draft layer become real convolutions."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        layers = model.dflash_module.layers
        assert len(layers) == NUM_DRAFT_LAYERS
        for layer in layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                assert isinstance(conv, DFlashGroupedConv)
                assert conv.taps == CONV_KERNEL_SIZE
                assert conv.group_size == CONV_GROUP_SIZE

    def test_selector_shapes(self):
        """The candidate selector's codebooks and projection are sized from the config."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        selector = model.dflash_module.candidate_selector
        vocab = model.dflash_config.vocab_size
        assert selector.top_k == SELECTOR_TOP_K
        assert selector.predecessor_codebook.shape == (vocab, SELECTOR_RANK)
        assert selector.successor_codebook.shape == (vocab, SELECTOR_RANK)
        assert selector.hidden_projection.out_features == SELECTOR_RANK
        assert selector.hidden_projection.bias is None

    def test_new_params_trainable(self):
        """The convolution and selector parameters are trainable."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        new = [
            (n, p)
            for n, p in model.named_parameters()
            if "_conv." in n or "candidate_selector" in n
        ]
        assert len(new) >= 2 * NUM_DRAFT_LAYERS * 2 + 3
        assert all(p.requires_grad for _, p in new)

    @pytest.mark.parametrize("field", ARCH_FIELDS)
    def test_missing_architecture_field_raises(self, field):
        """projector_type=dflash2 without a required architecture field is an error."""
        config = _get_dflash2_config()
        del config["dflash_architecture_config"][field]
        model = get_tiny_llama(num_hidden_layers=4)
        with pytest.raises(ValueError, match=field):
            mtsp.convert(model, [("dflash", config)])

    def test_conv_kernel_larger_than_block_raises(self):
        """A convolution tap count exceeding the block size is an error."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash2_config(conv_kernel_size=BLOCK_SIZE + 1)
        with pytest.raises(ValueError, match="conv_kernel_size"):
            mtsp.convert(model, [("dflash", config)])

    def test_conv_group_size_must_divide_hidden(self):
        """A conv_group_size that does not divide hidden_size is an error."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash2_config(conv_group_size=model.config.hidden_size - 1)
        with pytest.raises(ValueError, match="conv_group_size"):
            mtsp.convert(model, [("dflash", config)])

    def test_dflash_mode_still_creates_plain_dflash(self):
        """Without projector_type=dflash2, conversion still yields a plain DFlash model."""
        config = deepcopy(DFLASH_DEFAULT_CFG["config"])
        config["dflash_mask_token_id"] = 0
        config["dflash_architecture_config"] = {"num_hidden_layers": NUM_DRAFT_LAYERS}
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)
        assert not isinstance(model, HFDFlash2Model)
        assert type(model.dflash_module) is DFlashModule
        # The sublayer seam stays a parameterless no-op for a plain DFlash draft.
        for layer in model.dflash_module.layers:
            assert isinstance(layer.attention_conv, _IdentitySublayerWrapper)
            assert isinstance(layer.mlp_conv, _IdentitySublayerWrapper)
        assert not any("_conv." in n for n, _ in model.dflash_module.named_parameters())


class TestDFlashGroupedConv:
    """Test the convolution's structural invariants directly."""

    def _conv(self, hidden_size=32, taps=2):
        torch.manual_seed(0)
        return DFlashGroupedConv(
            hidden_size=hidden_size, block_size=BLOCK_SIZE, taps=taps, group_size=CONV_GROUP_SIZE
        ).double()

    def test_identity_at_initialization(self):
        """With the dynamic kernel zeroed, the identity base kernel is a no-op.

        This is what makes enabling DFlash2 a stable extension of a DFlash backbone
        rather than a perturbation of it.
        """
        conv = self._conv()
        with torch.no_grad():
            conv.kernel_projection.weight.zero_()
        x = torch.randn(2, SEQ_LEN, 32, dtype=torch.double)
        out = conv.finish(*conv.prepare(x))
        assert torch.allclose(out, x, atol=1e-12)

    def test_taps_do_not_cross_the_block_boundary(self):
        """Perturbing the last position of a block leaves later blocks untouched."""
        conv = self._conv()
        x = torch.randn(2, SEQ_LEN, 32, dtype=torch.double)
        baseline = conv.finish(*conv.prepare(x))

        perturbed_input = x.clone()
        perturbed_input[:, BLOCK_SIZE - 1] += 5.0
        perturbed = conv.finish(*conv.prepare(perturbed_input))

        assert torch.allclose(baseline[:, BLOCK_SIZE:], perturbed[:, BLOCK_SIZE:], atol=1e-12)
        assert not torch.allclose(baseline[:, :BLOCK_SIZE], perturbed[:, :BLOCK_SIZE])

    def test_intra_block_dependency_is_backward_only(self):
        """A position influences its successors inside the block, never its predecessors.

        This is the point of the convolution: it injects the sequential dependency the
        parallel backbone lacks, without letting a position see the future.
        """
        conv = self._conv()
        x = torch.randn(2, SEQ_LEN, 32, dtype=torch.double)
        baseline = conv.finish(*conv.prepare(x))

        perturbed_input = x.clone()
        perturbed_input[:, 1] += 5.0
        perturbed = conv.finish(*conv.prepare(perturbed_input))

        assert torch.allclose(baseline[:, 0], perturbed[:, 0], atol=1e-12)
        assert not torch.allclose(baseline[:, 2], perturbed[:, 2])

    def test_sequence_length_must_be_block_aligned(self):
        """A sequence length not divisible by the block size is an error."""
        conv = self._conv()
        with pytest.raises(ValueError, match="block_size"):
            conv.prepare(torch.randn(1, BLOCK_SIZE + 1, 32, dtype=torch.double))


class TestDFlash2Forward:
    """Test the DFlash2 training forward (online path on CPU)."""

    def test_forward_grads_reach_conv_and_selector(self):
        """Backward fills gradients on the convolutions, the selector and the backbone."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        model.train()

        input_ids, attention_mask, labels = _make_batch(model.dflash_config.vocab_size)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out.loss.requires_grad
        assert out.loss.dim() == 0
        out.loss.backward()

        module = model.dflash_module
        for grad in (
            module.candidate_selector.predecessor_codebook.grad,
            module.candidate_selector.successor_codebook.grad,
            module.layers[0].attention_conv.base_kernel.grad,
            module.layers[0].mlp_conv.kernel_projection.weight.grad,
            module.fc.weight.grad,
        ):
            assert grad is not None and torch.isfinite(grad).all()
            assert grad.abs().sum() > 0

    def test_selector_metrics_reported(self):
        """The forward records selector accuracy and top-k coverage."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        model.train()

        input_ids, attention_mask, labels = _make_batch(model.dflash_config.vocab_size)
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        metrics = model._selector_metrics
        for key in ("selector_accuracy", "selector_coverage"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_selector_alpha_zero_disables_the_term(self):
        """alpha=0 trains the backbone and convolutions only; the selector gets no grad."""
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config(selector_loss_alpha=0.0))])
        model.train()

        input_ids, attention_mask, labels = _make_batch(model.dflash_config.vocab_size)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()

        module = model.dflash_module
        codebook_grad = module.candidate_selector.predecessor_codebook.grad
        assert codebook_grad is None or codebook_grad.abs().sum() == 0
        # The convolutions still train: they live inside the backbone.
        conv_grad = module.layers[0].attention_conv.kernel_projection.weight.grad
        assert conv_grad is not None and conv_grad.abs().sum() > 0

    def test_selector_loss_increases_total_loss(self):
        """The selector term adds to the backbone loss rather than replacing it."""
        input_ids, attention_mask, labels = _make_batch(32)

        losses = {}
        for alpha in (0.0, 1.0):
            torch.manual_seed(0)
            model = get_tiny_llama(num_hidden_layers=4)
            mtsp.convert(model, [("dflash", _get_dflash2_config(selector_loss_alpha=alpha))])
            model.train()
            torch.manual_seed(0)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses[alpha] = float(out.loss.detach())
        assert losses[1.0] > losses[0.0]

    def test_overfits_a_single_batch(self):
        """A few steps on one batch drive backbone and selector accuracy up.

        Guards the target/predecessor alignment: a misaligned selector objective still
        produces a finite decreasing loss, but its accuracy does not reach 1.
        """
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        model.train()

        input_ids, attention_mask, labels = _make_batch(model.dflash_config.vocab_size)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
        for _ in range(60):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

        assert out.train_acc[0][0] > 0.9
        assert model._selector_metrics["selector_accuracy"] > 0.9


class TestDFlash2Export:
    """Test the DFlash2 export format (weights + config)."""

    def _export(self, tmp_path):
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", _get_dflash2_config())])
        export_dir = tmp_path / "exported"
        model.get_exporter().export(export_dir)
        return export_dir

    def test_export_weight_keys_match_reference(self, tmp_path):
        """Exported weights carry the DFlash2 tensors under reference names, no prefix."""
        sd = load_file(str(self._export(tmp_path) / "model.safetensors"))
        for key in sd:
            assert "dflash_module." not in key
            assert "rotary_emb" not in key

        assert "candidate_selector.predecessor_codebook" in sd
        assert "candidate_selector.successor_codebook" in sd
        assert "candidate_selector.hidden_projection.weight" in sd
        for layer_idx in range(NUM_DRAFT_LAYERS):
            for wrapper in ("attention_conv", "mlp_conv"):
                assert f"layers.{layer_idx}.{wrapper}.base_kernel" in sd
                assert f"layers.{layer_idx}.{wrapper}.kernel_projection.weight" in sd

    def test_export_config_declares_dflash2_architecture(self, tmp_path):
        """config.json selects the DFlash2 serving path and carries its fields.

        The architecture name matters: a checkpoint declaring ``DFlashDraftModel``
        loads as a plain DFlash draft and silently ignores these weights.
        """
        with open(self._export(tmp_path) / "config.json") as f:
            cfg = json.load(f)

        assert cfg["architectures"] == ["DFlash2DraftModel"]
        dflash_config = cfg["dflash_config"]
        assert dflash_config["projector_type"] == "dflash2"
        assert dflash_config["conv_kernel_size"] == CONV_KERNEL_SIZE
        assert dflash_config["conv_group_size"] == CONV_GROUP_SIZE
        assert dflash_config["selector_rank"] == SELECTOR_RANK
        assert dflash_config["selector_top_k"] == SELECTOR_TOP_K
        assert "mask_token_id" in dflash_config
        assert "target_layer_ids" in dflash_config
