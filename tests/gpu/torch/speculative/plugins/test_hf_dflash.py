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

"""GPU tests for DFlash speculative decoding plugin.

These tests require a CUDA GPU. CPU-only tests are in tests/unit/.
"""

import pytest
import torch
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


@pytest.fixture
def dflash_model():
    """Create a tiny DFlash model on GPU."""
    model = get_tiny_llama(num_hidden_layers=4)
    config = get_dflash_config()
    mtsp.convert(model, [("dflash", config)])
    model = model.cuda()
    return model


class TestDFlashModuleGPU:
    """Test DFlash draft module forward pass on GPU."""

    def test_dflash_module_forward_shape(self, dflash_model):
        """Test that draft module produces correct output shape."""
        model = dflash_model
        bsz = 2
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)

        dtype = next(model.dflash_module.parameters()).dtype
        target_hidden = torch.randn(
            bsz, SEQ_LEN, num_layers * hidden_size, device="cuda", dtype=dtype
        )
        noise_emb = torch.randn(bsz, SEQ_LEN, hidden_size, device="cuda", dtype=dtype)
        pos_ids = (
            torch.cat([torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)])
            .unsqueeze(0)
            .expand(bsz, -1)
            .cuda()
        )

        output = model.dflash_module(
            noise_embedding=noise_emb,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=None,
        )
        assert output.shape == (bsz, SEQ_LEN, hidden_size)

    def test_dflash_module_deterministic(self, dflash_model):
        """Test that draft module produces identical outputs for same input."""
        model = dflash_model
        model.eval()
        bsz = 1
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)

        dtype = next(model.dflash_module.parameters()).dtype
        target_hidden = torch.randn(
            bsz, SEQ_LEN, num_layers * hidden_size, device="cuda", dtype=dtype
        )
        noise_emb = torch.randn(bsz, SEQ_LEN, hidden_size, device="cuda", dtype=dtype)
        pos_ids = torch.cat([torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)]).unsqueeze(0).cuda()

        with torch.no_grad():
            out1 = model.dflash_module(
                noise_embedding=noise_emb,
                target_hidden=target_hidden,
                position_ids=pos_ids,
            )
            out2 = model.dflash_module(
                noise_embedding=noise_emb,
                target_hidden=target_hidden,
                position_ids=pos_ids,
            )
        assert torch.allclose(out1, out2)


class TestDFlashTrainingForwardGPU:
    """Test DFlash training forward pass end-to-end on GPU."""

    @pytest.fixture
    def model(self):
        """Create a tiny DFlash model in training mode on GPU."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        model = model.cuda()
        model.train()
        return model

    def test_training_forward_returns_loss(self, model):
        """Test that training forward returns a differentiable loss."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "loss")
        assert output.loss.requires_grad

    def test_training_forward_returns_accuracy(self, model):
        """Test that training forward returns train_acc."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "train_acc")

    def test_training_forward_with_labels(self, model):
        """Test that labels are used for response-only loss masking."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        # Labels with -100 for first half (masked), real labels for second half
        labels = torch.full((bsz, SEQ_LEN), -100, dtype=torch.long, device="cuda")
        labels[:, SEQ_LEN // 2 :] = input_ids[:, SEQ_LEN // 2 :]

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert hasattr(output, "loss")
        assert output.loss.requires_grad

    def test_training_forward_all_masked_labels(self, model):
        """Test that all-masked labels produce zero loss without crashing."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")
        labels = torch.full((bsz, SEQ_LEN), -100, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert output.loss.item() == 0.0

    def test_training_backward(self, model):
        """Test that gradients flow to dflash_module."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        output.loss.backward()

        has_grad = False
        for name, param in model.dflash_module.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "DFlash module should receive gradients"

    def test_eval_forward_uses_base_model(self, model):
        """In eval mode, forward should use base model (not DFlash training)."""
        model.eval()
        bsz = 1
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")

        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (bsz, SEQ_LEN, model.config.vocab_size)


class TestDFlashOfflineForwardGPU:
    """Test DFlash training forward pass with dflash_offline=True on GPU."""

    NUM_BASE_LAYERS = 4

    @pytest.fixture
    def offline_model(self):
        """Create a tiny DFlash model with dflash_offline=True on GPU."""
        model = get_tiny_llama(num_hidden_layers=self.NUM_BASE_LAYERS)
        model.config.num_orig_hidden_layers = self.NUM_BASE_LAYERS
        config = get_dflash_config()
        config["dflash_offline"] = True
        mtsp.convert(model, [("dflash", config)])
        model = model.cuda()
        model.train()
        return model

    def _make_base_model_outputs(self, model, bsz):
        """Build a base_model_outputs dict matching DFlashBaseModelOutput.from_offline_dict.

        Production EagleOfflineDataCollator only emits ``aux_hidden_states`` and
        ``base_model_hidden_states`` — ``base_model_logits`` is never in the batch,
        so this fixture matches that shape exactly.
        """
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)
        dtype = next(model.dflash_module.parameters()).dtype
        return {
            "aux_hidden_states": torch.randn(
                bsz, SEQ_LEN, num_layers * hidden_size, device="cuda", dtype=dtype
            ),
            "base_model_hidden_states": torch.randn(
                bsz, SEQ_LEN, hidden_size, device="cuda", dtype=dtype
            ),
        }

    def test_offline_forward_returns_loss(self, offline_model):
        """Offline forward consumes precomputed base_model_outputs and returns a finite loss."""
        bsz = 2
        input_ids = torch.randint(0, offline_model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")
        base_model_outputs = self._make_base_model_outputs(offline_model, bsz)

        output = offline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            base_model_outputs=base_model_outputs,
        )
        assert hasattr(output, "loss")
        assert output.loss.requires_grad
        assert torch.isfinite(output.loss).item()

    def test_offline_forward_self_logit_distillation_reads_hidden_states(self, offline_model):
        """Without base_model_logits, self-distillation projects the captured hidden states.

        Only the rows the loss reads are projected, so no full-sequence logits come back.
        """
        assert offline_model.dflash_self_logit_distillation
        bsz = 2
        input_ids = torch.randint(0, offline_model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")
        base_model_outputs = self._make_base_model_outputs(offline_model, bsz)

        output = offline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            base_model_outputs=base_model_outputs,
        )
        assert output.logits is None
        assert output.loss.requires_grad
        assert torch.isfinite(output.loss).item()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs to shard the base model across devices"
)
class TestShardedBaseGeneration:
    """AR validation loads with ``device_map="auto"``, which shards the base model.

    ``target_layer_ids`` spans early and late layers, so on a multi-GPU box the selected
    hidden states come off different devices. Concatenating them without gathering first
    raises ``Expected all tensors to be on the same device`` for every sample, which is
    what broke ``test_dflash_ar_validate`` on the 2-GPU nightly runner.
    """

    @staticmethod
    def _shard_hidden_states(model, spare_device):
        """Make the base model report early hidden states from another device.

        Standing in for accelerate's dispatch: the base forward stays whole, but its output
        is spread the way a real ``device_map="auto"`` split would spread it.
        """
        base_forward = model._base_model.forward

        def sharded_forward(*args, input_ids=None, **kwargs):
            # accelerate's AlignDevicesHook moves inputs to each submodule's device.
            if input_ids is not None:
                input_ids = input_ids.to(next(model._base_model.layers[-1].parameters()).device)
            out = base_forward(*args, input_ids=input_ids, **kwargs)
            hidden = list(out.hidden_states)
            for i in range(len(hidden) // 2):
                hidden[i] = hidden[i].to(spare_device)
            out.hidden_states = tuple(hidden)
            return out

        model._base_model.forward = sharded_forward

    def test_generate_gathers_sharded_hidden_states(self):
        model = get_tiny_llama(num_hidden_layers=4)
        mtsp.convert(model, [("dflash", get_dflash_config(block_size=4))])
        model = model.to("cuda:1").eval()

        # The fix is only exercised if the split actually separates two target layers.
        assert (
            min(model.target_layer_ids)
            < len(model._base_model.layers) // 2
            <= max(model.target_layer_ids)
        ), model.target_layer_ids

        self._shard_hidden_states(model, spare_device=torch.device("cuda:0"))

        input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
        with torch.no_grad():
            base_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=2)

        # validate_online cats both onto the running sequence, so both must come back on
        # the caller's device rather than wherever the draft happened to run.
        assert base_token.device == input_ids.device
        assert draft_tokens.device == input_ids.device
        assert draft_tokens.shape == (1, 2)


class TestDFlashFlexAttentionGPU:
    """FlexAttention path must match the dense-mask SDPA path it replaces.

    The model is wider than the rest of this file because FlexAttention's Triton templates
    need head_dim >= 16 (``get_tiny_llama`` defaults to head_dim 2).
    """

    SEQ = 64
    BASE_KWARGS = {
        "num_hidden_layers": 4,
        "hidden_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 64,
        "max_position_embeddings": 256,
        "vocab_size": 64,
    }

    @classmethod
    def _model(cls, **cfg_overrides):
        model = get_tiny_llama(**cls.BASE_KWARGS)
        config = get_dflash_config()
        config.update(cfg_overrides)
        mtsp.convert(model, [("dflash", config)])
        return model.cuda().train()

    @classmethod
    def _pair(cls, **cfg_overrides):
        """A dense model and a flex model with identical draft weights."""
        dense = cls._model(**cfg_overrides)
        flex = cls._model(dflash_use_flex_attention=True, **cfg_overrides)
        flex.dflash_module.load_state_dict(dense.dflash_module.state_dict())
        return dense, flex

    @classmethod
    def _inputs(cls, bsz=2):
        input_ids = torch.randint(0, cls.BASE_KWARGS["vocab_size"], (bsz, cls.SEQ), device="cuda")
        attention_mask = torch.ones(bsz, cls.SEQ, dtype=torch.long, device="cuda")
        return input_ids, attention_mask

    def test_mask_builder_returns_block_mask(self):
        """With the flag on, the mask builder hands back a BlockMask, not a dense tensor."""
        pytest.importorskip("torch.nn.attention.flex_attention")
        from modelopt.torch.speculative.plugins.dflash_flex_attention import is_block_mask

        model = self._model(dflash_use_flex_attention=True)
        mask = model._build_draft_attention_mask(
            self.SEQ,
            torch.tensor([[4, 8]], device="cuda"),
            torch.tensor([[True, True]], device="cuda"),
            2,
            torch.float32,
            torch.device("cuda"),
            window=None,
        )
        assert is_block_mask(mask)
        assert not torch.is_tensor(mask)

    @pytest.mark.parametrize("window", [None, 8])
    def test_matches_dense_mask_path(self, window):
        """Loss agrees with the dense path to bf16 tolerance, with and without SWA."""
        pytest.importorskip("torch.nn.attention.flex_attention")
        overrides = {} if window is None else {"dflash_swa_window_size": window}
        dense, flex = self._pair(**overrides)
        input_ids, attention_mask = self._inputs()

        # Anchors are resampled every forward: same seed, so only the kernel differs.
        torch.manual_seed(1234)
        out_dense = dense(input_ids=input_ids, attention_mask=attention_mask)
        torch.manual_seed(1234)
        out_flex = flex(input_ids=input_ids, attention_mask=attention_mask)

        torch.testing.assert_close(out_flex.loss, out_dense.loss, rtol=2e-2, atol=2e-2)

    def test_matches_dense_mask_path_with_invalid_blocks(self):
        """Fully-masked query rows (invalid blocks), a softmax over nothing, match too."""
        pytest.importorskip("torch.nn.attention.flex_attention")
        dense, flex = self._pair()
        input_ids, attention_mask = self._inputs()
        # Row 1 has few supervised positions, so its trailing blocks are invalid.
        labels = input_ids.clone()
        labels[1, : self.SEQ - BLOCK_SIZE] = -100

        torch.manual_seed(7)
        out_dense = dense(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        torch.manual_seed(7)
        out_flex = flex(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert torch.isfinite(out_flex.loss), "flex produced a non-finite loss"
        torch.testing.assert_close(out_flex.loss, out_dense.loss, rtol=2e-2, atol=2e-2)

    def test_backward_produces_finite_grads(self):
        """The flex path is differentiable and its grads are finite."""
        pytest.importorskip("torch.nn.attention.flex_attention")
        flex = self._model(dflash_use_flex_attention=True)
        input_ids, attention_mask = self._inputs()

        flex(input_ids=input_ids, attention_mask=attention_mask).loss.backward()
        grads = [
            p.grad
            for p in flex.dflash_module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads, "no draft gradients were produced"
        assert all(torch.isfinite(g).all() for g in grads)
