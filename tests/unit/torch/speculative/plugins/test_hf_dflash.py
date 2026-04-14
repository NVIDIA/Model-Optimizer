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

import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from _test_utils.torch.transformers_models import (
    get_tiny_llama,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_dflash import (
    DFlashModule,
    HFDFlashModel,
    build_target_layer_ids,
)

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


def _get_dflash_config(block_size=BLOCK_SIZE, num_layers=NUM_DRAFT_LAYERS):
    """Create a DFlash config for testing."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0  # use token 0 as mask for tiny model
    config["dflash_architecture_config"] = {
        "num_hidden_layers": num_layers,
    }
    return config


class TestDFlashConvert:
    """Test DFlash model conversion."""

    def test_convert_creates_dflash_model(self):
        """Test that convert produces an HFDFlashModel."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)

    def test_convert_creates_dflash_module(self):
        """Test that convert attaches a DFlashModule."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "dflash_module")
        assert isinstance(model.dflash_module, DFlashModule)

    def test_convert_freezes_base_model(self):
        """Test that base model parameters are frozen after convert."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        for name, param in model.named_parameters():
            if "dflash_module" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_convert_dflash_module_trainable(self):
        """Test that DFlash module parameters are trainable after convert."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        dflash_params = [(n, p) for n, p in model.named_parameters() if "dflash_module" in n]
        assert len(dflash_params) > 0
        for name, param in dflash_params:
            assert param.requires_grad, f"DFlash param {name} should be trainable"

    def test_convert_sets_target_layer_ids(self):
        """Test that target layer IDs are set correctly."""
        model = get_tiny_llama(num_hidden_layers=8)
        config = _get_dflash_config(num_layers=3)
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "target_layer_ids")
        assert len(model.target_layer_ids) == 3
        for lid in model.target_layer_ids:
            assert 0 <= lid < 8

    def test_convert_sets_mask_token_id(self):
        """Test that mask_token_id is set from config."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "mask_token_id")
        assert model.mask_token_id == 0

    def test_convert_missing_mask_token_id_errors(self):
        """Test that missing mask_token_id raises ValueError."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        del config["dflash_mask_token_id"]
        with pytest.raises(ValueError, match="dflash_mask_token_id is required"):
            mtsp.convert(model, [("dflash", config)])


class TestDFlashSaveRestore:
    """Test DFlash model save and restore."""

    def test_save_and_restore(self, tmp_path):
        """Test round-trip save/load preserves modelopt state and outputs."""
        mto.enable_huggingface_checkpointing()
        model_ref = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model_ref, [("dflash", config)])

        model_ref.save_pretrained(tmp_path / "modelopt_model")
        assert os.path.exists(tmp_path / "modelopt_model/modelopt_state.pth")

        model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
        assert isinstance(model_test, HFDFlashModel)
        tf_modelopt_state_and_output_tester(model_ref, model_test)


class TestDFlashLazyRotaryEmb:
    """Test lazy rotary embedding initialization (matching EAGLE3 pattern).

    rotary_emb is not created in __init__ — it's lazily initialized on first
    forward call to avoid meta-tensor issues during from_pretrained restore.
    """

    def test_rotary_emb_not_created_in_init(self):
        """rotary_emb should not exist after convert (before forward)."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert not hasattr(model.dflash_module, "rotary_emb")

    def test_rotary_emb_created_on_forward(self):
        """rotary_emb should be created on first forward call."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        dflash_mod = model.dflash_module
        # Call _maybe_init_rotary_emb directly
        dflash_mod._maybe_init_rotary_emb(device="cpu")
        assert hasattr(dflash_mod, "rotary_emb")
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

        from modelopt.torch.speculative.plugins.hf_dflash import DFlashAttention

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

        from modelopt.torch.speculative.plugins.hf_dflash import DFlashAttention

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


class TestValidateOnline:
    """Test validate_online acceptance counting logic."""

    def test_all_accepted(self):
        """When all draft tokens match posterior, AR = 1 + steps."""
        from modelopt.torch.speculative.utils import AcceptanceRateValidation

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
        from modelopt.torch.speculative.utils import AcceptanceRateValidation

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
        config = _get_dflash_config()
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
        config = _get_dflash_config()
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
        import json

        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
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

    def test_export_tensor_count(self, tmp_path):
        """Exported model should have the right number of tensors."""
        from safetensors.torch import load_file

        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
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
    """Test _ensure_generation_tags with a real tokenizer (Qwen3-0.6B from HF)."""

    @pytest.fixture
    def qwen3_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    @pytest.fixture
    def qwen3_chat_template(self):
        template_path = (
            Path(__file__).parents[5]
            / "tools/launcher/examples/Qwen/Qwen3-8B/chat_template_train.jinja"
        )
        return template_path.read_text()

    def test_chatml_think_template_produces_assistant_mask(
        self, qwen3_tokenizer, qwen3_chat_template
    ):
        """Verify generation-tagged chat template produces correct assistant masks."""
        from modelopt.torch.utils.plugins.transformers_dataset import LanguageDataCollator

        collator = LanguageDataCollator(
            tokenizer=qwen3_tokenizer,
            train_len=128,
            return_labels=True,
            answer_only_loss=True,
            chat_template=qwen3_chat_template,
        )

        # Verify template was replaced with generation-tagged version
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
        decoded = qwen3_tokenizer.decode(non_masked)
        assert "The answer is 4" in decoded

    def test_multi_turn_masks_only_assistant(self, qwen3_tokenizer, qwen3_chat_template):
        """Verify multi-turn: only assistant turns are unmasked."""
        from modelopt.torch.utils.plugins.transformers_dataset import LanguageDataCollator

        collator = LanguageDataCollator(
            tokenizer=qwen3_tokenizer,
            train_len=256,
            return_labels=True,
            answer_only_loss=True,
            chat_template=qwen3_chat_template,
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
        decoded = qwen3_tokenizer.decode(non_masked)
        # Both assistant responses should appear in unmasked tokens
        assert "Hi there" in decoded
        assert "I am fine" in decoded
        # User/system content should NOT appear in unmasked tokens
        assert "You are helpful" not in decoded
        assert "How are you?" not in decoded
