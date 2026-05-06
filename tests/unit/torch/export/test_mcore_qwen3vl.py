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

"""Unit tests for Qwen3-VL Megatron Core export/import plugin."""

import pytest

from modelopt.torch.export.plugins.mcore_custom import (
    COL_TP,
    REPLICATE,
    ROW_TP,
    GatedMLPMerging,
    GatedMLPSlicing,
    NameRemapping,
    QKVMerging,
    QKVSlicing,
)
from modelopt.torch.export.plugins.mcore_qwen3vl import (
    qwen3vl_causal_lm_export,
    qwen3vl_causal_lm_import,
)


# All mcore keys that a dense (non-MoE) Qwen3-VL model should have
DENSE_MCORE_KEYS = {
    "word_embeddings",
    "final_layernorm",
    "output_layer",
    "input_layernorm",
    "linear_qkv",
    "linear_proj",
    "q_layernorm",
    "k_layernorm",
    "pre_mlp_layernorm",
    "linear_fc1",
    "linear_fc2",
}

# Additional MoE keys
MOE_MCORE_KEYS = {
    "router",
    "local_experts.linear_fc1",
    "local_experts.linear_fc2",
}


class TestQwen3VLRegistration:
    """Test that Qwen3-VL is registered in the global mapping."""

    def test_registered_in_export_mapping(self):
        from modelopt.torch.export.plugins.mcore_common import (
            all_mcore_hf_export_mapping,
        )

        assert "Qwen3VLForConditionalGeneration" in all_mcore_hf_export_mapping
        assert (
            all_mcore_hf_export_mapping["Qwen3VLForConditionalGeneration"]
            is qwen3vl_causal_lm_export
        )

    def test_registered_in_import_mapping(self):
        from modelopt.torch.export.plugins.mcore_common import (
            all_mcore_hf_import_mapping,
        )

        assert "Qwen3VLForConditionalGeneration" in all_mcore_hf_import_mapping
        assert (
            all_mcore_hf_import_mapping["Qwen3VLForConditionalGeneration"]
            is qwen3vl_causal_lm_import
        )


class TestQwen3VLImportMapping:
    """Test the HuggingFace -> Megatron Core import mapping."""

    def test_has_all_dense_keys(self):
        assert DENSE_MCORE_KEYS.issubset(qwen3vl_causal_lm_import.keys())

    def test_has_all_moe_keys(self):
        assert MOE_MCORE_KEYS.issubset(qwen3vl_causal_lm_import.keys())

    def test_language_model_prefix(self):
        """Qwen3-VL uses model.language_model. prefix (not model.)."""
        prefix_keys = [
            "word_embeddings",
            "final_layernorm",
            "input_layernorm",
            "linear_qkv",
            "linear_proj",
            "q_layernorm",
            "k_layernorm",
            "pre_mlp_layernorm",
            "linear_fc1",
            "linear_fc2",
        ]
        for key in prefix_keys:
            mapping = qwen3vl_causal_lm_import[key]
            assert "model.language_model." in mapping.target_name_or_prefix, (
                f"{key}: expected 'model.language_model.' prefix, "
                f"got '{mapping.target_name_or_prefix}'"
            )

    def test_output_layer_at_root(self):
        """lm_head is at root level, not under language_model."""
        mapping = qwen3vl_causal_lm_import["output_layer"]
        assert mapping.target_name_or_prefix == "lm_head."

    def test_qkv_uses_merging(self):
        assert isinstance(qwen3vl_causal_lm_import["linear_qkv"], QKVMerging)

    def test_mlp_uses_gated_merging(self):
        assert isinstance(
            qwen3vl_causal_lm_import["linear_fc1"], GatedMLPMerging
        )

    @pytest.mark.parametrize(
        "key",
        [
            "input_layernorm",
            "q_layernorm",
            "k_layernorm",
            "pre_mlp_layernorm",
            "final_layernorm",
        ],
    )
    def test_layernorms_are_replicated(self, key):
        """Layernorms should use REPLICATE (empty func_kwargs)."""
        mapping = qwen3vl_causal_lm_import[key]
        assert isinstance(mapping, NameRemapping)
        assert mapping.func_kwargs == REPLICATE

    @pytest.mark.parametrize(
        "key,expected_kwargs",
        [
            ("word_embeddings", COL_TP),
            ("output_layer", COL_TP),
            ("linear_proj", ROW_TP),
        ],
    )
    def test_tp_sharding(self, key, expected_kwargs):
        mapping = qwen3vl_causal_lm_import[key]
        assert mapping.func_kwargs == expected_kwargs


class TestQwen3VLExportMapping:
    """Test the Megatron Core -> HuggingFace export mapping."""

    def test_has_all_dense_keys(self):
        assert DENSE_MCORE_KEYS.issubset(qwen3vl_causal_lm_export.keys())

    def test_has_all_moe_keys(self):
        assert MOE_MCORE_KEYS.issubset(qwen3vl_causal_lm_export.keys())

    def test_language_model_prefix(self):
        """Export paths should also use model.language_model. prefix."""
        prefix_keys = [
            "word_embeddings",
            "final_layernorm",
            "input_layernorm",
            "linear_qkv",
            "linear_proj",
            "q_layernorm",
            "k_layernorm",
            "pre_mlp_layernorm",
            "linear_fc1",
            "linear_fc2",
        ]
        for key in prefix_keys:
            mapping = qwen3vl_causal_lm_export[key]
            assert "model.language_model." in mapping.target_name_or_prefix, (
                f"{key}: expected 'model.language_model.' prefix, "
                f"got '{mapping.target_name_or_prefix}'"
            )

    def test_output_layer_at_root(self):
        mapping = qwen3vl_causal_lm_export["output_layer"]
        assert mapping.target_name_or_prefix == "lm_head."

    def test_qkv_uses_slicing(self):
        assert isinstance(qwen3vl_causal_lm_export["linear_qkv"], QKVSlicing)

    def test_mlp_uses_gated_slicing(self):
        assert isinstance(
            qwen3vl_causal_lm_export["linear_fc1"], GatedMLPSlicing
        )

    def test_export_has_no_parallel_config(self):
        """Export mappings should not have parallel configs."""
        for key in ["word_embeddings", "final_layernorm", "output_layer",
                     "input_layernorm", "linear_proj"]:
            mapping = qwen3vl_causal_lm_export[key]
            assert "parallel_config" not in mapping.func_kwargs


class TestQwen3VLImportExportSymmetry:
    """Test that import and export mappings are consistent."""

    def test_same_mcore_keys(self):
        assert set(qwen3vl_causal_lm_import.keys()) == set(
            qwen3vl_causal_lm_export.keys()
        )

    @pytest.mark.parametrize(
        "key",
        [
            "word_embeddings",
            "final_layernorm",
            "output_layer",
            "input_layernorm",
            "linear_proj",
            "q_layernorm",
            "k_layernorm",
            "pre_mlp_layernorm",
            "linear_fc2",
            "router",
        ],
    )
    def test_matching_hf_prefixes(self, key):
        """Import and export should map to the same HF prefix."""
        imp = qwen3vl_causal_lm_import[key]
        exp = qwen3vl_causal_lm_export[key]
        assert imp.target_name_or_prefix == exp.target_name_or_prefix, (
            f"{key}: import prefix '{imp.target_name_or_prefix}' != "
            f"export prefix '{exp.target_name_or_prefix}'"
        )

    def test_qkv_matching_prefix(self):
        imp = qwen3vl_causal_lm_import["linear_qkv"]
        exp = qwen3vl_causal_lm_export["linear_qkv"]
        assert imp.target_name_or_prefix == exp.target_name_or_prefix

    def test_mlp_fc1_matching_prefix(self):
        imp = qwen3vl_causal_lm_import["linear_fc1"]
        exp = qwen3vl_causal_lm_export["linear_fc1"]
        assert imp.target_name_or_prefix == exp.target_name_or_prefix


class TestQwen3VLvsQwen3Difference:
    """Test that Qwen3-VL differs from Qwen3 only in the language_model prefix."""

    def test_same_keys_as_qwen3(self):
        from modelopt.torch.export.plugins.mcore_qwen import (
            qwen3_causal_lm_export,
            qwen3_causal_lm_import,
        )

        assert set(qwen3vl_causal_lm_import.keys()) == set(
            qwen3_causal_lm_import.keys()
        )
        assert set(qwen3vl_causal_lm_export.keys()) == set(
            qwen3_causal_lm_export.keys()
        )

    @pytest.mark.parametrize(
        "key",
        [
            "word_embeddings",
            "final_layernorm",
            "input_layernorm",
            "linear_qkv",
            "linear_proj",
            "q_layernorm",
            "k_layernorm",
            "pre_mlp_layernorm",
            "linear_fc1",
            "linear_fc2",
            "router",
            "local_experts.linear_fc1",
            "local_experts.linear_fc2",
        ],
    )
    def test_vl_adds_language_model_prefix(self, key):
        """Qwen3-VL should have 'language_model.' inserted after 'model.'."""
        from modelopt.torch.export.plugins.mcore_qwen import (
            qwen3_causal_lm_import,
        )

        qwen3_prefix = qwen3_causal_lm_import[key].target_name_or_prefix
        qwen3vl_prefix = qwen3vl_causal_lm_import[key].target_name_or_prefix
        expected = qwen3_prefix.replace("model.", "model.language_model.", 1)
        assert qwen3vl_prefix == expected, (
            f"{key}: expected '{expected}', got '{qwen3vl_prefix}'"
        )

    def test_output_layer_same(self):
        """lm_head is at root level for both Qwen3 and Qwen3-VL."""
        from modelopt.torch.export.plugins.mcore_qwen import (
            qwen3_causal_lm_import,
        )

        assert (
            qwen3vl_causal_lm_import["output_layer"].target_name_or_prefix
            == qwen3_causal_lm_import["output_layer"].target_name_or_prefix
        )
