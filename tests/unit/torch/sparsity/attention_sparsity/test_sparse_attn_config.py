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

"""Unit tests for sparse attention checkpoint config helpers."""

import types

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)


class TestMatchSparseConfig:
    """Test match_sparse_config name-pattern matching."""

    def test_matches_glob(self):
        """Test that a glob pattern matches a module name."""
        cfg = {"sparse_cfg": {"*self_attn*": {"sparsity_n": 2}, "default": {"enable": False}}}
        assert match_sparse_config("model.layers.3.self_attn", cfg) == {"sparsity_n": 2}

    def test_returns_none_for_no_match(self):
        """Test that a non-matching module name returns None."""
        cfg = {"sparse_cfg": {"*self_attn*": {"sparsity_n": 2}, "default": {"enable": False}}}
        assert match_sparse_config("embed_tokens", cfg) is None

    def test_skips_default_and_calibration_keys(self):
        """Test that ``default`` and ``calibration`` keys are treated as metadata."""
        cfg = {
            "sparse_cfg": {
                "default": {"enable": False},
                "calibration": {"dataset": "x"},
                "*attn*": {"sparsity_n": 2},
            }
        }
        assert match_sparse_config("default", cfg) is None
        assert match_sparse_config("calibration", cfg) is None
        assert match_sparse_config("model.layers.0.self_attn", cfg) == {"sparsity_n": 2}

    def test_accepts_bare_sparse_cfg(self):
        """Test that the bare inner dict is accepted alongside ``{sparse_cfg: {...}}``."""
        bare = {"*attn*": {"sparsity_n": 2}, "default": {"enable": False}}
        assert match_sparse_config("self_attn", bare) == {"sparsity_n": 2}

    def test_first_match_wins(self):
        """Test that patterns are tried in insertion order with first hit winning."""
        cfg = {
            "sparse_cfg": {
                "*self_attn*": {"sparsity_n": 2, "scope": "broad"},
                "*layers.0.self_attn*": {"scope": "specific"},
                "default": {"enable": False},
            }
        }
        matched = match_sparse_config("model.layers.0.self_attn", cfg)
        assert matched["scope"] == "broad"


class TestLoadFromCheckpointMetadata:
    """Test load_from_checkpoint_metadata reading from a HF config object."""

    def test_returns_none_for_missing_hf_config(self):
        """Test that a None hf_config returns None."""
        assert load_from_checkpoint_metadata(None) is None

    def test_returns_none_when_attribute_missing(self):
        """Test that an hf_config without sparse_attention_config returns None."""
        hf_config = types.SimpleNamespace()
        assert load_from_checkpoint_metadata(hf_config) is None

    def test_returns_none_for_unknown_algo(self):
        """Test that an unrecognized sparse_algo returns None."""
        meta = {"config_groups": {"group_0": {"algorithm": "future_algo_v9000"}}}
        hf_config = types.SimpleNamespace(sparse_attention_config=meta)
        assert load_from_checkpoint_metadata(hf_config) is None

    def test_maps_uncalibrated_softmax_skip_to_preset(self):
        """Test that uncalibrated softmax_skip uses the static Triton preset."""
        meta = {
            "config_groups": {"group_0": {"algorithm": "skip_softmax"}},
            "producer": {"name": "modelopt", "version": "0.37.0"},
        }
        hf_config = types.SimpleNamespace(sparse_attention_config=meta)
        result = load_from_checkpoint_metadata(hf_config)
        assert result is not None
        cfg, preset_name = result
        assert preset_name == "SKIP_SOFTMAX_TRITON_DEFAULT"
        assert cfg is mtsa.SKIP_SOFTMAX_TRITON_DEFAULT

    def test_maps_calibrated_softmax_skip_to_dynamic_config(self):
        """Test that calibrated softmax_skip preserves checkpoint coefficients."""
        threshold_scale_factor = {
            "formula": "a * exp(b * target_sparsity)",
            "prefill": {"a": 2.0, "b": 3.0},
            "decode": {"a": 5.0, "b": 7.0},
        }
        meta = {
            "config_groups": {
                "group_0": {
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": threshold_scale_factor,
                    "target_sparsity": {"prefill": 0.4, "decode": 0.6},
                }
            },
            "producer": {"name": "modelopt", "version": "0.45.0"},
        }
        hf_config = types.SimpleNamespace(sparse_attention_config=meta)

        result = load_from_checkpoint_metadata(hf_config)

        assert result is not None
        cfg, preset_name = result
        assert preset_name == "CHECKPOINT_CALIBRATED_SOFTMAX_SKIP"
        layer_cfg = match_sparse_config("model.layers.0.self_attn", cfg)
        assert layer_cfg == {
            "method": "triton_skip_softmax",
            "threshold_scale_factor": threshold_scale_factor,
            "target_sparse_ratio": {"prefill": 0.4, "decode": 0.6},
            "backend": "triton",
            "enable": True,
        }

    def test_maps_sparse_softmax_to_dynamic_config(self):
        """Test that checkpoint N:M sparse-softmax metadata restores layer params."""
        meta = {
            "config_groups": {
                "group_0": {
                    "algorithm": "sparse_softmax",
                    "sparsity_n": 2,
                    "sparsity_m": 4,
                    "dense_sink_tokens": 4,
                    "dense_recent_tokens": 128,
                }
            },
            "producer": {"name": "modelopt", "version": "0.45.0"},
        }
        hf_config = types.SimpleNamespace(sparse_attention_config=meta)

        result = load_from_checkpoint_metadata(hf_config)

        assert result is not None
        cfg, preset_name = result
        assert preset_name == "CHECKPOINT_SPARSE_SOFTMAX"
        layer_cfg = match_sparse_config("model.layers.0.self_attn", cfg)
        assert layer_cfg == {
            "method": "triton_sparse_softmax",
            "sparsity_n": 2,
            "sparsity_m": 4,
            "dense_sink_tokens": 4,
            "dense_recent_tokens": 128,
            "backend": "triton",
            "enable": True,
        }

    def test_maps_calibrated_softmax_skip_with_sparse_softmax_overlay(self):
        """Test that vLLM restores combined skip-softmax plus N:M sparse-softmax."""
        threshold_scale_factor = {
            "formula": "a * exp(b * target_sparsity)",
            "prefill": {"a": 2.0, "b": 3.0},
        }
        meta = {
            "config_groups": {
                "group_0": {
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": threshold_scale_factor,
                    "target_sparsity": {"prefill": 0.4, "decode": 0.6},
                },
                "group_1": {
                    "algorithm": "sparse_softmax",
                    "sparsity_n": 2,
                    "sparsity_m": 4,
                },
            },
            "producer": {"name": "modelopt", "version": "0.45.0"},
        }
        hf_config = types.SimpleNamespace(sparse_attention_config=meta)

        result = load_from_checkpoint_metadata(hf_config)

        assert result is not None
        cfg, preset_name = result
        assert preset_name == "CHECKPOINT_CALIBRATED_SOFTMAX_SKIP_SPARSE_SOFTMAX"
        layer_cfg = match_sparse_config("model.layers.0.self_attn", cfg)
        assert layer_cfg == {
            "method": "triton_skip_softmax",
            "threshold_scale_factor": threshold_scale_factor,
            "target_sparse_ratio": {"prefill": 0.4, "decode": 0.6},
            "sparsity_n": 2,
            "sparsity_m": 4,
            "dense_sink_tokens": 0,
            "dense_recent_tokens": 64,
            "backend": "triton",
            "enable": True,
        }

    def test_handles_non_dict_metadata(self):
        """Test that a non-dict sparse_attention_config returns None."""
        hf_config = types.SimpleNamespace(sparse_attention_config="not a dict")
        assert load_from_checkpoint_metadata(hf_config) is None

    def test_handles_empty_config_groups(self):
        """Test that an empty config_groups returns None."""
        hf_config = types.SimpleNamespace(sparse_attention_config={"config_groups": {}})
        assert load_from_checkpoint_metadata(hf_config) is None

    def test_calibrated_softmax_skip_honors_ignore(self):
        """Layers recorded under ``ignore`` stay dense on load; others are sparsified."""
        meta = {
            "config_groups": {
                "group_0": {
                    "algorithm": "skip_softmax",
                    "ignore": ["blocks.0.attn1", "blocks.0.attn2"],
                    "threshold_scale_factor": {
                        "formula": "a * exp(b * target_sparsity)",
                        "prefill": {"a": 2.0, "b": 3.0},
                    },
                    "target_sparsity": {"prefill": 0.5},
                }
            },
            "producer": {"name": "modelopt", "version": "0.45.0"},
        }
        hf_config = types.SimpleNamespace(sparse_attention_config=meta)
        result = load_from_checkpoint_metadata(hf_config)
        assert result is not None
        cfg, _ = result
        # ``ignore``'d layers stay dense.
        assert match_sparse_config("transformer.blocks.0.attn1", cfg) == {"enable": False}
        assert match_sparse_config("transformer.blocks.0.attn2", cfg) == {"enable": False}
        # A non-ignored self-attention layer is sparsified.
        sparsified = match_sparse_config("transformer.blocks.2.attn1", cfg)
        assert sparsified["method"] == "triton_skip_softmax"
        assert sparsified["enable"] is True
