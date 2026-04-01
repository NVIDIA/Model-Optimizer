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

"""Test of quantization config validations."""

import pytest

from modelopt.torch.quantization.config import (
    FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    FP8_DEFAULT_CFG,
    FP8_PER_CHANNEL_PER_TOKEN_CFG,
    INT4_AWQ_CFG,
    NVFP4_DEFAULT_CFG,
    W4A8_AWQ_BETA_CFG,
    find_quant_cfg_entry,
    need_calibration,
    normalize_quant_cfg_list,
)


def test_need_calibration():
    assert need_calibration(FP8_DEFAULT_CFG)
    assert not need_calibration(FP8_PER_CHANNEL_PER_TOKEN_CFG)
    assert not need_calibration(FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert need_calibration(INT4_AWQ_CFG)
    assert need_calibration(W4A8_AWQ_BETA_CFG)
    assert need_calibration(NVFP4_DEFAULT_CFG)


def test_need_calibration_with_list_cfg():
    """need_calibration must handle sequential (list) cfg entries without crashing."""
    # Static list-cfg on a non-weight quantizer → needs calibration
    cfg_static = {
        "quant_cfg": [
            {
                "quantizer_path": "*input_quantizer",
                "cfg": [
                    {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                    {"num_bits": (4, 3)},
                ],
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    assert need_calibration(cfg_static)

    # Dynamic list-cfg on a non-weight quantizer → no calibration needed
    cfg_dynamic = {
        "quant_cfg": [
            {
                "quantizer_path": "*input_quantizer",
                "cfg": [{"num_bits": (4, 3), "type": "dynamic"}],
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    assert not need_calibration(cfg_dynamic)


class TestNormalizeQuantCfgList:
    def test_new_format_passthrough(self):
        """New-format entries are returned unchanged (only canonical defaults added)."""
        raw = [{"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["quantizer_path"] == "*weight_quantizer"
        assert result[0]["cfg"] == {"num_bits": 8, "axis": 0}
        assert result[0]["enable"] is True  # defaulted

    def test_new_format_enable_false(self):
        """Explicit enable=False is preserved."""
        raw = [{"quantizer_path": "*", "enable": False}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None  # defaulted

    def test_new_format_explicit_enable_true_no_cfg(self):
        """Explicit enable=True with no cfg is valid and cfg defaults to None."""
        raw = [{"quantizer_path": "*", "enable": True}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is True
        assert result[0]["cfg"] is None

    def test_legacy_single_key_dict(self):
        """Legacy {'*path': {attrs}} is converted to new format."""
        raw = [{"*weight_quantizer": {"num_bits": 8, "axis": 0}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_path"] == "*weight_quantizer"
        assert result[0]["cfg"] == {"num_bits": 8, "axis": 0}
        assert result[0]["enable"] is True  # defaulted

    def test_legacy_single_key_dict_with_enable(self):
        """Legacy {'*path': {'enable': False}} splits enable out from cfg."""
        raw = [{"*input_quantizer": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_path"] == "*input_quantizer"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None

    def test_legacy_nn_class_scoped(self):
        """Legacy {'nn.Linear': {'*': {attrs}}} is converted with parent_class."""
        raw = [{"nn.Linear": {"*": {"enable": False}}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["parent_class"] == "nn.Linear"
        assert result[0]["quantizer_path"] == "*"
        assert result[0]["enable"] is False

    def test_normalization_cfg_defaults_to_none(self):
        """Entries without cfg get cfg=None after normalization."""
        raw = [{"quantizer_path": "*lm_head*", "enable": False}]
        result = normalize_quant_cfg_list(raw)
        assert "cfg" in result[0]
        assert result[0]["cfg"] is None

    def test_normalization_enable_defaults_to_true(self):
        """Entries with cfg but no enable get enable=True after normalization."""
        raw = [{"quantizer_path": "*", "cfg": {"num_bits": 4}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is True

    def test_empty_list(self):
        """Empty list is returned unchanged."""
        assert normalize_quant_cfg_list([]) == []

    def test_multiple_entries_order_preserved(self):
        """The order of entries is preserved."""
        raw = [
            {"quantizer_path": "*", "enable": False},
            {"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 8}},
        ]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_path"] == "*"
        assert result[1]["quantizer_path"] == "*weight_quantizer"

    def test_error_on_quantizer_path_only(self):
        """Entry with only quantizer_path and no cfg or enable is rejected."""
        with pytest.raises(ValueError, match="must specify 'cfg', 'enable'"):
            normalize_quant_cfg_list([{"quantizer_path": "*"}])

    def test_error_on_empty_dict(self):
        """An empty dict entry is rejected."""
        with pytest.raises(ValueError):
            normalize_quant_cfg_list([{}])

    def test_error_on_multi_key_legacy_dict(self):
        """A multi-key legacy dict (no quantizer_path, no nn.* keys) is rejected."""
        with pytest.raises(ValueError):
            normalize_quant_cfg_list([{"*weight_quantizer": {}, "*input_quantizer": {}}])

    def test_new_format_with_list_cfg(self):
        """cfg can be a list of dicts for SequentialQuantizer."""
        raw = [
            {
                "quantizer_path": "*weight_quantizer",
                "cfg": [
                    {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                    {"num_bits": (4, 3)},
                ],
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["cfg"] == raw[0]["cfg"]
        assert result[0]["enable"] is True

    def test_legacy_flat_dict_conversion(self):
        """Legacy flat dict {'*': {...}, '*weight_quantizer': {...}} is converted to list."""
        raw = {"*": {"enable": False}, "*weight_quantizer": {"num_bits": 8, "axis": 0}}
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 2
        assert result[0]["quantizer_path"] == "*"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None
        assert result[1]["quantizer_path"] == "*weight_quantizer"
        assert result[1]["cfg"] == {"num_bits": 8, "axis": 0}
        assert result[1]["enable"] is True

    def test_legacy_enable_only_produces_cfg_none(self):
        """Legacy {'*': {'enable': False}} should produce cfg=None, not cfg={}."""
        raw = [{"*": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["cfg"] is None
        assert result[0]["enable"] is False

    def test_legacy_nn_class_enable_only_produces_cfg_none(self):
        """Legacy nn.* scoped format with only enable produces cfg=None."""
        raw = [{"nn.Linear": {"*": {"enable": False}}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["cfg"] is None
        assert result[0]["enable"] is False
        assert result[0]["parent_class"] == "nn.Linear"

    def test_legacy_default_key(self):
        """Legacy 'default' key is converted to quantizer_path='*'."""
        raw = [{"default": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_path"] == "*"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None

    def test_legacy_default_key_with_cfg(self):
        """Legacy 'default' key with cfg attributes maps to '*'."""
        raw = [{"default": {"num_bits": 8, "axis": None}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_path"] == "*"
        assert result[0]["cfg"] == {"num_bits": 8, "axis": None}
        assert result[0]["enable"] is True

    def test_legacy_flat_dict_with_default_key(self):
        """Legacy flat dict containing 'default' key converts it to '*'."""
        raw = {"default": {"enable": False}, "*weight_quantizer": {"num_bits": 8}}
        result = normalize_quant_cfg_list(raw)
        default_entries = [e for e in result if e["quantizer_path"] == "*"]
        assert len(default_entries) == 1
        assert default_entries[0]["enable"] is False

    def test_legacy_nn_class_multi_key(self):
        """Legacy nn.* scoped format with multiple sub-keys produces multiple entries."""
        raw = [
            {
                "nn.Linear": {
                    "*input_quantizer": {"enable": False},
                    "*weight_quantizer": {"num_bits": 4},
                }
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 2
        paths = {e["quantizer_path"] for e in result}
        assert paths == {"*input_quantizer", "*weight_quantizer"}
        for e in result:
            assert e["parent_class"] == "nn.Linear"


class TestFindQuantCfgEntry:
    def test_finds_last_match(self):
        """When multiple entries share the same quantizer_path, returns the last one."""
        entries = normalize_quant_cfg_list(
            [
                {"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 8}},
                {"quantizer_path": "*input_quantizer", "cfg": {"num_bits": 4}},
                {"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        result = find_quant_cfg_entry(entries, "*weight_quantizer")
        assert result["cfg"] == {"num_bits": 4}

    def test_exact_match_only(self):
        """Does not do fnmatch — only exact string equality on quantizer_path."""
        entries = normalize_quant_cfg_list(
            [{"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        with pytest.raises(KeyError):
            find_quant_cfg_entry(entries, "model.layer.weight_quantizer")

    def test_raises_on_missing(self):
        """Raises KeyError when no entry matches."""
        entries = normalize_quant_cfg_list(
            [{"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        with pytest.raises(KeyError):
            find_quant_cfg_entry(entries, "*input_quantizer")

    def test_single_entry(self):
        entries = normalize_quant_cfg_list([{"quantizer_path": "*", "enable": False}])
        result = find_quant_cfg_entry(entries, "*")
        assert result["enable"] is False

    def test_empty_list(self):
        with pytest.raises(KeyError):
            find_quant_cfg_entry([], "*")


def test_need_calibration_with_legacy_dict_format():
    """need_calibration should accept legacy dict-format quant_cfg without crashing."""
    legacy_config = {
        "quant_cfg": {"*input_quantizer": {"num_bits": 8, "axis": None}},
        "algorithm": "max",
    }
    assert need_calibration(legacy_config)


def test_need_calibration_with_legacy_list_of_single_key_dicts():
    """need_calibration should accept legacy list-of-single-key-dicts format."""
    legacy_config = {
        "quant_cfg": [{"*input_quantizer": {"num_bits": 8, "axis": None}}],
        "algorithm": "max",
    }
    assert need_calibration(legacy_config)
