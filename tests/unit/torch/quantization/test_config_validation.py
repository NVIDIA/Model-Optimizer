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
from pydantic import ValidationError

from modelopt.torch.quantization.config import (
    FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    FP8_DEFAULT_CFG,
    FP8_PER_CHANNEL_PER_TOKEN_CFG,
    INT4_AWQ_CFG,
    NVFP4_DEFAULT_CFG,
    W4A8_AWQ_BETA_CFG,
    QuantizeConfig,
    find_quant_cfg_entry_by_path,
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
                "quantizer_name": "*input_quantizer",
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
                "quantizer_name": "*input_quantizer",
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
        raw = [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert result[0]["cfg"] == {"num_bits": 8, "axis": 0}
        assert result[0]["enable"] is True  # defaulted

    def test_new_format_enable_false(self):
        """Explicit enable=False is preserved."""
        raw = [{"quantizer_name": "*", "enable": False}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None  # defaulted

    def test_new_format_explicit_enable_true_no_cfg(self):
        """Explicit enable=True with no cfg is valid and cfg defaults to None."""
        raw = [{"quantizer_name": "*", "enable": True}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is True
        assert result[0]["cfg"] is None

    def test_legacy_single_key_dict(self):
        """Legacy {'*path': {attrs}} is converted to new format."""
        raw = [{"*weight_quantizer": {"num_bits": 8, "axis": 0}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert result[0]["cfg"] == {"num_bits": 8, "axis": 0}
        assert result[0]["enable"] is True  # defaulted

    def test_legacy_single_key_dict_with_enable(self):
        """Legacy {'*path': {'enable': False}} splits enable out from cfg."""
        raw = [{"*input_quantizer": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*input_quantizer"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None

    def test_legacy_nn_class_scoped(self):
        """Legacy {'nn.Linear': {'*': {attrs}}} is converted with parent_class."""
        raw = [{"nn.Linear": {"*": {"enable": False}}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["parent_class"] == "nn.Linear"
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["enable"] is False

    def test_normalization_cfg_defaults_to_none(self):
        """Entries without cfg get cfg=None after normalization."""
        raw = [{"quantizer_name": "*lm_head*", "enable": False}]
        result = normalize_quant_cfg_list(raw)
        assert "cfg" in result[0]
        assert result[0]["cfg"] is None

    def test_normalization_enable_defaults_to_true(self):
        """Entries with cfg but no enable get enable=True after normalization."""
        raw = [{"quantizer_name": "*", "cfg": {"num_bits": 4}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is True

    def test_empty_list(self):
        """Empty list is returned unchanged."""
        assert normalize_quant_cfg_list([]) == []

    def test_multiple_entries_order_preserved(self):
        """The order of entries is preserved."""
        raw = [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
        ]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*"
        assert result[1]["quantizer_name"] == "*weight_quantizer"

    def test_error_on_quantizer_name_only(self):
        """Entry with only quantizer_name and no cfg or enable is rejected."""
        with pytest.raises(ValueError, match="must specify 'cfg', 'enable'"):
            normalize_quant_cfg_list([{"quantizer_name": "*"}])

    def test_error_on_empty_dict(self):
        """An empty dict entry is rejected."""
        with pytest.raises(ValueError):
            normalize_quant_cfg_list([{}])

    def test_error_on_multi_key_legacy_dict(self):
        """A multi-key legacy dict (no quantizer_name, no nn.* keys) is rejected."""
        with pytest.raises(ValueError):
            normalize_quant_cfg_list([{"*weight_quantizer": {}, "*input_quantizer": {}}])

    def test_error_on_empty_cfg_dict_implicit_enable(self):
        """Entry with cfg={} and implicit enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list([{"quantizer_name": "*weight_quantizer", "cfg": {}}])

    def test_error_on_empty_cfg_dict_explicit_enable_true(self):
        """Entry with cfg={} and explicit enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": {}, "enable": True}]
            )

    def test_error_on_empty_cfg_list_enable_true(self):
        """Entry with cfg=[] and enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": [], "enable": True}]
            )

    def test_error_on_non_dict_non_list_cfg_enable_true(self):
        """Entry with cfg of invalid type (e.g. int) and enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": 42, "enable": True}]
            )

    def test_error_on_cfg_list_with_empty_dict_enable_true(self):
        """Entry with cfg=[{}] and enable=True is rejected (empty dict element)."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": [{}], "enable": True}]
            )

    def test_error_on_cfg_list_with_non_dict_element_enable_true(self):
        """Entry with cfg=[42] and enable=True is rejected (non-dict element)."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": [42], "enable": True}]
            )

    def test_empty_cfg_dict_enable_false_accepted(self):
        """Entry with cfg={} and enable=False is allowed (disable-only entry)."""
        result = normalize_quant_cfg_list(
            [{"quantizer_name": "*input_quantizer", "cfg": {}, "enable": False}]
        )
        assert result[0]["enable"] is False

    def test_empty_cfg_list_enable_false_accepted(self):
        """Entry with cfg=[] and enable=False is allowed (disable-only entry)."""
        result = normalize_quant_cfg_list(
            [{"quantizer_name": "*input_quantizer", "cfg": [], "enable": False}]
        )
        assert result[0]["enable"] is False

    def test_new_format_with_list_cfg(self):
        """cfg can be a list of dicts for SequentialQuantizer."""
        raw = [
            {
                "quantizer_name": "*weight_quantizer",
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
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None
        assert result[1]["quantizer_name"] == "*weight_quantizer"
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
        """Legacy 'default' key is converted to quantizer_name='*'."""
        raw = [{"default": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None

    def test_legacy_default_key_with_cfg(self):
        """Legacy 'default' key with cfg attributes maps to '*'."""
        raw = [{"default": {"num_bits": 8, "axis": None}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["cfg"] == {"num_bits": 8, "axis": None}
        assert result[0]["enable"] is True

    def test_legacy_flat_dict_with_default_key(self):
        """Legacy flat dict containing 'default' key converts it to '*'."""
        raw = {"default": {"enable": False}, "*weight_quantizer": {"num_bits": 8}}
        result = normalize_quant_cfg_list(raw)
        default_entries = [e for e in result if e["quantizer_name"] == "*"]
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
        paths = {e["quantizer_name"] for e in result}
        assert paths == {"*input_quantizer", "*weight_quantizer"}
        for e in result:
            assert e["parent_class"] == "nn.Linear"

    def test_legacy_nn_class_with_cfg(self):
        """Legacy nn.* scoped format with actual quantizer attributes (not just enable)."""
        raw = [{"nn.Linear": {"*weight_quantizer": {"num_bits": 4, "axis": 0}}}]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["parent_class"] == "nn.Linear"
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert result[0]["cfg"] == {"num_bits": 4, "axis": 0}
        assert result[0]["enable"] is True

    def test_legacy_list_valued_cfg(self):
        """Legacy dict format with list-valued cfg (SequentialQuantizer) normalizes correctly."""
        raw = [
            {
                "*weight_quantizer": [
                    {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                    {"num_bits": 8, "axis": 0},
                ]
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert isinstance(result[0]["cfg"], list)
        assert len(result[0]["cfg"]) == 2
        assert result[0]["cfg"][0]["num_bits"] == 4
        assert result[0]["cfg"][1]["num_bits"] == 8
        assert result[0]["enable"] is True


class TestFindQuantCfgEntry:
    def test_finds_last_match(self):
        """When multiple entries share the same quantizer_name, returns the last one."""
        entries = normalize_quant_cfg_list(
            [
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
                {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 4}},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        result = find_quant_cfg_entry_by_path(entries, "*weight_quantizer")
        assert result["cfg"] == {"num_bits": 4}

    def test_exact_match_only(self):
        """Does not do fnmatch — only exact string equality on quantizer_name."""
        entries = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        with pytest.raises(KeyError):
            find_quant_cfg_entry_by_path(entries, "model.layer.weight_quantizer")

    def test_raises_on_missing(self):
        """Raises KeyError when no entry matches."""
        entries = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        with pytest.raises(KeyError):
            find_quant_cfg_entry_by_path(entries, "*input_quantizer")

    def test_single_entry(self):
        entries = normalize_quant_cfg_list([{"quantizer_name": "*", "enable": False}])
        result = find_quant_cfg_entry_by_path(entries, "*")
        assert result["enable"] is False

    def test_empty_list(self):
        with pytest.raises(KeyError):
            find_quant_cfg_entry_by_path([], "*")


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


class TestMatchQuantizerCfg:
    """Tests for _match_quantizer_cfg in algorithms.py."""

    def test_wildcard_matches_bare_name(self):
        """'*weight_quantizer' matches bare 'weight_quantizer'."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched == {"num_bits": 8}
        assert enable is True

    def test_star_matches_any_bare_name(self):
        """'*' matches any bare quantizer name."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list([{"quantizer_name": "*", "enable": False}])
        matched, enable = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched is None  # enable-only entry has cfg=None
        assert enable is False

    def test_path_scoped_pattern_matches_matching_suffix(self):
        """'*mlp*weight_quantizer' matches bare 'weight_quantizer' (suffix match)."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*mlp*weight_quantizer", "cfg": {"num_bits": 4}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched == {"num_bits": 4}

    def test_path_scoped_pattern_does_not_match_different_suffix(self):
        """'*mlp*weight_quantizer' does NOT match bare 'input_quantizer'."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*mlp*weight_quantizer", "cfg": {"num_bits": 4}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "input_quantizer")
        assert matched is None
        assert enable is None

    def test_last_match_wins(self):
        """Later entries override earlier ones."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        matched, _ = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched == {"num_bits": 4}

    def test_no_match_returns_none(self):
        """No matching entry returns (None, None)."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "output_quantizer")
        assert matched is None
        assert enable is None

    def test_bracket_pattern_matches_correctly(self):
        """'*[kv]_bmm_quantizer' matches 'k_bmm_quantizer' and 'v_bmm_quantizer'."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*[kv]_bmm_quantizer", "cfg": {"num_bits": (4, 3)}}]
        )
        matched_k, _ = _match_quantizer_cfg(quant_cfg, "k_bmm_quantizer")
        matched_v, _ = _match_quantizer_cfg(quant_cfg, "v_bmm_quantizer")
        matched_w, _ = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched_k is not None
        assert matched_v is not None
        assert matched_w is None

    def test_path_scoped_does_not_overmatch(self):
        """'*mixer*weight_quantizer' should NOT match 'input_quantizer'.

        Regression test: the old rsplit('*') logic would strip to 'weight_quantizer' and
        overmatch any quantizer ending in 'weight_quantizer', but should not match unrelated names.
        """
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*mixer*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        # input_quantizer should only match the disable-all, not the mixer pattern
        matched, enable = _match_quantizer_cfg(quant_cfg, "input_quantizer")
        assert matched is None  # cfg is None (enable-only entry)
        assert enable is False


class TestQuantizeConfigValidators:
    """Tests for QuantizeConfig Pydantic field validators."""

    def test_normalize_validator_converts_legacy_dict(self):
        """The 'before' validator auto-normalizes legacy dict format."""
        cfg = QuantizeConfig(
            quant_cfg={"*": {"enable": False}, "*weight_quantizer": {"num_bits": 8}},
            algorithm="max",
        )
        assert isinstance(cfg.quant_cfg, list)
        assert all("quantizer_name" in e for e in cfg.quant_cfg)

    def test_validate_quant_cfg_entries_catches_invalid_cfg(self):
        """The 'after' validator surfaces QuantizerAttributeConfig errors early."""
        with pytest.raises(ValidationError):
            QuantizeConfig(
                quant_cfg=[
                    {
                        "quantizer_name": "*weight_quantizer",
                        "cfg": {"num_bits": 8, "axis": 0, "block_sizes": {-1: 128}},
                    }
                ],
                algorithm="max",
            )

    def test_validate_quant_cfg_entries_accepts_valid_cfg(self):
        """The 'after' validator passes for valid configs."""
        cfg = QuantizeConfig(
            quant_cfg=[
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
                {"quantizer_name": "*input_quantizer", "enable": False},
            ],
            algorithm="max",
        )
        assert len(cfg.quant_cfg) == 2
