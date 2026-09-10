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

"""YAML fixture-driven tests for the recipe system.

Each YAML file in tests/unit/torch/recipes/fixtures/ is a complete test case:
the recipe itself plus a ``_test:`` metadata block with per-API assertions.

Supported ``_test:`` sections:

  validate:                          # RecipeConfig.model_validate()
    expect_success: true             # validation should succeed
    expect_error: "substring"        # validation should fail with this message

  resolve:                           # resolve_recipe()
    check_has_key: quantize_config   # result must have this key
    check_no_key: export             # result must NOT have this key
    check_algorithm: max             # shorthand for result[quantize_config][algorithm]
    check_quant_cfg_has_keys: [...]  # keys present in quant_cfg
    check_kv_patterns_present: true  # bmm/kv quantizer patterns exist
    check_format_count: N            # number of quantization_formats
    check_config:                    # nested dict value checks
      quantize_config:
        algorithm: max
    check_block_sizes:               # block_sizes specific checks
      path: dotted.path.to.block_sizes
      expected: {-1: 16}

  # Future: plan, execute sections added when those APIs land in PR
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from modelopt.torch.recipes.schema.models import RecipeConfig
from modelopt.torch.recipes.schema.resolver import resolve_recipe

# ---------------------------------------------------------------------------
# Discover YAML fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _discover_fixtures():
    """Load all YAML files with _test metadata."""
    cases = []
    for path in sorted(FIXTURES_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data and data.get("_test"):
            cases.append(pytest.param(data, id=path.stem))
    return cases


FIXTURE_CASES = _discover_fixtures()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(d: dict, dotted_path: str):
    """Navigate a dict by dotted path (e.g., 'quantize_config.quant_cfg.*weight_quantizer')."""
    for key in dotted_path.split("."):
        d = d[key]
    return d


def _assert_dict_subset(actual: dict, expected: dict, path: str = ""):
    """Assert that expected is a subset of actual (recursive)."""
    for key, exp_val in expected.items():
        full_path = f"{path}.{key}" if path else str(key)
        assert key in actual, f"Missing key '{full_path}' in {list(actual.keys())}"
        act_val = actual[key]
        if isinstance(exp_val, dict) and isinstance(act_val, dict):
            _assert_dict_subset(act_val, exp_val, full_path)
        elif isinstance(exp_val, list):
            assert act_val == exp_val, f"At '{full_path}': expected {exp_val}, got {act_val}"
        else:
            assert act_val == exp_val, f"At '{full_path}': expected {exp_val}, got {act_val}"


# ---------------------------------------------------------------------------
# Validate tests — RecipeConfig.model_validate()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_data", FIXTURE_CASES)
def test_validate(fixture_data):
    """Test schema validation for each YAML fixture."""
    test_meta = fixture_data["_test"]
    validate_meta = test_meta.get("validate")
    if validate_meta is None:
        pytest.skip("No validate section in _test")

    recipe_dict = {k: v for k, v in fixture_data.items() if k != "_test"}

    expected_error = validate_meta.get("expect_error")
    if expected_error:
        with pytest.raises(ValidationError) as exc_info:
            RecipeConfig.model_validate(recipe_dict)
        assert expected_error in str(exc_info.value), (
            f"Expected '{expected_error}' in error, got: {exc_info.value}"
        )
        return

    # expect_success (default)
    recipe = RecipeConfig.model_validate(recipe_dict)
    assert recipe is not None


# ---------------------------------------------------------------------------
# Resolve tests — resolve_recipe()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_data", FIXTURE_CASES)
def test_resolve(fixture_data):
    """Test recipe resolution for each YAML fixture."""
    test_meta = fixture_data["_test"]
    resolve_meta = test_meta.get("resolve")
    if resolve_meta is None:
        pytest.skip("No resolve section in _test")

    # Validation errors don't reach resolver
    validate_meta = test_meta.get("validate", {})
    if validate_meta.get("expect_error"):
        pytest.skip("Validation error fixture — no resolve")

    recipe_dict = {k: v for k, v in fixture_data.items() if k != "_test"}
    recipe = RecipeConfig.model_validate(recipe_dict)

    # resolve.expect_error — resolver should raise
    expected_error = resolve_meta.get("expect_error")
    if expected_error:
        with pytest.raises(Exception) as exc_info:
            resolve_recipe(recipe)
        assert expected_error in str(exc_info.value), (
            f"Expected '{expected_error}' in error, got: {exc_info.value}"
        )
        return

    result = resolve_recipe(recipe)

    # check_has_key
    has_key = resolve_meta.get("check_has_key")
    if has_key:
        assert has_key in result, f"Expected key '{has_key}' in result, got {list(result.keys())}"

    # check_no_key
    no_key = resolve_meta.get("check_no_key")
    if no_key:
        assert no_key not in result, f"Key '{no_key}' should not be in result"

    # check_algorithm
    check_algo = resolve_meta.get("check_algorithm")
    if check_algo:
        assert result["quantize_config"]["algorithm"] == check_algo, (
            f"Expected algorithm '{check_algo}', got {result['quantize_config']['algorithm']}"
        )

    # check_quant_cfg_has_keys
    cfg_keys = resolve_meta.get("check_quant_cfg_has_keys")
    if cfg_keys:
        qcfg = result["quantize_config"]["quant_cfg"]
        for key in cfg_keys:
            assert key in qcfg, f"Expected '{key}' in quant_cfg, got {list(qcfg.keys())}"

    # check_kv_patterns_present
    if resolve_meta.get("check_kv_patterns_present"):
        qcfg = result["quantize_config"]["quant_cfg"]
        kv_keys = [k for k in qcfg if "bmm_quantizer" in k or "kv" in k.lower()]
        assert len(kv_keys) > 0, f"Expected KV cache patterns, got none in {list(qcfg.keys())}"

    # check_format_count
    fmt_count = resolve_meta.get("check_format_count")
    if fmt_count is not None:
        formats = result["auto_quantize_kwargs"]["quantization_formats"]
        assert len(formats) == fmt_count, f"Expected {fmt_count} formats, got {len(formats)}"

    # check_config (nested dict subset)
    check_config = resolve_meta.get("check_config")
    if check_config:
        _assert_dict_subset(result, check_config)

    # check_block_sizes (special path-based check)
    check_bs = resolve_meta.get("check_block_sizes")
    if check_bs:
        bs = _get_nested(result, check_bs["path"])
        for key, expected_val in check_bs["expected"].items():
            # YAML loads integer keys as int, but block_sizes may have int or str keys
            actual_val = bs.get(key)
            if actual_val is None:
                actual_val = bs.get(str(key))
            assert actual_val == expected_val, (
                f"block_sizes[{key}]: expected {expected_val}, got {actual_val}"
            )
