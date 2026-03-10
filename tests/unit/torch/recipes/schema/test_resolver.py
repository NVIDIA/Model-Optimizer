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

"""Tests for resolver (schema/resolver.py)."""

import yaml

from modelopt.torch.recipes.schema.models import RecipeConfig
from modelopt.torch.recipes.schema.resolver import resolve_recipe


def test_custom_weights_activations():
    """Custom weights/activations without preset."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      weights:
        format: int8
      activations:
        format: int8
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    assert "*weight_quantizer" in qcfg
    assert "*input_quantizer" in qcfg
    assert qcfg["*weight_quantizer"]["num_bits"] == 8


def test_disabled_patterns():
    """disabled_patterns creates disable entries."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      disabled_patterns:
        - "*layers.0*"
        - "*layers.1*"
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    assert qcfg["*layers.0*"] == {"enable": False}
    assert qcfg["*layers.1*"] == {"enable": False}


def test_module_class_override():
    """Module-class override in resolver."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      overrides:
        - module_class: "nn.Linear"
          weights:
            format: int8
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    assert "nn.Linear" in qcfg
    assert "*weight_quantizer" in qcfg["nn.Linear"]


def test_module_class_disable_override():
    """Module-class override with enable=False and no weights/activations."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      overrides:
        - module_class: "nn.Embedding"
          enable: false
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    assert qcfg["nn.Embedding"] == {"*": {"enable": False}}


def test_override_with_format_and_num_bits():
    """Pattern override with format, num_bits, and axis."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      overrides:
        - pattern: "*mlp*weight_quantizer"
          format: int4
          num_bits: 4
          axis: 0
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    entry = qcfg["*mlp*weight_quantizer"]
    assert entry["num_bits"] == 4
    assert entry["axis"] == 0


def test_scale_type_override():
    """scale_type shorthand merges into block_sizes.type, preserving existing block_sizes."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: nvfp4_local_hessian
      overrides:
        - pattern: "*weight_quantizer"
          scale_type: dynamic
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    wq = qcfg["*weight_quantizer"]
    assert wq["block_sizes"]["type"] == "dynamic"
    assert wq["block_sizes"][-1] == 16
    assert wq["block_sizes"]["scale_bits"] in ([4, 3], (4, 3))
    assert wq["num_bits"] in ([2, 1], (2, 1))


def test_scale_type_new_pattern():
    """scale_type on a new pattern creates entry with just scale_type."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: nvfp4_local_hessian
      overrides:
        - pattern: "*self_attn*weight_quantizer"
          scale_type: dynamic
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    attn = qcfg["*self_attn*weight_quantizer"]
    assert attn["block_sizes"]["type"] == "dynamic"


def test_algorithm_string_override():
    """Algorithm override as a plain string."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      algorithm: awq_lite
    """)
    )
    result = resolve_recipe(recipe)
    assert result["quantize_config"]["algorithm"] == "awq_lite"


def test_algorithm_dict_override():
    """Algorithm override as a dict with method + extra fields."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      algorithm:
        method: awq_lite
        alpha_step: 0.1
    """)
    )
    result = resolve_recipe(recipe)
    algo = result["quantize_config"]["algorithm"]
    assert algo["method"] == "awq_lite"
    assert algo["alpha_step"] == 0.1


def test_auto_quantize_with_kv_cache_and_disabled():
    """Auto-quantize with kv_cache and disabled_patterns."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    auto_quantize:
      effective_bits: 4.5
      formats:
        - preset: fp8
      kv_cache:
        format: fp8
      disabled_patterns:
        - "*lm_head*"
    """)
    )
    result = resolve_recipe(recipe)
    kwargs = result["auto_quantize_kwargs"]
    assert "disabled_layers" in kwargs
    assert kwargs["disabled_layers"] == ["*lm_head*"]


def test_block_sizes_string_keys_via_resolve():
    """Block sizes with string dimension keys resolve through public API."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      weights:
        format: nvfp4
        block_sizes:
          last_dim: 16
          second_last_dim: 32
          scale_bits: [4, 3]
      activations:
        enable: false
    """)
    )
    result = resolve_recipe(recipe)
    bs = result["quantize_config"]["quant_cfg"]["*weight_quantizer"]["block_sizes"]
    assert bs[-1] == 16
    assert bs[-2] == 32
    assert bs["scale_bits"] == [4, 3]


def test_block_sizes_passthrough_via_resolve():
    """Block sizes with integer string keys and type passthrough."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      weights:
        block_sizes:
          type: dynamic
          "0": 128
      activations:
        enable: false
    """)
    )
    result = resolve_recipe(recipe)
    bs = result["quantize_config"]["quant_cfg"]["*weight_quantizer"]["block_sizes"]
    assert bs["type"] == "dynamic"
    assert bs[0] == 128


# ── KV cache via resolve_recipe ──


def test_kv_cache_adds_patterns():
    """KV cache section adds quantizer patterns via resolve_recipe."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      kv_cache:
        format: fp8
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    # KV patterns should be present from kv_cache section
    kv_keys = [k for k in qcfg if "bmm_quantizer" in k or "kv" in k.lower()]
    assert len(kv_keys) > 0


def test_kv_cache_sets_algorithm_default():
    """KV cache merge sets algorithm to 'max' when not specified."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      weights:
        format: int8
      activations:
        format: int8
      kv_cache:
        format: fp8
    """)
    )
    result = resolve_recipe(recipe)
    assert result["quantize_config"]["algorithm"] == "max"


def test_kv_cache_preserves_explicit_algorithm():
    """KV cache merge does not overwrite an explicit algorithm."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
      algorithm: awq_lite
      kv_cache:
        format: fp8
    """)
    )
    result = resolve_recipe(recipe)
    assert result["quantize_config"]["algorithm"] == "awq_lite"
