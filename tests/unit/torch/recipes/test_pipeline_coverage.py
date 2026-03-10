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

"""Additional coverage tests for pipeline, resolver, bridge, and formats."""

import yaml

from modelopt.torch.recipes.pipeline import PipelinePlan, PipelineStep, plan_pipeline
from modelopt.torch.recipes.schema.models import RecipeConfig


# ── pipeline.py coverage ──


def test_pruning_step():
    """Pruning recipe produces a pruning step."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    pruning:
      mode: gradnas
      constraints:
        flops: 0.5
      calibration:
        dataset: pile
        num_samples: 256
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 1
    assert plan.steps[0].technique == "pruning"
    assert "prune()" in plan.steps[0].api_call
    assert plan.steps[0].config["mode"] == "gradnas"
    assert plan.steps[0].calibration["dataset"] == "pile"


def test_pruning_with_training():
    """Pruning with training config."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    pruning:
      mode: gradnas
      constraints:
        flops: 0.5
      training:
        learning_rate: 1e-4
        num_epochs: 3
    """)
    )
    plan = plan_pipeline(recipe)
    assert plan.steps[0].training["learning_rate"] == 1e-4


def test_auto_quantize_step():
    """Auto-quantize via plan_pipeline."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    auto_quantize:
      effective_bits: 4.5
      formats:
        - preset: fp8
        - preset: nvfp4
      calibration:
        dataset: cnn_dailymail
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 1
    assert plan.steps[0].technique == "auto_quantize"
    assert "auto_quantize()" in plan.steps[0].api_call
    assert plan.steps[0].calibration["dataset"] == "cnn_dailymail"


def test_model_and_metadata_in_plan():
    """Model and metadata fields appear in the plan."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    model:
      path: meta-llama/Llama-3-8B
      trust_remote_code: true
    metadata:
      name: test-recipe
      description: A test recipe
    quantization:
      preset: fp8
    """)
    )
    plan = plan_pipeline(recipe)
    assert plan.model["path"] == "meta-llama/Llama-3-8B"
    assert plan.model["trust_remote_code"] is True
    assert plan.metadata["name"] == "test-recipe"


def test_four_technique_pipeline():
    """All four techniques in correct order."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    pruning:
      mode: gradnas
      constraints: {}
    sparsity:
      method: wanda
    quantization:
      preset: fp8
    distillation:
      teacher: "meta-llama/Llama-3-70B"
      training:
        learning_rate: 1e-5
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 4
    assert [s.technique for s in plan.steps] == [
        "pruning",
        "sparsity",
        "quantization (ptq)",
        "distillation",
    ]


def test_dry_run_with_model_metadata_export():
    """Dry-run output includes model, metadata, and export sections."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    model:
      path: meta-llama/Llama-3-8B
      trust_remote_code: true
      attn_implementation: flash_attention_2
    metadata:
      name: test-recipe
      description: A test recipe
    quantization:
      preset: fp8
      calibration:
        dataset: pile
        num_samples: 128
    export:
      format: tensorrt_llm
      tensor_parallel: 8
      output_dir: ./my-output
    """)
    )
    plan = plan_pipeline(recipe)
    output = plan.dry_run()
    assert "meta-llama/Llama-3-8B" in output
    assert "trust_remote_code: True" in output
    assert "flash_attention_2" in output
    assert "test-recipe" in output
    assert "A test recipe" in output
    assert "tensorrt_llm" in output
    assert "Tensor parallel: 8" in output
    assert "./my-output" in output
    assert "dataset=pile" in output


def test_dry_run_verbose():
    """Verbose dry-run includes resolved config."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
    """)
    )
    plan = plan_pipeline(recipe)
    output = plan.dry_run(verbose=True)
    assert "Resolved config:" in output


def test_dry_run_with_training():
    """Dry-run shows training info."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      mode: qat
      preset: fp8
      training:
        learning_rate: 1e-5
        num_epochs: 2
        max_steps: 100
    """)
    )
    plan = plan_pipeline(recipe)
    output = plan.dry_run()
    assert "Training:" in output
    assert "max_steps=100" in output


def test_make_serializable():
    """_make_serializable converts tuples and nested structures."""
    from modelopt.torch.recipes.pipeline import _make_serializable

    result = _make_serializable({"a": (1, 2), "b": {"c": (3,)}, "d": [4, 5]})
    assert result == {"a": [1, 2], "b": {"c": [3]}, "d": [4, 5]}


# ── resolver.py coverage ──


def test_custom_weights_activations():
    """Custom weights/activations without preset."""
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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


def test_algorithm_string_override():
    """Algorithm override as a plain string."""
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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
    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

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


def test_block_sizes_string_keys():
    """Block sizes with string dimension keys."""
    from modelopt.torch.recipes.schema.resolver import _resolve_block_sizes

    result = _resolve_block_sizes({"last_dim": 16, "second_last_dim": 32, "scale_bits": [4, 3]})
    assert result[-1] == 16
    assert result[-2] == 32
    assert result["scale_bits"] == (4, 3)


def test_block_sizes_passthrough():
    """Block sizes with passthrough keys."""
    from modelopt.torch.recipes.schema.resolver import _resolve_block_sizes

    result = _resolve_block_sizes({"type": "dynamic", "0": 128})
    assert result["type"] == "dynamic"
    assert result[0] == 128


# ── formats.py coverage ──


def test_unknown_format_raises():
    from modelopt.torch.recipes.schema.formats import get_format

    import pytest

    with pytest.raises(KeyError, match="Unknown format"):
        get_format("nonexistent_format")


def test_unknown_kv_format_raises():
    from modelopt.torch.recipes.schema.formats import get_kv_format

    import pytest

    with pytest.raises(KeyError, match="Unknown KV cache format"):
        get_kv_format("nonexistent_kv_format")


# ── bridge.py coverage ──


def test_auto_quantize_bridge():
    """Auto-quantize path in recipe_to_hf_ptq_args."""
    from modelopt.torch.recipes.bridge import recipe_to_hf_ptq_args

    resolved = {"auto_quantize_kwargs": {"quantization_formats": [], "method": "gradient"}}
    result = recipe_to_hf_ptq_args(resolved)
    assert "_resolved_auto_quantize_kwargs" in result


def test_summarize_empty_recipe():
    """Summarize with no quantization or auto_quantize."""
    from modelopt.torch.recipes.bridge import summarize_recipe

    summary = summarize_recipe("test.yaml", {}, {})
    assert summary["type"] == "unknown"


# ── config.py coverage ──


def test_sweep_config_validate():
    """SweepConfig.validate() returns errors for empty config."""
    from modelopt.torch.recipes.experiment.config import SweepConfig

    config = SweepConfig()
    errors = config.validate()
    assert "No models specified" in errors
    assert "No recipes specified" in errors
    assert "No launchers specified" in errors
    assert "No eval tasks specified" in errors


def test_sweep_eval_resolve_with_overrides():
    """SweepEvalConfig.resolve_for_job applies model and launcher overrides."""
    from modelopt.torch.recipes.experiment.config import SweepEvalConfig

    eval_cfg = SweepEvalConfig(
        engine="vllm",
        tensor_parallel_size=8,
        tasks=["mmlu", "gsm8k"],
        model_overrides={
            "code-model": {"tasks": ["humaneval"], "engine": "trtllm", "tensor_parallel_size": 4}
        },
        launcher_overrides={"gb200": {"tensor_parallel_size": 2, "engine": "sglang"}},
    )

    # No overrides
    result = eval_cfg.resolve_for_job("generic-model", "h100")
    assert result["deployment"]["engine"] == "vllm"
    assert result["deployment"]["tensor_parallel_size"] == 8

    # Model override
    result = eval_cfg.resolve_for_job("code-model", "h100")
    assert result["deployment"]["engine"] == "trtllm"
    assert result["deployment"]["tensor_parallel_size"] == 4
    assert result["evaluation"]["tasks"] == [{"name": "humaneval"}]

    # Launcher override takes precedence for tp and engine
    result = eval_cfg.resolve_for_job("generic-model", "gb200")
    assert result["deployment"]["tensor_parallel_size"] == 2
    assert result["deployment"]["engine"] == "sglang"


def test_sweep_eval_benchmark_set():
    """Benchmark set appears in resolved config."""
    from modelopt.torch.recipes.experiment.config import SweepEvalConfig

    eval_cfg = SweepEvalConfig(tasks=["mmlu"], benchmark_set="standard_v2")
    result = eval_cfg.resolve_for_job("model", "launcher")
    assert result["evaluation"]["benchmark_set"] == "standard_v2"
