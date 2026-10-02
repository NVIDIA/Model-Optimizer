# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""End-to-end tests for the vLLM fakequant dynamic modules.

Boots ``vllm.LLM`` on tiny HF models (saved via
``_test_utils.torch.transformers_models``) and runs ``mtq.quantize`` inside the
worker via ``LLM.collective_rpc``. Asserts every ``_QuantVLLM…`` class is
installed and every enabled quantizer ends up with a registered tensor-level
``_amax`` after calibration. Mirrors the
``examples/vllm_serve/fakequant_worker.py`` production path.

Architectures: TinyLlama (Linear + Attention), TinyQwen3MoE (+ FusedMoE),
TinyDeepseekV3 (+ MLAAttention).
"""

from __future__ import annotations

import builtins
import copy
import gc
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _test_utils.torch.transformers_models import (
    create_tiny_deepseek_v3_dir,
    create_tiny_deepseek_v4_config_dir,
    create_tiny_glm5_next_config_dir,
    create_tiny_llama_dir,
    create_tiny_qwen3_moe_dir,
)
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt
from vllm.utils.import_utils import has_deep_gemm

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.config_loader import load_config
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins import vllm as vllm_plugin
from modelopt.torch.quantization.plugins.vllm import (
    _ATTENTION_TYPES,
    VllmMLAAttention,
    _QuantFusedMoEBase,
    _QuantVLLMAttention,
    _VLLMParallelLinear,
    build_vllm_attention_quant_cfg,
    configure_vllm_nvfp4_attention_quantizers,
    disable_compilation,
)
from modelopt.torch.quantization.plugins.vllm_indexer import _QuantVLLMIndexerBase


def _load_example_module(name: str):
    """Import a module from ``examples/vllm_serve/`` by path (not an installed package)."""
    path = Path(__file__).parents[4] / "examples/vllm_serve" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fakequant_launcher(monkeypatch):
    examples_dir = Path(__file__).parents[4] / "examples/vllm_serve"
    monkeypatch.syspath_prepend(str(examples_dir))
    return _load_example_module("vllm_serve_fakequant")


@pytest.fixture
def clean_launcher_env(monkeypatch):
    """Keep launch settings from affecting other CLI cases."""
    for key in (
        "QUANT_CFG",
        "KV_QUANT_CFG",
        "QUANT_FILE_PATH",
        "MODELOPT_STATE_PATH",
        "RECIPE_PATH",
        "QUANT_DATASET",
        "QUANT_CALIB_SIZE",
        "CALIB_BATCH_SIZE",
        "TRUST_REMOTE_CODE",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


def _stub_launcher_runtime(monkeypatch, launcher):
    vllm_main = Mock()
    ray_registration = Mock()
    moe_support = Mock(return_value=launcher._vllm_supports_moe_backend())
    monkeypatch.setattr(launcher, "vllm_main", vllm_main)
    monkeypatch.setattr(launcher, "_register_ray_env_vars", ray_registration)
    monkeypatch.setattr(launcher, "_vllm_supports_moe_backend", moe_support)
    return vllm_main, ray_registration, moe_support


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["serve", "/models/qwen", "--port", "8000"], "/models/qwen"),
        (["serve", "--port", "8000", "/models/qwen"], "/models/qwen"),
        (["serve", "--host", "0.0.0.0", "--model", "/models/qwen"], "/models/qwen"),
    ],
)
def test_fakequant_launcher_finds_model_with_options_before_or_after_it(
    monkeypatch, argv, expected
):
    launcher = _load_fakequant_launcher(monkeypatch)
    assert launcher._find_serve_model(argv) == expected


@pytest.mark.parametrize(
    ("extra_args", "error"),
    [
        (
            ["--modelopt-quant-file-path", "/tmp/quantizer_state.pth"],
            "requires --modelopt-quant-cfg",
        ),
        (
            [
                "--modelopt-state-path",
                "/tmp/modelopt_state.pth",
                "--modelopt-quant-file-path",
                "/tmp/quantizer_state.pth",
            ],
            "cannot be combined with --modelopt-state-path",
        ),
    ],
)
def test_fakequant_launcher_rejects_unusable_quant_file(monkeypatch, extra_args, error):
    for key in (
        "QUANT_CFG",
        "KV_QUANT_CFG",
        "RECIPE_PATH",
        "MODELOPT_STATE_PATH",
        "QUANT_FILE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    launcher = _load_fakequant_launcher(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "/models/qwen", *extra_args])
    with pytest.raises(SystemExit, match=error):
        launcher.main()


def _calibration_worker(
    num_blocks: int,
    *,
    cache_specs=("attention", "mamba"),
    needs_kv_cache_zeroing=True,
):
    cache_groups = [SimpleNamespace(kv_cache_spec=spec) for spec in cache_specs]
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=cache_groups,
                num_blocks=num_blocks,
                needs_kv_cache_zeroing=needs_kv_cache_zeroing,
            )
        )
    )


def _patch_vllm_imports(monkeypatch, modules):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            imported = modules[name]
            if isinstance(imported, BaseException):
                raise imported
            return imported
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _launcher_import_modules():
    """Build isolated vLLM module stubs for launcher parser compatibility tests."""
    legacy_parser = Mock(name="legacy_parser")
    current_parser = Mock(name="current_parser")
    modules = {
        "vllm": SimpleNamespace(__version__="0.30.0"),
        "vllm_mlflow_utils": SimpleNamespace(
            MLFLOW_ENV_VARS=set(),
            add_mlflow_args=Mock(),
            resolve_mlflow_args=Mock(),
        ),
        "vllm.entrypoints.openai.cli_args": SimpleNamespace(make_arg_parser=legacy_parser),
        "vllm.entrypoints.cli.serve": SimpleNamespace(make_arg_parser=current_parser),
        "vllm.entrypoints.cli.main": SimpleNamespace(main=Mock()),
        "vllm.utils.argparse_utils": SimpleNamespace(FlexibleArgumentParser=Mock()),
    }
    return modules, legacy_parser, current_parser


@pytest.mark.parametrize(
    ("missing_module", "use_current"),
    [
        (None, False),
        ("vllm.entrypoints.openai.cli_args", True),
        ("vllm.entrypoints", True),
    ],
)
def test_vllm_serve_parser_layouts(monkeypatch, missing_module, use_current):
    modules, legacy_parser, current_parser = _launcher_import_modules()
    if missing_module is not None:
        modules["vllm.entrypoints.openai.cli_args"] = ModuleNotFoundError(name=missing_module)
    _patch_vllm_imports(monkeypatch, modules)

    launcher = _load_example_module("vllm_serve_fakequant")
    parser = launcher._make_vllm_serve_parser()

    expected = current_parser if use_current else legacy_parser
    expected.assert_called_once_with(parser)


def test_vllm_serve_parser_dependency_error_propagates(monkeypatch):
    modules, _, _ = _launcher_import_modules()
    dependency_error = ModuleNotFoundError(name="vllm_dependency")
    modules["vllm.entrypoints.openai.cli_args"] = dependency_error
    modules["vllm.entrypoints.cli.serve"] = AssertionError("fallback must not be imported")
    _patch_vllm_imports(monkeypatch, modules)

    launcher = _load_example_module("vllm_serve_fakequant")
    with pytest.raises(ModuleNotFoundError) as raised:
        launcher._make_vllm_serve_parser()

    assert raised.value is dependency_error


@pytest.mark.parametrize("quant_cfg", [None, "FP8_DEFAULT_CFG"])
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["vllm", "launch", "render", "--help"], id="launch"),
        pytest.param(["vllm", "--help"], id="top-level-help"),
    ],
)
def test_fakequant_launcher_passes_through_non_serve_commands(monkeypatch, quant_cfg, argv):
    for key in (
        "QUANT_CFG",
        "KV_QUANT_CFG",
        "RECIPE_PATH",
        "MODELOPT_STATE_PATH",
        "QUANT_FILE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    if quant_cfg:
        monkeypatch.setenv("QUANT_CFG", quant_cfg)
    launcher = _load_fakequant_launcher(monkeypatch)
    monkeypatch.setattr(launcher, "resolve_mlflow_args", Mock())
    vllm_main = Mock()
    monkeypatch.setattr(launcher, "vllm_main", vllm_main)
    monkeypatch.setattr(sys, "argv", argv)

    launcher.main()

    assert sys.argv == argv
    vllm_main.assert_called_once_with()


@pytest.mark.parametrize(
    ("argv", "initial_env", "forwarded", "expected_env"),
    [
        pytest.param(
            ["vllm", "serve", "/models/qwen", "--port", "8000"],
            {},
            ["serve", "/models/qwen", "--port", "8000"],
            {},
            id="stock-serve",
        ),
        pytest.param(
            [
                "vllm",
                "serve",
                "/models/qwen",
                "--modelopt-quant-cfg",
                "FP8_DEFAULT_CFG",
                "--modelopt-quant-calib-size",
                "8",
                "--trust-remote-code",
            ],
            {},
            ["serve", "/models/qwen", "--trust-remote-code"],
            {
                "QUANT_CFG": "FP8_DEFAULT_CFG",
                "QUANT_CALIB_SIZE": "8",
                "TRUST_REMOTE_CODE": "true",
            },
            id="cli-settings",
        ),
        pytest.param(
            ["vllm", "serve", "/models/qwen"],
            {"QUANT_CFG": "NVFP4_DEFAULT_CFG"},
            ["serve", "/models/qwen"],
            {"QUANT_CFG": "NVFP4_DEFAULT_CFG"},
            id="environment-fallback",
        ),
        pytest.param(
            ["vllm", "serve", "/models/qwen"],
            {"MODELOPT_STATE_PATH": "/tmp/modelopt_state.pth"},
            ["serve", "/models/qwen"],
            {"MODELOPT_STATE_PATH": "/tmp/modelopt_state.pth"},
            id="state-restore",
        ),
        pytest.param(
            ["vllm_serve_fakequant.py", "/models/qwen", "--modelopt-quant-cfg", "FP8_DEFAULT_CFG"],
            {},
            ["serve", "/models/qwen"],
            {"QUANT_CFG": "FP8_DEFAULT_CFG"},
            id="legacy-direct-script",
        ),
        pytest.param(
            [
                "vllm_serve_fakequant.py",
                "--port",
                "8000",
                "/models/qwen",
                "--modelopt-quant-cfg",
                "FP8_DEFAULT_CFG",
            ],
            {},
            ["serve", "--port", "8000", "/models/qwen"],
            {"QUANT_CFG": "FP8_DEFAULT_CFG"},
            id="legacy-option-leading-direct-script",
        ),
    ],
)
def test_fakequant_launcher_serving_paths(
    monkeypatch, clean_launcher_env, argv, initial_env, forwarded, expected_env
):
    """Serve through stock vLLM or publish settings and select the fakequant worker."""
    for key, value in initial_env.items():
        monkeypatch.setenv(key, value)
    launcher = _load_fakequant_launcher(monkeypatch)
    monkeypatch.setattr(launcher, "resolve_mlflow_args", Mock())
    vllm_main, ray_registration, moe_support = _stub_launcher_runtime(monkeypatch, launcher)
    monkeypatch.setattr(sys, "argv", argv)

    launcher.main()

    expected_argv = ["vllm", *forwarded]
    if expected_env:
        expected_argv.extend(["--worker-cls", "fakequant_worker.FakeQuantWorker"])
        if moe_support.return_value:
            expected_argv.extend(["--moe-backend", "triton"])
        for key, value in expected_env.items():
            assert os.environ[key] == value
        assert os.environ["QUANT_DATASET"] == "cnn_dailymail"
    else:
        assert "QUANT_CFG" not in os.environ
        assert "MODELOPT_STATE_PATH" not in os.environ
    assert sys.argv == expected_argv
    _, unknown = launcher._make_vllm_serve_parser().parse_known_args(sys.argv[2:])
    assert not unknown
    vllm_main.assert_called_once_with()
    assert ray_registration.call_count == bool(expected_env)
    assert moe_support.call_count == bool(expected_env)


@pytest.mark.parametrize(
    "overrides",
    [
        ["--worker-cls=custom.Worker", "--moe-backend=auto"],
        ["--worker-cls", "custom.Worker", "--moe-backend", "auto"],
    ],
    ids=("equals", "separate-values"),
)
def test_fakequant_launcher_preserves_explicit_worker_and_moe_overrides(
    monkeypatch, clean_launcher_env, overrides
):
    launcher = _load_fakequant_launcher(monkeypatch)
    if not launcher._vllm_supports_moe_backend():
        pytest.skip("vLLM does not expose --moe-backend")
    monkeypatch.setattr(launcher, "resolve_mlflow_args", Mock())
    vllm_main, ray_registration, _ = _stub_launcher_runtime(monkeypatch, launcher)
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/qwen", "--modelopt-quant-cfg", "FP8_DEFAULT_CFG", *overrides],
    )

    launcher.main()

    assert sys.argv == ["vllm", "serve", "/models/qwen", *overrides]
    _, unknown = launcher._make_vllm_serve_parser().parse_known_args(sys.argv[2:])
    assert not unknown
    assert os.environ["QUANT_CFG"] == "FP8_DEFAULT_CFG"
    vllm_main.assert_called_once_with()
    ray_registration.assert_called_once_with()


@pytest.mark.parametrize(
    ("settings", "initial_env", "expected_variant", "env_key", "env_value"),
    [
        (
            ["--modelopt-quant-cfg", "FP8_DEFAULT_CFG"],
            {},
            "FP8_DEFAULT_CFG",
            "QUANT_CFG",
            "FP8_DEFAULT_CFG",
        ),
        (
            ["--modelopt-recipe-path", "/recipes/nvfp4.yaml"],
            {},
            "nvfp4",
            "RECIPE_PATH",
            "/recipes/nvfp4.yaml",
        ),
        (
            ["--modelopt-quant-cfg", "FP8_DEFAULT_CFG"],
            {"QUANT_CFG": "INT8_DEFAULT_CFG"},
            "FP8_DEFAULT_CFG",
            "QUANT_CFG",
            "FP8_DEFAULT_CFG",
        ),
        (
            ["--modelopt-recipe-path", "/recipes/nvfp4.yaml"],
            {"RECIPE_PATH": "/recipes/old.yaml"},
            "nvfp4",
            "RECIPE_PATH",
            "/recipes/nvfp4.yaml",
        ),
    ],
    ids=(
        "cli-preset-only",
        "cli-recipe-only",
        "cli-preset-overrides-env",
        "cli-recipe-overrides-env",
    ),
)
def test_fakequant_launcher_mlflow_uses_effective_cli_settings(
    monkeypatch, clean_launcher_env, settings, initial_env, expected_variant, env_key, env_value
):
    """Default experiment naming sees CLI overrides before MLflow resolves them."""
    for key, value in initial_env.items():
        monkeypatch.setenv(key, value)
    launcher = _load_fakequant_launcher(monkeypatch)
    vllm_main, _, _ = _stub_launcher_runtime(monkeypatch, launcher)
    mlflow_utils = sys.modules["vllm_mlflow_utils"]
    monkeypatch.setattr(
        mlflow_utils,
        "resolve_tracking_uri",
        lambda _args, _parser: ("https://mlflow.example.com", True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm",
            "serve",
            "/models/qwen",
            "--mlflow",
            "https://mlflow.example.com",
            *settings,
        ],
    )

    launcher.main()

    assert os.environ[env_key] == env_value
    assert os.environ["MLFLOW_EXPERIMENT_NAME"].endswith(f"/qwen-{expected_variant}")
    vllm_main.assert_called_once_with()


def test_get_calibration_block_count_uses_vllm_028_reservation_helper(monkeypatch):
    """The current vLLM adapter must forward every warmup reservation argument."""
    module = _load_example_module("vllm_ptq_utils")
    reserved_block_count = Mock(return_value=4)
    _patch_vllm_imports(
        monkeypatch,
        {"vllm.v1.worker.gpu.warmup": SimpleNamespace(_reserved_block_count=reserved_block_count)},
    )
    model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(num_lookahead_tokens=3),
        max_model_len=2048,
    )
    kv_cache_spec = object()

    block_count = module._get_calibration_block_count(model_runner)

    assert block_count is not None
    assert block_count(128, kv_cache_spec) == 4
    reserved_block_count.assert_called_once_with(
        128,
        kv_cache_spec,
        num_lookahead_tokens=3,
        max_model_len=2048,
        max_encoder_len=0,
    )


def test_get_calibration_block_count_reserves_one_kpool_tail_block(monkeypatch):
    """Kpool's circular tail cache owns exactly one physical block per request."""
    module = _load_example_module("vllm_ptq_utils")

    class KpoolTailSpec:
        pass

    class UniformTypeKVCacheSpecs:
        def __init__(self, first_spec):
            self.first_spec = first_spec

    reserved_block_count = Mock(return_value=32)
    _patch_vllm_imports(
        monkeypatch,
        {
            "vllm.v1.worker.gpu.warmup": SimpleNamespace(
                _reserved_block_count=reserved_block_count
            ),
            "vllm.v1.kv_cache_interface": SimpleNamespace(
                KpoolTailSpec=KpoolTailSpec,
                UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
            ),
        },
    )
    model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(num_lookahead_tokens=0),
        max_model_len=1024,
    )
    block_count = module._get_calibration_block_count(model_runner)

    assert block_count is not None
    assert block_count(128, KpoolTailSpec()) == 1
    assert block_count(128, UniformTypeKVCacheSpecs(KpoolTailSpec())) == 1
    reserved_block_count.assert_not_called()


def test_get_calibration_block_count_uses_vllm_026_reservation_policy(monkeypatch):
    """The vLLM 0.26 adapter must preserve its cross-attention and Mamba rules."""
    module = _load_example_module("vllm_ptq_utils")

    class CrossAttentionSpec:
        block_size = 16

    class MambaSpec:
        block_size = 16
        mamba_cache_mode = "align"
        num_speculative_blocks = 2

    cdiv = Mock(
        side_effect=lambda numerator, denominator: (numerator + denominator - 1) // denominator
    )
    _patch_vllm_imports(
        monkeypatch,
        {
            "vllm.v1.worker.gpu.warmup": ImportError("0.28 helper unavailable"),
            "vllm.utils.math_utils": SimpleNamespace(cdiv=cdiv),
            "vllm.v1.kv_cache_interface": SimpleNamespace(
                CrossAttentionSpec=CrossAttentionSpec,
                MambaSpec=MambaSpec,
            ),
        },
    )
    model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        max_model_len=2048,
    )

    block_count = module._get_calibration_block_count(model_runner)

    assert block_count is not None
    assert block_count(33, SimpleNamespace(block_size=16)) == 3
    assert block_count(33, CrossAttentionSpec()) == 0
    assert block_count(33, MambaSpec()) == 5
    assert cdiv.call_args_list == [
        ((33, 16),),
        ((0, 16),),
        ((33, 16),),
    ]


def test_allocate_calibration_blocks_assigns_non_null_blocks(monkeypatch):
    """Scratch block tables must use unique non-null blocks for every request and group."""
    module = _load_example_module("vllm_ptq_utils")
    block_count = Mock(side_effect=[1, 2, 2, 1])
    monkeypatch.setattr(
        module,
        "_get_calibration_block_count",
        Mock(return_value=block_count),
    )

    block_tables, blocks_to_zero = module._allocate_calibration_blocks(
        _calibration_worker(num_blocks=7),
        sequence_lengths=[8, 16],
    )

    assert block_tables == [
        ([1], [2, 3]),
        ([4, 5], [6]),
    ]
    assert block_count.call_args_list == [
        ((8, "attention"),),
        ((8, "mamba"),),
        ((16, "attention"),),
        ((16, "mamba"),),
    ]

    scheduler_fields = {field.name for field in module.dataclasses.fields(module.SchedulerOutput)}
    expected_blocks_to_zero = (
        [1, 2, 3, 4, 5, 6] if "new_block_ids_to_zero" in scheduler_fields else None
    )
    assert blocks_to_zero == expected_blocks_to_zero


def test_allocate_calibration_blocks_skips_zeroing_for_attention_only_cache(monkeypatch):
    """Attention-only caches have no block zeroer and must receive an empty zeroing list."""
    module = _load_example_module("vllm_ptq_utils")
    monkeypatch.setattr(
        module,
        "_get_calibration_block_count",
        Mock(return_value=Mock(return_value=1)),
    )

    block_tables, blocks_to_zero = module._allocate_calibration_blocks(
        _calibration_worker(
            num_blocks=4,
            cache_specs=("attention",),
            needs_kv_cache_zeroing=False,
        ),
        sequence_lengths=[8],
    )

    assert block_tables == [([1],)]
    scheduler_fields = {field.name for field in module.dataclasses.fields(module.SchedulerOutput)}
    expected_blocks_to_zero = [] if "new_block_ids_to_zero" in scheduler_fields else None
    assert blocks_to_zero == expected_blocks_to_zero


def test_allocate_calibration_blocks_rejects_insufficient_capacity(monkeypatch):
    """Scratch block allocation must account for block 0 being unavailable."""
    module = _load_example_module("vllm_ptq_utils")
    monkeypatch.setattr(
        module,
        "_get_calibration_block_count",
        Mock(return_value=Mock(side_effect=[1, 2, 2, 1])),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Calibration batch requires 6 KV cache blocks, "
            r"but only 5 non-null blocks are available\."
        ),
    ):
        module._allocate_calibration_blocks(
            _calibration_worker(num_blocks=6),
            sequence_lengths=[8, 16],
        )


@pytest.mark.parametrize("has_calibration_error", [False, True])
def test_cleanup_failure_preserves_calibration_error(has_calibration_error):
    """Cleanup must fail closed without replacing an active calibration error."""
    module = _load_example_module("vllm_ptq_utils")
    execute_error = RuntimeError("scheduler cleanup failed")
    finish_error = RuntimeError("legacy cleanup failed")
    calibration_error = ValueError("calibration failed") if has_calibration_error else None
    worker = SimpleNamespace(
        execute_model=Mock(side_effect=execute_error),
        model_runner=SimpleNamespace(finish_requests=Mock(side_effect=finish_error)),
    )

    expected_error = calibration_error or finish_error
    with pytest.raises(type(expected_error)) as raised:
        module._cleanup_calibration_requests(worker, object(), calibration_error)

    assert raised.value is expected_error
    if calibration_error is not None:
        assert calibration_error.__cause__ is finish_error
    assert finish_error.__cause__ is execute_error


@pytest.mark.parametrize("has_calibration_error", [False, True])
def test_cleanup_without_legacy_fallback_preserves_primary_error(has_calibration_error):
    """Missing legacy cleanup must preserve the most useful primary error."""
    module = _load_example_module("vllm_ptq_utils")
    execute_error = RuntimeError("scheduler cleanup failed")
    calibration_error = ValueError("calibration failed") if has_calibration_error else None
    worker = SimpleNamespace(
        execute_model=Mock(side_effect=execute_error),
        model_runner=SimpleNamespace(),
    )

    expected_error = calibration_error or execute_error
    with pytest.raises(type(expected_error)) as raised:
        module._cleanup_calibration_requests(worker, object(), calibration_error)

    assert raised.value is expected_error
    if calibration_error is not None:
        assert calibration_error.__cause__ is execute_error


def test_cleanup_uses_legacy_fallback():
    """A successful legacy cleanup may recover from an unsupported scheduler step."""
    module = _load_example_module("vllm_ptq_utils")
    cleanup_output = object()
    finish_requests = Mock()
    worker = SimpleNamespace(
        execute_model=Mock(side_effect=RuntimeError("unsupported scheduler cleanup")),
        model_runner=SimpleNamespace(finish_requests=finish_requests),
    )

    module._cleanup_calibration_requests(worker, cleanup_output, calibration_error=None)

    finish_requests.assert_called_once_with(cleanup_output)


class _NativeAttention(torch.nn.Module):
    def forward(self, query, key, value, *args, **kwargs):
        return query, key, value


class _TestQuantVLLMAttention(_QuantVLLMAttention, _NativeAttention):
    pass


def _new_attention(cls):
    attention = object.__new__(cls)
    torch.nn.Module.__init__(attention)
    return attention


def _nvfp4_quantizer(*, block_size=16, enabled=True):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: block_size, "type": "dynamic", "scale_bits": (4, 3)},
            enable=enabled,
        )
    )
    return quantizer


def test_attention_setup_keeps_qkv_only_checkpoint_surface(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = _new_attention(_TestQuantVLLMAttention)

    attention._setup()

    quantizer_names = ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")
    assert set(dict(attention.named_children())) == set(quantizer_names)
    for name in quantizer_names:
        getattr(attention, name).amax = torch.tensor(1.0)
    assert set(attention.state_dict()) == {f"{name}._amax" for name in quantizer_names}
    assert not hasattr(attention, "_query_quant_in_kernel")
    assert not hasattr(attention, "_value_quant_in_kernel")

    attention.k_bmm_quantizer = _nvfp4_quantizer()
    attention.v_bmm_quantizer = _nvfp4_quantizer()
    attention.device, attention.dtype = torch.device("cpu"), torch.float32
    attention.modelopt_post_restore()
    assert not hasattr(attention.k_bmm_quantizer, "_amax")
    assert not hasattr(attention.v_bmm_quantizer, "_amax")


def test_configure_vllm_nvfp4_attention_quantizers_is_attention_scoped(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = object.__new__(vllm_plugin.vllm_attention.Attention)
    torch.nn.Module.__init__(attention)
    linear = torch.nn.Linear(4, 4)
    attention.unrelated_linear = linear
    original_linear_type = type(linear)

    converted = configure_vllm_nvfp4_attention_quantizers(
        attention, device="cpu", dtype=torch.bfloat16
    )

    assert converted is attention
    assert isinstance(converted, _QuantVLLMAttention)
    assert converted.device == torch.device("cpu")
    assert converted.dtype == torch.bfloat16
    assert type(linear) is original_linear_type
    for name in ("q", "k", "p", "v"):
        quantizer = getattr(converted, f"{name}_bmm_quantizer")
        assert quantizer.is_enabled
        assert quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    assert not hasattr(converted.q_bmm_quantizer, "_amax")
    assert not hasattr(converted.p_bmm_quantizer, "_amax")
    assert converted.k_bmm_quantizer._amax == 6.0 * 448.0
    assert converted.v_bmm_quantizer._amax == 6.0 * 448.0
    assert not hasattr(converted, "_query_quant_in_kernel")
    assert not hasattr(converted, "_value_quant_in_kernel")


def test_configure_vllm_nvfp4_attention_quantizers_preserves_and_moves_amax(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = object.__new__(vllm_plugin.vllm_attention.Attention)
    torch.nn.Module.__init__(attention)
    converted = configure_vllm_nvfp4_attention_quantizers(
        attention, device="cpu", dtype=torch.float16
    )
    for name, value in zip(("q", "k", "p", "v"), (13.0, 17.0, 23.0, 19.0), strict=True):
        getattr(converted, f"{name}_bmm_quantizer").amax = torch.tensor(value)
    reconfigured = configure_vllm_nvfp4_attention_quantizers(
        converted, device="cpu", dtype=torch.float16
    )

    assert reconfigured is converted
    assert converted.q_bmm_quantizer._amax == 13.0
    assert converted.k_bmm_quantizer._amax == 17.0
    assert converted.p_bmm_quantizer._amax == 23.0
    assert converted.v_bmm_quantizer._amax == 19.0

    configure_vllm_nvfp4_attention_quantizers(converted, device="meta", dtype=torch.float16)
    for name in ("q", "k", "p", "v"):
        assert getattr(converted, f"{name}_bmm_quantizer")._amax.device.type == "meta"


def test_quant_vllm_attention_forward_skips_only_in_kernel_qv_quantization():
    attention = _new_attention(_TestQuantVLLMAttention)
    attention.q_bmm_quantizer = Mock(side_effect=lambda inputs: inputs + 1)
    attention.k_bmm_quantizer = Mock(side_effect=lambda inputs: inputs + 2)
    attention.v_bmm_quantizer = Mock(side_effect=lambda inputs: inputs + 3)
    query = torch.tensor(10)
    key = torch.tensor(20)
    value = torch.tensor(30)

    assert not hasattr(attention, "_query_quant_in_kernel")
    assert not hasattr(attention, "_value_quant_in_kernel")
    quantized = attention(query, key, value)
    attention._query_quant_in_kernel = True
    query_in_kernel = attention(query, key, value)
    attention._value_quant_in_kernel = True
    qv_in_kernel = attention(query, key, value)

    assert quantized[:3] == (torch.tensor(11), torch.tensor(22), torch.tensor(33))
    assert query_in_kernel[:3] == (query, torch.tensor(22), torch.tensor(33))
    assert qv_in_kernel[:3] == (query, torch.tensor(22), value)
    assert attention.q_bmm_quantizer.call_count == 1
    assert attention.k_bmm_quantizer.call_count == 3
    assert attention.v_bmm_quantizer.call_count == 2


def test_disable_compilation_warns_without_installing_marker():
    """A non-compile-wrapped model remains unchanged while the no-op risk is visible."""
    model = torch.nn.Module()

    with pytest.warns(UserWarning, match="rerun with --enforce-eager"), disable_compilation(model):
        assert not hasattr(model, "do_not_compile")

    assert not hasattr(model, "do_not_compile")


def test_disable_compilation_updates_all_markers_and_restores_after_error():
    """Every language and vision compile wrapper is restored after an exceptional exit."""

    class CompileWrappedModule(torch.nn.Module):
        do_not_compile = False

    model = CompileWrappedModule()
    model.do_not_compile = False
    model.vision_model = CompileWrappedModule()
    model.vision_model.do_not_compile = True
    model.language_model = CompileWrappedModule()

    with pytest.raises(RuntimeError, match="quantization failed"), disable_compilation(model):
        assert model.do_not_compile is True
        assert model.vision_model.do_not_compile is True
        assert model.language_model.do_not_compile is True
        raise RuntimeError("quantization failed")

    assert model.do_not_compile is False
    assert model.vision_model.do_not_compile is True
    assert model.language_model.do_not_compile is False
    assert "do_not_compile" not in vars(model.language_model)


def test_attention_kv_defaults_set_only_uncalibrated_dynamic_block16_quantizers():
    calibrated_amax = 7.25
    layer = SimpleNamespace(
        q_bmm_quantizer=_nvfp4_quantizer(),
        k_bmm_quantizer=_nvfp4_quantizer(),
        v_bmm_quantizer=_nvfp4_quantizer(),
        p_bmm_quantizer=_nvfp4_quantizer(),
    )
    layer.v_bmm_quantizer.amax = calibrated_amax

    vllm_plugin._set_vllm_attention_kv_default_amax(layer, torch.device("cpu"))

    assert layer.k_bmm_quantizer._amax.item() == 6.0 * 448.0
    assert layer.v_bmm_quantizer._amax.item() == calibrated_amax
    assert not hasattr(layer.q_bmm_quantizer, "_amax")
    assert not hasattr(layer.p_bmm_quantizer, "_amax")


def test_attention_kv_defaults_ignore_unsupported_quantizers():
    for quantizer in (
        TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3))),
        _nvfp4_quantizer(block_size=32),
        _nvfp4_quantizer(enabled=False),
    ):
        layer = SimpleNamespace(k_bmm_quantizer=quantizer, v_bmm_quantizer=quantizer)
        vllm_plugin._set_vllm_attention_kv_default_amax(layer, torch.device("cpu"))
        assert not hasattr(quantizer, "_amax")


def test_get_device_dtype_ignores_kv_cache_dtype():
    """The dtype is the layer's compute dtype, whatever the KV-cache format (--kv-cache-dtype)."""

    def attention_like(**attrs):
        module = torch.nn.Module()
        module.register_buffer("_k_scale", torch.tensor(1.0))  # vLLM's float32 KV scales
        module.kv_cache = torch.zeros(2, 16, 8, dtype=torch.uint8)
        for name, value in attrs.items():
            setattr(module, name, value)
        return module

    def linear_like(weight_dtype):
        # vLLM linears keep the model dtype in ``params_dtype``, also with pre-quantized weights.
        linear = torch.nn.Linear(4, 4).to(weight_dtype)
        linear.params_dtype = torch.bfloat16
        return linear

    attention = attention_like(dtype=torch.bfloat16)  # vLLM Attention: dtype but no device attr
    mla = attention_like(kv_b_proj=linear_like(torch.bfloat16))  # MLAAttention: neither
    mla_fp8 = attention_like(kv_b_proj=linear_like(torch.float8_e4m3fn))  # FP8 checkpoint
    for cache_dtype in ("auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_ds_mla"):
        for module in (attention, mla, mla_fp8):
            module.kv_cache_dtype = cache_dtype
            assert vllm_plugin._get_device_dtype(module) == (torch.device("cpu"), torch.bfloat16)


class _PrequantizedMethod:
    """Stands in for a vLLM real-quant method such as ``Fp8LinearMethod``."""

    def apply(self, layer, x, bias=None):
        return x + 1


class _NativeLinear(torch.nn.Module):
    def forward(self, input_):
        return self.quant_method.apply(self, input_)


class _TestQuantVLLMLinear(_VLLMParallelLinear, _NativeLinear):
    pass


class _TestQuantFusedMoE(_QuantFusedMoEBase):
    pass


def _packed_weight(*shape):
    # An int32 packed weight: any float round trip (e.g. a disabled-quantizer fold) corrupts it.
    return torch.nn.Parameter(
        torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32), requires_grad=False
    )


def _prequantized_module(monkeypatch, cls, prefix, **weights):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    module = _new_attention(cls)
    module.prefix = prefix
    module.quant_method = _PrequantizedMethod()
    for name, weight in weights.items():
        setattr(module, name, weight)
    module._setup()
    return module


def test_prequantized_linear_passes_through(monkeypatch):
    """A layer of a pre-quantized (e.g. FP8) checkpoint runs untouched under a KV-only config."""
    linear = _prequantized_module(
        monkeypatch,
        _TestQuantVLLMLinear,
        "model.layers.0.mlp.down_proj",
        weight=_packed_weight(4, 4),
    )
    for name in _VLLMParallelLinear._QUANTIZER_NAMES:
        getattr(linear, name).disable()
    weight = linear.weight.detach().clone()

    assert linear._prequantized
    assert torch.equal(linear(torch.zeros(2, 4)), torch.ones(2, 4))
    assert isinstance(linear.quant_method, _PrequantizedMethod)
    assert list(linear.iter_weights_for_calibration()) == []
    linear.fold_weight()
    assert torch.equal(linear.weight, weight)


def test_prequantized_linear_rejects_enabled_quantizers(monkeypatch):
    linear = _prequantized_module(
        monkeypatch,
        _TestQuantVLLMLinear,
        "model.layers.0.mlp.down_proj",
        weight=_packed_weight(4, 4),
    )
    linear.input_quantizer.disable()
    with pytest.raises(
        RuntimeError,
        match=r"model\.layers\.0\.mlp\.down_proj uses vLLM's _PrequantizedMethod.*weight_quantizer",
    ):
        linear(torch.zeros(2, 4))

    unquantized = _new_attention(_TestQuantVLLMLinear)
    unquantized.quant_method = vllm_plugin.vllm_linear.UnquantizedLinearMethod()
    unquantized._setup()
    assert not unquantized._prequantized


def test_prequantized_fused_moe_passes_through(monkeypatch):
    moe = _prequantized_module(
        monkeypatch,
        _TestQuantFusedMoE,
        "model.layers.1.mlp.experts",
        w13_weight=_packed_weight(2, 8, 4),
        w2_weight=_packed_weight(2, 4, 4),
    )
    for name in _QuantFusedMoEBase._QUANTIZER_NAMES:
        getattr(moe, name).disable()
    kernels = [getattr(module, name) for module, name in vllm_plugin._FUSED_MOE_KERNEL_TARGETS]
    w13, w2 = moe.w13_weight.detach().clone(), moe.w2_weight.detach().clone()

    assert moe._prequantized
    with moe._fakequant_moe_kernels():
        # The real-quant experts keep vLLM's own kernels.
        assert [getattr(m, n) for m, n in vllm_plugin._FUSED_MOE_KERNEL_TARGETS] == kernels
    assert list(moe.iter_weights_for_calibration()) == []
    moe.fold_weight()
    assert torch.equal(moe.w13_weight, w13)
    assert torch.equal(moe.w2_weight, w2)

    moe.w13_input_quantizer.enable()
    with pytest.raises(RuntimeError, match="w13_input_quantizer"), moe._fakequant_moe_kernels():
        pass


@pytest.mark.skipif(VllmMLAAttention is None, reason="this vLLM has no MLAAttention")
@pytest.mark.parametrize(
    ("rope_dim", "expected"),
    [(64, ["*kv_c_bmm_quantizer", "*k_pe_bmm_quantizer"]), (0, ["*kv_c_bmm_quantizer"])],
    ids=("rope", "nope"),
)
def test_update_kv_cfg_for_mla_extends_kv_presets(rope_dim, expected):
    """KV_QUANT_CFG presets reach the MLA quantizers; NoPE models (GLM-5.3-Flash) have no RoPE key."""
    ptq_utils = _load_example_module("vllm_ptq_utils")
    mla = _new_attention(VllmMLAAttention)
    mla.qk_rope_head_dim = rope_dim
    kv_cfg = copy.deepcopy(mtq.NVFP4_KV_CFG["quant_cfg"])
    base_cfg = kv_cfg[0]["cfg"]

    updated = ptq_utils.update_kv_cfg_for_mla(torch.nn.Sequential(mla), kv_cfg)

    assert [entry["quantizer_name"] for entry in updated[1:]] == expected
    assert all(entry["cfg"] == base_cfg and entry["enable"] for entry in updated[1:])


@pytest.mark.skipif(VllmMLAAttention is None, reason="this vLLM has no MLAAttention")
def test_update_kv_cfg_for_mla_skips_non_mla_and_warns_on_affine():
    ptq_utils = _load_example_module("vllm_ptq_utils")
    kv_cfg = copy.deepcopy(mtq.NVFP4_AFFINE_KV_CFG["quant_cfg"])
    assert ptq_utils.update_kv_cfg_for_mla(torch.nn.Linear(2, 2), copy.deepcopy(kv_cfg)) == kv_cfg

    mla = _new_attention(VllmMLAAttention)
    mla.qk_rope_head_dim = 0
    with pytest.warns(UserWarning, match="affine"):
        updated = ptq_utils.update_kv_cfg_for_mla(torch.nn.Sequential(mla), copy.deepcopy(kv_cfg))
    # The MLA latent gets the base (non-affine) format.
    assert updated[-1] == {
        "quantizer_name": "*kv_c_bmm_quantizer",
        "cfg": kv_cfg[0]["cfg"],
        "enable": True,
    }


class _NativeMLAAttention(torch.nn.Module):
    def forward(self, query, kv_c, k_pe, *args, **kwargs):
        return query, kv_c, k_pe


@pytest.mark.skipif(VllmMLAAttention is None, reason="this vLLM has no MLAAttention")
@pytest.mark.parametrize("rope_dim", [0, 64], ids=("nope", "rope"))  # GLM-5.3-Flash, DeepSeek-V3
def test_kv_nvfp4_mla_unit_quantizes_the_mla_kv_cache(monkeypatch, rope_dim):
    """Like vLLM's ``nvfp4_ds_mla`` cache: an NVFP4 latent and an unscaled FP8 RoPE key ``k_pe``.
    An empty NoPE ``k_pe`` passes through."""
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )

    class _TestQuantVLLMMLAAttention(vllm_plugin._QuantVLLMMLAAttention, _NativeMLAAttention):
        pass

    mla = _new_attention(_TestQuantVLLMMLAAttention)
    mla._setup()
    set_quantizer_by_cfg(
        mla,
        [{"quantizer_name": "*", "enable": False}, *load_config("configs/ptq/units/kv_nvfp4_mla")],
    )

    assert mla.kv_c_bmm_quantizer.is_enabled
    assert mla.kv_c_bmm_quantizer.amax == 6.0 * 448.0
    assert mla.k_pe_bmm_quantizer.is_enabled
    assert not mla.q_bmm_quantizer.is_enabled

    mla.to("cuda")
    query = torch.randn(5, 4, 512, device="cuda", dtype=torch.bfloat16)
    kv_c = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.randn(5, 1, rope_dim, device="cuda", dtype=torch.bfloat16)
    out_query, out_kv_c, out_k_pe = mla(query, kv_c, k_pe)
    assert out_query is query
    assert not torch.equal(out_kv_c, kv_c)
    # NVFP4 with a fixed global scale is idempotent: the output is already on the grid.
    assert torch.equal(mla.kv_c_bmm_quantizer(out_kv_c), out_kv_c)
    if rope_dim:
        assert not torch.equal(out_k_pe, k_pe)
        assert torch.equal(out_k_pe, k_pe.to(torch.float8_e4m3fn).to(torch.bfloat16))
    else:
        assert out_k_pe is k_pe


@pytest.mark.parametrize(
    ("name", "shape"),
    [("kv_c_bmm_quantizer", (8, 512)), ("k_pe_bmm_quantizer", (8, 1, 64))],
    ids=("kv_c", "k_pe"),
)
def test_kv_nvfp4_mla_quantizer_replays_in_cuda_graph(name, shape):
    """The latent's constant amax is created on the CPU; on the GPU (FakeQuantWorker moves every
    quantizer there before CUDA graph capture) each fake quant of the unit captures and replays
    like eager."""
    layer = torch.nn.Module()
    setattr(layer, name, TensorQuantizer())
    set_quantizer_by_cfg(layer, load_config("configs/ptq/units/kv_nvfp4_mla"))
    quantizer = getattr(layer, name)
    if name == "kv_c_bmm_quantizer":
        assert quantizer._amax.device.type == "cpu"
    quantizer.to("cuda")

    static_in = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantizer(static_in)  # compile the kernel before capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = quantizer(static_in)

    new_in = torch.randn_like(static_in)
    static_in.copy_(new_in)
    graph.replay()
    assert torch.equal(static_out, quantizer(new_in))


def _quantize_and_summarize(self):
    """Run on the worker via ``LLM.collective_rpc``.

    Module-level so it survives pickle over engine-core IPC. ``self`` is the
    vLLM worker — needed to drive ``model_runner._dummy_run`` from the
    calibration forward_loop. Returns a JSON-able summary.
    """
    model = self.get_model()

    def _forward_loop(_model):
        # ``num_tokens=1`` is enough for the ``"max"`` calibrator.
        self.model_runner._dummy_run(1)

    with disable_compilation(model):
        mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=_forward_loop)

    parallel_linear_counts: dict[str, int] = {}
    moe_count = 0
    attention_count = 0
    mla_count = 0
    missing_quantizers: list[str] = []
    quantizers_without_amax: list[str] = []
    enabled_quantizer_count = 0

    def _missing(module, name, slots):
        return (
            f"{name}.{slot}"
            for slot in slots
            if not isinstance(getattr(module, slot, None), TensorQuantizer)
        )

    for name, module in model.named_modules():
        if isinstance(module, _VLLMParallelLinear):
            kind = type(module).__name__
            parallel_linear_counts[kind] = parallel_linear_counts.get(kind, 0) + 1
            missing_quantizers.extend(
                _missing(module, name, ("input_quantizer", "weight_quantizer", "output_quantizer"))
            )
        elif isinstance(module, _QuantFusedMoEBase):
            moe_count += 1
            missing_quantizers.extend(
                _missing(
                    module,
                    name,
                    (
                        "w13_input_quantizer",
                        "w2_input_quantizer",
                        "w13_weight_quantizer",
                        "w2_weight_quantizer",
                    ),
                )
            )
        elif VllmMLAAttention is not None and isinstance(module, VllmMLAAttention):
            mla_count += 1
            missing_quantizers.extend(
                _missing(
                    module, name, ("q_bmm_quantizer", "kv_c_bmm_quantizer", "k_pe_bmm_quantizer")
                )
            )
        elif isinstance(module, _ATTENTION_TYPES):
            attention_count += 1
            missing_quantizers.extend(
                _missing(module, name, ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"))
            )

        # Static-amax invariant: every enabled quantizer must own an ``_amax``
        # after calibration. ``kv_b_proj`` is exempt — vLLM's MLA decode path
        # reads its weight directly and never calls its forward.
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            enabled_quantizer_count += 1
            if not hasattr(module, "_amax") and "kv_b_proj" not in name:
                quantizers_without_amax.append(name)

    return {
        "parallel_linear_counts": parallel_linear_counts,
        "moe_count": moe_count,
        "attention_count": attention_count,
        "mla_count": mla_count,
        "missing_quantizers": missing_quantizers,
        "quantizers_without_amax": quantizers_without_amax,
        "enabled_quantizer_count": enabled_quantizer_count,
        "quantizer_names": sorted(
            name for name, m in model.named_modules() if isinstance(m, TensorQuantizer)
        ),
    }


def _boot_llm(model_dir, max_model_len=64, **extra):
    """Construct a vLLM engine on a tiny model.

    MoE fixtures override with ``moe_backend="triton"`` (pins the Triton
    experts kernel whose module-level entries the modelopt plugin patches —
    FlashInfer/TRTLLM kernels bypass them) and ``enable_expert_parallel=True``
    (keeps modelopt's MoE-specific calibration paths live).
    """
    return LLM(
        model=str(model_dir),
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=max_model_len,
        max_num_seqs=1,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        **extra,
    )


def _shutdown_llm(llm):
    del llm
    gc.collect()
    cleanup_dist_env_and_memory(shutdown_ray=False)


@pytest.fixture(scope="module")
def tiny_llama_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_llama")
    # Helper default ``max_position_embeddings=32`` would clash with vLLM's ``max_model_len=64`` set in ``_boot_llm``.
    # head_dim=64 with num_attention_heads=2 is broadly supported by vLLM's attention backends.
    model_dir = create_tiny_llama_dir(
        tmp,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        head_dim=64,
    )
    llm = _boot_llm(model_dir)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


@pytest.fixture(scope="module")
def tiny_qwen3_moe_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_qwen3_moe")
    # head_dim=64 with num_attention_heads=2 is broadly supported by vLLM's attention backends.
    model_dir = create_tiny_qwen3_moe_dir(
        tmp,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        vocab_size=128,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    llm = _boot_llm(model_dir, moe_backend="triton", enable_expert_parallel=True)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


@pytest.fixture(scope="module")
def tiny_deepseek_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_deepseek")
    # vLLM >= 0.26 only accepts a fixed set of MLA dimension triples (its MLA prefill backend
    # selector rejects the helper's 16/16/16 default); use the real DeepSeek 128/64/128 one.
    model_dir = create_tiny_deepseek_v3_dir(
        tmp, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
    )
    llm = _boot_llm(model_dir, moe_backend="triton", enable_expert_parallel=True)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


_INDEXER_FP8_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*indexer_k_quantizer", "cfg": {"num_bits": (4, 3)}, "enable": True},
        {"quantizer_name": "*indexer_q_quantizer", "cfg": {"num_bits": (4, 3)}, "enable": True},
    ],
    "algorithm": "max",
}

# model -> (architecture vLLM must know, tiny checkpoint builder, extra LLM kwargs)
_SPARSE_ATTN_MODELS = {
    "glm5_next": (
        "Glm5NextForCausalLM",
        create_tiny_glm5_next_config_dir,
        {"load_format": "dummy"},
    ),
    # DeepSeek-V4 computes the indexer query only when top-k selection is needed, i.e. beyond
    # compress_ratio * index_topk = 4 * 1024 tokens.
    "deepseek_v4": (
        "DeepseekV4ForCausalLM",
        create_tiny_deepseek_v4_config_dir,
        {"load_format": "dummy", "max_model_len": 4608, "max_num_batched_tokens": 4608},
    ),
}


@pytest.fixture(scope="module", params=list(_SPARSE_ATTN_MODELS))
def tiny_sparse_attn_llm(request, tmp_path_factory):
    """Tiny sparse-attention models with an indexer K cache: GLM-5.3-Flash and DeepSeek-V4-Pro."""
    arch, build, extra = _SPARSE_ATTN_MODELS[request.param]
    if arch not in ModelRegistry.get_supported_archs():
        pytest.skip(f"this vLLM release has no {arch}")
    if not has_deep_gemm():
        pytest.skip("vLLM's sparse-attention indexer needs DeepGEMM")
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("vLLM's sparse-attention indexer backends need Hopper or Blackwell")
    llm = _boot_llm(build(tmp_path_factory.mktemp(request.param)), **extra)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


def _cache_row_signature(kv_cache, chunk=2048):
    """Per-row checksum of the uint8 indexer cache, chunked to avoid copying the KV pool."""
    weights = torch.arange(1, kv_cache.shape[-1] + 1, device=kv_cache.device) * 1000003 % 998244353
    return torch.cat(
        [
            (kv_cache[i : i + chunk].to(torch.int64) * weights).sum(-1)
            for i in range(0, kv_cache.shape[0], chunk)
        ]
    )


def _indexer_cache_rows(kv_cache, mask):
    """Dequantized values and raw fp32 scale bits of the cache rows selected by ``mask``."""
    num_blocks, block_size, row_bytes = kv_cache.shape
    head_dim = row_bytes - 4
    block, pos = mask.nonzero(as_tuple=True)
    flat = kv_cache.view(num_blocks, block_size * row_bytes)
    values = flat[block[:, None], pos[:, None] * head_dim + torch.arange(head_dim).cuda()]
    scales = flat[block[:, None], block_size * head_dim + pos[:, None] * 4 + torch.arange(4).cuda()]
    values = values.contiguous().view(torch.float8_e4m3fn).float()
    return values, scales.contiguous().view(torch.int32).squeeze(-1)


def _calibrate_and_clip_indexer_k(self):
    """Run on the worker: calibrate FP8 indexer q and K quantizers, then clip K to 1/8 of its amax.

    Calibration goes through real scheduled prefills: the fused indexers write their cache only
    when ``attn_metadata`` is set, which a dummy run does not do. The second prompt fills the
    context, so that DeepSeek-V4 selects top-k and computes its query.
    """
    model = self.get_model()
    lengths = (40, self.model_config.max_model_len - 8)
    batches = [{"input_ids": torch.randint(1, 100, (1, n))} for n in lengths]
    forward_loop = _load_example_module("vllm_ptq_utils").calibrate_fun(batches, self)
    with disable_compilation(model):
        mtq.quantize(model, _INDEXER_FP8_CFG, forward_loop=forward_loop)

    amaxes, self.indexer_k_snapshots = {"k": {}, "q": {}}, {}
    for name, module in model.named_modules():
        if isinstance(module, _QuantVLLMIndexerBase):
            for kind in amaxes:
                amax = getattr(module, f"indexer_{kind}_quantizer").amax
                amaxes[kind][name] = None if amax is None else amax.item()
            if amaxes["k"][name]:
                module.indexer_k_quantizer.amax = module.indexer_k_quantizer.amax / 8
            self.indexer_k_snapshots[name] = _cache_row_signature(module.k_cache.kv_cache)
    return amaxes


def _indexer_k_rows_written(self):
    """Run on the worker: rows the indexer kernels wrote since the snapshot, max over the clip."""
    torch.cuda.synchronize()
    result = {}
    for name, module in self.get_model().named_modules():
        if name not in self.indexer_k_snapshots:
            continue
        cache = module.k_cache.kv_cache
        changed = (_cache_row_signature(cache) != self.indexer_k_snapshots[name]).view(
            cache.shape[:2]
        )
        changed[0] = False  # vLLM's null block, where other layers write scratch data
        values, scale_bits = _indexer_cache_rows(cache, changed)
        # Hybrid models alias one KV pool across cache groups; the indexer kernels' own rows have a
        # power-of-two scale and use the FP8 range (or the fixed scale of the 1e-4 amax floor).
        fp8_max = values.abs().amax(-1)
        power_of_two = (scale_bits > 0) & ((scale_bits & 0x7FFFFF) == 0)
        floor_scale = scale_bits == torch.tensor(2.0**-22).view(torch.int32).item()
        kernel_rows = power_of_two & (((fp8_max > 224) & (fp8_max <= 448)) | floor_scale)
        scale = torch.ldexp(torch.ones_like(fp8_max), ((scale_bits >> 23) & 0xFF) - 127)
        row_max = (fp8_max * scale)[kernel_rows]
        clip = module.indexer_k_quantizer.amax.item()
        result[name] = (
            int(kernel_rows.sum()),
            row_max.max().item() / clip if row_max.numel() else 0,
        )
    return result


def _assert_quantizer_amax_is_static(summary):
    """Every enabled quantizer must own a registered ``_amax`` after
    calibration. Missing ``_amax`` → repr ``amax=dynamic`` → regression.
    """
    assert summary["enabled_quantizer_count"] > 0, summary
    assert summary["quantizers_without_amax"] == [], summary["quantizers_without_amax"]


def test_tiny_llama_quantize(tiny_llama_llm):
    """Covers QKV/Row/MergedColumn ParallelLinear + Attention on a dense Llama."""
    summaries = tiny_llama_llm.collective_rpc(_quantize_and_summarize)
    summary = summaries[0]

    assert summary["missing_quantizers"] == [], summary["missing_quantizers"]

    parallel_linear_counts = summary["parallel_linear_counts"]
    # Each decoder layer contributes one of each. With num_hidden_layers=2:
    assert parallel_linear_counts.get("QuantQKVParallelLinear", 0) >= 2, parallel_linear_counts
    # o_proj + down_proj per layer
    assert parallel_linear_counts.get("QuantRowParallelLinear", 0) >= 4, parallel_linear_counts
    assert parallel_linear_counts.get("QuantMergedColumnParallelLinear", 0) >= 2, (
        parallel_linear_counts
    )

    # Llama uses the base Attention type — one per decoder layer.
    assert summary["attention_count"] >= 2, summary

    # No MoE in a dense Llama.
    assert summary["moe_count"] == 0

    _assert_quantizer_amax_is_static(summary)


def test_tiny_qwen3_moe_quantize(tiny_qwen3_moe_llm):
    """Tiny Qwen3-MoE adds FusedMoE coverage on top of the dense linears."""
    summaries = tiny_qwen3_moe_llm.collective_rpc(_quantize_and_summarize)
    summary = summaries[0]

    assert summary["missing_quantizers"] == [], summary["missing_quantizers"]

    parallel_linear_counts = summary["parallel_linear_counts"]
    assert parallel_linear_counts.get("QuantQKVParallelLinear", 0) >= 2, parallel_linear_counts
    assert parallel_linear_counts.get("QuantRowParallelLinear", 0) >= 2, parallel_linear_counts

    # decoder_sparse_step=1 → every layer is MoE. With 2 layers we expect ≥2 FusedMoE.
    assert summary["moe_count"] >= 2, summary
    assert summary["attention_count"] >= 2, summary

    _assert_quantizer_amax_is_static(summary)

    # The vllm_serve reload helper must map HF expert keys onto module paths that exist here:
    # a stale mapping is dropped silently at load and serves uncalibrated experts.
    reload_utils = _load_example_module("vllm_reload_utils")
    for hf_key, expected_quantizer in (
        ("model.layers.0.mlp.experts.0.gate_proj.input_quantizer._amax", "w13_input_quantizer"),
        ("model.layers.0.mlp.experts.0.down_proj.weight_quantizer._amax", "w2_weight_quantizer"),
    ):
        action, vllm_key, _ = reload_utils._convert_key_for_vllm(hf_key, 1.0)
        assert action == "group", (hf_key, action)
        module_path = vllm_key.rsplit("._amax", 1)[0]
        assert module_path.endswith(expected_quantizer), vllm_key
        assert module_path in summary["quantizer_names"], (vllm_key, summary["quantizer_names"])


def test_tiny_deepseek_mla_quantize(tiny_deepseek_llm):
    """Tiny DeepSeek-V3 covers MLAAttention (and again FusedMoE)."""
    summaries = tiny_deepseek_llm.collective_rpc(_quantize_and_summarize)
    summary = summaries[0]

    assert summary["missing_quantizers"] == [], summary["missing_quantizers"]
    assert summary["mla_count"] >= 2, summary
    # ``first_k_dense_replace=0`` → every layer is MoE.
    assert summary["moe_count"] >= 2, summary

    _assert_quantizer_amax_is_static(summary)

    # ``n_shared_experts=1``: vLLM merges the shared expert's gate/up into ``gate_up_proj``, so
    # the reload helper must merge those HF keys too rather than copying them through.
    reload_utils = _load_example_module("vllm_reload_utils")
    action, vllm_key, _ = reload_utils._convert_key_for_vllm(
        "model.layers.0.mlp.shared_experts.gate_proj.input_quantizer._amax", 1.0
    )
    assert action == "group", (action, vllm_key)
    assert vllm_key.rsplit("._amax", 1)[0] in summary["quantizer_names"], vllm_key


@pytest.mark.timeout(600)  # engine boot and the DeepGEMM JIT dominate
def test_tiny_sparse_attn_indexer_quantize(tiny_sparse_attn_llm):
    """The indexer query is fake-quantized and the K cache the kernels read holds QDQ keys."""
    amaxes = tiny_sparse_attn_llm.collective_rpc(_calibrate_and_clip_indexer_k)[0]
    assert amaxes["k"], "no indexer was converted"
    for kind_amaxes in amaxes.values():  # calibrated: the quantizer saw the kernels' tensors
        assert all(a is not None and 0 < a < float("inf") for a in kind_amaxes.values()), amaxes

    # Prefill plus decode steps that complete further pools / compression groups.
    prompts = [TokensPrompt(prompt_token_ids=list(range(1 + i, 41 + i))) for i in range(2)]
    params = SamplingParams(max_tokens=12, ignore_eos=True, temperature=0.0, detokenize=False)
    tiny_sparse_attn_llm.generate(prompts, params)

    for name, (rows, max_over_clip) in tiny_sparse_attn_llm.collective_rpc(_indexer_k_rows_written)[
        0
    ].items():
        assert rows > 0, f"{name}: the serving step wrote no indexer cache rows"
        # Re-storing a clipped key in the FP8 cache rounds it up by at most 2**-4.
        assert max_over_clip <= 1.07, (name, rows, max_over_clip)


def _quantize_kv_preset_and_summarize(self):
    """Worker RPC: ``KV_QUANT_CFG=NVFP4_KV_CFG`` as FakeQuantWorker resolves it, fold, one more
    forward; JSON-able summary."""
    model = self.get_model()
    config = _load_example_module("vllm_ptq_utils").get_quant_config(
        {"recipe_path": None, "quant_cfg": None, "kv_quant_cfg": "NVFP4_KV_CFG"}, model
    )
    with disable_compilation(model):
        mtq.quantize(model, config, forward_loop=lambda _: self.model_runner._dummy_run(1))
    mtq.fold_weight(model)
    self.model_runner._dummy_run(1)

    methods: dict[str, list[str]] = {"passthrough": [], "fake_quant": []}
    enabled = []
    for name, module in model.named_modules():
        if isinstance(module, (_VLLMParallelLinear, _QuantFusedMoEBase)):
            kind = "passthrough" if module._prequantized else "fake_quant"
            methods[kind].append(type(module.quant_method).__name__)
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            enabled.append((name, float(module.amax)))
    return {**methods, "enabled": enabled}


@pytest.fixture
def tiny_deepseek_fp8_llm(tmp_path):
    # vLLM's online FP8 quantization turns the linears and experts into real-quant FP8 layers.
    # kv_lora_rank=512 keeps DeepSeek's 576-wide cache row, which every vLLM MLA backend accepts.
    model_dir = create_tiny_deepseek_v3_dir(
        tmp_path, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
    )
    llm = _boot_llm(
        model_dir, quantization="fp8", moe_backend="triton", enable_expert_parallel=True
    )
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


def test_tiny_deepseek_fp8_kv_quant_cfg(tiny_deepseek_fp8_llm):
    """FP8 layers pass through while the KV_QUANT_CFG preset fake-quantizes the MLA KV cache."""
    summary = tiny_deepseek_fp8_llm.collective_rpc(_quantize_kv_preset_and_summarize)[0]

    assert summary["passthrough"], summary
    assert any("MoE" in method for method in summary["passthrough"]), summary
    enabled = {name.rsplit(".", 1)[-1] for name, _ in summary["enabled"]}
    assert enabled == {"kv_c_bmm_quantizer", "k_pe_bmm_quantizer"}, summary
    assert all(amax > 0 for _, amax in summary["enabled"]), summary


def test_configure_vllm_attention_quantizers_fp8_bmm2(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = object.__new__(vllm_plugin.vllm_attention.Attention)
    torch.nn.Module.__init__(attention)

    converted = configure_vllm_nvfp4_attention_quantizers(
        attention,
        device="cpu",
        dtype=torch.bfloat16,
        cfg=build_vllm_attention_quant_cfg(p_format="fp8", v_format="fp8"),
    )

    # BMM1 unchanged: Q/K dynamic block-16 NVFP4 (F1)
    for name in ("q", "k"):
        quantizer = getattr(converted, f"{name}_bmm_quantizer")
        assert quantizer.is_enabled and quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    assert converted.k_bmm_quantizer._amax == 6.0 * 448.0
    # BMM2: P/V per-tensor FP8 E4M3 with fixed amax (P=1.0, V=448) (F3)
    for name, amax in (("p", 1.0), ("v", 448.0)):
        quantizer = getattr(converted, f"{name}_bmm_quantizer")
        assert quantizer.is_enabled
        assert quantizer.num_bits == (4, 3)
        assert not quantizer.block_sizes
        assert float(quantizer._amax) == amax
    # idempotent: calibrated amax survives reconfiguration
    converted.v_bmm_quantizer.amax = torch.tensor(96.0)
    reconfigured = configure_vllm_nvfp4_attention_quantizers(
        converted,
        device="cpu",
        dtype=torch.bfloat16,
        cfg=build_vllm_attention_quant_cfg(p_format="fp8", v_format="fp8"),
    )
    assert float(reconfigured.v_bmm_quantizer._amax) == 96.0
