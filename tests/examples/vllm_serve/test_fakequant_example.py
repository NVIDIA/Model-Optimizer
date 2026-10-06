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

"""Tests for the vLLM fakequant launcher and PTQ example helpers."""

import builtins
import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.vllm import VllmMLAAttention

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples/vllm_serve"


def _load_example_module(name: str):
    """Import a module from ``examples/vllm_serve/`` by path (not an installed package)."""
    path = _EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fakequant_launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return _load_example_module("vllm_serve_fakequant")


@pytest.fixture
def clean_launcher_env():
    """Keep launch settings from affecting other CLI cases."""
    with patch.dict(os.environ):
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
            "VLLM_DISABLE_COMPILE_CACHE",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT_NAME",
            "MODELOPT_MLFLOW_REQUIRED",
            "MODELOPT_MLFLOW_COMMAND",
            "MODELOPT_MLFLOW_RUN_NAME",
        ):
            os.environ.pop(key, None)
        yield


def _stub_launcher_runtime(monkeypatch, launcher):
    vllm_main = Mock()
    ray_registration = Mock()
    moe_support = Mock(return_value=launcher._vllm_supports_moe_backend())
    monkeypatch.setattr(launcher, "vllm_main", vllm_main)
    monkeypatch.setattr(launcher, "_register_ray_env_vars", ray_registration)
    monkeypatch.setattr(launcher, "_vllm_supports_moe_backend", moe_support)
    return vllm_main, ray_registration, moe_support


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
def test_fakequant_launcher_rejects_unusable_quant_file(
    monkeypatch, clean_launcher_env, extra_args, error
):
    launcher = _load_fakequant_launcher(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["vllm_serve_fakequant.py", "serve", "/models/qwen", *extra_args]
    )
    with pytest.raises(SystemExit, match=error):
        launcher.main()


def test_manual_quantizer_state_and_cfg_prevent_full_state_autodetection(
    monkeypatch, clean_launcher_env, tmp_path
):
    launcher = _load_fakequant_launcher(monkeypatch)
    (tmp_path / "vllm_fq_modelopt_state.pth").touch()
    explicit_quantizer_state = "/explicit/quantizer_state.pth"
    args = SimpleNamespace(
        model=str(tmp_path),
        modelopt_quant_cfg="FP8_DEFAULT_CFG",
        modelopt_kv_quant_cfg=None,
        modelopt_quant_file_path=explicit_quantizer_state,
        modelopt_recipe_path=None,
        modelopt_state_path=None,
    )

    launcher._autodetect_fakequant_paths(args)

    assert args.modelopt_quant_file_path == explicit_quantizer_state
    assert args.modelopt_state_path is None


def test_fakequant_launcher_autodetects_megatron_sidecars(
    monkeypatch, clean_launcher_env, tmp_path
):
    (tmp_path / "quantizer_state.pth").touch()
    (tmp_path / "quant_recipe.yaml").write_text("quantizer: {}")
    launcher = _load_fakequant_launcher(monkeypatch)
    monkeypatch.setattr(launcher, "resolve_mlflow_args", Mock())
    vllm_main, ray_registration, _ = _stub_launcher_runtime(monkeypatch, launcher)
    monkeypatch.setattr(sys, "argv", ["vllm_serve_fakequant.py", str(tmp_path)])

    launcher.main()

    assert os.environ["QUANT_FILE_PATH"] == str(tmp_path / "quantizer_state.pth")
    assert os.environ["RECIPE_PATH"] == str(tmp_path / "quant_recipe.yaml")
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert "--worker-cls" in sys.argv
    assert "fakequant_worker.FakeQuantWorker" in sys.argv
    vllm_main.assert_called_once_with()
    ray_registration.assert_called_once_with()


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
    entrypoints = SimpleNamespace(
        current_arg_parser=Mock(name="current_arg_parser"),
        legacy_arg_parser=Mock(name="legacy_arg_parser"),
    )
    modules = {
        "vllm": SimpleNamespace(__version__="0.30.0"),
        "vllm_mlflow_utils": SimpleNamespace(
            MLFLOW_ENV_VARS=set(),
            add_mlflow_args=Mock(),
            resolve_mlflow_args=Mock(),
        ),
        "vllm.entrypoints.openai.cli_args": SimpleNamespace(
            make_arg_parser=entrypoints.legacy_arg_parser
        ),
        "vllm.entrypoints.cli.serve": SimpleNamespace(
            make_arg_parser=entrypoints.current_arg_parser
        ),
        "vllm.entrypoints.cli.main": SimpleNamespace(main=Mock()),
        "vllm.utils.argparse_utils": SimpleNamespace(FlexibleArgumentParser=Mock()),
    }
    return modules, entrypoints


@pytest.mark.parametrize(
    ("arg_parser_missing", "uses_current"),
    [(None, False), ("vllm.entrypoints.openai.cli_args", True)],
    ids=("legacy", "current"),
)
def test_vllm_serve_entrypoint_layouts(monkeypatch, arg_parser_missing, uses_current):
    """Resolve the serve parser in both supported vLLM module layouts."""
    modules, entrypoints = _launcher_import_modules()
    if arg_parser_missing is not None:
        modules["vllm.entrypoints.openai.cli_args"] = ModuleNotFoundError(name=arg_parser_missing)
    _patch_vllm_imports(monkeypatch, modules)

    launcher = _load_example_module("vllm_serve_fakequant")
    parser = launcher._make_vllm_serve_parser()

    expected_arg_parser = (
        entrypoints.current_arg_parser if uses_current else entrypoints.legacy_arg_parser
    )
    expected_arg_parser.assert_called_once_with(parser)


def test_vllm_serve_main_disables_compile_cache(monkeypatch, clean_launcher_env):
    """A cached torch.compile graph of the same model without the fake quant must not be reused."""
    os.environ["MODELOPT_STATE_PATH"] = "/tmp/modelopt_state.pth"
    launcher = _load_fakequant_launcher(monkeypatch)
    monkeypatch.setattr(launcher, "resolve_mlflow_args", Mock())
    vllm_main, _, _ = _stub_launcher_runtime(monkeypatch, launcher)
    monkeypatch.setattr(sys, "argv", ["vllm_serve_fakequant.py", "serve", "/models/qwen"])

    launcher.main()

    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert os.environ["MODELOPT_STATE_PATH"] == "/tmp/modelopt_state.pth"
    vllm_main.assert_called_once_with()


def test_vllm_serve_entrypoint_dependency_error_propagates(monkeypatch):
    """Do not replace a missing parser dependency with a fallback import error."""
    modules, _ = _launcher_import_modules()
    dependency_error = ModuleNotFoundError(name="vllm_dependency")
    modules["vllm.entrypoints.openai.cli_args"] = dependency_error
    modules["vllm.entrypoints.cli.serve"] = AssertionError("fallback must not be imported")
    _patch_vllm_imports(monkeypatch, modules)

    launcher = _load_example_module("vllm_serve_fakequant")
    with pytest.raises(ModuleNotFoundError) as raised:
        launcher._make_vllm_serve_parser()

    assert raised.value is dependency_error


def test_fakequant_launcher_passes_through_non_serve_commands(monkeypatch, clean_launcher_env):
    os.environ["QUANT_CFG"] = "FP8_DEFAULT_CFG"
    launcher = _load_fakequant_launcher(monkeypatch)
    vllm_main = Mock()
    mlflow_resolution = Mock()
    monkeypatch.setattr(launcher, "vllm_main", vllm_main)
    monkeypatch.setattr(launcher, "resolve_mlflow_args", mlflow_resolution)
    monkeypatch.setattr(sys, "argv", ["vllm_serve_fakequant.py", "launch", "render", "--help"])

    launcher.main()

    assert sys.argv == ["vllm", "launch", "render", "--help"]
    vllm_main.assert_called_once_with()
    mlflow_resolution.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "initial_env", "forwarded", "expected_env"),
    [
        pytest.param(
            ["vllm_serve_fakequant.py", "serve", "/models/qwen", "--port", "8000"],
            {},
            ["serve", "/models/qwen", "--port", "8000"],
            {},
            id="stock-serve",
        ),
        pytest.param(
            [
                "vllm_serve_fakequant.py",
                "--port",
                "8000",
                "/models/qwen",
                "--modelopt-quant-cfg",
                "FP8_DEFAULT_CFG",
                "--modelopt-quant-calib-size",
                "8",
                "--trust-remote-code",
            ],
            {},
            ["serve", "--port", "8000", "/models/qwen", "--trust-remote-code"],
            {"QUANT_CFG": "FP8_DEFAULT_CFG", "QUANT_CALIB_SIZE": "8", "TRUST_REMOTE_CODE": "true"},
            id="implicit-serve-cli-settings",
        ),
        pytest.param(
            ["vllm_serve_fakequant.py", "serve", "/models/qwen"],
            {"QUANT_CFG": "NVFP4_DEFAULT_CFG", "VLLM_DISABLE_COMPILE_CACHE": "0"},
            ["serve", "/models/qwen"],
            {"QUANT_CFG": "NVFP4_DEFAULT_CFG"},
            id="environment-fallback",
        ),
    ],
)
def test_fakequant_launcher_serving_paths(
    monkeypatch, clean_launcher_env, argv, initial_env, forwarded, expected_env
):
    """Serve through stock vLLM or publish settings and select the fakequant worker."""
    os.environ.update(initial_env)
    launcher = _load_fakequant_launcher(monkeypatch)
    mlflow_resolution = Mock()
    monkeypatch.setattr(launcher, "resolve_mlflow_args", mlflow_resolution)
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
        if "VLLM_DISABLE_COMPILE_CACHE" in initial_env:
            assert (
                os.environ["VLLM_DISABLE_COMPILE_CACHE"]
                == initial_env["VLLM_DISABLE_COMPILE_CACHE"]
            )
    else:
        assert "QUANT_CFG" not in os.environ
        assert "MODELOPT_STATE_PATH" not in os.environ
        assert "VLLM_DISABLE_COMPILE_CACHE" not in os.environ
    assert sys.argv == expected_argv
    _, unknown = launcher._make_vllm_serve_parser().parse_known_args(sys.argv[2:])
    assert not unknown
    assert mlflow_resolution.call_args.args[0].model == "/models/qwen"
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
        [
            "vllm_serve_fakequant.py",
            "serve",
            "/models/qwen",
            "--modelopt-quant-cfg",
            "FP8_DEFAULT_CFG",
            *overrides,
        ],
    )

    launcher.main()

    assert sys.argv == ["vllm", "serve", "/models/qwen", *overrides]
    _, unknown = launcher._make_vllm_serve_parser().parse_known_args(sys.argv[2:])
    assert not unknown
    assert os.environ["QUANT_CFG"] == "FP8_DEFAULT_CFG"
    vllm_main.assert_called_once_with()
    ray_registration.assert_called_once_with()


def test_fakequant_launcher_mlflow_uses_effective_cli_settings(monkeypatch, clean_launcher_env):
    """MLflow sees the recipe selected on the CLI, not the earlier environment value."""
    os.environ["RECIPE_PATH"] = "/recipes/old.yaml"
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
            "vllm_serve_fakequant.py",
            "serve",
            "/models/qwen",
            "--mlflow",
            "https://mlflow.example.com",
            "--modelopt-recipe-path",
            "/recipes/nvfp4.yaml",
        ],
    )

    launcher.main()

    assert os.environ["RECIPE_PATH"] == "/recipes/nvfp4.yaml"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"].endswith("/qwen-nvfp4")
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


def _new_attention(cls):
    attention = object.__new__(cls)
    torch.nn.Module.__init__(attention)
    return attention


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
