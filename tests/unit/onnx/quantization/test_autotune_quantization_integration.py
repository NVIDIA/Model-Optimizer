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

import contextlib
import importlib
import json
import sys

import pytest


def test_quantization_cli_parser_imports_without_tensorrt():
    """Verify the CLI parser can be constructed without TensorRT installed."""
    with pytest.MonkeyPatch.context() as mp:
        # Force tensorrt import to fail, even if it's actually installed
        mp.setitem(sys.modules, "tensorrt", None)

        # Reload the autotune package so it picks up the blocked import
        import modelopt.onnx.quantization.autotune

        importlib.reload(modelopt.onnx.quantization.autotune)

        from modelopt.onnx.quantization.__main__ import get_parser

        parser = get_parser()
        args = parser.parse_args(["--onnx_path", "dummy.onnx"])
        assert args.onnx_path == "dummy.onnx"
        assert args.quantize_mode == "int8"


def test_quantization_cli_parses_inline_input_shapes_profile():
    from modelopt.onnx.quantization.__main__ import get_parser

    profile = [{"nv_profile_min_shapes": "input_ids:1x1"}, {}]
    args = get_parser().parse_args(
        [
            "--onnx_path",
            "dummy.onnx",
            "--input_shapes_profile",
            json.dumps(profile),
        ]
    )

    assert args.input_shapes_profile == profile


def test_quantization_cli_parses_input_shapes_profile_file(tmp_path):
    from modelopt.onnx.quantization.__main__ import get_parser

    profile = [{"trt_profile_min_shapes": "input_ids:1x1"}, {}]
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    args = get_parser().parse_args(
        [
            "--onnx_path",
            "dummy.onnx",
            "--input_shapes_profile",
            str(profile_path),
        ]
    )

    assert args.input_shapes_profile == profile


def test_quantization_cli_forwards_input_shapes_profile(monkeypatch, tmp_path):
    import modelopt.onnx.quantization.__main__ as quantization_cli

    profile = [{"nv_profile_min_shapes": "input_ids:1x1"}, {}]
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    captured = {}

    def fake_quantize(onnx_path_arg, **kwargs):
        captured["onnx_path"] = onnx_path_arg
        captured.update(kwargs)

    monkeypatch.setattr(quantization_cli, "quantize", fake_quantize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelopt.onnx.quantization",
            "--onnx_path",
            str(onnx_path),
            "--calibration_eps",
            "NvTensorRtRtx",
            "cpu",
            "--input_shapes_profile",
            json.dumps(profile),
        ],
    )

    quantization_cli.main()

    assert captured["onnx_path"] == str(onnx_path)
    assert captured["input_shapes_profile"] == profile


# --- autotune_network_timeout_minutes: CLI → quantize() ---


def test_quantization_cli_network_timeout_default_is_10(monkeypatch, tmp_path):
    """Without ``--autotune_network_timeout_minutes`` the default of 10 reaches ``quantize``."""
    import modelopt.onnx.quantization.__main__ as quantization_cli

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    captured = {}

    def fake_quantize(onnx_path_arg, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(quantization_cli, "quantize", fake_quantize)
    monkeypatch.setattr(sys, "argv", ["modelopt.onnx.quantization", "--onnx_path", str(onnx_path)])

    quantization_cli.main()

    assert captured["autotune_network_timeout_minutes"] == 10


def test_quantization_cli_network_timeout_custom_value_forwarded(monkeypatch, tmp_path):
    """``--autotune_network_timeout_minutes N`` is forwarded to ``quantize``."""
    import modelopt.onnx.quantization.__main__ as quantization_cli

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    captured = {}

    def fake_quantize(onnx_path_arg, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(quantization_cli, "quantize", fake_quantize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelopt.onnx.quantization",
            "--onnx_path",
            str(onnx_path),
            "--autotune_network_timeout_minutes",
            "25",
        ],
    )

    quantization_cli.main()

    assert captured["autotune_network_timeout_minutes"] == 25


def test_quantization_cli_network_timeout_is_integer(monkeypatch, tmp_path):
    """``--autotune_network_timeout_minutes`` is parsed as int, not string."""
    import modelopt.onnx.quantization.__main__ as quantization_cli

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    captured = {}

    def fake_quantize(onnx_path_arg, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(quantization_cli, "quantize", fake_quantize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelopt.onnx.quantization",
            "--onnx_path",
            str(onnx_path),
            "--autotune_network_timeout_minutes",
            "7",
        ],
    )

    quantization_cli.main()

    assert isinstance(captured["autotune_network_timeout_minutes"], int)


# --- quantize(): autotune_network_timeout_minutes → _find_nodes_to_quantize_autotune ---


def test_quantize_threads_network_timeout_to_autotune(monkeypatch, tmp_path):
    """``autotune_network_timeout_minutes`` reaches ``_find_nodes_to_quantize_autotune``.

    Follows the same heavy-mock pattern used in ``test_quantize_api.py`` to
    exercise the wiring inside ``quantize()`` without running TensorRT or real
    calibration.
    """
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")
    captured = {}

    def fake_preprocess(*args, **kwargs):
        return str(onnx_path), object(), [], False, False, False, {}, {}

    def fake_find_nodes_autotune(*args, network_timeout_minutes=10, **kwargs):
        captured["network_timeout_minutes"] = network_timeout_minutes
        return [], [], [], []

    monkeypatch.setattr(quantize_module, "_preprocess_onnx", fake_preprocess)
    monkeypatch.setattr(quantize_module, "update_trt_ep_support", lambda *a, **k: None)
    monkeypatch.setattr(quantize_module, "validate_op_types_spelling", lambda *a: None)
    monkeypatch.setattr(quantize_module, "find_nodes_from_mha_to_exclude", lambda *a, **k: [])
    monkeypatch.setattr(
        quantize_module, "_find_nodes_to_quantize_autotune", fake_find_nodes_autotune
    )
    monkeypatch.setattr(quantize_module, "quantize_int8", lambda **k: None)
    monkeypatch.setattr(quantize_module.onnx.checker, "check_model", lambda *a: None)

    quantize_module.quantize(
        str(onnx_path),
        autotune=True,
        calibration_data_reader=object(),
        autotune_network_timeout_minutes=42,
    )

    assert captured.get("network_timeout_minutes") == 42


def test_find_nodes_to_quantize_autotune_passes_timeout_to_init(monkeypatch):
    """``_find_nodes_to_quantize_autotune`` forwards ``network_timeout_minutes`` to init."""
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    captured = {}

    def fake_init(*, network_timeout_minutes=10, **kwargs):
        captured["network_timeout_minutes"] = network_timeout_minutes

    monkeypatch.setattr(
        "modelopt.onnx.quantization.autotune.workflows.init_benchmark_instance",
        fake_init,
    )

    with contextlib.suppress(RuntimeError):
        quantize_module._find_nodes_to_quantize_autotune(
            onnx_model=None,
            quantize_mode="int8",
            trt_plugins=None,
            network_timeout_minutes=15,
        )

    assert captured.get("network_timeout_minutes") == 15
