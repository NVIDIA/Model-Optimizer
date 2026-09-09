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
