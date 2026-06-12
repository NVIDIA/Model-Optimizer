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
import sys

import pytest

from modelopt.onnx.quantization.__main__ import get_parser


def test_quantization_cli_parser_imports_without_tensorrt():
    """Verify the CLI parser can be constructed without TensorRT installed."""
    with pytest.MonkeyPatch.context() as mp:
        # Force tensorrt import to fail, even if it's actually installed
        mp.setitem(sys.modules, "tensorrt", None)

        # Reload the autotune package so it picks up the blocked import
        import modelopt.onnx.quantization.autotune

        importlib.reload(modelopt.onnx.quantization.autotune)

        from modelopt.onnx.quantization.__main__ import get_parser as get_parser_no_trt

        parser = get_parser_no_trt()
        args = parser.parse_args(["--onnx_path", "dummy.onnx"])
        assert args.onnx_path == "dummy.onnx"
        assert args.quantize_mode == "int8"


def test_trust_calibration_data_combines_with_calibration_data_path():
    """--trust_calibration_data toggles allow_pickle for --calibration_data_path/--calibration_cache_path.

    It must not be mutually exclusive with them, otherwise the secure pickle-opt-in path that
    main() documents in its error message is unreachable from the CLI.
    """
    parser = get_parser()

    args = parser.parse_args(["--onnx_path", "dummy.onnx", "--calibration_data_path", "calib.npy"])
    assert args.trust_calibration_data is False

    args = parser.parse_args(
        [
            "--onnx_path",
            "dummy.onnx",
            "--calibration_data_path",
            "calib.npy",
            "--trust_calibration_data",
        ]
    )
    assert args.calibration_data_path == "calib.npy"
    assert args.trust_calibration_data is True

    args = parser.parse_args(
        [
            "--onnx_path",
            "dummy.onnx",
            "--calibration_cache_path",
            "calib.cache",
            "--trust_calibration_data",
        ]
    )
    assert args.calibration_cache_path == "calib.cache"
    assert args.trust_calibration_data is True


def test_calibration_data_and_cache_paths_remain_mutually_exclusive():
    """--calibration_data_path and --calibration_cache_path are still mutually exclusive."""
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--onnx_path",
                "dummy.onnx",
                "--calibration_data_path",
                "calib.npy",
                "--calibration_cache_path",
                "calib.cache",
            ]
        )
