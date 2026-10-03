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
