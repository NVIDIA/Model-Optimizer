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

"""Tests for CLI argument parsing and mode presets."""

import argparse
from types import SimpleNamespace

import pytest

from modelopt.onnx.quantization.autotune.__main__ import (
    DEFAULT_NUM_SCHEMES,
    DEFAULT_TIMING_RUNS,
    DEFAULT_WARMUP_RUNS,
    MODE_PRESETS,
    apply_mode_presets,
    get_parser,
)


class TestGetParser:
    def _parse(self, *args):
        return get_parser().parse_args(list(args))

    def test_required_onnx_path(self):
        with pytest.raises(SystemExit):
            self._parse()

    def test_minimal_args(self):
        args = self._parse("-m", "model.onnx")
        assert args.onnx_path == "model.onnx"
        assert args.output_dir == "./autotuner_output"

    def test_output_dir(self):
        args = self._parse("-m", "model.onnx", "-o", "/tmp/out")
        assert args.output_dir == "/tmp/out"

    def test_workflow_region_default(self):
        args = self._parse("-m", "model.onnx")
        assert args.workflow == "region"

    def test_workflow_subgraph(self):
        args = self._parse("-m", "model.onnx", "--workflow", "subgraph")
        assert args.workflow == "subgraph"

    def test_workflow_invalid_choice(self):
        with pytest.raises(SystemExit):
            self._parse("-m", "model.onnx", "--workflow", "invalid")

    def test_graph_json(self):
        args = self._parse("-m", "model.onnx", "--graph_json", "graph.json")
        assert args.graph_json == "graph.json"

    def test_incremental_validation_default(self):
        args = self._parse("-m", "model.onnx")
        assert args.incremental_validation is True

    def test_no_incremental_validation(self):
        args = self._parse("-m", "model.onnx", "--no-incremental_validation")
        assert args.incremental_validation is False

    def test_quant_type_int8_default(self):
        args = self._parse("-m", "model.onnx")
        assert args.quant_type == "int8"

    def test_quant_type_fp8(self):
        args = self._parse("-m", "model.onnx", "--quant_type", "fp8")
        assert args.quant_type == "fp8"

    def test_use_trtexec(self):
        args = self._parse("-m", "model.onnx", "--use_trtexec")
        assert args.use_trtexec is True

    def test_use_trtexec_default_false(self):
        args = self._parse("-m", "model.onnx")
        assert args.use_trtexec is False

    def test_schemes_per_region(self):
        args = self._parse("-m", "model.onnx", "-s", "30")
        assert args.num_schemes == 30

    def test_mode_choices(self):
        for mode in ["quick", "default", "extensive"]:
            args = self._parse("-m", "model.onnx", "--mode", mode)
            assert args.mode == mode

    def test_verbose(self):
        args = self._parse("-m", "model.onnx", "-v")
        assert args.verbose is True

    def test_trtexec_benchmark_args(self):
        args = self._parse("-m", "model.onnx", "--trtexec_benchmark_args", "--fp16 --verbose")
        assert args.trtexec_benchmark_args == "--fp16 --verbose"

    def test_plugin_libraries(self):
        args = self._parse("-m", "model.onnx", "--plugins", "a.so", "b.so")
        assert args.plugin_libraries == ["a.so", "b.so"]


class TestApplyModePresets:
    def _make_args(self, mode="default"):
        return SimpleNamespace(
            mode=mode,
            num_schemes=DEFAULT_NUM_SCHEMES,
            warmup_runs=DEFAULT_WARMUP_RUNS,
            timing_runs=DEFAULT_TIMING_RUNS,
            _explicit_num_schemes=False,
            _explicit_warmup_runs=False,
            _explicit_timing_runs=False,
        )

    def test_quick_preset(self):
        args = self._make_args("quick")
        apply_mode_presets(args)
        assert args.num_schemes == MODE_PRESETS["quick"]["schemes_per_region"]
        assert args.warmup_runs == MODE_PRESETS["quick"]["warmup_runs"]
        assert args.timing_runs == MODE_PRESETS["quick"]["timing_runs"]

    def test_extensive_preset(self):
        args = self._make_args("extensive")
        apply_mode_presets(args)
        assert args.num_schemes == 200
        assert args.timing_runs == 200

    def test_explicit_override_preserved(self):
        args = self._make_args("quick")
        args.num_schemes = 999
        args._explicit_num_schemes = True
        apply_mode_presets(args)
        assert args.num_schemes == 999
        assert args.warmup_runs == MODE_PRESETS["quick"]["warmup_runs"]

    def test_unknown_mode_noop(self):
        args = self._make_args("unknown")
        original_schemes = args.num_schemes
        apply_mode_presets(args)
        assert args.num_schemes == original_schemes
