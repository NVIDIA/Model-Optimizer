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

"""
Tests for the Config class and CLI mode presets in the autotuner.

Tests configuration parameter validation, defaults, and CLI --mode preset
selection and explicit-flag precedence.
"""

from modelopt.onnx.quantization.autotune.__main__ import (
    MODE_PRESETS,
    apply_mode_presets,
    get_parser,
)
from modelopt.onnx.quantization.autotune.common import Config


class TestConfig:
    """Test Config class functionality."""

    def test_default_values(self):
        """Test that Config has correct default values."""
        config = Config()

        # Logging
        assert not config.verbose

        # Performance thresholds

        # Q/DQ defaults
        assert config.default_q_scale == 0.1
        assert config.default_q_zero_point == 0
        assert config.default_quant_type == "int8"

        # Region builder settings
        assert config.maximum_sequence_region_size == 10
        assert config.minimum_topdown_search_size == 10

        # Scheme generation parameters
        assert config.top_percent_to_mutate == 0.1
        assert config.minimum_schemes_to_mutate == 10
        assert config.maximum_mutations == 3
        assert config.maximum_generation_attempts == 100

        # Pattern cache parameters
        assert config.pattern_cache_minimum_distance == 4
        assert config.pattern_cache_max_entries_per_pattern == 32

    def test_custom_values(self):
        """Test creating Config with custom values."""
        config = Config(
            verbose=True,
            default_q_scale=0.05,
            default_q_zero_point=128,
            default_quant_type="fp8",
            maximum_sequence_region_size=20,
        )

        assert config.verbose
        assert config.default_q_scale == 0.05
        assert config.default_q_zero_point == 128
        assert config.default_quant_type == "fp8"
        assert config.maximum_sequence_region_size == 20

    def test_region_size_validation(self):
        """Test that region size parameters are positive."""
        config = Config(maximum_sequence_region_size=50, minimum_topdown_search_size=5)
        assert config.maximum_sequence_region_size > 0
        assert config.minimum_topdown_search_size > 0

    def test_genetic_algorithm_params(self):
        """Test genetic algorithm parameters."""
        config = Config(
            top_percent_to_mutate=0.2,
            minimum_schemes_to_mutate=2,
            maximum_mutations=5,
            maximum_generation_attempts=50,
        )

        assert config.top_percent_to_mutate == 0.2
        assert config.minimum_schemes_to_mutate == 2
        assert config.maximum_mutations == 5
        assert config.maximum_generation_attempts == 50

    def test_pattern_cache_params(self):
        """Test pattern cache parameters."""
        config = Config(pattern_cache_minimum_distance=3, pattern_cache_max_entries_per_pattern=10)

        assert config.pattern_cache_minimum_distance == 3
        assert config.pattern_cache_max_entries_per_pattern == 10


class TestModePresets:
    """Test --mode preset selection and explicit-flag precedence."""

    @staticmethod
    def _parse_cli(argv):
        """Parse argv with the autotune CLI parser and apply mode presets."""
        parser = get_parser()
        args = parser.parse_args(argv)
        apply_mode_presets(args)
        return args

    def test_mode_quick_applies_preset_when_no_explicit_flags(self):
        """With --mode quick and no explicit schemes/warmup/timing, preset values are used."""
        args = self._parse_cli(["--onnx_path", "model.onnx", "--mode", "quick"])
        preset = MODE_PRESETS["quick"]
        assert args.num_schemes == preset["schemes_per_region"]
        assert args.warmup_runs == preset["warmup_runs"]
        assert args.timing_runs == preset["timing_runs"]

    def test_mode_default_applies_preset_when_no_explicit_flags(self):
        """With --mode default and no explicit flags, preset values are used."""
        args = self._parse_cli(["--onnx_path", "model.onnx", "--mode", "default"])
        preset = MODE_PRESETS["default"]
        assert args.num_schemes == preset["schemes_per_region"]
        assert args.warmup_runs == preset["warmup_runs"]
        assert args.timing_runs == preset["timing_runs"]

    def test_mode_extensive_applies_preset_when_no_explicit_flags(self):
        """With --mode extensive and no explicit flags, preset values are used."""
        args = self._parse_cli(["--onnx_path", "model.onnx", "--mode", "extensive"])
        preset = MODE_PRESETS["extensive"]
        assert args.num_schemes == preset["schemes_per_region"]
        assert args.warmup_runs == preset["warmup_runs"]
        assert args.timing_runs == preset["timing_runs"]

    def test_explicit_schemes_per_region_overrides_mode_preset(self):
        """Explicit --schemes_per_region is kept even when it differs from preset."""
        args = self._parse_cli(
            ["--onnx_path", "model.onnx", "--mode", "default", "--schemes_per_region", "99"]
        )
        assert args.num_schemes == 99
        assert args.warmup_runs == MODE_PRESETS["default"]["warmup_runs"]
        assert args.timing_runs == MODE_PRESETS["default"]["timing_runs"]

    def test_explicit_default_value_not_overridden_by_mode(self):
        """Explicit --schemes_per_region 30 (parser default) is not overridden by --mode default."""
        args = self._parse_cli(
            ["--onnx_path", "model.onnx", "--mode", "default", "--schemes_per_region", "30"]
        )
        assert args.num_schemes == 30

    def test_explicit_warmup_runs_overrides_mode_preset(self):
        """Explicit --warmup_runs is kept and not overridden by preset."""
        args = self._parse_cli(
            ["--onnx_path", "model.onnx", "--mode", "extensive", "--warmup_runs", "3"]
        )
        assert args.warmup_runs == 3
        assert args.num_schemes == MODE_PRESETS["extensive"]["schemes_per_region"]
        assert args.timing_runs == MODE_PRESETS["extensive"]["timing_runs"]

    def test_explicit_timing_runs_overrides_mode_preset(self):
        """Explicit --timing_runs is kept and not overridden by preset."""
        args = self._parse_cli(
            ["--onnx_path", "model.onnx", "--mode", "quick", "--timing_runs", "7"]
        )
        assert args.timing_runs == 7
        assert args.num_schemes == MODE_PRESETS["quick"]["schemes_per_region"]
        assert args.warmup_runs == MODE_PRESETS["quick"]["warmup_runs"]

    def test_multiple_explicit_overrides_mode_preset(self):
        """Multiple explicit flags override only their respective preset values."""
        args = self._parse_cli(
            [
                "--onnx_path",
                "model.onnx",
                "--mode",
                "extensive",
                "--schemes_per_region",
                "10",
                "--timing_runs",
                "5",
            ]
        )
        assert args.num_schemes == 10
        assert args.timing_runs == 5
        assert args.warmup_runs == MODE_PRESETS["extensive"]["warmup_runs"]

    def test_short_flag_schemes_per_region_overrides_mode(self):
        """Short form -s for schemes_per_region is treated as explicit and overrides preset."""
        args = self._parse_cli(["--onnx_path", "model.onnx", "--mode", "default", "-s", "25"])
        assert args.num_schemes == 25
