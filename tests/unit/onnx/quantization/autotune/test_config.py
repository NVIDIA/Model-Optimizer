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
Tests for the Config class in the autotuner.

Tests configuration parameter validation and defaults.
"""

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
