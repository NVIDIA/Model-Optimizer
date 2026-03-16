#!/usr/bin/env python3
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

"""Tests for the Config class in the autotuner."""

import unittest

from modelopt.onnx.quantization.autotune.common import Config


class TestConfig(unittest.TestCase):
    """Test Config class functionality."""

    def test_default_values(self):
        """Test that Config has correct default values."""
        config = Config()

        # Logging
        self.assertEqual(config.verbose, False)

        # Performance thresholds
        self.assertEqual(config.performance_threshold, 1.02)

        # Q/DQ defaults
        self.assertEqual(config.default_q_scale, 0.1)
        self.assertEqual(config.default_q_zero_point, 0)
        self.assertEqual(config.default_quant_type, "int8")

        # Region builder settings
        self.assertEqual(config.maximum_sequence_region_size, 10)
        self.assertEqual(config.minimum_topdown_search_size, 10)

        # Scheme generation parameters
        self.assertEqual(config.top_percent_to_mutate, 0.1)
        self.assertEqual(config.minimum_schemes_to_mutate, 10)
        self.assertEqual(config.maximum_mutations, 3)
        self.assertEqual(config.maximum_generation_attempts, 100)

        # Pattern cache parameters
        self.assertEqual(config.pattern_cache_minimum_distance, 4)
        self.assertEqual(config.pattern_cache_max_entries_per_pattern, 32)


    def test_custom_values(self):
        """Test creating Config with custom values."""
        config = Config(
            verbose=True,
            performance_threshold=1.05,
            default_q_scale=0.05,
            default_q_zero_point=128,
            default_quant_type="fp8",
            maximum_sequence_region_size=20,
        )

        self.assertEqual(config.verbose, True)
        self.assertEqual(config.performance_threshold, 1.05)
        self.assertEqual(config.default_q_scale, 0.05)
        self.assertEqual(config.default_q_zero_point, 128)
        self.assertEqual(config.default_quant_type, "fp8")
        self.assertEqual(config.maximum_sequence_region_size, 20)

    def test_performance_threshold_validation(self):
        """Test that performance_threshold must be >= 1.0."""
        config1 = Config(performance_threshold=1.0)
        self.assertEqual(config1.performance_threshold, 1.0)

        config2 = Config(performance_threshold=1.5)
        self.assertEqual(config2.performance_threshold, 1.5)

    def test_region_size_validation(self):
        """Test that region size parameters are positive."""
        config = Config(maximum_sequence_region_size=50, minimum_topdown_search_size=5)
        self.assertGreater(config.maximum_sequence_region_size, 0)
        self.assertGreater(config.minimum_topdown_search_size, 0)

    def test_genetic_algorithm_params(self):
        """Test genetic algorithm parameters."""
        config = Config(
            top_percent_to_mutate=0.2,
            minimum_schemes_to_mutate=2,
            maximum_mutations=5,
            maximum_generation_attempts=50,
        )

        self.assertEqual(config.top_percent_to_mutate, 0.2)
        self.assertEqual(config.minimum_schemes_to_mutate, 2)
        self.assertEqual(config.maximum_mutations, 5)
        self.assertEqual(config.maximum_generation_attempts, 50)

    def test_pattern_cache_params(self):
        """Test pattern cache parameters."""
        config = Config(pattern_cache_minimum_distance=3, pattern_cache_max_entries_per_pattern=10)

        self.assertEqual(config.pattern_cache_minimum_distance, 3)
        self.assertEqual(config.pattern_cache_max_entries_per_pattern, 10)


if __name__ == "__main__":
    unittest.main()
