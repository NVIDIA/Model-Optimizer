#!/usr/bin/env python3
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
Tests for QDQAutotuner class.

Tests the main autotuner class public API.
Note: Full integration tests with TensorRT benchmarking should be in separate integration test files.
"""

import os
import sys
import tempfile
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
import onnx_graphsurgeon as gs
from onnx import helper

from modelopt.onnx.quantization.autotune import Config, QDQAutotuner, RegionPattern
from modelopt.onnx.quantization.autotune.common import PatternCache, RegionType


def create_simple_conv_model():
    """
    Create a simple ONNX model: Input -> Conv -> Relu -> Output.

    This is a minimal model for testing autotuner initialization.
    """
    # Input
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])

    # Output
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 64, 224, 224]
    )

    # Conv node
    conv_node = helper.make_node(
        "Conv", inputs=["input", "conv_weight"], outputs=["conv_out"], name="conv"
    )

    # Relu node
    relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"], name="relu")

    # Create graph
    graph = helper.make_graph(
        [conv_node, relu_node],
        "simple_conv",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor(
                "conv_weight", onnx.TensorProto.FLOAT, [64, 3, 3, 3], [0.1] * (64 * 3 * 3 * 3)
            )
        ],
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    return model


class TestQDQAutotuner(unittest.TestCase):
    """Test QDQAutotuner functionality."""

    @staticmethod
    def _create_test_config():
        """
        Create a reasonable config for testing.

        Uses sensible defaults suitable for unit tests:
        - verbose=False: Keep test output clean
        - maximum_sequence_region_size=50: Allow larger test regions
        - Other parameters: Match Config defaults for typical behavior
        """
        return Config(
            # Logging
            verbose=False,
            # Performance Requirements
            # Quantization Parameters
            default_q_scale=0.1,
            default_q_zero_point=0,
            default_quant_type="int8",
            # Region Builder Settings
            maximum_sequence_region_size=50,
            minimum_topdown_search_size=10,
            # Scheme Generation Settings
            top_percent_to_mutate=0.1,
            minimum_schemes_to_mutate=10,
            maximum_mutations=3,
            maximum_generation_attempts=100,
            # Pattern Cache Settings
            pattern_cache_minimum_distance=4,
            pattern_cache_max_entries_per_pattern=32,
        )

    def test_creation_with_onnx_model(self):
        """Test creating autotuner with ONNX ModelProto."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)

        assert autotuner is not None
        assert autotuner.onnx_model is not None
        assert autotuner.graph is not None
        print("✓ QDQAutotuner creation with ONNX model")

    def test_creation_with_gs_graph(self):
        """Test creating autotuner with GraphSurgeon graph."""
        model = create_simple_conv_model()
        gs_graph = gs.import_onnx(model)

        autotuner = QDQAutotuner(gs_graph)

        assert autotuner is not None
        assert autotuner.graph is not None
        print("✓ QDQAutotuner creation with GS graph")

    def test_initialize_with_default_config(self):
        """Test initialization with default test config."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)

        config = self._create_test_config()
        autotuner.initialize(config)

        # Should have provided config
        assert autotuner.config is not None
        assert autotuner.config.maximum_sequence_region_size == 50

        # Should have discovered regions
        assert len(autotuner.regions) > 0
        print("✓ QDQAutotuner initialize with default config")

    def test_initialize_with_config(self):
        """Test initialization with custom config (different from default)."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)

        # Create custom config with different values
        config = Config(
            verbose=True,
            default_q_scale=0.05,
            default_q_zero_point=128,
            default_quant_type="fp8",
            maximum_sequence_region_size=20,
            minimum_topdown_search_size=5,
            top_percent_to_mutate=0.2,
            minimum_schemes_to_mutate=5,
            maximum_mutations=5,
            maximum_generation_attempts=50,
            pattern_cache_minimum_distance=2,
            pattern_cache_max_entries_per_pattern=16,
        )
        autotuner.initialize(config)

        # Should use provided custom config values
        assert autotuner.config.verbose
        assert autotuner.config.default_q_scale == 0.05
        assert autotuner.config.default_q_zero_point == 128
        assert autotuner.config.default_quant_type == "fp8"
        assert autotuner.config.maximum_sequence_region_size == 20
        assert autotuner.config.minimum_topdown_search_size == 5
        assert autotuner.config.top_percent_to_mutate == 0.2
        assert autotuner.config.minimum_schemes_to_mutate == 5
        assert autotuner.config.maximum_mutations == 5
        assert autotuner.config.maximum_generation_attempts == 50
        assert autotuner.config.pattern_cache_minimum_distance == 2
        assert autotuner.config.pattern_cache_max_entries_per_pattern == 16
        print("✓ QDQAutotuner initialize with config")

    def test_initialize_with_pattern_cache(self):
        """Test initialization with pattern cache."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)

        config = self._create_test_config()
        pattern_cache = PatternCache()
        autotuner.initialize(config, pattern_cache=pattern_cache)

        assert autotuner.pattern_cache is not None
        print("✓ QDQAutotuner initialize with pattern cache")

    def test_region_discovery(self):
        """Test that regions are automatically discovered."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)

        config = self._create_test_config()
        autotuner.initialize(config)

        # Should discover at least one region
        assert len(autotuner.regions) > 0

        # Regions should be valid
        for region in autotuner.regions:
            assert region.get_id() is not None
            assert region.get_type() in [RegionType.LEAF, RegionType.COMPOSITE, RegionType.ROOT]

        print("✓ QDQAutotuner region discovery")

    def test_export_baseline_model(self):
        """Test exporting baseline model without Q/DQ."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        try:
            # Export baseline without Q/DQ insertion
            autotuner.export_onnx(output_path, insert_qdq=False)

            # Verify file was created
            assert os.path.exists(output_path)

            # Verify it's a valid ONNX model
            exported_model = onnx.load(output_path)
            assert exported_model is not None
            print("✓ QDQAutotuner export baseline model")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_set_profile_region(self):
        """Test setting a region for profiling."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        if len(autotuner.regions) > 0:
            region = autotuner.regions[0]
            autotuner.set_profile_region(region)

            # Should set current profile region
            assert autotuner.current_profile_region == region
            assert autotuner.current_profile_pattern_schemes is not None
            print("✓ QDQAutotuner set_profile_region")
        else:
            self.skipTest("No regions discovered")

    def test_generate_scheme(self):
        """Test generating an insertion scheme."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        if len(autotuner.regions) > 0:
            region = autotuner.regions[0]
            autotuner.set_profile_region(region)

            # Generate a scheme
            scheme_idx = autotuner.generate()

            # Should return a valid index (>= 0) or -1 if no more unique schemes
            assert isinstance(scheme_idx, int)
            print("✓ QDQAutotuner generate scheme")
        else:
            self.skipTest("No regions discovered")

    def test_submit_latency(self):
        """Test submitting performance measurement."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        # Submit baseline latency
        autotuner.submit(10.5)

        # Baseline should be recorded
        assert autotuner.baseline_latency_ms == 10.5
        print("✓ QDQAutotuner submit latency")

    def test_save_and_load_state(self):
        """Test saving and loading autotuner state."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        # Submit some results
        autotuner.submit(10.5)  # baseline

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            state_path = f.name

        try:
            # Save state
            autotuner.save_state(state_path)
            assert os.path.exists(state_path)

            # Create new autotuner and load state
            autotuner2 = QDQAutotuner(model)
            config2 = self._create_test_config()
            autotuner2.initialize(config2)
            autotuner2.load_state(state_path)

            # Baseline should match
            assert autotuner2.baseline_latency_ms == 10.5
            print("✓ QDQAutotuner save and load state")
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_regions_prioritization(self):
        """Test that LEAF regions are prioritized."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        # Check that LEAF regions come before non-LEAF
        leaf_indices = [
            i for i, r in enumerate(autotuner.regions) if r.get_type() == RegionType.LEAF
        ]
        non_leaf_indices = [
            i for i, r in enumerate(autotuner.regions) if r.get_type() != RegionType.LEAF
        ]

        if leaf_indices and non_leaf_indices:
            # All LEAF should come before non-LEAF
            assert max(leaf_indices) < min(non_leaf_indices)
            print("✓ QDQAutotuner LEAF region prioritization")
        else:
            print("✓ QDQAutotuner regions (not enough for prioritization test)")

    def test_profiled_patterns_tracking(self):
        """Test that profiled patterns are tracked."""
        model = create_simple_conv_model()
        autotuner = QDQAutotuner(model)
        config = self._create_test_config()
        autotuner.initialize(config)

        # Submit baseline latency first
        autotuner.submit(10.0)

        if len(autotuner.regions) > 0:
            region = autotuner.regions[0]
            autotuner.set_profile_region(region)

            # Generate and submit a scheme
            scheme_idx = autotuner.generate()
            if scheme_idx >= 0:
                autotuner.submit(12.0)
                autotuner.set_profile_region(None, commit=True)

                # Pattern should be tracked
                pattern_sig = RegionPattern.from_region(region, autotuner.graph).signature
                profiled_patterns = [p.pattern.signature for p in autotuner.profiled_patterns]
                assert pattern_sig in profiled_patterns
                print("✓ QDQAutotuner profiled patterns tracking")
            else:
                print("✓ QDQAutotuner (no schemes to test tracking)")
        else:
            self.skipTest("No regions discovered")


def run_tests():
    """Run all QDQAutotuner tests."""
    print("=" * 70)
    print("QDQAutotuner Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestQDQAutotuner))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All QDQAutotuner tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
