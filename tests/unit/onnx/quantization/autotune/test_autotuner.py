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

"""
Tests for QDQAutotuner class.

Tests the main autotuner class public API.
Note: Full integration tests with TensorRT benchmarking should be in separate integration test files.
"""

import os
import tempfile

import models as _test_models
import onnx
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.autotune import Config, QDQAutotuner, RegionPattern
from modelopt.onnx.quantization.autotune.common import PatternCache, RegionType


@pytest.fixture
def simple_conv_model():
    """Simple ONNX model: Input -> Conv -> Relu -> Output. Created via models.py."""
    return _test_models._create_simple_conv_onnx_model()


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


class TestQDQAutotuner:
    """Test QDQAutotuner functionality."""

    def test_creation_with_onnx_model(self, simple_conv_model):
        """Test creating autotuner with ONNX ModelProto."""
        autotuner = QDQAutotuner(simple_conv_model)

        assert autotuner is not None
        assert autotuner.onnx_model is not None
        assert autotuner.graph is not None

    def test_creation_with_gs_graph(self, simple_conv_model):
        """Test creating autotuner with GraphSurgeon graph."""
        gs_graph = gs.import_onnx(simple_conv_model)
        autotuner = QDQAutotuner(gs_graph)

        assert autotuner is not None
        assert autotuner.graph is not None

    def test_initialize_with_default_config(self, simple_conv_model):
        """Test initialization with default test config."""
        autotuner = QDQAutotuner(simple_conv_model)

        config = _create_test_config()
        autotuner.initialize(config)

        # Should have provided config
        assert autotuner.config is not None
        assert autotuner.config.maximum_sequence_region_size == 50

        # Should have discovered regions
        assert len(autotuner.regions) > 0

    def test_initialize_with_config(self, simple_conv_model):
        """Test initialization with custom config (different from default)."""
        autotuner = QDQAutotuner(simple_conv_model)

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

    def test_initialize_with_pattern_cache(self, simple_conv_model):
        """Test initialization with pattern cache."""
        autotuner = QDQAutotuner(simple_conv_model)

        config = _create_test_config()
        pattern_cache = PatternCache()
        autotuner.initialize(config, pattern_cache=pattern_cache)

        assert autotuner.pattern_cache is not None

    def test_region_discovery(self, simple_conv_model):
        """Test that regions are automatically discovered."""
        autotuner = QDQAutotuner(simple_conv_model)

        config = _create_test_config()
        autotuner.initialize(config)

        # Should discover at least one region
        assert len(autotuner.regions) > 0

        # Regions should be valid
        for region in autotuner.regions:
            assert region.get_id() is not None
            assert region.get_type() in [RegionType.LEAF, RegionType.COMPOSITE, RegionType.ROOT]

    def test_export_baseline_model(self, simple_conv_model):
        """Test exporting baseline model without Q/DQ."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
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
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_set_profile_region(self, simple_conv_model):
        """Test setting a region for profiling."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
        autotuner.initialize(config)

        if len(autotuner.regions) > 0:
            region = autotuner.regions[0]
            autotuner.set_profile_region(region)
            # Should set current profile region
            assert autotuner.current_profile_region == region
            assert autotuner.current_profile_pattern_schemes is not None
        else:
            pytest.skip("No regions discovered")

    def test_generate_scheme(self, simple_conv_model):
        """Test generating multiple schemes and that Q/DQ nodes appear in exported model."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
        autotuner.initialize(config)

        if len(autotuner.regions) == 0:
            pytest.skip("No regions discovered")

        autotuner.submit(10.0)  # baseline
        region = autotuner.regions[0]
        autotuner.set_profile_region(region)

        # Generate multiple schemes and submit a latency for each
        num_generated = 0
        while True:
            scheme_idx = autotuner.generate()
            if scheme_idx < 0:
                break
            assert isinstance(scheme_idx, int)
            autotuner.submit(10.0 + num_generated * 0.1)  # dummy latency
            num_generated += 1
            if num_generated >= 5:  # cap iterations
                break

        assert num_generated > 0, "Expected at least one scheme to be generated"
        autotuner.set_profile_region(None, commit=True)

        # Export with Q/DQ and verify Q/DQ nodes are in the model
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name
        try:
            autotuner.export_onnx(output_path, insert_qdq=True)
            exported = onnx.load(output_path)
            node_ops = [n.op_type for n in exported.graph.node]
            assert "QuantizeLinear" in node_ops, "Expected QuantizeLinear nodes in exported model"
            assert "DequantizeLinear" in node_ops, (
                "Expected DequantizeLinear nodes in exported model"
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_submit_latency(self, simple_conv_model):
        """Test submitting performance measurement."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
        autotuner.initialize(config)
        # Submit baseline latency
        autotuner.submit(10.5)
        # Baseline should be recorded
        assert autotuner.baseline_latency_ms == 10.5

    def test_save_and_load_state(self, simple_conv_model):
        """Test saving and loading autotuner state."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
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
            autotuner2 = QDQAutotuner(simple_conv_model)
            config2 = _create_test_config()
            autotuner2.initialize(config2)
            autotuner2.load_state(state_path)

            # Baseline should match
            assert autotuner2.baseline_latency_ms == 10.5
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_regions_prioritization(self, simple_conv_model):
        """Test that LEAF regions are prioritized."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
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

    def test_profiled_patterns_tracking(self, simple_conv_model):
        """Test that profiled patterns are tracked."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
        autotuner.initialize(config)
        autotuner.submit(10.0)

        if len(autotuner.regions) > 0:
            region = autotuner.regions[0]
            autotuner.set_profile_region(region)

            scheme_idx = autotuner.generate()
            if scheme_idx >= 0:
                autotuner.submit(12.0)
                autotuner.set_profile_region(None, commit=True)
                pattern_sig = RegionPattern.from_region(region, autotuner.graph).signature
                profiled_patterns = [p.pattern.signature for p in autotuner.profiled_patterns]
                assert pattern_sig in profiled_patterns
        else:
            pytest.skip("No regions discovered")
