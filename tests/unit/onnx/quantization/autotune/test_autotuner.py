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

import copy
import os
import tempfile

import onnx
import onnx_graphsurgeon as gs
import pytest
from _test_utils.onnx.quantization.autotune.models import _create_simple_conv_onnx_model

import modelopt.onnx.quantization.autotune.export_utils as export_utils
from modelopt.onnx.quantization.autotune import Config, QDQAutotuner, RegionPattern
from modelopt.onnx.quantization.autotune.common import (
    InsertionScheme,
    NodeInputInsertionPoint,
    PatternCache,
    PatternSchemes,
    Region,
    RegionType,
    SchemeAction,
)
from modelopt.onnx.quantization.autotune.insertion_points import (
    ResolvedInsertionPoint,
    get_autotuner_quantizable_ops,
)


@pytest.fixture
def simple_conv_model():
    """Simple ONNX model: Input -> Conv -> Relu -> Output. Created via _test_utils models."""
    return _create_simple_conv_onnx_model()


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


def _measure_pattern_controls(autotuner, region):
    if autotuner.baseline_latency_ms is None:
        autotuner.submit(10.0)
    autotuner.set_profile_region(region)
    assert autotuner.begin_inherit_profile() == 0
    autotuner.submit_inherit(10.0)
    assert autotuner.generate() == 1
    autotuner.submit(10.0)


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
            assert region.id is not None
            assert region.type in [RegionType.LEAF, RegionType.COMPOSITE, RegionType.ROOT]

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

    def test_qdq_export_orders_merged_insertion_points(self, simple_conv_model, monkeypatch):
        insertion_points = [
            ResolvedInsertionPoint("input", node_index=0, input_index=0),
            ResolvedInsertionPoint("conv_weight", node_index=0, input_index=1),
        ]
        created_tensor_names = []
        create_qdq_nodes = export_utils.create_qdq_nodes

        def record_qdq_creation(tensor_name, *args, **kwargs):
            created_tensor_names.append(tensor_name)
            return create_qdq_nodes(tensor_name, *args, **kwargs)

        monkeypatch.setattr(export_utils, "create_qdq_nodes", record_qdq_creation)
        exported_models = []
        for merged_points in (insertion_points, list(reversed(insertion_points))):
            monkeypatch.setattr(
                export_utils,
                "merge_resolved_insertion_points",
                lambda *_, points=merged_points: points,
            )
            exported_models.append(
                export_utils.export_qdq_onnx(
                    simple_conv_model,
                    set(insertion_points),
                    _create_test_config(),
                )
            )

        assert created_tensor_names == ["conv_weight_n0_i1", "input_n0_i0"] * 2
        assert exported_models[0].SerializeToString(deterministic=True) == exported_models[
            1
        ].SerializeToString(deterministic=True)

    @pytest.mark.parametrize("force_no_qdq", [False, True])
    def test_empty_ort_quantization_config(self, simple_conv_model, force_no_qdq):
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.set_force_no_qdq(force_no_qdq)

        assert autotuner.get_ort_quantization_config() == ([], [], [], [])

    @pytest.mark.parametrize(
        ("quant_type", "expected_op_types"),
        [
            ("fp8", {"Conv", "Gemm", "MatMul", "Add"}),
            ("int8", get_autotuner_quantizable_ops()),
        ],
    )
    def test_ort_quantization_config_op_types(
        self, simple_conv_model, monkeypatch, quant_type, expected_op_types
    ):
        config = _create_test_config()
        config.default_quant_type = quant_type
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(config)
        conv_index = next(i for i, node in enumerate(autotuner.graph.nodes) if node.op == "Conv")
        relu_index = next(i for i, node in enumerate(autotuner.graph.nodes) if node.op == "Relu")
        resolved_ips = {
            ResolvedInsertionPoint(
                tensor_name=autotuner.graph.nodes[conv_index].inputs[0].name,
                node_index=conv_index,
                input_index=0,
            ),
            ResolvedInsertionPoint(
                tensor_name=autotuner.graph.nodes[relu_index].inputs[0].name,
                node_index=relu_index,
                input_index=0,
            ),
        }
        monkeypatch.setattr(
            autotuner,
            "get_resolved_insertion_points",
            lambda **_: resolved_ips,
        )

        (
            nodes_to_quantize,
            op_types_to_quantize,
            _,
            op_types_needing_output_quant,
        ) = autotuner.get_ort_quantization_config()

        assert nodes_to_quantize == [
            autotuner.graph.nodes[index].name for index in sorted({conv_index, relu_index})
        ]
        assert op_types_to_quantize == sorted(expected_op_types)
        assert op_types_needing_output_quant == ["Conv"]

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

    def test_consecutive_matching_regions_profile_once(self, simple_conv_model):
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        first_region = autotuner.regions[0]
        duplicate_region = copy.deepcopy(first_region)
        duplicate_region.id = max(region.id for region in autotuner.regions) + 1
        autotuner.regions.append(duplicate_region)

        _measure_pattern_controls(autotuner, first_region)
        autotuner.set_profile_region(duplicate_region)

        assert len(autotuner.profiled_patterns) == 1
        assert autotuner.current_profile_region is None
        assert autotuner.current_profile_pattern_schemes is None

    def test_switch_to_profiled_pattern_commits_active_region(self, simple_conv_model):
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        first_region = autotuner.regions[0]
        duplicate_region = copy.deepcopy(first_region)
        duplicate_region.id = max(region.id for region in autotuner.regions) + 1
        active_region = copy.deepcopy(first_region)
        active_region.id = duplicate_region.id + 1
        active_region.nodes = {1}
        active_region.inputs = ["conv_out"]
        active_region.outputs = ["output"]
        autotuner.regions.extend([duplicate_region, active_region])

        _measure_pattern_controls(autotuner, first_region)
        autotuner.set_profile_region(None, commit=True)
        _measure_pattern_controls(autotuner, active_region)
        autotuner.set_profile_region(duplicate_region)

        assert len(autotuner.profiled_patterns) == 2
        assert autotuner.current_profile_region is None
        assert autotuner.current_profile_pattern_schemes is None

    def test_generate_scheme(self, simple_conv_model):
        """Test generating multiple schemes and that Q/DQ nodes appear in exported model."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
        autotuner.initialize(config)

        if len(autotuner.regions) == 0:
            pytest.skip("No regions discovered")

        autotuner.submit(10.0)
        region = autotuner.regions[0]
        autotuner.set_profile_region(region)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

            has_q = False
            has_dq = False
            for _ in range(5):
                scheme_idx = autotuner.generate()
                assert isinstance(scheme_idx, int)
                autotuner.submit(10.0 + _ * 0.1)

                autotuner.export_onnx(output_path, insert_qdq=True)
                exported = onnx.load(output_path)
                node_ops = [n.op_type for n in exported.graph.node]
                for node_op in node_ops:
                    if node_op == "QuantizeLinear":
                        has_q = True
                    if node_op == "DequantizeLinear":
                        has_dq = True
                if has_q and has_dq:
                    break
            assert has_q and has_dq, (
                "Expected QuantizeLinear and DequantizeLinear nodes in exported model"
            )

    def test_submit_latency(self, simple_conv_model):
        """Test submitting performance measurement."""
        autotuner = QDQAutotuner(simple_conv_model)
        config = _create_test_config()
        autotuner.initialize(config)
        # Submit baseline latency
        autotuner.submit(10.5)
        # Baseline should be recorded
        assert autotuner.baseline_latency_ms == 10.5

    def test_control_order_and_invalid_inherit(self, simple_conv_model):
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.submit(10.0)
        autotuner.set_profile_region(autotuner.regions[0])

        assert autotuner.begin_inherit_profile() == 0
        autotuner.submit_inherit(float("inf"), success=False)
        schemes = autotuner.current_profile_pattern_schemes
        assert schemes is not None
        assert schemes.selected_scheme.action == SchemeAction.INHERIT

        assert autotuner.generate() == 1
        assert schemes.schemes[1].action == SchemeAction.NO_QDQ

    def test_invalid_no_qdq_generates_full_qdq_seed(self, simple_conv_model):
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.submit(10.0)
        autotuner.set_profile_region(autotuner.regions[0])
        autotuner.begin_inherit_profile()
        autotuner.submit_inherit(10.0)

        assert autotuner.generate() == 1
        autotuner.submit(float("inf"), success=False)
        scheme_index = autotuner.generate()

        assert scheme_index == 2
        scheme = autotuner.current_profile_pattern_schemes.schemes[scheme_index]
        full_scheme = autotuner.current_profile_pattern_schemes.pattern.get_full_insertion_scheme(
            autotuner.current_profile_region, autotuner.graph
        )
        assert scheme.is_qdq
        assert scheme.hash == full_scheme.hash

    def test_v2_state_restores_partial_pattern(self, simple_conv_model, tmp_path):
        state_path = tmp_path / "state.yaml"
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.set_resume_fingerprint(backend="test", device="cpu")
        autotuner.submit(10.0)
        autotuner.set_profile_region(autotuner.regions[0])
        autotuner.begin_inherit_profile()
        autotuner.submit_inherit(10.0)
        assert autotuner.generate() == 1
        autotuner.submit(9.9)
        autotuner.set_force_no_qdq()
        autotuner.record_proxy_decision(
            proxy_selection="no_qdq",
            baseline_latency_ms=10.0,
            candidate_latency_ms=9.9,
            candidate_quantization_site_count=1,
            candidate_model_sha256="proxy-candidate",
            selected_model_sha256="proxy-selected",
        )
        autotuner.record_final_baseline_measurement(
            decision_stage="calibrated_baseline",
            baseline_final_latency_ms=10.0,
            baseline_model_sha256="baseline",
        )
        autotuner.record_final_decision(
            decision_stage="calibrated_final",
            final_selection="no_qdq",
            candidate_final_latency_ms=9.9,
            final_latency_ms=10.0,
            candidate_quantization_site_count=1,
            candidate_model_sha256="candidate",
            selected_model_sha256="selected",
        )
        autotuner.save_state(str(state_path))

        restored = QDQAutotuner(simple_conv_model)
        restored.initialize(_create_test_config())
        restored.set_resume_fingerprint(backend="test", device="cpu")
        restored.load_state(str(state_path))

        assert restored.baseline_latency_ms == 10.0
        assert restored.current_profile_region is not None
        assert restored.begin_inherit_profile() == -1
        assert restored.current_profile_pattern_schemes.profiled_override_count == 1
        assert restored.force_no_qdq
        assert restored.proxy_selection == "no_qdq"
        assert restored.proxy_candidate_quantization_site_count == 1
        assert restored.final_baseline_measurement["decision_stage"] == "calibrated_baseline"
        assert restored.decision_stage == "calibrated_final"
        assert restored.final_selection == "no_qdq"
        assert restored.candidate_quantization_site_count == 1

    def test_v2_state_restores_calibrated_baseline_checkpoint(self, simple_conv_model, tmp_path):
        state_path = tmp_path / "state.yaml"
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.record_final_baseline_measurement(
            decision_stage="calibrated_baseline",
            baseline_final_latency_ms=10.0,
            baseline_model_sha256="baseline",
        )
        autotuner.save_state(str(state_path))

        restored = QDQAutotuner(simple_conv_model)
        restored.initialize(_create_test_config())
        restored.load_state(str(state_path))

        assert restored.decision_stage == "calibrated_baseline"
        assert restored.baseline_final_latency_ms == 10.0
        assert restored.baseline_model_sha256 == "baseline"
        assert restored.final_decision is None

    def test_partial_pattern_generation_is_deterministic_after_resume(
        self, simple_conv_model, tmp_path
    ):
        state_path = tmp_path / "state.yaml"
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.set_resume_fingerprint(backend="test", device="cpu")
        autotuner.submit(10.0)
        autotuner.set_profile_region(autotuner.regions[0])
        autotuner.begin_inherit_profile()
        autotuner.submit_inherit(10.0)
        assert autotuner.generate() == 1
        autotuner.submit(10.1)
        autotuner.save_state(str(state_path))

        expected_index = autotuner.generate()
        assert expected_index >= 0
        expected_hash = autotuner.current_profile_pattern_schemes.schemes[expected_index].hash

        restored = QDQAutotuner(simple_conv_model)
        restored.initialize(_create_test_config())
        restored.set_resume_fingerprint(backend="test", device="cpu")
        restored.load_state(str(state_path))

        actual_index = restored.generate()
        assert actual_index == expected_index
        assert restored.current_profile_pattern_schemes.schemes[actual_index].hash == expected_hash

    def test_completed_pattern_skips_unprofiled_cache_seed_after_resume(
        self, simple_conv_model, tmp_path
    ):
        state_path = tmp_path / "state.yaml"
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.set_resume_fingerprint(backend="test", device="cpu")
        autotuner.submit(10.0)
        region = autotuner.regions[0]
        autotuner.set_profile_region(region)
        autotuner.begin_inherit_profile()
        autotuner.submit_inherit(10.0)
        assert autotuner.generate() == 1
        autotuner.submit(10.1)
        autotuner.current_profile_pattern_schemes.schemes.append(
            InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)])
        )
        autotuner.set_profile_region(None, commit=True)
        assert autotuner.profiled_patterns[0].completed
        autotuner.save_state(str(state_path))

        restored = QDQAutotuner(simple_conv_model)
        restored.initialize(_create_test_config())
        restored.set_resume_fingerprint(backend="test", device="cpu")
        restored.load_state(str(state_path))

        assert restored.profiled_patterns[0].completed
        restored.set_profile_region(restored.regions[0])
        assert restored.current_profile_region is None
        assert restored.current_profile_pattern_schemes is None

    def test_measurement_fingerprint_change_reuses_only_qdq_candidates(
        self, simple_conv_model, tmp_path
    ):
        state_path = tmp_path / "state.yaml"
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        autotuner.set_resume_fingerprint(backend="old")
        autotuner.submit(10.0)
        autotuner.set_profile_region(autotuner.regions[0])
        schemes = autotuner.current_profile_pattern_schemes
        schemes.schemes.append(
            InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=9.0)
        )
        autotuner.save_state(str(state_path))

        restored = QDQAutotuner(simple_conv_model)
        restored.initialize(_create_test_config())
        restored.set_resume_fingerprint(backend="new")
        restored.load_state(str(state_path))

        assert restored.baseline_latency_ms is None
        assert restored.current_profile_pattern_schemes is None
        assert restored.pattern_cache.total_schemes == 1
        assert restored.pattern_cache.pattern_schemes[0].schemes[0].latency_ms == float("inf")

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
        leaf_indices = [i for i, r in enumerate(autotuner.regions) if r.type == RegionType.LEAF]
        non_leaf_indices = [i for i, r in enumerate(autotuner.regions) if r.type != RegionType.LEAF]

        if leaf_indices and non_leaf_indices:
            # All LEAF should come before non-LEAF
            assert max(leaf_indices) < min(non_leaf_indices)

    def test_nested_regions_are_ordered_descendant_first(self):
        root = Region(region_id=0, level=0, region_type=RegionType.ROOT)
        composite = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        sibling_leaf = Region(region_id=2, level=1, region_type=RegionType.LEAF)
        nested_leaf = Region(region_id=3, level=2, region_type=RegionType.LEAF)
        root.add_child(composite)
        root.add_child(sibling_leaf)
        composite.add_child(nested_leaf)

        ordered = QDQAutotuner._sort_regions_for_profiling(
            [root, composite, nested_leaf, sibling_leaf]
        )

        assert [region.id for region in ordered] == [2, 3, 1, 0]
        positions = {region.id: index for index, region in enumerate(ordered)}
        for region in ordered:
            assert all(positions[child.id] < positions[region.id] for child in region.children)

    def test_parent_no_qdq_clears_child_qdq_coverage(self, simple_conv_model):
        autotuner = QDQAutotuner(simple_conv_model)
        autotuner.initialize(_create_test_config())
        child = Region(region_id=1, level=1, region_type=RegionType.LEAF)
        child.nodes = {0}
        child.inputs = ["input", "conv_weight"]
        child.outputs = ["conv_out"]
        parent = Region(region_id=0, level=0, region_type=RegionType.COMPOSITE)
        parent.nodes = {1}
        parent.inputs = ["input", "conv_weight"]
        parent.outputs = ["output"]
        parent.add_child(child)
        autotuner.regions = QDQAutotuner._sort_regions_for_profiling([parent, child])

        child_pattern = RegionPattern.from_region(child, autotuner.graph)
        child_qdq = InsertionScheme(
            action=SchemeAction.QDQ,
            node_inputs=[NodeInputInsertionPoint(node_index=0, input_index=0)],
            latency_ms=8.0,
        )
        child_schemes = PatternSchemes(pattern=child_pattern, schemes=[child_qdq])
        child_schemes.selected_scheme_hash = child_qdq.hash

        parent_pattern = RegionPattern.from_region(parent, autotuner.graph)
        parent_no_qdq = InsertionScheme(action=SchemeAction.NO_QDQ, latency_ms=8.0)
        parent_schemes = PatternSchemes(pattern=parent_pattern, schemes=[parent_no_qdq])
        parent_schemes.selected_scheme_hash = parent_no_qdq.hash
        autotuner.profiled_patterns = [child_schemes, parent_schemes]

        assert autotuner.get_resolved_insertion_points() == set()

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
