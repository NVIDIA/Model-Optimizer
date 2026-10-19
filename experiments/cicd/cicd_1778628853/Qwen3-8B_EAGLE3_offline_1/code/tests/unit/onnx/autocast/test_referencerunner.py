# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile
from collections import OrderedDict

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.referencerunner import ReferenceRunner, TensorStats


def create_multi_io_model():
    # Create a model with 2 inputs and 2 outputs
    # y1 = x1 + x2
    # y2 = x1 * x2
    input1_shape = [1, 3]
    input2_shape = [1, 3]
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, input1_shape)
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, input2_shape)
    y1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3])
    y2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3])

    add_node = helper.make_node("Add", ["X1", "X2"], ["Y1"], name="add")
    mul_node = helper.make_node("Mul", ["X1", "X2"], ["Y2"], name="mul")

    graph = helper.make_graph([add_node, mul_node], "model_multi_io", [x1, x2], [y1, y2])
    model = helper.make_model(graph, producer_name="model_multi_io")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)
    model = onnx_utils.infer_shapes(model)
    return model


@pytest.fixture
def simple_model():
    return create_multi_io_model()


@pytest.fixture
def reference_runner(simple_model):
    return ReferenceRunner(simple_model)


def test_init(simple_model):
    """Test initialization of ReferenceRunner."""
    runner = ReferenceRunner(simple_model)
    assert sorted(runner.input_names) == ["X1", "X2"]
    assert isinstance(runner.model, onnx.ModelProto)


def test_run_with_random_inputs(reference_runner):
    """Test running inference with random inputs."""
    results = reference_runner.run()
    assert isinstance(results, OrderedDict)
    assert "Y1" in results
    assert "Y2" in results
    assert results["Y1"].shape == (1, 3)
    assert results["Y2"].shape == (1, 3)


def test_run_with_json_inputs(reference_runner):
    """Test running inference with JSON inputs."""
    # Create test inputs
    inputs = {
        "X1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "X2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    }

    # Save inputs to temporary JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        from polygraphy.json import save_json

        input_path = f.name
        save_json([inputs], input_path, description="test input data")
    try:
        results = reference_runner.run(input_path)
        assert isinstance(results, OrderedDict)
        assert "Y1" in results
        assert "Y2" in results
        assert results["Y1"].shape == (1, 3)
        assert results["Y2"].shape == (1, 3)
    finally:
        os.remove(input_path)


def test_run_with_npz_inputs(reference_runner):
    """Test running inference with NPZ inputs."""
    # Create test inputs
    inputs = {
        "X1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "X2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    }

    # Save inputs to temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, **inputs)
        input_path = f.name

    try:
        results = reference_runner.run(input_path)
        assert isinstance(results, OrderedDict)
        assert "Y1" in results
        assert "Y2" in results
        assert results["Y1"].shape == (1, 3)
        assert results["Y2"].shape == (1, 3)
    finally:
        os.remove(input_path)


def test_run_with_dict_inputs(reference_runner):
    """Test running inference with dict inputs."""
    inputs = {
        "X1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "X2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    }
    results = reference_runner.run(inputs)
    assert isinstance(results, OrderedDict)
    assert "Y1" in results
    assert "Y2" in results
    assert results["Y1"].shape == (1, 3)
    assert results["Y2"].shape == (1, 3)


def test_invalid_input_format(reference_runner):
    """Test error handling for invalid input format."""
    with pytest.raises(ValueError, match=r"Supported input types:.*"):
        reference_runner.run("invalid.txt")


def test_mismatched_input_names(reference_runner):
    """Test error handling for mismatched input names."""
    inputs = {
        "wrong_name1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "wrong_name2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        from polygraphy.json import save_json

        input_path = f.name
        save_json([inputs], input_path, description="test input data")

    try:
        with pytest.raises(ValueError, match="Input names from ONNX model do not match"):
            reference_runner.run(input_path)
    finally:
        os.remove(input_path)


def test_invalid_json(reference_runner):
    """Test error handling for non-Polygraphy JSON format."""
    inputs = {"X1": [[1.0, 2.0, 3.0]], "X2": [[4.0, 5.0, 6.0]]}

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        import json

        json.dump(inputs, f)
        input_path = f.name
    try:
        with pytest.raises(ValueError, match=r"Invalid input file\."):
            reference_runner.run(input_path)
    finally:
        os.remove(input_path)


def test_invalid_npz_file(reference_runner):
    """Test error handling for invalid NPZ file."""
    # Create a single numpy array and save as NPZ (incorrect format)
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.save(f, data)
        input_path = f.name
    try:
        with pytest.raises(ValueError, match=r"Invalid input file\."):
            reference_runner.run(input_path)
    finally:
        os.remove(input_path)


def test_compare_outputs(reference_runner):
    """Test running with different inputs and comparing outputs."""
    # First run with input set 1
    inputs1 = {
        "X1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "X2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        from polygraphy.json import save_json

        input_path = f.name
        save_json([inputs1], input_path, description="test input data 1")

    try:
        outputs1 = reference_runner.run(input_path)
        # Expected: Y1 = X1 + X2, Y2 = X1 * X2
        expected_y1 = inputs1["X1"] + inputs1["X2"]  # [5.0, 7.0, 9.0]
        expected_y2 = inputs1["X1"] * inputs1["X2"]  # [4.0, 10.0, 18.0]
        np.testing.assert_allclose(outputs1["Y1"], expected_y1)
        np.testing.assert_allclose(outputs1["Y2"], expected_y2)
    finally:
        os.remove(input_path)

    # Second run with input set 2
    inputs2 = {
        "X1": np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        "X2": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        from polygraphy.json import save_json

        input_path = f.name
        save_json([inputs2], input_path, description="test input data 2")

    try:
        outputs2 = reference_runner.run(input_path)
        # Expected: Y1 = X1 + X2, Y2 = X1 * X2
        expected_y1 = inputs2["X1"] + inputs2["X2"]  # [3.0, 5.0, 7.0]
        expected_y2 = inputs2["X1"] * inputs2["X2"]  # [2.0, 6.0, 12.0]
        np.testing.assert_allclose(outputs2["Y1"], expected_y1)
        np.testing.assert_allclose(outputs2["Y2"], expected_y2)
    finally:
        os.remove(input_path)


def test_tensor_stats():
    """Test TensorStats dataclass functionality."""
    stats = TensorStats(absmax=10.5, min_val=-5.0, max_val=10.5, shape=(2, 3))

    assert stats.absmax == 10.5
    assert stats.min_val == -5.0
    assert stats.max_val == 10.5
    assert stats.shape == (2, 3)
    assert stats.size == 6
    assert abs(stats) == 10.5  # Test __abs__ method


def test_run_with_multi_batch_npz_directory(reference_runner):
    """Test running inference with directory containing multiple NPZ files."""
    # Create temporary directory with multiple NPZ files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create batch 1
        inputs1 = {
            "X1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "X2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        }
        np.savez(os.path.join(temp_dir, "batch_001.npz"), **inputs1)

        # Create batch 2
        inputs2 = {
            "X1": np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            "X2": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }
        np.savez(os.path.join(temp_dir, "batch_002.npz"), **inputs2)

        # Create batch 3
        inputs3 = {
            "X1": np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
            "X2": np.array([[8.0, 9.0, 10.0]], dtype=np.float32),
        }
        np.savez(os.path.join(temp_dir, "batch_003.npz"), **inputs3)

        # Run with directory path
        results = reference_runner.run(temp_dir)

        # Should return TensorStats objects for multi-batch
        assert isinstance(results, OrderedDict)
        assert "Y1" in results
        assert "Y2" in results

        # Check that results are TensorStats objects
        assert isinstance(results["Y1"], TensorStats)
        assert isinstance(results["Y2"], TensorStats)

        # Verify aggregated statistics
        # Y1 = X1 + X2
        # Batch 1: [5.0, 7.0, 9.0]
        # Batch 2: [3.0, 5.0, 7.0]
        # Batch 3: [13.0, 15.0, 17.0]
        assert results["Y1"].absmax == 17.0
        assert results["Y1"].min_val == 3.0
        assert results["Y1"].max_val == 17.0
        assert results["Y1"].shape == (1, 3)

        # Y2 = X1 * X2
        # Batch 1: [4.0, 10.0, 18.0]
        # Batch 2: [2.0, 6.0, 12.0]
        # Batch 3: [40.0, 54.0, 70.0]
        assert results["Y2"].absmax == 70.0
        assert results["Y2"].min_val == 2.0
        assert results["Y2"].max_val == 70.0
        assert results["Y2"].shape == (1, 3)


def test_run_with_empty_npz_directory(reference_runner):
    """Test error handling for empty NPZ directory."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        pytest.raises(ValueError, match="No NPZ files found in directory"),
    ):
        reference_runner.run(temp_dir)


def test_single_batch_backward_compatibility(reference_runner):
    """Test that single batch still returns raw numpy arrays (backward compatibility)."""
    inputs = {
        "X1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "X2": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, **inputs)
        input_path = f.name

    try:
        results = reference_runner.run(input_path)
        # Single batch should return raw numpy arrays, not TensorStats
        assert isinstance(results, OrderedDict)
        assert "Y1" in results
        assert "Y2" in results
        assert isinstance(results["Y1"], np.ndarray)
        assert isinstance(results["Y2"], np.ndarray)
        assert not isinstance(results["Y1"], TensorStats)
        assert not isinstance(results["Y2"], TensorStats)
    finally:
        os.remove(input_path)


def test_multi_batch_aggregation_statistics(reference_runner):
    """Test that multi-batch aggregation correctly computes statistics across batches."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create batches with different value ranges
        # Batch 1: small values
        inputs1 = {
            "X1": np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
            "X2": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }
        np.savez(os.path.join(temp_dir, "batch_001.npz"), **inputs1)

        # Batch 2: large values
        inputs2 = {
            "X1": np.array([[-10.0, 0.0, 10.0]], dtype=np.float32),
            "X2": np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
        }
        np.savez(os.path.join(temp_dir, "batch_002.npz"), **inputs2)

        # Batch 3: mixed values
        inputs3 = {
            "X1": np.array([[5.0, -5.0, 0.0]], dtype=np.float32),
            "X2": np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        }
        np.savez(os.path.join(temp_dir, "batch_003.npz"), **inputs3)

        results = reference_runner.run(temp_dir)

        # Y1 = X1 + X2
        # Batch 1: [0.0, 2.0, 4.0] -> absmax=4.0, min=0.0, max=4.0
        # Batch 2: [-5.0, 6.0, 17.0] -> absmax=17.0, min=-5.0, max=17.0
        # Batch 3: [7.0, -2.0, 4.0] -> absmax=7.0, min=-2.0, max=7.0
        # Aggregated: absmax=17.0, min=-5.0, max=17.0
        assert results["Y1"].absmax == 17.0
        assert results["Y1"].min_val == -5.0
        assert results["Y1"].max_val == 17.0

        # Y2 = X1 * X2
        # Batch 1: [-1.0, 0.0, 3.0] -> absmax=3.0, min=-1.0, max=3.0
        # Batch 2: [-50.0, 0.0, 70.0] -> absmax=70.0, min=-50.0, max=70.0
        # Batch 3: [10.0, -15.0, 0.0] -> absmax=15.0, min=-15.0, max=10.0
        # Aggregated: absmax=70.0, min=-50.0, max=70.0
        assert results["Y2"].absmax == 70.0
        assert results["Y2"].min_val == -50.0
        assert results["Y2"].max_val == 70.0
