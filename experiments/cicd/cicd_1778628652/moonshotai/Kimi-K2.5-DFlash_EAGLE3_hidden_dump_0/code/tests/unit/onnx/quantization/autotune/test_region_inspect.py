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

"""Unit tests for region_inspect module."""

import os
from unittest.mock import Mock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def create_simple_onnx_model():
    """Create a simple ONNX model for testing.

    Creates a model with: Input -> Conv -> Relu -> MatMul -> Output
    """
    # Create input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1000])

    # Create weights for Conv
    conv_weight = np.random.randn(64, 3, 7, 7).astype(np.float32)
    conv_weight_tensor = numpy_helper.from_array(conv_weight, "conv_weight")

    # Create weights for MatMul
    matmul_weight = np.random.randn(64, 1000).astype(np.float32)
    matmul_weight_tensor = numpy_helper.from_array(matmul_weight, "matmul_weight")

    # Create nodes
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["conv_output"],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_output"],
        outputs=["relu_output"],
    )

    flatten_node = helper.make_node(
        "Flatten",
        inputs=["relu_output"],
        outputs=["flatten_output"],
        axis=1,
    )

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["flatten_output", "matmul_weight"],
        outputs=["output"],
    )

    # Create graph
    graph = helper.make_graph(
        [conv_node, relu_node, flatten_node, matmul_node],
        "test_model",
        [input_tensor],
        [output_tensor],
        [conv_weight_tensor, matmul_weight_tensor],
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 13

    return model


@pytest.fixture
def simple_onnx_model():
    """Fixture that provides a simple ONNX model."""
    return create_simple_onnx_model()


@pytest.fixture
def onnx_model_file(tmp_path, simple_onnx_model):
    """Fixture that provides a path to a saved ONNX model."""
    model_path = os.path.join(tmp_path, "test_model.onnx")
    onnx.save(simple_onnx_model, model_path)
    return model_path


class TestRegionInspectImports:
    """Test that the region_inspect module can be imported."""

    def test_module_imports(self):
        """Test that the module imports without errors when dependencies exist."""
        # This test will skip if the required dependencies don't exist
        try:
            from modelopt.onnx.quantization.autotune import region_inspect

            assert hasattr(region_inspect, "inspect_region_search")
            assert hasattr(region_inspect, "main")
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


class TestRegionInspectWithMocks:
    """Test region_inspect functionality with mocked dependencies."""

    @patch("modelopt.onnx.quantization.autotune.region_inspect.CombinedRegionSearch")
    @patch("modelopt.onnx.quantization.autotune.region_inspect.has_quantizable_operations")
    def test_inspect_region_search_basic(
        self, mock_has_quantizable, mock_combined_search, onnx_model_file
    ):
        """Test basic functionality of inspect_region_search with mocked dependencies."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import inspect_region_search
        except ImportError:
            pytest.skip("Required dependencies not available")

        # Setup mocks
        mock_region = Mock()
        mock_region.type = Mock(value="LEAF")
        mock_region.inputs = ["input1"]
        mock_region.outputs = ["output1"]
        mock_region.children = []
        mock_region.get_region_nodes_and_descendants.return_value = [Mock(), Mock()]
        mock_region.get_children.return_value = []

        mock_search_instance = Mock()
        mock_search_instance.search_regions.return_value = [mock_region]
        mock_search_instance.print_tree = Mock()
        mock_combined_search.return_value = mock_search_instance

        mock_has_quantizable.return_value = True

        # Call the function
        result = inspect_region_search(
            onnx_path=onnx_model_file, max_sequence_size=10, include_all_regions=False
        )

        # Verify the function was called correctly
        assert mock_combined_search.called
        assert mock_search_instance.search_regions.called
        assert isinstance(result, list)

    @patch("modelopt.onnx.quantization.autotune.region_inspect.CombinedRegionSearch")
    @patch("modelopt.onnx.quantization.autotune.region_inspect.has_quantizable_operations")
    def test_inspect_region_search_with_custom_params(
        self, mock_has_quantizable, mock_combined_search, onnx_model_file
    ):
        """Test inspect_region_search with custom parameters."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import inspect_region_search
        except ImportError:
            pytest.skip("Required dependencies not available")

        # Setup mocks
        mock_region = Mock()
        mock_region.type = Mock(value="COMPOSITE")
        mock_region.inputs = ["input1"]
        mock_region.outputs = ["output1"]
        mock_region.children = []
        mock_region.get_region_nodes_and_descendants.return_value = [Mock()]
        mock_region.get_children.return_value = []

        mock_search_instance = Mock()
        mock_search_instance.search_regions.return_value = [mock_region]
        mock_search_instance.print_tree = Mock()
        mock_combined_search.return_value = mock_search_instance

        mock_has_quantizable.return_value = True

        # Call with custom parameters
        result = inspect_region_search(
            onnx_path=onnx_model_file, max_sequence_size=20, include_all_regions=True
        )

        # Verify custom parameters were used
        assert mock_combined_search.called
        call_kwargs = mock_combined_search.call_args[1]
        assert call_kwargs.get("maximum_sequence_region_size") == 20
        assert isinstance(result, list)

    @patch("modelopt.onnx.quantization.autotune.region_inspect.CombinedRegionSearch")
    @patch("modelopt.onnx.quantization.autotune.region_inspect.has_quantizable_operations")
    def test_inspect_region_search_filtering(
        self, mock_has_quantizable, mock_combined_search, onnx_model_file
    ):
        """Test that regions without quantizable operations are filtered out."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import inspect_region_search
        except ImportError:
            pytest.skip("Required dependencies not available")

        # Setup mocks - one region with quantizable ops, one without
        mock_region_quantizable = Mock()
        mock_region_quantizable.type = Mock(value="LEAF")
        mock_region_quantizable.inputs = ["input1"]
        mock_region_quantizable.outputs = ["output1"]
        mock_region_quantizable.get_region_nodes_and_descendants.return_value = [Mock()]
        mock_region_quantizable.get_children.return_value = []

        mock_region_non_quantizable = Mock()
        mock_region_non_quantizable.type = Mock(value="LEAF")
        mock_region_non_quantizable.inputs = ["input2"]
        mock_region_non_quantizable.outputs = ["output2"]
        mock_region_non_quantizable.get_region_nodes_and_descendants.return_value = [Mock()]
        mock_region_non_quantizable.get_children.return_value = []

        mock_search_instance = Mock()
        mock_search_instance.search_regions.return_value = [
            mock_region_quantizable,
            mock_region_non_quantizable,
        ]
        mock_search_instance.print_tree = Mock()
        mock_combined_search.return_value = mock_search_instance

        # First region has quantizable ops, second doesn't
        mock_has_quantizable.side_effect = [True, False]

        # Call with filtering enabled
        result = inspect_region_search(
            onnx_path=onnx_model_file, max_sequence_size=10, include_all_regions=False
        )

        # Should only return the quantizable region
        assert len(result) == 1


class TestRegionInspectMain:
    """Test the main CLI entry point."""

    @patch("modelopt.onnx.quantization.autotune.region_inspect.inspect_region_search")
    def test_main_success(self, mock_inspect, onnx_model_file):
        """Test main function with successful execution."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import main
        except ImportError:
            pytest.skip("Required dependencies not available")

        mock_inspect.return_value = [Mock(), Mock()]

        with patch("sys.argv", ["region_inspect", "--model", onnx_model_file]):
            exit_code = main()
            assert exit_code == 0
            assert mock_inspect.called

    @patch("modelopt.onnx.quantization.autotune.region_inspect.inspect_region_search")
    def test_main_with_verbose(self, mock_inspect, onnx_model_file):
        """Test main function with verbose flag."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import main
        except ImportError:
            pytest.skip("Required dependencies not available")

        mock_inspect.return_value = [Mock()]

        with patch("sys.argv", ["region_inspect", "--model", onnx_model_file, "--verbose"]):
            exit_code = main()
            assert exit_code == 0

    @patch("modelopt.onnx.quantization.autotune.region_inspect.inspect_region_search")
    def test_main_with_custom_max_sequence_size(self, mock_inspect, onnx_model_file):
        """Test main function with custom max_sequence_size."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import main
        except ImportError:
            pytest.skip("Required dependencies not available")

        mock_inspect.return_value = [Mock()]

        with patch(
            "sys.argv", ["region_inspect", "--model", onnx_model_file, "--max-sequence-size", "20"]
        ):
            exit_code = main()
            assert exit_code == 0
            # Verify max_sequence_size parameter was passed
            call_kwargs = mock_inspect.call_args[1]
            assert call_kwargs.get("max_sequence_size") == 20

    @patch("modelopt.onnx.quantization.autotune.region_inspect.inspect_region_search")
    def test_main_with_include_all_regions(self, mock_inspect, onnx_model_file):
        """Test main function with include_all_regions flag."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import main
        except ImportError:
            pytest.skip("Required dependencies not available")

        mock_inspect.return_value = [Mock()]

        with patch(
            "sys.argv", ["region_inspect", "--model", onnx_model_file, "--include-all-regions"]
        ):
            exit_code = main()
            assert exit_code == 0
            # Verify include_all_regions parameter was passed
            call_kwargs = mock_inspect.call_args[1]
            assert call_kwargs.get("include_all_regions") is True

    @patch("modelopt.onnx.quantization.autotune.region_inspect.inspect_region_search")
    def test_main_failure(self, mock_inspect, onnx_model_file):
        """Test main function with execution failure."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import main
        except ImportError:
            pytest.skip("Required dependencies not available")

        mock_inspect.side_effect = Exception("Test error")

        with patch("sys.argv", ["region_inspect", "--model", onnx_model_file]):
            exit_code = main()
            assert exit_code == 1


class TestRegionInspectModelLoading:
    """Test model loading functionality."""

    @patch("modelopt.onnx.quantization.autotune.region_inspect.CombinedRegionSearch")
    @patch("modelopt.onnx.quantization.autotune.region_inspect.has_quantizable_operations")
    def test_loads_valid_onnx_model(
        self, mock_has_quantizable, mock_combined_search, onnx_model_file
    ):
        """Test that a valid ONNX model can be loaded."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import inspect_region_search
        except ImportError:
            pytest.skip("Required dependencies not available")

        # Setup minimal mocks
        mock_region = Mock()
        mock_region.type = Mock(value="LEAF")
        mock_region.inputs = []
        mock_region.outputs = []
        mock_region.get_region_nodes_and_descendants.return_value = []
        mock_region.get_children.return_value = []

        mock_search_instance = Mock()
        mock_search_instance.search_regions.return_value = [mock_region]
        mock_search_instance.print_tree = Mock()
        mock_combined_search.return_value = mock_search_instance
        mock_has_quantizable.return_value = False

        # Should not raise an exception
        result = inspect_region_search(onnx_model_file)
        assert isinstance(result, list)

    def test_fails_on_nonexistent_file(self):
        """Test that loading a non-existent file raises an error."""
        try:
            from modelopt.onnx.quantization.autotune.region_inspect import inspect_region_search
        except ImportError:
            pytest.skip("Required dependencies not available")

        with pytest.raises(Exception):  # Could be FileNotFoundError or other
            inspect_region_search("/nonexistent/path/to/model.onnx")
