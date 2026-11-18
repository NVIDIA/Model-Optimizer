# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for modelopt.onnx.quantization.ort_patching module."""

from unittest.mock import Mock, patch

import numpy as np
import onnx
import pytest
from onnx import onnx_pb
from onnxruntime.quantization.base_quantizer import BaseQuantizer
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    DistributionCalibrater,
    EntropyCalibrater,
    HistogramCollector,
    MinMaxCalibrater,
    PercentileCalibrater,
    TensorData,
    TensorsData,
)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import QuantType
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from modelopt.onnx.quantization.ort_patching import (
    _adjust_tensor_ranges,
    _augment_graph_min_max_calibrater_single_node_calibration,
    _check_opset_version,
    _collect_absolute_value,
    _collect_data_histogram_calibrator,
    _collect_data_min_max_calibrater_single_node_calibration,
    _collect_data_minmax_calibrator,
    _collect_histogram_collector_single_node_calibration,
    _collect_value,
    _collect_value_histogram_collector_single_node_calibration,
    _compute_data_min_max_calibrater_single_node_calibration,
    _compute_data_minmax_calibrator,
    _create_calibrator_with_extra_options,
    _create_inference_session_with_ep_config,
    _init_calibrater_base,
    _merge_range_min_max_calibrater_single_node_calibration,
    _merge_range_minmax_calibrator,
    _quantize_static,
    _select_tensors_to_calibrate,
    load_model_with_shape_infer,
)


@pytest.fixture
def simple_onnx_model():
    """Create a simple ONNX model for testing."""
    # Create a simple model with one Add operation
    input1 = onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 3])
    input2 = onnx.helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [1, 3])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])

    add_node = onnx.helper.make_node("Add", ["input1", "input2"], ["output"], name="add_node")

    graph = onnx.helper.make_graph([add_node], "simple_model", [input1, input2], [output])

    model = onnx.helper.make_model(graph)
    model.opset_import[0].version = 11
    return model


@pytest.fixture
def mock_calibrator():
    """Create a mock calibrator for testing."""
    calibrator = Mock(spec=MinMaxCalibrater)
    calibrator.intermediate_outputs = []
    calibrator.model_original_outputs = ["output"]
    calibrator.num_model_outputs = 1
    calibrator.symmetric = False
    calibrator.moving_average = False
    calibrator.calibrate_tensors_range = None
    calibrator.group_qdq_tensors = None
    calibrator.single_node_model_path_map = {}
    calibrator.providers = []
    calibrator.trt_extra_plugin_lib_paths = None
    return calibrator


@pytest.fixture
def mock_histogram_collector():
    """Create a mock histogram collector for testing."""
    collector = Mock(spec=HistogramCollector)
    collector.histogram_dict = {}
    collector.num_bins = 128
    collector.method = "entropy"
    collector.symmetric = False
    return collector


@pytest.fixture
def sample_tensor_data():
    """Create sample tensor data for testing."""
    return {
        "tensor1": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        "tensor2": [np.array([-1.0, 0.0, 1.0]), np.array([2.0, -2.0, 0.5])],
    }


class TestModelLoading:
    """Test model loading functions."""

    def test_load_model_with_shape_infer(self, simple_onnx_model, tmp_path):
        """Test loading model with shape inference."""
        model_path = tmp_path / "test_model.onnx"
        onnx.save(simple_onnx_model, str(model_path))

        with patch("modelopt.onnx.quantization.ort_patching.onnx_utils") as mock_ssi:
            mock_ssi.infer_shapes.return_value = simple_onnx_model
            with patch("modelopt.onnx.quantization.ort_patching.add_infer_metadata") as mock_aim:
                result = load_model_with_shape_infer(model_path)

                mock_ssi.infer_shapes.assert_called_once()
                mock_aim.assert_called_once_with(simple_onnx_model)
                assert result == simple_onnx_model


class TestHistogramCollection:
    """Test histogram collection functions."""

    def test_collect_value(self, mock_histogram_collector, sample_tensor_data):
        """Test _collect_value function."""
        mock_histogram_collector.merge_histogram.return_value = (
            np.array([1, 2, 3]),
            np.array([0, 1, 2, 3]),
            -1.0,
            6.0,
            6.0,
        )

        _collect_value(mock_histogram_collector, sample_tensor_data)

        # Check that histogram_dict was populated
        assert len(mock_histogram_collector.histogram_dict) == 2
        assert "tensor1" in mock_histogram_collector.histogram_dict
        assert "tensor2" in mock_histogram_collector.histogram_dict

    def test_collect_absolute_value(self, mock_histogram_collector, sample_tensor_data):
        """Test _collect_absolute_value function."""
        # Convert to float32 to avoid the float64 assertion error
        sample_tensor_data_f32 = {
            "tensor1": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
            "tensor2": [
                np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                np.array([2.0, -2.0, 0.5], dtype=np.float32),
            ],
        }
        _collect_absolute_value(mock_histogram_collector, sample_tensor_data_f32)

        # Check that histogram_dict was populated
        assert len(mock_histogram_collector.histogram_dict) == 2
        assert "tensor1" in mock_histogram_collector.histogram_dict
        assert "tensor2" in mock_histogram_collector.histogram_dict

    def test_collect_value_histogram_collector_single_node(self, mock_histogram_collector):
        """Test single node histogram collection."""
        name_to_arr = {"tensor1": np.array([1.0, 2.0, 3.0, -1.0])}
        mock_histogram_collector.merge_histogram.return_value = (
            np.array([1, 2, 3]),
            np.array([0, 1, 2, 3]),
            -1.0,
            3.0,
            3.0,
        )

        _collect_value_histogram_collector_single_node_calibration(
            mock_histogram_collector, name_to_arr
        )

        assert "tensor1" in mock_histogram_collector.histogram_dict

    def test_collect_histogram_collector_single_node(self, mock_histogram_collector):
        """Test histogram collector for single node calibration."""
        name_to_arr = {"tensor1": np.array([1.0, 2.0, 3.0])}
        mock_histogram_collector.collect_value = Mock()
        mock_histogram_collector.collect_absolute_value = Mock()

        # Test entropy method
        mock_histogram_collector.method = "entropy"
        _collect_histogram_collector_single_node_calibration(mock_histogram_collector, name_to_arr)
        mock_histogram_collector.collect_value.assert_called_once()

        # Test percentile method with symmetric
        mock_histogram_collector.method = "percentile"
        mock_histogram_collector.symmetric = True
        _collect_histogram_collector_single_node_calibration(mock_histogram_collector, name_to_arr)
        mock_histogram_collector.collect_absolute_value.assert_called_once()

        # Test unsupported method
        mock_histogram_collector.method = "unsupported"
        with pytest.raises(ValueError):
            _collect_histogram_collector_single_node_calibration(
                mock_histogram_collector, name_to_arr
            )


class TestMinMaxCalibration:
    """Test MinMax calibration functions."""

    def test_compute_data_minmax_calibrator(self, mock_calibrator):
        """Test compute data for MinMax calibrator."""
        # Setup mock data - empty intermediate_outputs should return calibrate_tensors_range
        mock_calibrator.intermediate_outputs = []

        result = _compute_data_minmax_calibrator(mock_calibrator)

        assert result == mock_calibrator.calibrate_tensors_range

    def test_compute_data_single_node_calibration(self, mock_calibrator):
        """Test compute data for single node calibration."""
        # Empty intermediate outputs case
        mock_calibrator.intermediate_outputs = []

        result = _compute_data_min_max_calibrater_single_node_calibration(mock_calibrator)

        assert result == mock_calibrator.calibrate_tensors_range

    def test_collect_data_minmax_calibrator(self, mock_calibrator):
        """Test collect data for MinMax calibrator."""
        mock_data_reader = Mock(spec=CalibrationDataReader)
        mock_data_reader.get_next.side_effect = [
            {"input1": np.array([1.0, 2.0])},
            {"input1": np.array([3.0, 4.0])},
            None,
        ]

        mock_calibrator.infer_session = Mock()
        mock_calibrator.infer_session.run.return_value = [np.array([1.0, 2.0])]
        mock_calibrator.compute_data.return_value = TensorsData(CalibrationMethod.MinMax, {})
        mock_calibrator.clear_collected_data = Mock()

        _collect_data_minmax_calibrator(mock_calibrator, mock_data_reader)

        # Should be called twice (one for each input batch)
        assert mock_calibrator.compute_data.call_count == 2
        assert mock_calibrator.clear_collected_data.call_count == 2

    def test_merge_range_minmax_calibrator(self, mock_calibrator):
        """Test merge range for MinMax calibrator."""
        old_range = TensorsData(
            CalibrationMethod.MinMax,
            {"tensor1": TensorData(lowest=np.float32(1.0), highest=np.float32(3.0))},
        )
        new_range = TensorsData(
            CalibrationMethod.MinMax,
            {"tensor1": TensorData(lowest=np.float32(0.5), highest=np.float32(4.0))},
        )

        result = _merge_range_minmax_calibrator(mock_calibrator, old_range, new_range)

        assert result.data["tensor1"].range_value == (np.float32(0.5), np.float32(4.0))

    def test_merge_range_single_node_calibration(self, mock_calibrator):
        """Test merge range for single node calibration."""
        old_range = TensorsData(
            CalibrationMethod.MinMax,
            {"tensor1": TensorData(lowest=np.float32(1.0), highest=np.float32(3.0))},
        )
        new_range = TensorsData(
            CalibrationMethod.MinMax,
            {
                "tensor1": TensorData(lowest=np.float32(0.5), highest=np.float32(4.0)),
                "tensor2": TensorData(lowest=np.float32(-1.0), highest=np.float32(1.0)),
            },
        )

        result = _merge_range_min_max_calibrater_single_node_calibration(
            mock_calibrator, old_range, new_range
        )

        assert result.data["tensor1"].range_value == (np.float32(0.5), np.float32(4.0))
        assert result.data["tensor2"].range_value == (np.float32(-1.0), np.float32(1.0))


class TestHistogramCalibration:
    """Test histogram calibration functions."""

    def test_collect_data_histogram_calibrator(self, mock_calibrator):
        """Test collect data for histogram calibrator."""
        mock_data_reader = Mock(spec=CalibrationDataReader)
        mock_data_reader.get_next.side_effect = [{"input1": np.array([1.0, 2.0])}, None]

        mock_calibrator.infer_session = Mock()
        mock_calibrator.infer_session.run.return_value = [np.array([1.0, 2.0])]
        mock_calibrator.infer_session.get_outputs.return_value = [Mock(name="tensor1")]
        mock_calibrator.tensors_to_calibrate = {"tensor1"}
        mock_calibrator.collector = Mock()
        mock_calibrator.clear_collected_data = Mock()

        _collect_data_histogram_calibrator(mock_calibrator, mock_data_reader)

        mock_calibrator.collector.collect.assert_called_once()
        mock_calibrator.clear_collected_data.assert_called_once()


class TestTensorSelection:
    """Test tensor selection functions."""

    def test_select_tensors_to_calibrate(self, simple_onnx_model):
        """Test tensor selection for calibration."""
        mock_calibrator = Mock()
        mock_calibrator.op_types_to_calibrate = ["add_node"]  # Node names as op types

        tensors, value_infos = _select_tensors_to_calibrate(mock_calibrator, simple_onnx_model)

        assert isinstance(tensors, set)
        assert isinstance(value_infos, dict)


class TestInferenceSession:
    """Test inference session creation functions."""

    def test_create_inference_session_with_ep_config(self, mock_calibrator, tmp_path):
        """Test inference session creation with EP configuration."""
        model_path = tmp_path / "test_model.onnx"
        model_path.write_text("dummy")

        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_inference_session = Mock()
            mock_session.return_value = mock_inference_session

            _create_inference_session_with_ep_config(
                mock_calibrator,
                model_path=str(model_path),
                execution_providers=["CPUExecutionProvider"],
                trt_extra_plugin_lib_paths="/path/to/plugin",
                group_qdq_tensors={"tensor1": ["tensor2"]},
            )

            # Verify that calibrator attributes are set correctly
            assert mock_calibrator.trt_extra_plugin_lib_paths == "/path/to/plugin"
            assert mock_calibrator.infer_session == mock_inference_session
            assert mock_calibrator.group_qdq_tensors == {"tensor1": ["tensor2"]}

            # Verify InferenceSession was called with correct parameters
            mock_session.assert_called_once()
            call_args = mock_session.call_args

            # Check that the model path was passed correctly
            assert call_args[0][0] == str(model_path)

            # Check that session options were configured
            sess_options = call_args[1]["sess_options"]
            assert sess_options is not None

            # Check that providers were configured correctly
            providers = call_args[1]["providers"]
            assert len(providers) == 2
            # CPUExecutionProvider should be converted to tuple format with arena config
            provider_tuple = providers[1]
            assert provider_tuple[0] == "CPUExecutionProvider"
            assert isinstance(provider_tuple[1], dict)
            assert provider_tuple[1]["arena_extend_strategy"] == "kSameAsRequested"


class TestQuantizerFunctions:
    """Test quantizer-related functions."""

    def test_check_opset_version(self):
        """Test opset version checking."""
        mock_quantizer = Mock(spec=QDQQuantizer)
        # Create nested mock structure
        mock_quantizer.model = Mock()
        mock_quantizer.model.model = Mock()
        mock_quantizer.model.model.opset_import = [Mock(domain="ai.onnx", version=11)]
        mock_quantizer.weight_qType = onnx_pb.TensorProto.INT8
        mock_quantizer.fuse_dynamic_quant = False

        result = _check_opset_version(mock_quantizer)

        assert result == 11
        assert mock_quantizer.fuse_dynamic_quant is True

    def test_adjust_tensor_ranges(self):
        """Test tensor ranges adjustment."""
        mock_quantizer = Mock(spec=BaseQuantizer)

        # Create mock model with nodes
        mock_node1 = Mock()
        mock_node1.op_type = "Clip"
        mock_node1.input = ["input1"]
        mock_node1.output = ["output1"]

        mock_node2 = Mock()
        mock_node2.op_type = "Softmax"
        mock_node2.output = ["output2"]

        mock_quantizer.model = Mock()
        mock_quantizer.model.nodes.return_value = [mock_node1, mock_node2]
        mock_quantizer.tensors_range = {
            "input1": TensorData(lowest=np.float32(-2.0), highest=np.float32(2.0)),
            "output1": TensorData(lowest=np.float32(0.0), highest=np.float32(1.0)),
            "output2": TensorData(lowest=np.float32(0.2), highest=np.float32(0.8)),
        }
        mock_quantizer.is_activation_symmetric = False
        mock_quantizer.should_quantize_node.return_value = True
        mock_quantizer.model.input_name_to_nodes.return_value = {"input1": [mock_node1]}

        _adjust_tensor_ranges(mock_quantizer)

        # Check Softmax range was adjusted to [0, 1]
        softmax_range = mock_quantizer.tensors_range["output2"]
        assert softmax_range.range_value == (np.float32(0.0), np.float32(1.0))


class TestCalibratorCreation:
    """Test calibrator creation functions."""

    def test_create_calibrator_with_extra_options(self, simple_onnx_model, tmp_path):
        """Test calibrator creation with extra options."""
        model_path = tmp_path / "test_model.onnx"
        onnx.save(simple_onnx_model, str(model_path))

        with (
            patch.object(MinMaxCalibrater, "__init__", return_value=None) as mock_init,
            patch.object(MinMaxCalibrater, "augment_graph") as mock_augment,
            patch.object(MinMaxCalibrater, "create_inference_session") as mock_create,
        ):
            calibrator = _create_calibrator_with_extra_options(
                str(model_path),
                op_types_to_calibrate=["Conv"],
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"symmetric": True},
            )

            mock_init.assert_called_once()
            mock_augment.assert_called_once()
            mock_create.assert_called_once()
            assert isinstance(calibrator, MinMaxCalibrater)

    def test_create_calibrator_entropy_method(self, simple_onnx_model, tmp_path):
        """Test calibrator creation with entropy method."""
        model_path = tmp_path / "test_model.onnx"
        onnx.save(simple_onnx_model, str(model_path))

        with (
            patch.object(EntropyCalibrater, "__init__", return_value=None),
            patch.object(EntropyCalibrater, "augment_graph"),
            patch.object(EntropyCalibrater, "create_inference_session"),
        ):
            calibrator = _create_calibrator_with_extra_options(
                str(model_path),
                calibrate_method=CalibrationMethod.Entropy,
                extra_options={"num_bins": 256},
            )

            assert isinstance(calibrator, EntropyCalibrater)

    def test_create_calibrator_unsupported_method(self, simple_onnx_model, tmp_path):
        """Test calibrator creation with unsupported method."""
        model_path = tmp_path / "test_model.onnx"
        onnx.save(simple_onnx_model, str(model_path))

        with pytest.raises(ValueError, match="Unsupported calibration method"):
            _create_calibrator_with_extra_options(
                str(model_path), calibrate_method="UnsupportedMethod"
            )


class TestStaticQuantization:
    """Test static quantization function."""

    def test_quantize_static_basic(self, simple_onnx_model, tmp_path):
        """Test basic static quantization."""
        model_input = tmp_path / "input_model.onnx"
        model_output = tmp_path / "output_model.onnx"
        onnx.save(simple_onnx_model, str(model_input))

        mock_data_reader = Mock(spec=CalibrationDataReader)

        with (
            patch(
                "modelopt.onnx.quantization.ort_patching.calibrate.create_calibrator"
            ) as mock_create,
            patch("modelopt.onnx.quantization.ort_patching.QDQQuantizer") as mock_quantizer_class,
        ):
            mock_calibrator = Mock()
            mock_calibrator.collect_data = Mock()
            mock_calibrator.compute_data.return_value = TensorsData(CalibrationMethod.MinMax, {})
            mock_create.return_value = mock_calibrator

            mock_quantizer = Mock()
            mock_quantizer.quantize_model = Mock()
            mock_quantizer.model.save_model_to_file = Mock()
            mock_quantizer_class.return_value = mock_quantizer

            _quantize_static(
                str(model_input),
                str(model_output),
                mock_data_reader,
                extra_options={"TrtExtraPluginLibraryPaths": "/path/to/plugin"},
            )

            mock_create.assert_called_once()
            mock_calibrator.collect_data.assert_called_once_with(mock_data_reader)
            mock_quantizer.quantize_model.assert_called_once()


class TestInitialization:
    """Test initialization functions."""

    def test_init_calibrater_base(self, simple_onnx_model, tmp_path):
        """Test calibrater base initialization."""
        model_path = tmp_path / "test_model.onnx"
        onnx.save(simple_onnx_model, str(model_path))

        mock_calibrater = Mock()

        with patch(
            "modelopt.onnx.quantization.ort_patching.load_model_with_shape_infer"
        ) as mock_load:
            mock_load.return_value = simple_onnx_model

            _init_calibrater_base(
                mock_calibrater, str(model_path), op_types_to_calibrate=["Conv"], symmetric=True
            )

            assert mock_calibrater.model == simple_onnx_model
            assert mock_calibrater.op_types_to_calibrate == ["Conv"]
            assert mock_calibrater.symmetric is True
            assert mock_calibrater.single_node_model_path_map == {}
            assert mock_calibrater.providers == []
            assert mock_calibrater.trt_extra_plugin_lib_paths is None


class TestSingleNodeCalibration:
    """Test single node calibration specific functions."""

    def test_collect_data_single_node_calibration(self, mock_calibrator):
        """Test collect data for single node calibration."""
        mock_data_reader = Mock(spec=CalibrationDataReader)
        mock_data_reader.get_next.side_effect = [{"input1": np.array([1.0, 2.0])}, None]

        # Mock the single node model path map to be empty so the function returns early
        mock_calibrator.single_node_model_path_map = {}

        mock_model = Mock()
        mock_graph = Mock()
        mock_graph.input = []
        mock_graph.output = []
        mock_model.graph = mock_graph
        mock_calibrator.model = mock_model

        _collect_data_min_max_calibrater_single_node_calibration(mock_calibrator, mock_data_reader)

        # Function should complete without calling compute_data since no models to process
        # No assertions needed - just testing that it doesn't crash

    def test_augment_graph_single_node_calibration(self, mock_calibrator, simple_onnx_model):
        """Test graph augmentation for single node calibration."""
        # Create a proper model with value_info entries to avoid shape inference error
        enhanced_model = onnx.helper.make_model(simple_onnx_model.graph)
        enhanced_model = SymbolicShapeInference.infer_shapes(enhanced_model)
        enhanced_model.opset_import.extend(simple_onnx_model.opset_import)

        # Add value_info entries for the outputs of the Add node
        output_vi = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])
        enhanced_model.graph.value_info.append(output_vi)

        mock_calibrator.model = enhanced_model
        mock_calibrator.augmented_model_path = "augmented_model.onnx"
        mock_calibrator.use_external_data_format = False

        # Mock the tensor selection to return proper tensors
        input1_vi = onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 3])
        mock_calibrator.select_tensors_to_calibrate.return_value = (
            {"input1"},
            {"input1": input1_vi},
        )

        with patch("onnx.save") as mock_save:
            mock_model_after_inference = onnx.helper.make_model(enhanced_model.graph)
            mock_model_after_inference.opset_import.extend(enhanced_model.opset_import)
            # Ensure value_info is preserved
            mock_model_after_inference.graph.value_info.extend(enhanced_model.graph.value_info)

            with patch("uuid.uuid4") as mock_uuid:
                mock_uuid.return_value = "test-uuid"

                _augment_graph_min_max_calibrater_single_node_calibration(mock_calibrator)

                mock_save.assert_called()
                assert len(mock_calibrator.single_node_model_path_map) > 0


@pytest.mark.parametrize(
    "calibration_method",
    [
        CalibrationMethod.MinMax,
        CalibrationMethod.Entropy,
        CalibrationMethod.Percentile,
        CalibrationMethod.Distribution,
    ],
)
def test_calibration_methods(calibration_method, simple_onnx_model, tmp_path):
    """Test different calibration methods."""
    model_path = tmp_path / "test_model.onnx"
    onnx.save(simple_onnx_model, str(model_path))

    calibrator_classes = {
        CalibrationMethod.MinMax: MinMaxCalibrater,
        CalibrationMethod.Entropy: EntropyCalibrater,
        CalibrationMethod.Percentile: PercentileCalibrater,
        CalibrationMethod.Distribution: DistributionCalibrater,
    }

    expected_class = calibrator_classes[calibration_method]

    with (
        patch.object(expected_class, "__init__", return_value=None),
        patch.object(expected_class, "augment_graph"),
        patch.object(expected_class, "create_inference_session"),
    ):
        calibrator = _create_calibrator_with_extra_options(
            str(model_path), calibrate_method=calibration_method
        )

        assert isinstance(calibrator, expected_class)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_intermediate_outputs(self, mock_calibrator):
        """Test handling of empty intermediate outputs."""
        mock_calibrator.intermediate_outputs = []

        result = _compute_data_minmax_calibrator(mock_calibrator)

        assert result == mock_calibrator.calibrate_tensors_range

    def test_nan_values_in_tensor_data(self):
        """Test handling of NaN values in tensor data."""
        mock_quantizer = Mock(spec=BaseQuantizer)
        mock_quantizer.model = Mock()
        mock_quantizer.model.nodes.return_value = []
        mock_quantizer.tensors_range = {
            "tensor1": TensorData(lowest=np.float32(np.nan), highest=np.float32(np.nan))
        }

        _adjust_tensor_ranges(mock_quantizer)

        # Should replace NaN with default values
        adjusted_range = mock_quantizer.tensors_range["tensor1"]
        assert adjusted_range.range_value == (np.float32(0.0), np.float32(448.0))

    def test_invalid_model_path_type(self):
        """Test invalid model path type in initialization."""
        mock_calibrater = Mock()

        with pytest.raises(ValueError, match="model_path should be model path"):
            _init_calibrater_base(mock_calibrater, 123)  # Invalid type

    def test_float8_quantization_validation(self, simple_onnx_model, tmp_path):
        """Test validation for float8 quantization."""
        model_path = tmp_path / "test_model.onnx"
        onnx.save(simple_onnx_model, str(model_path))

        mock_data_reader = Mock(spec=CalibrationDataReader)

        with pytest.raises(ValueError, match="Only Distribution calibration method"):
            _quantize_static(
                str(model_path),
                str(model_path).replace(".onnx", "_quantized.onnx"),
                mock_data_reader,
                activation_type=QuantType.QFLOAT8E4M3FN,
                calibrate_method=CalibrationMethod.MinMax,  # Should fail
            )
