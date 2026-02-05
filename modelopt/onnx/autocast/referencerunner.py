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

"""Reference runner module for ONNX model execution.

This module provides functionality for running ONNX models using ONNXRuntime as a reference
implementation. It supports both random input generation and user-provided inputs through
NPZ or Polygraphy JSON files. The runner is used to analyze model behavior and validate
outputs during precision conversion.
"""

import copy
import io
import sys
import tempfile
from collections import OrderedDict

import numpy as np
import onnx

from modelopt.onnx import utils as onnx_utils
from modelopt.onnx.autocast.logging_config import configure_logging, logger
from modelopt.onnx.quantization.calib_utils import CalibrationDataProvider
from modelopt.onnx.quantization.ort_utils import _prepare_ep_list

configure_logging()


class ReferenceRunner:
    """A class to run ONNX models with ONNXRuntime for reference inference."""

    def __init__(
        self, model: onnx.ModelProto, providers: list[str] = ["cpu"], trt_plugins: list[str] = []
    ):
        """Initialize with ONNX model path."""
        self.model = model
        self.input_names = [input.name for input in self.model.graph.input]
        self.input_shapes = {
            input.name: [s.dim_value for s in input.type.tensor_type.shape.dim]
            for input in self.model.graph.input
        }
        self.providers = self._prepare_ep_list_with_trt_plugin_path(providers, trt_plugins)

    def _prepare_ep_list_with_trt_plugin_path(self, providers, trt_plugins):
        providers = _prepare_ep_list(providers) or providers
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
            # Ensure that the TRT EP is the first in the providers list to avoid fallback issues
            trt_ep_options = (
                {"trt_extra_plugin_lib_paths": ";".join(trt_plugins)} if trt_plugins else {}
            )
            providers.insert(0, ("TensorrtExecutionProvider", trt_ep_options))
            logger.info(f"Successfully updated EPs for ORT: {providers}")
        return providers

    def _load_inputs_from_json(self, input_data_path):
        """Load inputs from Polygraphy JSON format."""
        from polygraphy.json import load_json

        return load_json(input_data_path, description="input data")

    def _load_inputs_from_npz(self, input_data_path):
        """Load inputs from NPZ format."""
        calib_data = np.load(input_data_path)

        if isinstance(calib_data, np.lib.npyio.NpzFile):
            # Wrap data into a CalibDataProvider to support a single NPZ file containing data from multiple batches
            data_loader = {key: calib_data[key] for key in calib_data.files}
            return CalibrationDataProvider(self.model, data_loader).calibration_data_list
        return [calib_data]

    def _validate_inputs(self, data_loader):
        """Validate that input names and shapes match the model."""
        if isinstance(data_loader, list) and (
            isinstance(data_loader[0], (dict, np.lib.npyio.NpzFile))
        ):
            if sorted(self.input_names) != sorted(data_loader[0].keys()):
                raise ValueError("Input names from ONNX model do not match provided input names.")
            for inp_name, inp_shape in data_loader[0].items():
                # Get model and data shapes as numpy arrays
                inp_shape_model = np.array(self.input_shapes[inp_name])
                inp_shape_data = np.array(inp_shape.shape)
                # Compare input rank
                raise_value_error = len(inp_shape_model) != len(inp_shape_data)
                if not raise_value_error:
                    # Compare input shape, skipping check for unknown dimensions (shape = -1)
                    mask = (inp_shape_model != -1) & (inp_shape_data != -1)
                    raise_value_error = np.any(inp_shape_model[mask] != inp_shape_data[mask])
                if raise_value_error:
                    raise ValueError(
                        f"Input shape from '{inp_name}' does not match provided input shape: "
                        f"{self.input_shapes[inp_name]} vs {list(inp_shape.shape)}. "
                        f"Please make sure that your calibration data matches the ONNX input shapes."
                    )
        else:
            raise ValueError("Invalid input file.")

    def _load_inputs(self, inputs):
        """Get data loader from inputs or create random data loader if no inputs provided."""
        from polygraphy.comparator import DataLoader

        # If no inputs are provided, use random inputs
        data_loader = DataLoader(val_range={"": (-1, 1)})

        if inputs is not None:
            if isinstance(inputs, str):
                if inputs.endswith(".json"):
                    data_loader = self._load_inputs_from_json(inputs)
                elif inputs.endswith(".npz"):
                    data_loader = self._load_inputs_from_npz(inputs)
                else:
                    raise ValueError(
                        f"Invalid input file: {inputs}. Supported input file types: .json (Polygraphy JSON format), "
                        ".npz (Numpy)"
                    )
            elif isinstance(inputs, (dict, OrderedDict)):
                data_loader = [inputs]
            else:
                raise ValueError(
                    f"Invalid input type: {type(inputs)}. Supported input types: dict, OrderedDict, or a path to a "
                    "JSON or NPZ file."
                )
            self._validate_inputs(data_loader)

        return data_loader

    def _get_ort_runner(self, model):
        from polygraphy.backend.onnx import BytesFromOnnx
        from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx

        # Check if model has external data by checking:
        # 1. If any initializer has data_location set to EXTERNAL (even if data is loaded)
        # 2. If model size would exceed 2GB (indicating need for external data)
        needs_external_data = onnx_utils.check_model_uses_external_data(
            self.model
        ) or self.model.ByteSize() > 2 * (1024**3)
        if needs_external_data:
            logger.debug("Model has external data, using file-based approach")
            # Get the actual ONNX ModelProto from ModifyOutputs wrapper
            modified_model = model()

            # Use a persistent temp file, because we need the file to be present in an broader context
            tmp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            tmp_file.close()
            tmp_file_path = tmp_file.name
            onnx_utils.save_onnx(modified_model, tmp_file_path, save_as_external_data=True)
            logger.debug(f"Model with all outputs saved to {tmp_file_path}")
            build_onnxrt_session = SessionFromOnnx(tmp_file_path, providers=self.providers)

        else:
            # For models without external data, use the original BytesFromOnnx approach (no tmp files)
            logger.debug("Model has no external data, using BytesFromOnnx approach")
            serialize_onnx = BytesFromOnnx(model)
            build_onnxrt_session = SessionFromOnnx(serialize_onnx, providers=self.providers)
        runners = [OnnxrtRunner(build_onnxrt_session)]
        return runners

    def run(self, inputs=None):
        """Run FP32 inference with provided or random inputs."""
        import onnxruntime as ort
        from polygraphy import constants
        from polygraphy.backend.onnx import ModifyOutputs as ModifyOnnxOutputs
        from polygraphy.comparator import Comparator

        logger.info("Running ONNX Runtime to obtain reference outputs (this may take a while)...")
        # Set ONNX Runtime log level to ERROR to suppress warnings
        ort.set_default_logger_severity(3)

        model_copy = copy.deepcopy(self.model)
        modify_outputs = ModifyOnnxOutputs(model_copy, outputs=constants.MARK_ALL)

        # Load the modified model and create an inference session
        runners = self._get_ort_runner(modify_outputs)

        # Comparator is used despite the fact that we are using ONNXRuntime
        # because it provides the ability to generate random inputs using DataLoader
        data_loader = self._load_inputs(inputs)

        # Temporarily redirect stdout to suppress Comparator.run() output
        stdout = sys.stdout
        string_buffer = io.StringIO()
        sys.stdout = string_buffer
        try:
            results = Comparator.run(runners, data_loader=data_loader)
        finally:
            # Capture the output before restoring stdout
            captured_output = string_buffer.getvalue()
            sys.stdout = stdout

        if not results:
            logger.error(f"ONNXRuntime execution failed with output:\n{captured_output}")
            raise Exception("ONNXRuntime failed to run, see logs for details")

        # Get the output results
        output_dict = OrderedDict(results[0][1][0])

        # Include input data for completeness
        input_data = next(iter(data_loader))

        # Combine inputs and outputs in the returned dictionary
        combined_dict = OrderedDict()
        combined_dict.update(input_data)
        combined_dict.update(output_dict)

        return combined_dict
