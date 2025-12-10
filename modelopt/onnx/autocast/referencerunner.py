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
from collections import OrderedDict

import numpy as np
import onnx

from modelopt.onnx.autocast.logging_config import configure_logging, logger
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
        return [np.load(input_data_path)]

    def _validate_inputs(self, data_loader):
        """Validate that input names and shapes match the model."""
        if isinstance(data_loader, list) and (
            isinstance(data_loader[0], (dict, np.lib.npyio.NpzFile))
        ):
            if sorted(self.input_names) != sorted(data_loader[0].keys()):
                raise ValueError("Input names from ONNX model do not match provided input names.")
            for inp_name, inp_shape in data_loader[0].items():
                if self.input_shapes[inp_name] != list(inp_shape.shape):
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

    def run(self, inputs=None):
        """Run FP32 inference with provided or random inputs."""
        import onnxruntime as ort
        from polygraphy import constants
        from polygraphy.backend.onnx import BytesFromOnnx
        from polygraphy.backend.onnx import ModifyOutputs as ModifyOnnxOutputs
        from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
        from polygraphy.comparator import Comparator

        logger.info("Running ONNX Runtime to obtain reference outputs (this may take a while)...")
        # Set ONNX Runtime log level to ERROR to suppress warnings
        ort.set_default_logger_severity(3)

        model_copy = copy.deepcopy(self.model)
        modify_outputs = ModifyOnnxOutputs(model_copy, outputs=constants.MARK_ALL)
        serialize_onnx = BytesFromOnnx(modify_outputs)
        build_onnxrt_session = SessionFromOnnx(serialize_onnx, providers=self.providers)
        runners = [OnnxrtRunner(build_onnxrt_session)]

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
