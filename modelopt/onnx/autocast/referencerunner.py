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

When multiple batches of calibration data are provided, the runner aggregates statistics
across all batches to provide more robust range information for precision conversion decisions.
"""

import copy
import io
import os
import sys
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
from polygraphy.common import TensorMetadata

from modelopt.onnx import utils as onnx_utils
from modelopt.onnx.autocast.logging_config import configure_logging, logger
from modelopt.onnx.quantization.calib_utils import CalibrationDataProvider
from modelopt.onnx.quantization.ort_utils import _prepare_ep_list

configure_logging()


@dataclass
class TensorStats:
    """Statistics for a tensor aggregated across multiple batches."""

    absmax: float
    """Maximum absolute value across all batches."""
    min_val: float
    """Minimum value across all batches."""
    max_val: float
    """Maximum value across all batches."""
    shape: tuple
    """Shape of the tensor (from first batch)."""

    def __abs__(self):
        """Return the maximum absolute value (for compatibility with np.abs)."""
        return self.absmax

    @property
    def size(self):
        """Return total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


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
        """Load inputs from NPZ format.

        Supports both single NPZ file and directory containing multiple NPZ files for multi-batch calibration.

        Args:
            input_data_path: Path to NPZ file or directory containing NPZ files.

        Returns:
            List of input dictionaries, one per batch.
        """
        if os.path.isdir(input_data_path):
            # Load all NPZ files in the directory as multiple batches
            npz_files = sorted([f for f in os.listdir(input_data_path) if f.endswith(".npz")])
            if not npz_files:
                raise ValueError(f"No NPZ files found in directory: {input_data_path}")
            logger.info(
                f"Loading {len(npz_files)} NPZ files from directory for multi-batch calibration"
            )
            return [np.load(os.path.join(input_data_path, f)) for f in npz_files]
        else:
            calib_data = np.load(input_data_path)
            if isinstance(calib_data, np.lib.npyio.NpzFile):
                # Wrap data into a CalibDataProvider to support a single NPZ file
                # containing data from multiple batches
                data_loader = {key: calib_data[key] for key in calib_data.files}
                return CalibrationDataProvider(self.model, data_loader).calibration_data_list
            return [calib_data]

    def _validate_inputs(self, data_loader):
        """Validate that input names and shapes match the model."""
        if isinstance(data_loader, list) and (
            isinstance(data_loader[0], dict | np.lib.npyio.NpzFile)
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
                    # Compare input shape, skipping check for unknown dimensions
                    mask = inp_shape_model > 0
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
                elif inputs.endswith(".npz") or os.path.isdir(inputs):
                    data_loader = self._load_inputs_from_npz(inputs)
                else:
                    raise ValueError(
                        f"Invalid input file: {inputs}. Supported input types: .json (Polygraphy JSON format), "
                        ".npz (Numpy), or a directory containing .npz files"
                    )
            elif isinstance(inputs, dict | OrderedDict):
                data_loader = [inputs]
            else:
                raise ValueError(
                    f"Invalid input type: {type(inputs)}. Supported input types: dict, OrderedDict, or a path to a "
                    "JSON or NPZ file."
                )
            self._validate_inputs(data_loader)

        return data_loader

    def _get_ort_runner(self, model, disable_cpu_mem_arena: bool = False):
        from polygraphy.backend.onnx import BytesFromOnnx
        from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx

        # Get the actual ONNX ModelProto from ModifyOutputs wrapper
        modified_model = model()

        needs_file_backed_model = onnx_utils.check_model_uses_external_data(
            modified_model
        ) or onnx_utils.is_model_too_large_for_protobuf(modified_model)
        model_temp_dir = None
        try:
            if needs_file_backed_model:
                logger.debug("Model has external data, using file-based approach")
                model_temp_dir = tempfile.TemporaryDirectory()
                tmp_file_path = os.path.join(model_temp_dir.name, "model.onnx")
                onnx_utils.save_onnx(modified_model, tmp_file_path, save_as_external_data=True)
                logger.debug(f"Model with all outputs saved to {tmp_file_path}")
                model_bytes_or_path = tmp_file_path
            else:
                # For models without external data, use the original BytesFromOnnx approach (no tmp files)
                logger.debug("Model has no external data, using BytesFromOnnx approach")
                model_bytes_or_path = BytesFromOnnx(modified_model)

            if disable_cpu_mem_arena:

                def build_onnxrt_session():
                    model = (
                        model_bytes_or_path()
                        if callable(model_bytes_or_path)
                        else model_bytes_or_path
                    )
                    options = ort.SessionOptions()
                    options.enable_cpu_mem_arena = False
                    try:
                        process_cpu_count = len(os.sched_getaffinity(0))
                    except AttributeError:
                        process_cpu_count = 1
                    options.intra_op_num_threads = process_cpu_count
                    options.inter_op_num_threads = process_cpu_count
                    return ort.InferenceSession(
                        model, providers=self.providers, sess_options=options
                    )

            else:
                build_onnxrt_session = SessionFromOnnx(
                    model_bytes_or_path, providers=self.providers
                )
            runners = [OnnxrtRunner(build_onnxrt_session)]
        except Exception:
            if model_temp_dir is not None:
                model_temp_dir.cleanup()
            raise
        return runners, model_temp_dir

    def _aggregate_tensor_stats(self, all_batch_data: list[OrderedDict]) -> OrderedDict:
        """Aggregate tensor statistics across multiple batches.

        Args:
            all_batch_data: List of dictionaries containing tensor data for each batch.

        Returns:
            OrderedDict mapping tensor names to TensorStats objects.
        """
        if len(all_batch_data) == 1:
            # Single batch - return raw data for backward compatibility
            return all_batch_data[0]

        logger.info(f"Aggregating statistics across {len(all_batch_data)} batches...")

        aggregated = OrderedDict()
        tensor_names = all_batch_data[0].keys()

        for name in tensor_names:
            absmax = -np.inf
            min_val = np.inf
            max_val = -np.inf
            shape = None

            for batch_data in all_batch_data:
                if name not in batch_data:
                    continue
                data = batch_data[name]
                if shape is None:
                    shape = data.shape

                batch_absmax = np.max(np.abs(data)) if data.size > 0 else 0
                batch_min = np.min(data) if data.size > 0 else 0
                batch_max = np.max(data) if data.size > 0 else 0

                absmax = max(absmax, batch_absmax)
                min_val = min(min_val, batch_min)
                max_val = max(max_val, batch_max)

            if shape is not None:
                aggregated[name] = TensorStats(
                    absmax=absmax,
                    min_val=min_val,
                    max_val=max_val,
                    shape=shape,
                )

        return aggregated

    def _make_single_node_model(
        self,
        node: onnx.NodeProto,
        dependency_names: list[str],
        value_info_map: dict[str, onnx.ValueInfoProto],
        initializer_map: dict[str, onnx.TensorProto],
        runtime_values: dict[str, np.ndarray],
    ) -> onnx.ModelProto:
        """Build a runnable model containing one node from the reference graph."""
        model_inputs = []
        model_initializers = []
        for input_name in dependency_names:
            if input_name in initializer_map:
                model_initializers.append(initializer_map[input_name])
            elif input_name in value_info_map:
                runtime_input = copy.deepcopy(value_info_map[input_name])
                runtime_shape = runtime_input.type.tensor_type.shape
                runtime_shape.ClearField("dim")
                for dimension in runtime_values[input_name].shape:
                    runtime_shape.dim.add().dim_value = dimension
                model_inputs.append(runtime_input)
            else:
                raise ValueError(
                    f"Tensor {input_name!r} required by node {node.name!r} has no type metadata."
                )

        model_outputs = []
        for output_name in node.output:
            if not output_name:
                continue
            if output_name not in value_info_map:
                raise ValueError(
                    f"Tensor {output_name!r} produced by node {node.name!r} has no type metadata."
                )
            output = copy.deepcopy(value_info_map[output_name])
            if node.domain in ("", "ai.onnx"):
                output.type.tensor_type.ClearField("shape")
            model_outputs.append(output)

        graph = onnx.helper.make_graph(
            [node],
            f"{self.model.graph.name}_{node.name}",
            model_inputs,
            model_outputs,
        )
        for initializer in model_initializers:
            graph.initializer.add().CopyFrom(initializer)
        model = onnx.helper.make_model(
            graph,
            opset_imports=self.model.opset_import,
            functions=self.model.functions,
            ir_version=self.model.ir_version,
        )
        if node.domain in ("", "ai.onnx"):
            model = onnx_utils.infer_shapes(model)
        return model

    @classmethod
    def _get_graph_captured_inputs(cls, graph: onnx.GraphProto) -> list[str]:
        """Return tensors referenced by a subgraph but defined in an outer scope."""
        defined_names = {
            *(value.name for value in graph.input),
            *(initializer.name for initializer in graph.initializer),
            *(output for node in graph.node for output in node.output if output),
        }
        referenced_names = []
        for node in graph.node:
            referenced_names.extend(name for name in node.input if name)
            referenced_names.extend(cls._get_node_captured_inputs(node))
        return list(dict.fromkeys(name for name in referenced_names if name not in defined_names))

    @classmethod
    def _get_node_captured_inputs(cls, node: onnx.NodeProto) -> list[str]:
        """Return outer-scope tensors captured by graph attributes on a node."""
        captured_inputs = []
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                captured_inputs.extend(cls._get_graph_captured_inputs(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    captured_inputs.extend(cls._get_graph_captured_inputs(graph))
        return list(dict.fromkeys(captured_inputs))

    @classmethod
    def _get_node_dependencies(cls, node: onnx.NodeProto) -> list[str]:
        """Return explicit and control-flow-captured inputs required by a node."""
        return list(
            dict.fromkeys(
                [
                    *(name for name in node.input if name),
                    *cls._get_node_captured_inputs(node),
                ]
            )
        )

    @staticmethod
    def _update_tensor_stats(
        stats: OrderedDict[str, TensorStats], name: str, value: np.ndarray
    ) -> None:
        """Update compact tensor statistics with one calibration batch."""
        batch_absmax = np.max(np.abs(value)) if value.size else 0
        batch_min = np.min(value) if value.size else 0
        batch_max = np.max(value) if value.size else 0
        previous = stats.get(name)
        stats[name] = TensorStats(
            absmax=max(previous.absmax if previous else -np.inf, batch_absmax),
            min_val=min(previous.min_val if previous else np.inf, batch_min),
            max_val=max(previous.max_val if previous else -np.inf, batch_max),
            shape=previous.shape if previous else value.shape,
        )

    def _run_per_node(self, inputs=None) -> OrderedDict[str, TensorStats]:
        """Run each graph node separately and retain only live activation tensors."""
        value_info_map = {
            value_info.name: value_info
            for value_info in [
                *self.model.graph.input,
                *self.model.graph.output,
                *self.model.graph.value_info,
            ]
        }
        initializer_map = {
            initializer.name: initializer for initializer in self.model.graph.initializer
        }
        node_dependencies = [
            (node, self._get_node_dependencies(node)) for node in self.model.graph.node
        ]
        consumer_counts = Counter(
            input_name
            for _, dependency_names in node_dependencies
            for input_name in dependency_names
            if input_name not in initializer_map
        )
        stats = OrderedDict()
        data_loader = self._load_inputs(inputs)
        if inputs is None:
            input_metadata = TensorMetadata()
            for model_input in self.model.graph.input:
                tensor_type = model_input.type.tensor_type
                shape = tuple(
                    dimension.dim_value if dimension.HasField("dim_value") else None
                    for dimension in tensor_type.shape.dim
                )
                input_metadata.add(
                    model_input.name,
                    np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)),
                    shape,
                )
            data_loader.input_metadata = input_metadata

        for input_data in data_loader:
            live_values = {}
            for name, value in input_data.items():
                if name not in value_info_map:
                    raise ValueError(f"Tensor {name!r} has no type metadata.")
                elem_type = value_info_map[name].type.tensor_type.elem_type
                dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
                live_values[name] = value.astype(dtype, copy=False)
            remaining_consumers = consumer_counts.copy()
            for name in live_values:
                self._update_tensor_stats(stats, name, live_values[name])

            for node, dependency_names in node_dependencies:
                runtime_input_names = [
                    name for name in dependency_names if name not in initializer_map
                ]
                missing_inputs = [name for name in runtime_input_names if name not in live_values]
                if missing_inputs:
                    raise ValueError(
                        f"Node {node.name!r} is missing runtime inputs {missing_inputs!r}."
                    )
                feed_dict = {name: live_values[name] for name in runtime_input_names}
                single_node_model = self._make_single_node_model(
                    node, dependency_names, value_info_map, initializer_map, live_values
                )
                runners, model_temp_dir = self._get_ort_runner(
                    lambda: single_node_model, disable_cpu_mem_arena=True
                )
                try:
                    with runners[0] as runner:
                        output_dict = runner.infer(feed_dict)
                finally:
                    if model_temp_dir is not None:
                        model_temp_dir.cleanup()
                    runners.clear()

                for input_name in runtime_input_names:
                    remaining_consumers[input_name] -= 1
                    if remaining_consumers[input_name] == 0:
                        live_values.pop(input_name)
                feed_dict.clear()

                output_value = None
                for output_name in tuple(output_dict):
                    output_value = output_dict.pop(output_name)
                    self._update_tensor_stats(stats, output_name, output_value)
                    if remaining_consumers[output_name] > 0:
                        live_values[output_name] = output_value
                output_value = None

        return stats

    def run(self, inputs=None, calibrate_per_node: bool = False):
        """Run FP32 inference with provided or random inputs.

        When multiple batches of input data are provided, inference is run for each batch
        and statistics are aggregated across all batches for more robust range estimation.

        Args:
            inputs: Optional input data. Can be:
                - None: Random inputs will be generated
                - str: Path to JSON file, NPZ file, or directory containing NPZ files
                - dict/OrderedDict: Single batch of input data
            calibrate_per_node: Run each node separately and return compact statistics to reduce
                peak calibration memory.

        Returns:
            OrderedDict: Combined input and output data. Per-node mode and multi-batch normal mode
                return TensorStats objects; single-batch normal mode returns raw arrays.
        """
        from polygraphy import constants
        from polygraphy.backend.onnx import ModifyOutputs as ModifyOnnxOutputs
        from polygraphy.comparator import Comparator

        logger.info("Running ONNX Runtime to obtain reference outputs (this may take a while)...")
        # Set ONNX Runtime log level to ERROR to suppress warnings
        ort.set_default_logger_severity(3)
        if calibrate_per_node:
            return self._run_per_node(inputs)

        model_copy = copy.deepcopy(self.model)
        # Clear stale type metadata to prevent type check failures in ORT
        onnx_utils.clear_stale_value_info(model_copy)
        modify_outputs = ModifyOnnxOutputs(model_copy, outputs=constants.MARK_ALL)

        # Load the modified model and create an inference session
        runners, model_temp_dir = self._get_ort_runner(modify_outputs)
        try:
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
        finally:
            if model_temp_dir is not None:
                model_temp_dir.cleanup()

        if not results:
            logger.error(f"ONNXRuntime execution failed with output:\n{captured_output}")
            raise Exception("ONNXRuntime failed to run, see logs for details")

        # Collect all batch data (inputs + outputs)
        all_batch_data = []
        runner_results = results[0][1]  # Get all iteration results for the first runner
        data_loader_iter = iter(data_loader)

        for iter_idx, iter_result in enumerate(runner_results):
            output_dict = OrderedDict(iter_result)

            # Get corresponding input data
            try:
                input_data = next(data_loader_iter)
            except StopIteration:
                # If data_loader is exhausted, it might be a DataLoader that generates random data
                input_data = {}

            # Combine inputs and outputs for this batch
            batch_dict = OrderedDict()
            batch_dict.update(input_data)
            batch_dict.update(output_dict)
            all_batch_data.append(batch_dict)

        num_batches = len(all_batch_data)
        if num_batches > 1:
            logger.info(f"Processed {num_batches} batches of calibration data")

        # Aggregate statistics across all batches
        return self._aggregate_tensor_stats(all_batch_data)
