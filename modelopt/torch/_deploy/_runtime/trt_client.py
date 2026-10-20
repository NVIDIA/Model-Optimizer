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

from typing import Any

import tensorrt as trt
import torch

from modelopt.onnx.utils import get_node_names_from_bytes

from ..utils import OnnxBytes
from .registry import RuntimeRegistry
from .runtime_client import Deployment, DeploymentTable, DetailedResults, RuntimeClient
from .tensorrt.constants import SHA_256_HASH_LENGTH
from .tensorrt.engine_builder import build_engine, profile_engine
from .tensorrt.parse_trtexec_log import parse_profiling_log
from .tensorrt.tensorrt_utils import convert_trt_dtype_to_torch

__all__ = ["TRTLocalClient"]


@RuntimeRegistry.register("TRT")
class TRTLocalClient(RuntimeClient):
    """A client for using the local TRT runtime with GPU backend."""

    @property
    def default_deployment(self) -> Deployment:
        return {k: v[0] for k, v in self.deployment_table.items()}

    @property
    def deployment_table(self) -> DeploymentTable:
        return {
            "accelerator": ["GPU"],
            "precision": [
                "fp32",
                "fp16",
                "bf16",
                "fp8",
                "int8",
                "int4",
                "stronglyTyped",
                "best",
            ],
            # Support ONNX opsets 13-21
            "onnx_opset": [str(i) for i in range(13, 22)],
        }

    def __init__(self, deployment: Deployment):
        """Initialize a TRTLocalClient with the given deployment."""
        super().__init__(deployment)
        self.inference_sessions = {}
        logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(logger)
        assert trt.init_libnvinfer_plugins(logger, ""), "Failed to initialize nvinfer plugins."
        self.stream = torch.cuda.Stream()

    def _ir_to_compiled(
        self, ir_bytes: bytes, compilation_args: dict[str, Any] | None = None
    ) -> bytes:
        """Converts an ONNX model to a compiled TRT engine.

        Args:
            ir_bytes: The ONNX model bytes.
            compilation_args: A dictionary of compilation arguments.
                The following arguments are supported: dynamic_shapes, plugin_config, engine_path, timing_cache_path.

        Returns:
            The compiled TRT engine bytes.
        """
        onnx_bytes = OnnxBytes.from_bytes(ir_bytes)
        onnx_model_file_bytes = onnx_bytes.get_onnx_model_file_bytes()
        self.node_names = get_node_names_from_bytes(onnx_model_file_bytes)
        engine_bytes, _ = build_engine(
            onnx_bytes,
            dynamic_shapes=compilation_args.get("dynamic_shapes"),  # type: ignore[union-attr]
            plugin_config=compilation_args.get("plugin_config"),  # type: ignore[union-attr]
            engine_path=compilation_args.get("engine_path"),  # type: ignore[union-attr]
            timing_cache_path=compilation_args.get("timing_cache_path"),  # type: ignore[union-attr]
            trt_mode=self.deployment["precision"],
            verbose=(self.deployment.get("verbose", "false").lower() == "true"),
        )
        self.engine_bytes = engine_bytes
        return engine_bytes

    def _profile(
        self, compiled_model: bytes, compilation_args: dict[str, Any] | None = None
    ) -> tuple[float, DetailedResults]:
        node_names = None
        enable_layerwise_profiling = False
        if hasattr(self, "node_names"):
            node_names = self.node_names
            enable_layerwise_profiling = True
        _, trtexec_log = profile_engine(
            compiled_model,
            onnx_node_names=node_names,
            enable_layerwise_profiling=enable_layerwise_profiling,
            dynamic_shapes=compilation_args.get("dynamic_shapes"),  # type: ignore[union-attr]
        )
        profiling_results = parse_profiling_log(trtexec_log.decode())
        latency = 0.0
        detailed_results = {}
        if profiling_results is not None:
            # Return the mean of the GPU Compute Time
            latency = profiling_results["performance_summary"]["GPU Compute Time"][2]
            detailed_results = profiling_results
        return latency, detailed_results

    def _inference(
        self, compiled_model: bytes, inputs: list[torch.Tensor], io_shapes: dict[str, list] = {}
    ) -> list[torch.Tensor]:
        """Run inference with the compiled model and return the output as list of torch tensors."""
        assert compiled_model is not None, "Engine bytes are not set."

        model_hash = compiled_model[:SHA_256_HASH_LENGTH]
        if model_hash not in self.inference_sessions:
            model_bytes = compiled_model[SHA_256_HASH_LENGTH:]
            self.inference_sessions[model_hash] = self.TRTSession(
                model_bytes, self.trt_runtime, self.stream, inputs[0].device.type, io_shapes
            )
        return self.inference_sessions[model_hash].run(inputs)

    def _teardown_all_sessions(self):
        """Clean up all TRT sessions."""
        for session in self.inference_sessions.values():
            del session
        self.inference_sessions = {}

    class TRTSession:
        def __init__(self, compiled_model, trt_runtime, stream, device, io_shapes):
            self.engine = trt_runtime.deserialize_cuda_engine(compiled_model)
            assert self.engine is not None, "Engine deserialization failed."
            self.execution_context = self.engine.create_execution_context()
            assert self.execution_context is not None, "Execution context creation failed."
            self.stream = stream
            self.device = device
            self.io_shapes = io_shapes
            self.input_names = []
            self.output_names = []
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                names = (
                    self.input_names
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    else self.output_names
                )
                names.append(name)
            self.has_shape_inputs = any(
                self.engine.is_shape_inference_io(name) for name in self.input_names
            )
            self.input_tensors, self.output_tensors = self.initialize_input_output_tensors(
                self.engine
            )

        def _allocate_tensor(self, name, shape):
            dtype = convert_trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            device = (
                "cpu"
                if self.engine.get_tensor_location(name) == trt.TensorLocation.HOST
                else self.stream.device
            )
            with torch.cuda.stream(self.stream):
                return torch.empty(
                    tuple(shape), dtype=dtype, device=device, pin_memory=device == "cpu"
                )

        def initialize_input_output_tensors(self, engine):
            # Select the profile before setting any dynamic input dimensions.
            if not self.execution_context.set_optimization_profile_async(
                0, self.stream.cuda_stream
            ):
                raise RuntimeError("Failed to select TensorRT optimization profile 0.")
            input_tensors = []
            for name in self.input_names:
                shape = tuple(engine.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    minimum, _, maximum = engine.get_tensor_profile_shape(name, 0)
                    shape = tuple(self.io_shapes.get(name, minimum))
                    if len(shape) != len(minimum) or any(
                        dim < low or dim > high for dim, low, high in zip(shape, minimum, maximum)
                    ):
                        raise ValueError(
                            f"Input shape {shape} for {name} is outside optimization profile 0."
                        )
                else:
                    shape = tuple(self.io_shapes.get(name, shape))
                if not self.execution_context.set_input_shape(name, shape):
                    raise ValueError(f"Invalid TensorRT input shape {shape} for {name}.")
                tensor = self._allocate_tensor(name, shape)
                if not self.execution_context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Failed to bind TensorRT input {name}.")
                input_tensors.append(tensor)

            # Output shapes may depend on shape-tensor values, supplied by run().
            return input_tensors, []

        def _infer_shapes(self):
            missing = self.execution_context.infer_shapes()
            if missing:
                raise ValueError(f"Insufficiently specified TensorRT inputs: {missing}.")

        def _update_output_tensors(self):
            shape_outputs = [
                name for name in self.output_names if self.engine.is_shape_inference_io(name)
            ]
            # infer_shapes() can write shape outputs; do not use a previous call's allocation.
            for name in shape_outputs:
                if not self.execution_context.set_tensor_address(name, 0):
                    raise RuntimeError(f"Failed to clear TensorRT shape output {name}.")
            self._infer_shapes()
            output_tensors = []
            for index, name in enumerate(self.output_names):
                inferred = tuple(self.execution_context.get_tensor_shape(name))
                shape = tuple(self.io_shapes.get(name, inferred))
                if any(dim < 0 for dim in shape):
                    raise ValueError(
                        f"Shape for output {name} cannot be inferred: {inferred}. "
                        "Please provide the shape with io_shapes."
                    )
                if len(shape) != len(inferred) or any(
                    known >= 0 and dim != known for dim, known in zip(shape, inferred)
                ):
                    raise ValueError(
                        f"Output shape {shape} for {name} does not match inferred shape {inferred}."
                    )
                tensor = (
                    self.output_tensors[index]
                    if index < len(self.output_tensors)
                    and tuple(self.output_tensors[index].shape) == shape
                    else self._allocate_tensor(name, shape)
                )
                if not self.execution_context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Failed to bind TensorRT output {name}.")
                output_tensors.append(tensor)
            self.output_tensors = output_tensors
            if shape_outputs:
                self._infer_shapes()

        def run(self, inputs):
            assert self.engine is not None, "Engine is not set."
            if len(inputs) != len(self.input_tensors):
                raise ValueError(
                    f"Expected {len(self.input_tensors)} TensorRT inputs, got {len(inputs)}."
                )

            # Retain fixed-buffer padding for incomplete evaluation batches.
            self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
            with torch.cuda.stream(self.stream):
                for i, input_t in enumerate(inputs):
                    if input_t.ndim != self.input_tensors[i].ndim or any(
                        dim > size for dim, size in zip(input_t.shape, self.input_tensors[i].shape)
                    ):
                        raise ValueError(
                            f"Input {self.input_names[i]} shape {tuple(input_t.shape)} exceeds "
                            f"buffer shape {tuple(self.input_tensors[i].shape)}."
                        )
                    if self.input_tensors[i].shape == input_t.shape:
                        self.input_tensors[i].copy_(input_t, non_blocking=True)
                    else:
                        # Zero out the self.input_tensor[i]
                        self.input_tensors[i].zero_()

                        # Create slices for each dimension of input_t
                        slices = tuple(slice(0, dim) for dim in input_t.shape)

                        # Assign the portion of input_t to the self.input_tensor[i]
                        self.input_tensors[i][slices] = input_t

            # Shape inference has no stream argument and needs completed shape-input copies.
            if self.has_shape_inputs:
                self.stream.synchronize()
            if not self.output_tensors or self.has_shape_inputs:
                self._update_output_tensors()
            if not self.execution_context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference execution failed.")

            self.stream.synchronize()

            return self.output_tensors
