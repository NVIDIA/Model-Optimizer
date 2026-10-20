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

from math import prod
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
                model_bytes, self.trt_runtime, self.stream, self.stream.device, io_shapes
            )
        return self.inference_sessions[model_hash].run(inputs, io_shapes)

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
            self.stream = stream
            self.device = device
            self.io_shapes = io_shapes
            self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [
                name
                for name in self.names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [name for name in self.names if name not in self.input_names]
            self.storage = {}
            if not self.execution_context.set_optimization_profile_async(0, stream.cuda_stream):
                raise RuntimeError("Failed to select TensorRT optimization profile 0.")

        def _buffer(self, name, shape):
            dtype = convert_trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            device = (
                "cpu"
                if self.engine.get_tensor_location(name) == trt.TensorLocation.HOST
                else self.stream.device
            )
            numel = prod(shape)
            if name not in self.storage or self.storage[name].numel() < max(1, numel):
                self.storage[name] = torch.empty(max(1, numel), dtype=dtype, device=device)
            storage = self.storage[name]
            if not self.execution_context.set_tensor_address(name, storage.data_ptr()):
                raise RuntimeError(f"Failed to bind TensorRT tensor {name}.")
            return storage[:numel].reshape(shape)

        def initialize_input_output_tensors(self, inputs):
            if len(inputs) != len(self.input_names):
                raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(inputs)}.")
            for name, shape in self.io_shapes.items():
                if name not in self.names or any(dim < 0 for dim in shape):
                    raise ValueError(f"Invalid io_shapes entry for {name}: {shape}.")

            self.input_tensors = []
            for name, value in zip(self.input_names, inputs):
                declared = tuple(self.engine.get_tensor_shape(name))
                shape = tuple(self.io_shapes.get(name, declared))
                if name not in self.io_shapes and any(dim < 0 for dim in declared):
                    minimum = self.engine.get_tensor_profile_shape(name, 0)[0]
                    # Preserve partial-batch padding below profile MIN.
                    shape = tuple(max(actual, lower) for actual, lower in zip(value.shape, minimum))
                if (
                    len(declared) != value.ndim
                    or len(shape) != value.ndim
                    or any(actual > allocated for actual, allocated in zip(value.shape, shape))
                ):
                    raise ValueError(
                        f"Input {name} shape {tuple(value.shape)} does not fit {shape}."
                    )
                if not self.execution_context.set_input_shape(name, shape):
                    raise ValueError(f"Input shape for {name} is invalid for profile 0: {shape}.")
                buffer = self._buffer(name, shape)
                if tuple(value.shape) != shape:
                    buffer.zero_()
                buffer[tuple(slice(0, dim) for dim in value.shape)].copy_(value, non_blocking=True)
                self.input_tensors.append(buffer)

            # infer_shapes can read HOST values before enqueue, including GPU-to-host copies.
            if any(self.engine.is_shape_inference_io(name) for name in self.input_names):
                self.stream.synchronize()
            for name in self.output_names:
                shape = tuple(self.engine.get_tensor_shape(name))
                if all(dim >= 0 for dim in shape):
                    self._buffer(name, shape)
            missing = self.execution_context.infer_shapes()
            if missing:
                raise ValueError(f"TensorRT shapes cannot be inferred for tensors: {missing}.")

            self.output_tensors = []
            for name in self.output_names:
                shape = tuple(self.execution_context.get_tensor_shape(name))
                if name in self.io_shapes:
                    if all(dim >= 0 for dim in shape) and tuple(self.io_shapes[name]) != shape:
                        raise ValueError(
                            f"Output shape for {name} is {shape}, "
                            f"but io_shapes specifies {tuple(self.io_shapes[name])}."
                        )
                    shape = tuple(self.io_shapes[name])
                if any(dim < 0 for dim in shape):
                    raise ValueError(
                        f"Output shape for {name} cannot be inferred; provide io_shapes."
                    )
                self.output_tensors.append(self._buffer(name, shape))

        def run(self, inputs, io_shapes=None):
            if io_shapes is not None:
                self.io_shapes = io_shapes
            producer_stream = torch.cuda.current_stream(self.stream.device)
            with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
                self.stream.wait_stream(producer_stream)
                try:
                    self.initialize_input_output_tensors(inputs)
                    if not self.execution_context.execute_async_v3(
                        stream_handle=self.stream.cuda_stream
                    ):
                        raise RuntimeError("TensorRT execute_async_v3 failed.")
                finally:
                    self.stream.synchronize()
                return self.output_tensors
