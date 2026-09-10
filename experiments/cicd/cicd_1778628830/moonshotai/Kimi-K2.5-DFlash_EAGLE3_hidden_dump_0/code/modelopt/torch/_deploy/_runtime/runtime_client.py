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

from abc import ABC, abstractmethod
from typing import Any

import torch

__all__ = ["Deployment", "DeploymentTable", "DetailedResults", "RuntimeClient"]

Deployment = dict[str, str]
DeploymentTable = dict[str, list[str]]
DetailedResults = dict[str, Any]


class RuntimeClient(ABC):
    """A abstract client class for implementing various runtimes to be used for deployment.

    The RuntimeClient defines a common interfaces for accessing various runtimes within modelopt.
    """

    _runtime: str  # runtime of the client --> set by RuntimeRegistry.register

    def __init__(self, deployment: Deployment):
        super().__init__()
        self.deployment = self.sanitize_deployment_config(deployment)

    @property
    def runtime(self) -> str:
        return self._runtime

    def sanitize_deployment_config(self, deployment: Deployment) -> Deployment:
        """Cleans/checks the deployment config & fills in runtime-specific default values.

        Args:
            deployment: Deployment config with at least the ``runtime`` key specified.

        Returns:
            The sanitized deployment config with all runtime-specific default values filled
            in for missing keys.
        """
        # check runtime
        assert self.runtime == deployment["runtime"], "Runtime mismatch!"

        # fill in default values and update
        deployment = {**self.default_deployment, **deployment}

        # sanity check on keys (inverse doesn't have to be checked since we fill in defaults)
        table = self.deployment_table
        extra_keys = deployment.keys() - table.keys() - {"runtime", "verbose"}
        assert not extra_keys, f"Invalid deployment config keys detected: {extra_keys}!"

        # sanity checks on values
        invalid_values = {(k, deployment[k]): t for k, t in table.items() if deployment[k] not in t}
        assert not invalid_values, f"Invalid deployment config values detected: {invalid_values}!"

        return deployment

    def ir_to_compiled(self, ir_bytes: bytes, compilation_args: dict[str, Any] = {}) -> bytes:
        """Converts a model from its intermediate representation (IR) to a compiled device model.

        Args:
            ir_bytes: Intermediate representation (IR) of the model.
            compilation_args: Additional arguments for the compilation process.

        Returns: The compiled device model that can be used for further downstream tasks such as
            on-device inference and profiling,
        """
        # run model compilation
        compiled_model = self._ir_to_compiled(ir_bytes, compilation_args)
        assert compiled_model, "Device conversion failed!"

        return compiled_model

    def profile(
        self, compiled_model: bytes, compilation_args: dict[str, Any] = {}
    ) -> tuple[float, DetailedResults]:
        """Profiles a compiled device model and returns the latency & detailed profiling results.

        Args:
            compiled_model: Compiled device model from compilation service.
            compilation_args: Additional arguments for the compilation process.

        Returns: A tuple (latency, detailed_result) where
            ``latency`` is the latency of the compiled model in ms,
            ``detailed_result`` is a dictionary containing additional benchmarking results
        """
        # get latency & detailed results from client
        latency, detailed_result = self._profile(compiled_model, compilation_args)
        assert latency > 0.0, "Profiling failed!"

        return latency, detailed_result

    def inference(
        self, compiled_model: bytes, inputs: list[torch.Tensor], io_shapes: dict[str, list] = {}
    ) -> list[torch.Tensor]:
        """Run inference with the compiled model and return the output as list of torch Tensors.

        Args:
            compiled_model: Compiled device model from compilation service.
            inputs: Inputs to do inference on server.
            io_shapes: Defines the input and output shapes of the model. These shapes must be provided by the user
            when TensorRT cannot automatically infer them, such as in cases where dynamic dimensions are used for
            model outputs. To specify the shapes, use the following format:
                io_shapes = {"out.0": [2, 4, 64, 64]}

        Returns:
            A list of torch tensors from the inference outputs
        """
        # run inference
        outputs = self._inference(compiled_model, inputs, io_shapes)
        assert len(outputs) > 0, "Inference failed!"

        return outputs

    @property
    @abstractmethod
    def default_deployment(self) -> Deployment:
        """Provides the default deployment config without the device key."""
        raise NotImplementedError

    @property
    @abstractmethod
    def deployment_table(self) -> DeploymentTable:
        """Provides a set of supported values for each deployment config key."""
        raise NotImplementedError

    @abstractmethod
    def _ir_to_compiled(self, ir_bytes: bytes, compilation_args: dict[str, Any] = {}) -> bytes:
        """Converts a model from its intermediate representation (IR) to a compiled device model."""

    @abstractmethod
    def _profile(
        self, compiled_model: bytes, compilation_args: dict[str, Any] = {}
    ) -> tuple[float, DetailedResults]:
        """Profiles a compiled device model and returns the latency & detailed profiling results."""

    @abstractmethod
    def _inference(
        self, compiled_model: bytes, inputs: list[torch.Tensor], io_shapes: dict[str, list] = {}
    ) -> list[torch.Tensor]:
        """Run inference with the compiled model and return the output as list of torch tensors."""
