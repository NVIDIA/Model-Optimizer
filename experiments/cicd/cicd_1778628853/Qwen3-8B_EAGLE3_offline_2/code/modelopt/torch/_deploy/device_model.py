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

"""Class representing the compiled model for a particular device."""

import os
from typing import Any

from modelopt.torch.utils import unflatten_tree

from ._runtime import DetailedResults, RuntimeClient
from .utils import ModelMetadata, generate_onnx_input


class DeviceModel:
    """On-device model with profiling functions and PyTorch-like inference interface.

    This object should be generated from
    :meth:`compile <modelopt.torch._deploy.compilation.compile>`.
    """

    def __init__(
        self,
        client: RuntimeClient,
        compiled_model: bytes,
        metadata: ModelMetadata,
        compilation_args: dict[str, Any] = {},
        io_shapes: dict[str, list] = {},
        ignore_nesting: bool = False,
    ):
        """Initialize a device model with the corresponding model, onnx, and engine model.

        Args:
            client: the runtime client used to compile the model.
            compiled_model: Compiled device model created during runtime compilation.
            metadata: The model's metadata (needed for inference/profiling with compiled model).
            compilation_args: Additional arguments for the compilation process.
            io_shapes: Defines the input and output shapes of the model. These shapes must be provided by the user
            when TensorRT cannot automatically infer them, such as in cases where dynamic dimensions are used for
            model outputs. To specify the shapes, use the following format:
                io_shapes = {"out.0": [2, 4, 64, 64]}
            ignore_nesting: If True, only the last part of the nested input name will be considered during inference.
                eg. if the input name is x.y.z, only z will be considered.
        """
        self.client = client
        self.compiled_model = compiled_model
        self.model_metadata = metadata
        self.config = self.model_metadata.get("config", {})
        self.compilation_args = compilation_args
        self.io_shapes = io_shapes
        self.ignore_nesting = ignore_nesting

    def __call__(self, *args, **kwargs):
        """Execute forward function of the model on the specified deployment and return output."""
        return self.forward(*args, **kwargs)

    def get_latency(self) -> float:
        """Profiling API to let user get model latency with the compiled device model.

        Returns:
            The latency of the compiled model in ms.
        """
        latency, _ = self._profile_device()
        return latency

    def profile(self, verbose: bool = False) -> tuple[float, DetailedResults]:
        """Inference API to let user do inference with the compiled device model.

        Args:
            verbose: If True, print out the profiling results as a table.

        Returns: A tuple (latency, detailed_result) where
            ``latency`` is the latency of the compiled model in ms,
            ``detailed_result`` is a dictionary containing additional benchmarking results
        """
        latency, detailed_result = self._profile_device()

        if verbose:
            print(detailed_result)

        return latency, detailed_result

    def forward(self, *args, **kwargs) -> Any:
        """Execute forward of the model on the specified deployment and return output.

        Arguments:
            args: Non-keyword arguments to the model for inference.
            kwargs: Keyword arguments to the model for inference.

        Returns:
            The inference result in the same (nested) data structure as the original model.

        .. note::

            This API let the users do inference with the compiled device model.

        .. warning::

            All return values will be of type ``torch.Tensor`` even if the original model returned
            native python types such as bool/int/float.
        """
        if self.compiled_model is None:
            raise AttributeError("Please compile the model first.")

        # Flatten all args, kwargs into a single list of tensors for onnx/device inference.
        all_args = (*args, kwargs) if kwargs or (args and isinstance(args[-1], dict)) else args

        # If Model metadata is None then DeviceModel is instantiated with raw ONNX bytes instead of PyTorch module.
        onnx_inputs = all_args[0]
        if self.model_metadata:
            onnx_inputs = list(
                generate_onnx_input(self.model_metadata, all_args, self.ignore_nesting).values()
            )

        # run inference with the engine equivalent of the model
        onnx_outputs = self.client.inference(
            compiled_model=self.compiled_model, inputs=onnx_inputs, io_shapes=self.io_shapes
        )

        # Note that bool/float/ints will be returned as corresponding tensors
        # TODO: maybe eventually we want to compare this against the original types
        # generate expected returned data structure of the model
        if not self.model_metadata:
            return onnx_outputs
        return unflatten_tree(onnx_outputs, self.model_metadata["output_tree_spec"])

    def save_compile_model(self, path: str, remove_hash: bool = False):
        """Saves the compiled model to a file.

        Args:
            path: The path to save the compiled model.
            remove_hash: If True, the hash will be removed from the saved model.
        """
        compiled_model = self.compiled_model
        if remove_hash:
            compiled_model = compiled_model[32:]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(compiled_model)

    def _profile_device(self) -> tuple[float, DetailedResults]:
        """Profile the device model stored in self and return latency results."""
        return self.client.profile(
            compiled_model=self.compiled_model, compilation_args=self.compilation_args
        )
