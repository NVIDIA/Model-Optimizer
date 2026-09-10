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

"""Module to compile a model for a target device."""

from typing import Any

import torch.nn as nn

from modelopt.torch.utils import is_channels_last

from ._runtime import Deployment, RuntimeRegistry
from .device_model import DeviceModel
from .utils import DEFAULT_ONNX_OPSET, OnnxBytes, get_onnx_bytes_and_metadata

__all__ = ["compile"]


def compile(
    model: nn.Module,
    dummy_input: Any | tuple,
    deployment: Deployment,
    dynamic_axes: dict = {},
    compilation_args: dict = {},
) -> DeviceModel:
    """Compile a given torch model into a device model according to the provided deployment.

    Args:
        model: PyTorch model to compile for target device.
        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        deployment: Deployment configuration with keys as specified below.

            * ``runtime``: the desired runtime for deployment (*required*);
            * ``accelerator``: the accelerator on the device to be used (*optional*);
            * ``precision``: the desired precision (*optional*);
            * ``onnx_opset``: the opset version (*optional*).

            An example of a deployment configuration is:

            .. code-block:: python

                deployment = {
                    "runtime": "ORT",
                    "accelerator": "CPU",
                    "precision": "fp32",
                    "onnx_opset": 13,
                    "verbose": "false",
                }
        dynamic_axes: A dictionary that maps input names to the dynamic axes of the model. This is
            useful when the model has dynamic input shapes. The keys are the input names and the
            values are the dynamic axes. For example, if the model has a dynamic batch size, the
            dictionary would look like:

            .. code-block:: python

                dynamic_axes = {"input": {0: "batch", 1: "sequence"}}

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.
        compilation_args: Additional arguments for the compilation process.
    Returns:
        An instance of DeviceModel.
    """
    # Add check for nhwc format before we formally support NHWC models
    assert not is_channels_last(model), "Only NCHW models are supported!"

    # Export onnx model and get some required names from it
    onnx_bytes, metadata = get_onnx_bytes_and_metadata(
        model,
        dummy_input,
        dynamic_axes=dynamic_axes,
        onnx_opset=int(deployment.get("onnx_opset", DEFAULT_ONNX_OPSET)),
    )

    client = RuntimeRegistry.get(deployment)

    # For the ORTLocalClient, we need to pass the bytes of the onnx model instead of the OnnxBytes object
    if client.__class__.__name__ == "ORTLocalClient":
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

    compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)

    return DeviceModel(client, compiled_model, metadata)
