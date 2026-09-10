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

"""Module to profile a model on a target device."""

from typing import Any

import torch.nn as nn

from ._runtime import Deployment, DetailedResults
from .compilation import compile

__all__ = ["get_latency", "profile"]


def get_latency(
    model: nn.Module,
    dummy_input: Any | tuple,
    deployment: Deployment,
) -> float:
    """Obtain deployment latency of model by compiling and sending model to engine for profiling.

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

    Returns:
        The latency of the compiled model in ms.
    """
    device_model = compile(model, dummy_input, deployment)
    return device_model.get_latency()


def profile(
    model: nn.Module,
    dummy_input: Any | tuple,
    deployment: Deployment,
    verbose: bool = False,
) -> tuple[float, DetailedResults]:
    """Model profile method to help user to profile their model on target device.

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
                    "verbose": False,
                }
        verbose: If True, print out the profiling results as a table.

        Returns: A tuple (latency, detailed_result) where
            ``latency`` is the latency of the compiled model in ms,
            ``detailed_result`` is a dictionary containing additional benchmarking results
    """
    device_model = compile(model, dummy_input, deployment)
    return device_model.profile(verbose)
