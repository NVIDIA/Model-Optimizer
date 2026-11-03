# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/quant_utils.py
# and https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/calibrate.py
#
# MIT License
#
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""ONNX Runtime model loading and inference-session setup."""

__all__ = []

from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization.quant_utils import add_infer_metadata

import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.logging_config import logger


def load_model_with_shape_infer(model_path: Path) -> onnx.ModelProto:
    """Load model while performing symbolic shape infer and ONNX shape inference."""
    model = onnx.load(str(model_path), load_external_data=True)
    try:
        model = onnx_utils.infer_shapes(model)
        add_infer_metadata(model)
    except Exception as e:
        logger.info(f"Failed to infer shapes for model {model_path}: {e}")
    return model


def _configure_session_providers(
    sess_options: ort.SessionOptions,
    providers: list[str | tuple[str, dict]],
    trt_rtx_backend: str,
) -> dict[str, list[str | tuple[str, dict]]]:
    """Configure providers using the mechanism required by the selected EP.

    ``providers`` contains provider names or ``(name, options)`` pairs in priority order.
    ABI EPs are exposed as devices and must be added to ``sess_options``; passing them through
    ``InferenceSession(providers=...)`` overrides that configuration. This helper preserves the
    ABI device path while returning normal provider arguments for other EPs.
    """
    if trt_rtx_backend != "abi":
        return {"providers": providers}

    available_providers = set(ort.get_available_providers())
    ep_devices = ort.get_ep_devices()
    plugin_provider_names = {device.ep_name for device in ep_devices} - available_providers
    provider_names = {
        provider[0] if isinstance(provider, tuple) else provider for provider in providers
    }
    if not plugin_provider_names.intersection(provider_names):
        return {"providers": providers}

    for provider in providers:
        provider_name, provider_options = (
            provider if isinstance(provider, tuple) else (provider, {})
        )
        if provider_name in plugin_provider_names:
            selected_devices = [device for device in ep_devices if device.ep_name == provider_name]
            sess_options.add_provider_for_devices(selected_devices, provider_options)
        else:
            sess_options.add_provider(provider_name, provider_options)
    return {}


def _create_inference_session_with_ep_config(calibrator, **kwargs):
    """Create an ORT InferenceSession."""
    model_path = kwargs.get("model_path")
    logger.debug("Creating inference session with Execution Provider configuration")

    trt_rtx_backend = kwargs.get("trt_rtx_backend", "legacy")
    if trt_rtx_backend not in ("legacy", "abi"):
        raise ValueError(f"trt_rtx_backend must be 'legacy' or 'abi', got {trt_rtx_backend!r}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    sess_options.enable_cpu_mem_arena = False

    providers = kwargs.get("execution_providers", [])
    logger.debug(f"Execution providers: {providers}")

    # Note. This path can be an empty string, which denotes that the model has custom ops and TRT EP is needed.
    calibrator.trt_extra_plugin_lib_paths = kwargs.get("trt_extra_plugin_lib_paths")
    if calibrator.trt_extra_plugin_lib_paths is not None:
        logger.debug(f"TRT extra plugin paths: {calibrator.trt_extra_plugin_lib_paths}")
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                f"Could not find `TensorrtExecutionProvider`, only {ort.get_available_providers()}"
            )
        trt_ep_options = (
            {"trt_extra_plugin_lib_paths": calibrator.trt_extra_plugin_lib_paths}
            if calibrator.trt_extra_plugin_lib_paths
            else {}
        )

        # Set GPU memory usage limit
        trt_ep_options["trt_max_workspace_size"] = 80 * (1024**3)  # 80GB
        logger.debug(f"TRT EP options: {trt_ep_options}")

        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        providers.insert(0, ("TensorrtExecutionProvider", trt_ep_options))

    def _update_provider_config(provider, config):
        if isinstance(provider, tuple) and len(provider) > 1 and isinstance(provider[1], dict):
            provider[1].update(config)
        else:
            provider = (provider, config)
        return provider

    for i in range(len(providers)):
        if any(p in providers[i] for p in ["CPUExecutionProvider", "CUDAExecutionProvider"]):
            providers[i] = _update_provider_config(
                providers[i], {"arena_extend_strategy": "kSameAsRequested"}
            )

    session_path = calibrator.augmented_model_path if model_path is None else model_path
    provider_kwargs = _configure_session_providers(sess_options, providers, trt_rtx_backend)
    calibrator.infer_session = ort.InferenceSession(
        session_path,
        sess_options=sess_options,
        **provider_kwargs,
    )

    # Group qdq tensors will have the same scaling factor.
    calibrator.group_qdq_tensors = kwargs.get("group_qdq_tensors")
    if calibrator.group_qdq_tensors:
        logger.debug(f"Group QDQ tensors: {calibrator.group_qdq_tensors}")
