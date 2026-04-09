# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Quantization support for accelerate modified models."""

import warnings
from contextlib import contextmanager
from typing import Any

import torch
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import AlignDevicesHook, SequentialHook
from accelerate.utils import get_max_memory, infer_auto_device_map
from accelerate.utils.dataclasses import CustomDtype
from accelerate.utils.offload import PrefixedDataset

import modelopt.torch.quantization as mtq

__all__ = ["init_quantized_weights"]


def _get_cpu_offload_hook(hook):
    if isinstance(hook, AlignDevicesHook) and hook.offload and hook.weights_map is not None:
        assert "weight" in hook.weights_map
        if (
            isinstance(hook.weights_map, PrefixedDataset)
            and hook.weights_map.prefix + "weight" not in hook.weights_map.dataset.state_dict
        ):
            raise NotImplementedError(
                "This layer could be offloaded to disk. We don't support this yet."
            )
        return hook
    elif isinstance(hook, SequentialHook):
        for h in hook.hooks:
            align_hook = _get_cpu_offload_hook(h)
            if align_hook is not None:
                return align_hook
    return None


@contextmanager
def weight_access_and_writeback_context(module):
    """Context manager for weight access and writeback for modules managed by accelerate."""
    assert hasattr(module, "_hf_hook")
    align_hook = _get_cpu_offload_hook(module._hf_hook)

    if align_hook:
        # Accelerate uses AlignDevicesHook to offload weights to CPU/Disk and then reload them in the forward pass
        # The CPU/Disk offloaded weights are managed by PrefixDataset and OffloadedWeightsLoader
        # See https://github.com/huggingface/accelerate/blame/f48d95c4939b281505a45b3d6e0bf554b65cc1ea/src/accelerate/utils/offload.py#L104-L141
        # TODO: Add support for disk-offloaded models if needed (they will be really slow, hence low priority)

        # This will load the weights from CPU state_dict and move it to the GPU from meta device
        align_hook.pre_forward(module)
    try:
        yield
    finally:
        if align_hook:
            # Update the weight in the CPU state_dict
            if isinstance(align_hook.weights_map, PrefixedDataset):
                key = align_hook.weights_map.prefix + "weight"
                w_map = align_hook.weights_map.dataset.state_dict
            else:
                key, w_map = "weight", align_hook.weights_map
            w_map[key] = module.weight.data.to(w_map[key].device, dtype=w_map[key].dtype)
            align_hook.post_forward(module, None)


@contextmanager
def init_quantized_weights(
    quant_cfg: dict[str, Any], gpu_mem_percentage: float = 0.8, quant_gemm: bool = False
):
    """Context manager for initializing and loading HuggingFace models with quantized and compressed weights.

    This context manager patches `from_pretrained` to automatically:
    1. Initialize the model with an empty config
    2. Apply quantization configuration
    3. Compress the weights
    4. Load and dispatch the model to appropriate devices

    Args:
        quant_cfg: The quantization config to apply to the model
        gpu_mem_percentage: Percentage of GPU memory to use (0.0-1.0)
        quant_gemm: Whether to enable quantized GEMM

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from modelopt.torch.quantization.plugins.huggingface import init_quantized_weights
        import modelopt.torch as mtq

        # Load quantized and compressed model in one step
        with init_quantized_weights(mtq.NVFP4_DEFAULT_CFG):
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")


        # The model is already quantized, compressed, and loaded to appropriate devices

        # calibrate model
        mtq.calibrate(model, "max", forward_loop)
        ```
    """

    def get_no_split_module_classes(model):
        """Get no-split module classes for device mapping."""
        no_split_classes = set()

        # Get all named modules
        for name, module in model.named_modules():
            # Look for first layer patterns
            if name.endswith((".layers.0", ".layer.0", ".h.0", ".blocks.0")):
                no_split_classes.add(module.__class__.__name__)

        return list(no_split_classes)

    def get_model_device_map(model, gpu_mem_percentage):
        """Create optimized device map for the quantized and compressed model."""
        # Get compression ratio to adjust memory usage
        max_memory = get_max_memory()
        no_split_classes = get_no_split_module_classes(model)
        if len(no_split_classes) == 0:
            warnings.warn(
                "No no-split module classes found for the model. Default device map might be incorrect."
            )
        max_memory = {k: v * gpu_mem_percentage for k, v in max_memory.items()}

        def _get_special_dtype_map(model):
            special_dtype_map = {}

            def _get_byte_size(module):
                weight_num_bit = module.weight_quantizer.num_bits
                if isinstance(weight_num_bit, tuple):
                    assert len(weight_num_bit) == 2, (
                        "Weight num bit must be a tuple of two integers."
                    )
                    weight_num_bit = weight_num_bit[0] + weight_num_bit[1] + 1
                if weight_num_bit == 8:
                    return CustomDtype.FP8
                elif weight_num_bit == 4:
                    return CustomDtype.INT4
                else:
                    raise ValueError(f"Unsupported weight num bit: {weight_num_bit}")

            for name, module in model.named_modules():
                if (
                    hasattr(module, "weight")
                    and hasattr(module, "weight_quantizer")
                    and not module.weight_quantizer.fake_quant
                ):
                    special_dtype_map[name + ".weight"] = _get_byte_size(module)
            return special_dtype_map

        special_dtype_map = _get_special_dtype_map(model)
        inferred_device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_classes,
            special_dtypes=special_dtype_map,
        )
        return inferred_device_map

    # Store original from_pretrained methods that we'll override
    original_auto_causal_lm_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    # Create patched from_pretrained that applies quantization and compression
    def patched_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
        """Patched from_pretrained that handles quantization, compression and device mapping."""
        # Initialize with empty config first to avoid loading weights twice
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        if config is None:
            # Get config from pretrained model
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code
            )

        with init_empty_weights():
            # Fix torch_dtype to match original model
            torch_dtype = kwargs.get(
                "dtype", kwargs.get("torch_dtype", getattr(config, "torch_dtype", torch.float16))
            )
            model = cls.from_config(config, dtype=torch_dtype)

        mtq.quantize(model, quant_cfg)
        mtq.compress(model, config=mtq.CompressConfig(quant_gemm=quant_gemm))
        _device_map = get_model_device_map(model, gpu_mem_percentage)

        return load_checkpoint_and_dispatch(
            model,
            checkpoint=pretrained_model_name_or_path,
            device_map=_device_map,
            *args,
            **kwargs,
        )

    try:
        # Apply our patches
        transformers.AutoModelForCausalLM.from_pretrained = classmethod(patched_from_pretrained)
        yield
    finally:
        # Restore original methods
        transformers.AutoModelForCausalLM.from_pretrained = original_auto_causal_lm_from_pretrained
