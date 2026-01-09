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

import copy
import glob
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_max_memory
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.image_processor import BaseImageProcessor, MllamaImageProcessor

SPECULATIVE_MODEL_LIST = ["Eagle", "Medusa"]


def run_nemotron_vl_preview(
    full_model, tokenizer, input_ids, pyt_ckpt_path, stage_name, allow_fallback=False
):
    """Run text-only and VL preview generation for Nemotron VL models.

    Args:
        full_model: The full VL model
        tokenizer: The tokenizer
        input_ids: Input tensor for generation
        pyt_ckpt_path: Path to the model checkpoint
        stage_name: Description of the stage (e.g., "before quantization", "after quantization")
        allow_fallback: Whether to allow fallback to standard generate on failure

    Returns:
        Generated text response or None if generation failed
    """
    from vlm_utils import run_text_only_generation, run_vl_preview_generation

    print(f"Running text-only preview generation for Nemotron VL model ({stage_name})...")
    question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generation_config = {
        "max_new_tokens": 100,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Try text-only generation
    text_response = run_text_only_generation(
        full_model, tokenizer, question, generation_config, pyt_ckpt_path
    )

    if text_response is not None:
        print(f"✅ Text-only generation successful: {text_response[:100]}...")
        generated_ids = text_response
    elif allow_fallback:
        print("Text-only generation failed, falling back to standard generate...")
        generated_ids = full_model.generate(input_ids, max_new_tokens=100)
    else:
        generated_ids = None

    # Run additional VL test with images
    print(f"Running additional VL test with images ({stage_name})...")
    run_vl_preview_generation(full_model, tokenizer, pyt_ckpt_path, stage_name)

    return generated_ids


def _is_multimodal_config(config):
    """Check if a config indicates a multimodal model (config-only version of is_multimodal_model)."""
    return (
        hasattr(config, "vision_config")  # Standard vision config (e.g., Qwen2.5-VL)
        or getattr(config, "model_type", "") == "phi4mm"  # Phi-4 multimodal
        or hasattr(config, "vision_lora")  # Vision LoRA configurations
        or hasattr(config, "audio_processor")  # Audio processing capabilities
        or (
            hasattr(config, "embd_layer") and hasattr(config.embd_layer, "image_embd_layer")
        )  # Image embedding layers
    )


def is_nemotron_vl(model_or_config):
    """Check if model or config indicates a Nemotron VL model.

    Args:
        model_or_config: Either a model instance or a config object.

    Returns:
        bool: True if it's a Nemotron VL model, False otherwise.
    """
    # Try to get config from model, or use directly if it's a config
    if hasattr(model_or_config, "config"):
        config = model_or_config.config
        from modelopt.torch.export.model_utils import is_multimodal_model

        if not is_multimodal_model(model_or_config):
            return False
    else:
        config = model_or_config
        if not _is_multimodal_config(config):
            return False

    architectures = getattr(config, "architectures", [])
    return any("nemotron" in arch.lower() for arch in architectures)


def build_quant_cfg(
    qformat,
    kv_cache_qformat,
    awq_block_size,
    model_type,
    quant_cfg_choices,
    kv_quant_cfg_choices,
) -> dict[str, Any]:
    quant_cfg = {}
    assert qformat in quant_cfg_choices, (
        f"Unsupported quantization format: {qformat} with {kv_cache_qformat} KV cache"
    )

    quant_cfg = quant_cfg_choices[qformat]

    if "awq" in qformat:
        quant_cfg = copy.deepcopy(quant_cfg_choices[qformat])
        weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]
        # If awq_block_size argument is provided, update weight_quantizer
        if awq_block_size:
            weight_quantizer["block_sizes"][-1] = awq_block_size

        # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
        if qformat == "w4a8_awq" and model_type in ["gemma", "mpt"]:
            quant_cfg["algorithm"] = {"method": "awq_lite", "alpha_step": 1}

    enable_quant_kv_cache = kv_cache_qformat != "none"
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

    # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
    if enable_quant_kv_cache:
        quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
            quant_cfg,
            getattr(mtq, kv_quant_cfg_choices[kv_cache_qformat])["quant_cfg"],
        )

    # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
    if model_type == "gemma" and "int8_sq" in qformat:
        quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

    if model_type == "phi4mm":
        # Only quantize the language model
        quant_cfg["quant_cfg"]["*speech*"] = {"enable": False}
        quant_cfg["quant_cfg"]["*audio*"] = {"enable": False}
        quant_cfg["quant_cfg"]["*image*"] = {"enable": False}
        quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}

        if model_type in ["qwen3moe", "qwen3next"] and qformat == "nvfp4":
            # Disable the attention projection layers to retain accuracy
            quant_cfg["quant_cfg"]["model*.*attn*in_proj*"] = {"enable": False}
            quant_cfg["quant_cfg"]["model*.*attn*q_proj*"] = {"enable": False}
            quant_cfg["quant_cfg"]["model*.*attn*k_proj*"] = {"enable": False}
            quant_cfg["quant_cfg"]["model*.*attn*v_proj*"] = {"enable": False}

    return quant_cfg


def is_speculative(hf_config):
    """Check if the model architecture is a speculative model."""
    return hf_config.architectures and any(
        name in hf_config.architectures[0] for name in SPECULATIVE_MODEL_LIST
    )


def get_tokenizer(ckpt_path, trust_remote_code=False, **kwargs) -> PreTrainedTokenizerBase:
    print(f"Initializing tokenizer from {ckpt_path}")

    if "vila" in ckpt_path.lower():
        ckpt_path += "/llm"

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # can't set attribute 'pad_token' for "<unk>"
    # We skip this step for Nemo models
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer


def get_processor(
    ckpt_path,
    model_type,
    device: torch.device = "auto",
    trust_remote_code=False,
    attn_implementation=None,
) -> BaseImageProcessor | ProcessorMixin | None:
    """
    Returns a :class:`modelopt.torch.utils.image_processor.MllamaImageProcessor` object.
    """
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if model_type == "whisper":
        processor = AutoProcessor.from_pretrained(
            ckpt_path,
            padding_side="left",
            **model_kwargs,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        assert processor.tokenizer.pad_token is not None, (
            f"Pad token for {ckpt_path} cannot be set!"
        )

        return processor
    elif model_type == "mllama":
        processor = AutoProcessor.from_pretrained(
            ckpt_path,
            padding_side="left",
            **model_kwargs,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        assert processor.tokenizer.pad_token is not None, (
            f"Pad token for {ckpt_path} cannot be set!"
        )

        return MllamaImageProcessor(processor, device)

    return None


def get_dtype(dtype):
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    return dtype


def get_model(
    ckpt_path,
    device="cuda",
    gpu_mem_percentage=0.8,
    trust_remote_code=False,
    use_seq_device_map=False,
    attn_implementation=None,
):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Add VILA to sys.path before loading config if needed
    if "vila" in ckpt_path.lower():
        vila_path = os.path.join(ckpt_path, "..", "VILA")
        if vila_path not in sys.path:
            sys.path.append(vila_path)
        from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa: F401

    # Prepare config kwargs for loading
    config_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}

    # Load config once and handle VL model detection
    try:
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
        if is_nemotron_vl(hf_config):
            print(
                "Detected Nemotron VL model from config. "
                "Disabling automatic device mapping for compatibility."
            )
            device_map = None
    except Exception as e:
        print(f"Error: Could not load config from {ckpt_path}: {e}")
        raise RuntimeError(f"Failed to load model configuration from {ckpt_path}") from e
    if attn_implementation is not None:
        config_kwargs["attn_implementation"] = attn_implementation

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = config_kwargs.copy()
    # Don't set torch_dtype for VILA models as they handle it explicitly in their builder
    if "vila" not in ckpt_path.lower():
        model_kwargs.setdefault("torch_dtype", "auto")

    if "vila" in ckpt_path.lower():
        from transformers import AutoModel

        hf_vila = AutoModel.from_pretrained(
            ckpt_path,
            device_map=device_map,
            **model_kwargs,
        )
        model = hf_vila.llm
    else:
        if use_seq_device_map:
            device_map = "sequential"
            # If we use sequential, set max_memory limit to ensure that the model does not occupy the full GPU
            max_memory = get_max_memory()
            max_memory = {key: value * gpu_mem_percentage for key, value in max_memory.items()}
            model_kwargs["max_memory"] = max_memory

        if hf_config.model_type == "bart":
            # device_map "auto" and "cuda" triggers error regarding meta tensor from safetensors
            device_map = None

        if is_speculative(hf_config):
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
            )
        else:
            architecture = hf_config.architectures[0]

            if not hasattr(transformers, architecture) or "Deepseek" in architecture:
                if not hasattr(transformers, architecture):
                    warnings.warn(
                        f"Architecture {architecture} not found in transformers: {transformers.__version__}. "
                        "Falling back to AutoModelForCausalLM."
                    )
                assert trust_remote_code, (
                    "Please set trust_remote_code to True if you want to use this architecture"
                )

                auto_model_module = AutoModelForCausalLM
                from_config = auto_model_module.from_config
            else:
                auto_model_module = getattr(transformers, architecture)
                from_config = auto_model_module._from_config

            with init_empty_weights():
                # When computing the device_map, assuming half precision by default,
                # unless specified by the hf_config.
                torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
                model_kwargs2 = model_kwargs.copy()
                if auto_model_module != AutoModelForCausalLM:
                    model_kwargs2.pop("trust_remote_code", None)
                model_kwargs2["torch_dtype"] = torch_dtype
                model_kwargs2.pop("max_memory", None)
                model = from_config(hf_config, **model_kwargs2)

            max_memory = get_max_memory()
            inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

            on_cpu = "cpu" in inferred_device_map.values()

            if on_cpu:
                for _device in max_memory:
                    if isinstance(_device, int):
                        max_memory[_device] *= gpu_mem_percentage

                print(
                    "Model does not fit to the GPU mem. "
                    f"We apply the following memory limit for calibration: \n{max_memory}\n"
                    "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
                    "reduce the calibration `batch_size` manually."
                )
                model_kwargs["max_memory"] = max_memory

            model = auto_model_module.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
            )
    model.eval()

    # If device_map was disabled (None), manually move model to target device
    if device_map is None and device != "cpu":
        print(f"Moving model to {device} device...")
        model = model.to(device)

    if device == "cuda" and not is_model_on_gpu(model):
        print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def is_enc_dec(model_type) -> bool:
    """Return if the model is a encoder-decoder model."""
    return model_type in ["t5", "bart", "whisper"]


def _resolve_model_path(model_name_or_path: str, trust_remote_code: bool = False) -> str:
    """Resolve a model name or path to a local directory path.

    If the input is already a local directory, returns it as-is.
    If the input is a HuggingFace model ID, attempts to resolve it to the local cache path.

    Args:
        model_name_or_path: Either a local directory path or HuggingFace model ID
        trust_remote_code: Whether to trust remote code when loading the model

    Returns:
        Local directory path to the model files
    """
    # If it's already a local directory, return as-is
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # Try to resolve HuggingFace model ID to local cache path
    try:
        # First try to load the config to trigger caching
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        # The config object should have the local path information
        # Try different ways to get the cached path
        if hasattr(config, "_name_or_path") and os.path.isdir(config._name_or_path):
            return config._name_or_path

        # Alternative: use snapshot_download if available
        if snapshot_download is not None:
            try:
                local_path = snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=["*.py", "*.json"],  # Only download Python files and config
                )
                return local_path
            except Exception as e:
                print(f"Warning: Could not download model files using snapshot_download: {e}")

        # Fallback: try to find in HuggingFace cache
        from transformers.utils import TRANSFORMERS_CACHE

        # Look for the model in the cache directory
        cache_pattern = os.path.join(TRANSFORMERS_CACHE, "models--*")
        cache_dirs = glob.glob(cache_pattern)

        # Convert model name to cache directory format
        model_cache_name = model_name_or_path.replace("/", "--")
        for cache_dir in cache_dirs:
            if model_cache_name in cache_dir:
                # Look for the snapshots directory
                snapshots_dir = os.path.join(cache_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Get the latest snapshot
                    snapshot_dirs = [
                        d
                        for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))
                    ]
                    if snapshot_dirs:
                        latest_snapshot = max(snapshot_dirs)  # Use lexicographically latest
                        snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                        return snapshot_path

    except Exception as e:
        print(f"Warning: Could not resolve model path for {model_name_or_path}: {e}")

    # If all else fails, return the original path
    # This will cause the copy function to skip with a warning
    return model_name_or_path


def copy_custom_model_files(source_path: str, export_path: str, trust_remote_code: bool = False):
    """Copy custom model files (configuration_*.py, modeling_*.py, *.json, etc.) from source to export directory.

    This function copies custom Python files and JSON configuration files that are needed for
    models with custom code. It excludes config.json and model.safetensors.index.json as these
    are typically handled separately by the model export process.

    Args:
        source_path: Path to the original model directory or HuggingFace model ID
        export_path: Path to the exported model directory
        trust_remote_code: Whether trust_remote_code was used (only copy files if True)
    """
    if not trust_remote_code:
        return

    # Resolve the source path (handles both local paths and HF model IDs)
    resolved_source_path = _resolve_model_path(source_path, trust_remote_code)

    source_dir = Path(resolved_source_path)
    export_dir = Path(export_path)

    if not source_dir.exists():
        if resolved_source_path != source_path:
            print(
                f"Warning: Could not find local cache for HuggingFace model '{source_path}' "
                f"(resolved to '{resolved_source_path}')"
            )
        else:
            print(f"Warning: Source directory '{source_path}' does not exist")
        return

    if not export_dir.exists():
        print(f"Warning: Export directory {export_path} does not exist")
        return

    # Common patterns for custom model files that need to be copied
    custom_file_patterns = [
        "configuration_*.py",
        "modeling*.py",
        "tokenization_*.py",
        "processing_*.py",
        "image_processing*.py",
        "feature_extraction_*.py",
        "*.json",
    ]

    copied_files = []
    for pattern in custom_file_patterns:
        for file_path in source_dir.glob(pattern):
            if file_path.is_file():
                # Skip config.json and model.safetensors.index.json as they're handled separately
                if file_path.name in ["config.json", "model.safetensors.index.json"]:
                    continue
                dest_path = export_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
                    print(f"Copied custom model file: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Failed to copy {file_path.name}: {e}")

    if copied_files:
        print(f"Successfully copied {len(copied_files)} custom model files to {export_path}")
    else:
        print("No custom model files found to copy")
