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

import argparse
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
from accelerate.hooks import remove_hook_from_module
from example_utils import (
    build_quant_cfg,
    copy_custom_model_files,
    get_model,
    get_processor,
    get_tokenizer,
    is_enc_dec,
    is_nemotron_vl,
    run_nemotron_vl_preview,
)
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    WhisperProcessor,
)
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_max_memory
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
from modelopt.torch.export import (
    export_hf_checkpoint,
    export_hf_vllm_fq_checkpoint,
    export_tensorrt_llm_checkpoint,
    get_model_type,
)
from modelopt.torch.export.model_utils import get_language_model_from_vl, is_multimodal_model
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg, need_calibration
from modelopt.torch.quantization.plugins.accelerate import init_quantized_weights
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
    get_supported_datasets,
)
from modelopt.torch.utils.image_processor import BaseImageProcessor, MllamaImageProcessor
from modelopt.torch.utils.memory_monitor import launch_memory_monitor
from modelopt.torch.utils.speech_dataset_utils import get_speech_dataset_dataloader
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader

RAND_SEED = 1234

mto.enable_huggingface_checkpointing()

def load_model(
    ckpt_path,
    device="cuda",
    gpu_mem_percentage=0.8,
    trust_remote_code=False,
    use_seq_device_map=False,
):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Prepare config kwargs for loading
    config_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}

    # Load config once
    try:
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
    except Exception as e:
        print(f"Error: Could not load config from {ckpt_path}: {e}")
        raise RuntimeError(f"Failed to load model configuration from {ckpt_path}") from e

    model_kwargs = config_kwargs.copy()

    if use_seq_device_map:
        device_map = "sequential"
        # If we use sequential, set max_memory limit to ensure that the model does not occupy the full GPU
        max_memory = get_max_memory()
        max_memory = {key: value * gpu_mem_percentage for key, value in max_memory.items()}
        model_kwargs["max_memory"] = max_memory

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
            # When computing the device_map, assuming bfloat16 precision by default,
            # unless specified by the hf_config.
            torch_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
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


def quantize_and_export_model(
    args: argparse.Namespace,
):
    model = load_model(    args.pyt_ckpt_path,
    device=args.device,
    gpu_mem_percentage=args.gpu_max_mem_percentage,
    trust_remote_code=args.trust_remote_code,
    use_seq_device_map=args.use_seq_device_map,)

    args.batch_size = get_max_batch_size(
        model,
        max_sample_length=args.calib_seq,
    )
    args.batch_size = min(args.batch_size, sum(args.calib_size))

    print(f"Use calib batch_size {args.batch_size}")
    tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)
    device = model.device
    calib_dataloader = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        device=device,
        include_labels=False,
    )
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    mtq_cfg = getattr(mtq, args.quant_cfg)  # type: ignore [arg-type]
    if args.kv_cache_quant_cfg is not None:
        kv_cache_quant_cfg = getattr(mtq, args.kv_cache_quant_cfg)  # type: ignore [arg-type]
        mtq_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
                mtq_cfg["quant_cfg"], kv_cache_quant_cfg["quant_cfg"]
            )
    input_str = tokenizer.decode(next(iter(calib_dataloader))["input_ids"][0])
    generated_str_before_ptq = model.run(input_str)

    quantized_model = mtq.quantize(model, mtq_cfg, calibrate_loop)
    mtq.print_quant_summary(quantized_model)
    generated_str_after_ptq = model.run(input_str)

    print("--------")
    print(f"example test input: {input_str}")
    print("--------")
    print(f"example outputs before ptq: {generated_str_before_ptq}")
    print("--------")
    print(f"example outputs after ptq: {generated_str_after_ptq}")
    
    export_hf_vllm_fq_checkpoint(quantized_model, args.export_path)
    print(f"Model exported to {args.export_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        help="Specify where the PyTorch checkpoint path is",
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--quant_cfg",
        help=(
            "Quantization configuration."
        ),
        default="FP8_DEFAULT_CFG",
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for calibration. Default to 0 as we calculate max batch size on-the-fly",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--calib_size",
        help=(
            "Number of samples for calibration. If a comma separated list of values is provided, "
            "each value will be used as the calibration size for the corresponding dataset. "
            "This argument will be parsed and converted as a list of ints."
        ),
        type=str,
        default="512",
    )
    parser.add_argument(
        "--calib_seq",
        help="Maximum sequence length for calibration.",
        type=int,
        default=512,
    )
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument(
        "--dataset",
        help=(
            f"name of a dataset, or a comma separated list of datasets. "
            f"dataset choices are {get_supported_datasets()}"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--kv_cache_quant_cfg",
        required=False,
        default=None,
        help="Specify KV cache quantization configuration, default to None if not provided",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gpu_max_mem_percentage",
        help=(
            "Specify the percentage of available GPU memory to use for loading the model when "
            "device_map is set to sequential. "
            "By default, 80%% of the available GPU memory is used."
        ),
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--use_seq_device_map",
        help=(
            "Use device_map=sequential to load the model onto GPUs. This ensures the model is loaded "
            "utilizing the percentage of available GPU memory as specified by the value passed with gpu_max_mem flag."
            "Helpful in cases where device_map=auto loads model unevenly on GPUs causing GPU OOM during quantization."
        ),
        default=False,
        action="store_true",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise OSError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    # launch a memory monitor to read the currently used GPU memory.
    launch_memory_monitor()

    # Force eager execution for all model types.
    torch.compiler.set_stance("force_eager")

    # Quantize
    quantize_and_export_model(
        args,
        
    )


if __name__ == "__main__":
    args = parse_args()

    args.dataset = args.dataset.split(",") if args.dataset else None
    args.calib_size = [int(num_sample) for num_sample in args.calib_size.split(",")]
    main(args)
