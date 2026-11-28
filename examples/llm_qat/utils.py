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

import torch
import transformers
from peft import LoraConfig, TaskType
from transformers import default_data_collator


def load_model(model_name_or_path: str, **kwargs) -> transformers.PreTrainedModel:
    """Load a causal LM to quantize or train, an MXFP4 checkpoint dequantized to BF16.

    Transformers loads an MXFP4 checkpoint (``openai/gpt-oss-*``) packed wherever the ``kernels``
    package is installed, and ModelOpt can neither quantize nor train packed experts.
    """
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    quantization_config = getattr(config, "quantization_config", None) or {}
    if quantization_config.get("quant_method") != "mxfp4":
        return transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    from transformers.integrations.mxfp4 import Mxfp4Dequantize

    kwargs["quantization_config"] = transformers.Mxfp4Config(dequantize=True)
    if kwargs.get("device_map") == "auto":
        # Dequantizing across a balanced multi-GPU split trips an illegal memory access.
        kwargs["device_map"] = "sequential"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    # Saving reverses the loading converters, and transformers 5.5 to 5.18 reverse the dequantizing
    # ones into keys named by their patterns (``gate_up_proj$``), every layer's experts under one.
    # Without them the model saves as the BF16 checkpoint it now is.
    model._weight_conversions = [
        conversion
        for conversion in getattr(model, "_weight_conversions", None) or []
        if not any(isinstance(op, Mxfp4Dequantize) for op in getattr(conversion, "operations", []))
    ]
    return model


def make_supervised_data_module(
    data_args,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Build train/eval datasets and a default collator."""
    from dataset_utils import build_blend_dataset, load_blend_config

    config = load_blend_config(data_args.dataset_config)
    max_length = getattr(tokenizer, "model_max_length", 4096)

    ds = build_blend_dataset(
        config,
        tokenizer,
        max_length,
        seed=data_args.dataset_seed,
        cache_dir=data_args.dataset_cache_dir,
        shuffle=data_args.shuffle,
        shuffle_buffer=data_args.shuffle_buffer,
        num_proc=data_args.num_proc,
    )

    train_ds = ds["train"]
    if data_args.train_samples > 0 and data_args.train_samples < len(train_ds):
        train_ds = train_ds.select(range(data_args.train_samples))

    eval_ds = ds["eval"]
    if data_args.eval_samples > 0 and data_args.eval_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(data_args.eval_samples))

    return {
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": default_data_collator,
    }


def get_lora_config():
    return LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def get_metrics_with_perplexity(metrics):
    """Add perplexity to the metrics."""
    if "eval_loss" in metrics:
        metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])))
    return metrics
