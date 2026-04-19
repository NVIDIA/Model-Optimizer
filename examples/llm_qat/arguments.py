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

"""Shared argument dataclasses for llm_qat scripts (quantize.py, train.py)."""

from dataclasses import field

import transformers

from modelopt.torch.opt.plugins.transformers import ModelOptHFArguments, ModelOptTrainerArguments


class ModelArguments(ModelOptHFArguments):
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "HuggingFace model ID or local path to a pretrained model."},
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right-padded (and possibly truncated)."
            )
        },
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={
            "help": (
                "Attention implementation: 'flash_attention_2', 'flash_attention_3', "
                "'sdpa', or 'eager'."
            )
        },
    )


class DataArguments(ModelOptHFArguments):
    dataset_config: str = field(
        default="configs/dataset/blend.yaml",
        metadata={"help": "Path to a dataset blend YAML config file."},
    )
    train_samples: int = field(
        default=20000,
        metadata={"help": "Number of training samples to use."},
    )
    eval_samples: int = field(
        default=2000,
        metadata={"help": "Number of evaluation samples to use."},
    )
    dataset_seed: int = field(
        default=42,
        metadata={"help": "Random seed for dataset shuffling."},
    )
    dataset_cache_dir: str = field(
        default=".dataset_cache/tokenized",
        metadata={"help": "Directory for caching tokenized datasets."},
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle dataset sources (reservoir sampling)."},
    )
    shuffle_buffer: int = field(
        default=10000,
        metadata={"help": "Buffer size for streaming shuffle."},
    )
    num_proc: int = field(
        default=16,
        metadata={"help": "Number of CPU workers for tokenization."},
    )


class TrainingArguments(ModelOptTrainerArguments, transformers.TrainingArguments):
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    use_liger_kernel: bool = field(
        default=True,
        metadata={"help": "Use Liger kernel for fused loss computation. Reduces memory usage."},
    )
    lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add LoRA (Low-Rank Adaptation) adapter before training. When using real quantization, "
                "the LoRA adapter must be set, as quantized weights will be frozen during training."
            )
        },
    )


class QuantizeArguments(ModelOptHFArguments):
    recipe: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a quantization recipe YAML file (built-in or custom). "
                "Built-in recipes can be specified by relative path, e.g. "
                "'general/ptq/nvfp4_default-fp8_kv'."
            ),
        },
    )
    calib_size: int = field(
        default=512,
        metadata={
            "help": (
                "Specify the calibration size for quantization. The calibration dataset is used to"
                " setup the quantization scale parameters."
            )
        },
    )
    calib_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for calibration data during quantization."},
    )
    compress: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compress the model weights after quantization for QLoRA. "
                "This is useful for reducing the model size."
            )
        },
    )
    quantize_output_dir: str = field(
        default="quantized_model",
        metadata={"help": "Directory to save the quantized model checkpoint."},
    )
