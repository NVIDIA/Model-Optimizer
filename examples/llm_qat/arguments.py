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

from modelopt.torch.distill.plugins.huggingface import DistillArguments
from modelopt.torch.opt.plugins.transformers import ModelOptArgParser, ModelOptHFArguments
from modelopt.torch.quantization.plugins.transformers_trainer import (
    QuantizationArguments as ModelOptQuantizationArguments,
)


class ModelArguments(ModelOptHFArguments):
    model_name_or_path: str = field(
        default="Qwen/Qwen3-8B",
        metadata={
            "help": "HuggingFace model name or local path to the base model to quantize/train."
        },
    )
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right-padded (and possibly truncated)."
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


class TrainingArguments(ModelOptHFArguments, transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add LoRA (Low-Rank Adaptation) adapter before training. When using real quantization, "
                "the LoRA adapter must be set, as quantized weights will be frozen during training."
            )
        },
    )


class QuantizeArguments(ModelOptQuantizationArguments):
    calib_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for calibration data during quantization."},
    )
    output_dir: str = field(
        default="quantized_model",
        metadata={"help": "Directory to save the quantized model checkpoint."},
    )


TRAINING_ARG_TYPES = (ModelArguments, TrainingArguments, DataArguments, DistillArguments)
QUANTIZE_ARG_TYPES = (ModelArguments, DataArguments, QuantizeArguments)


def _unique_arg_types(*arg_type_groups):
    return tuple(dict.fromkeys(arg_type for group in arg_type_groups for arg_type in group))


def _build_usage_header():
    columns = {"quantize.py": QUANTIZE_ARG_TYPES, "train.py": TRAINING_ARG_TYPES}
    rows = _unique_arg_types(TRAINING_ARG_TYPES, QUANTIZE_ARG_TYPES)
    script_names = list(columns)

    lines = [
        "## Arguments by Script",
        "",
        "| Argument group | " + " | ".join(f"`{name}`" for name in script_names) + " |",
        "|---|" + ":---:|" * len(script_names),
    ]
    for dc in rows:
        cells = ["✅" if dc in columns[name] else "-" for name in script_names]
        lines.append(f"| {dc.__name__} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "**Note:** Each script accepts only the arguments in its supported groups ✅. Arguments from other "
        "groups are ignored (if set in a `--config` YAML) or error out (if passed as a CLI flag).",
    ]
    return "\n".join(lines)


def get_training_arg_parser():
    return ModelOptArgParser(TRAINING_ARG_TYPES)


def get_quantize_arg_parser():
    return ModelOptArgParser(QUANTIZE_ARG_TYPES)


def get_docs_arg_parser():
    return ModelOptArgParser(
        _unique_arg_types(TRAINING_ARG_TYPES, QUANTIZE_ARG_TYPES),
        conflict_handler="resolve",
        docs_header_extra=_build_usage_header(),
    )


def get_training_args(args=None):
    return get_training_arg_parser().parse_args_into_dataclasses(args=args)


def get_quantize_args(args=None):
    return get_quantize_arg_parser().parse_args_into_dataclasses(args=args)


if __name__ == "__main__":
    get_docs_arg_parser().parse_args_into_dataclasses()
