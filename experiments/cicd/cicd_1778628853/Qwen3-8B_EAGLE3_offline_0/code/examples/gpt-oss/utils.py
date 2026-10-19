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

import os

from datasets import load_dataset, load_from_disk
from transformers import AutoConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    # SFTTrainer, Use ModelOpt's version instead
    get_peft_config,
)


def is_distributed_job():
    return (os.environ.get("WORLD_SIZE", None) or os.environ.get("RANK", None)) is not None


def get_original_huggingface_quant_method(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "quantization_config") and config.quantization_config is not None:
        return config.quantization_config.get("quant_method")
    return None


def load_dataset_from_hub_or_local(script_args: ScriptArguments, training_args: SFTConfig):
    try:
        dataset = load_from_disk(script_args.dataset_name)
    except Exception:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # If the dataset does not have a test split, split the train split into train and test
    if training_args.eval_strategy != "no" and script_args.dataset_test_split not in dataset:
        dataset = dataset[script_args.dataset_train_split].train_test_split(test_size=0.1, seed=42)

    return dataset


def get_peft_config_for_moe(model, model_args: ModelConfig):
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        return None
    # Finetuning one layer from every 3 sections as shown in OpenAI's finetuning notebook
    # You may modify this behavior as you wish
    ft_every = min(len(model.model.layers) // 3, 1)
    peft_config.target_parameters = [
        f"{(i + 1) * ft_every - 1}.mlp.experts.gate_up_proj" for i in range(3)
    ] + [f"{(i + 1) * ft_every - 1}.mlp.experts.down_proj" for i in range(3)]
    return peft_config
