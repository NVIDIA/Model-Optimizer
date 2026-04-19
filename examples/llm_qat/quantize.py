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

"""Standalone quantization script for LLMs using ModelOpt recipes.

This script applies post-training quantization (PTQ) to a model and saves the
quantized checkpoint. The quantized model can then be used for QAT/QAD training
with train.py or exported with export.py.

Usage:
    python quantize.py \
        --model_name_or_path meta-llama/Meta-Llama-3-8B \
        --dataset_config configs/dataset/blend.yaml \
        --recipe general/ptq/nvfp4_default-fp8_kv \
        --quantize_output_dir llama3-quantized
"""

import os

import torch
import transformers
from arguments import DataArguments, ModelArguments, QuantizeArguments
from utils import make_supervised_data_module

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from modelopt.torch.opt.plugins.transformers import ModelOptArgParser
from modelopt.torch.utils import print_rank_0

# Enable automatic save/load of modelopt state with huggingface checkpointing
mto.enable_huggingface_checkpointing()


def _build_calib_dataloader(tokenizer, data_args, quant_args):
    """Build a calibration dataloader from the train dataset."""
    print_rank_0("Loading calibration dataset...")
    data_module = make_supervised_data_module(data_args, tokenizer)
    train_dataset = data_module["train_dataset"]
    num_samples = min(quant_args.calib_size, len(train_dataset))
    calib_dataset = torch.utils.data.Subset(train_dataset, list(range(num_samples)))
    return torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=quant_args.calib_batch_size,
        collate_fn=data_module["data_collator"],
    )


def quantize():
    parser = ModelOptArgParser((ModelArguments, DataArguments, QuantizeArguments))
    model_args, data_args, quant_args = parser.parse_args_into_dataclasses()

    # Load recipe
    print_rank_0(f"Loading recipe: {quant_args.recipe}")
    recipe = load_recipe(quant_args.recipe)
    if not isinstance(recipe, ModelOptPTQRecipe):
        raise ValueError(
            f"Expected PTQ recipe, but got {type(recipe).__name__} from {quant_args.recipe}"
        )
    ptq_cfg = recipe.quantize

    # Load model and tokenizer
    print_rank_0(f"Loading model: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=model_args.model_max_length
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    calib_dataloader = _build_calib_dataloader(tokenizer, data_args, quant_args)

    def forward_loop(model):
        for batch in calib_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)

    # Quantize
    print_rank_0("Quantizing the model...")
    mtq.quantize(model, ptq_cfg, forward_loop)
    mtq.print_quant_summary(model)

    if quant_args.compress:
        print_rank_0("Compressing model weights for QLoRA...")
        mtq.compress(model)

    # Save quantized checkpoint
    os.makedirs(quant_args.quantize_output_dir, exist_ok=True)
    print_rank_0(f"Saving quantized model to {quant_args.quantize_output_dir}")
    model.save_pretrained(quant_args.quantize_output_dir)
    tokenizer.save_pretrained(quant_args.quantize_output_dir)


if __name__ == "__main__":
    quantize()
