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
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts
from modelopt.torch.utils import get_dataset_dataloader

DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_tokenizer(ckpt_path: str, model_max_length: int, trust_remote_code: bool = False):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )

    return tokenizer


def get_model(ckpt_path, dtype="fp16", trust_remote_code=False):
    print(f"Initializing model from {ckpt_path} using dtype {dtype}.")
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")
    model_kwargs = {"dtype": dtype}

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, device_map="auto", **model_kwargs, trust_remote_code=trust_remote_code
    )
    model.eval()

    return model


def main(args):
    if not torch.cuda.is_available():
        raise OSError("GPU is required for inference.")

    random.seed(1234)
    np.random.seed(1234)

    model = get_model(args.model_name_or_path, args.dtype, trust_remote_code=args.trust_remote_code)
    tokenizer = get_tokenizer(
        args.model_name_or_path, args.model_max_length, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN},
            tokenizer=tokenizer,
            model=model,
        )

    # Get calibration dataloader
    calib_dataloader = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_sample_length=512,
        device=args.device,
    )

    # Sparsify the model
    print("Starting sparsifying...")
    total_time = -time.time()
    model = mts.sparsify(
        model,
        args.sparsity_fmt,
        config={"data_loader": calib_dataloader, "collect_func": lambda x: x},
    )
    total_time += time.time()
    print(f"Sparsification done. Total time used: {total_time:.2f}s")

    # Save the sparsity modelopt state
    os.makedirs(args.output_dir, exist_ok=True)
    modelopt_state_path = os.path.join(args.output_dir, "pts_modelopt_state.pth")
    print(f"Saving sparsity modelopt state to {modelopt_state_path}")
    mto.save(model, modelopt_state_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path", help="Specify where the PyTorch checkpoint path is", required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dataset",
        default="cnn_dailymail",
        help="Calibration dataset: a ModelOpt-registered name (see get_supported_datasets()), "
        "a HuggingFace dataset id, or a local .jsonl path.",
    )
    parser.add_argument("--dtype", help="Model data type.", default="fp16")
    parser.add_argument(
        "--model_max_length",
        default=2048,
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).",
    )
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument(
        "--sparsity_fmt", type=str, default="sparsegpt", choices=["sparsegpt", "sparse_magnitude"]
    )
    parser.add_argument(
        "--calib_size", help="Number of samples for calibration.", type=int, default=512
    )
    parser.add_argument("--output_dir", default="output_dir")
    parser.add_argument(
        "--inference_tensor_parallel",
        help="Number of tensor parallel groups for inference.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--inference_pipeline_parallel",
        help="Number of pipeline parallel groups for inference.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
