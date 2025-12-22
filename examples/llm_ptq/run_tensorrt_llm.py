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

"""An example script to run the tensorrt_llm inference."""

import argparse

import torch
from example_utils import get_tokenizer
from transformers import PreTrainedTokenizerBase

from modelopt.deploy.llm import LLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--max_output_len", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument(
        "--input_texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )

    return parser.parse_args()


def run(args):
    if not args.tokenizer:
        # Assume the tokenizer files are saved in the checkpoint_dir.
        args.tokenizer = args.checkpoint_dir

    if isinstance(args.tokenizer, PreTrainedTokenizerBase):
        tokenizer = args.tokenizer
    else:
        tokenizer = get_tokenizer(
            ckpt_path=args.tokenizer, trust_remote_code=args.trust_remote_code
        )

    input_texts = args.input_texts.split("|")
    assert input_texts, "input_text not specified"

    free_memory_before = torch.cuda.mem_get_info()

    print("TensorRT-LLM example outputs:")

    llm = LLM(args.checkpoint_dir, tokenizer=tokenizer, max_batch_size=len(input_texts))
    torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.generate_text(input_texts, args.max_output_len)
    torch.cuda.cudart().cudaProfilerStop()

    free_memory_after = torch.cuda.mem_get_info()
    print(
        f"Use GPU memory: {(free_memory_before[0] - free_memory_after[0]) / 1024 / 1024 / 1024} GB"
    )
    print(f"Generated outputs: {outputs}")

    outputs = llm.generate_tokens(input_texts, args.max_output_len)
    print(f"Generated tokens: {outputs}")

    logits = llm.generate_context_logits(input_texts)
    print(f"Generated logits: {logits}")


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
