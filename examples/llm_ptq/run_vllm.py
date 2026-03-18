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

"""Unified HF checkpoint inference with vLLM.

Usage:
    python run_vllm.py --model /path/to/quantized/model
    python run_vllm.py --model /path/to/model --tp 4
"""

from __future__ import annotations

import argparse

from example_utils import (
    ensure_tokenizer_files,
    get_model_type_from_config,
    get_quantization_format,
    get_sampling_params_from_config,
)
from transformers import AutoConfig, AutoProcessor
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Run unified hf checkpoint inference with vLLM")
    parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max model length (auto-detected from config if not specified)",
    )
    parser.add_argument("--prompt", type=str, default="What in Nvidia?", help="Text prompt")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Tokenizer ID or path (defaults to model path)"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling (-1 to disable)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code from HuggingFace model repos",
    )

    args = parser.parse_args()

    # Detect model type from config
    model_type = get_model_type_from_config(args.model)
    print(f"Detected model type: {model_type}")

    # Detect quantization format
    quantization = get_quantization_format(args.model)
    print(f"Detected quantization: {quantization}")

    # Get max_model_len from config if not specified
    if args.max_model_len is None:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        args.max_model_len = getattr(config, "max_position_embeddings", 4096)
        print(f"Using max_model_len from config: {args.max_model_len}")

    # Determine tokenizer source
    tokenizer_id = args.tokenizer or args.model

    # Load processor for chat template
    processor = AutoProcessor.from_pretrained(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )

    # Text-only conversations
    conversations = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            }
        ],
    ]

    # Apply chat template
    apply_chat_kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    # Qwen3Omni-specific: disable thinking mode
    if model_type == "qwen3omni":
        apply_chat_kwargs["enable_thinking"] = False

    texts = processor.apply_chat_template(conversations, **apply_chat_kwargs)

    # Ensure tokenizer files exist in local model dir (vLLM loads processor from model path)
    if args.tokenizer:
        ensure_tokenizer_files(args.model, args.tokenizer)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_id,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        quantization=quantization,
    )

    # Get sampling params from config, with CLI/defaults as fallback
    config_params = get_sampling_params_from_config(args.model)
    sampling_kwargs = {
        "temperature": config_params.get("temperature", args.temperature),
        "top_p": config_params.get("top_p", args.top_p),
        "max_tokens": config_params.get("max_tokens", args.max_tokens),
    }
    top_k = config_params.get("top_k", args.top_k)
    if top_k > 0:
        sampling_kwargs["top_k"] = top_k
    print(f"Sampling params: {sampling_kwargs}")
    sampling_params = SamplingParams(**sampling_kwargs)

    print("Running inference...")
    outputs = llm.generate(texts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print("-" * 80)
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
