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

"""Qwen3-Omni-30B-A3B text inference with vLLM.

Usage:
    python qwen3_omni_vllm.py
    python qwen3_omni_vllm.py --model /path/to/model --tp 4
"""

from __future__ import annotations

import argparse
import os
import shutil

# import vllm.model_executor.parameter as vllm_param
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


# # Debug patch to identify which weights cause shape mismatch
# def _patch_weight_loader_for_debug():
#     """Monkey-patch vLLM weight loader to print debug info on shape mismatch."""
#     original_load_column_parallel = vllm_param.ModelWeightParameter.load_column_parallel_weight

#     def debug_load_column_parallel(self, loaded_weight):
#         print(f"Loading param: {getattr(self, 'name', getattr(self, '_name', repr(self)))}")
#         print(f"  Parameter shape (expected): {self.data.shape}")
#         print(f"  Loaded weight shape (got):  {loaded_weight.shape}")

#         return original_load_column_parallel(self, loaded_weight)

#     vllm_param.ModelWeightParameter.load_column_parallel_weight = debug_load_column_parallel
#     print("DEBUG: Patched vLLM weight loader to print shape mismatch info")


# _patch_weight_loader_for_debug()

# Files needed for tokenizer/processor that vLLM loads from model path
TOKENIZER_FILES = [
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "chat_template.json",
]


def ensure_tokenizer_files(model_path: str, source_model_id: str) -> None:
    """Copy tokenizer files from HF model to local quantized model dir if missing."""
    if not os.path.isdir(model_path):
        return  # Not a local path, nothing to do

    # Check if tokenizer files are missing
    missing_files = [f for f in TOKENIZER_FILES if not os.path.exists(os.path.join(model_path, f))]
    if not missing_files:
        return

    print(f"Copying missing tokenizer files from {source_model_id}...")
    # Download only tokenizer files from HF
    cache_dir = snapshot_download(
        source_model_id,
        allow_patterns=TOKENIZER_FILES,
    )

    for fname in TOKENIZER_FILES:
        src = os.path.join(cache_dir, fname)
        dst = os.path.join(model_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni text inference with vLLM")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID or path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model length")

    args = parser.parse_args()

    # Load processor for chat template
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)

    # Text-only conversations
    conversations = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What are the key features of Qwen3-Omni?"}],
            }
        ],
    ]

    # Apply chat template with thinking disabled
    texts = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    # Process multimodal info (returns empty for text-only)
    # audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

    # Ensure tokenizer files exist in local model dir (vLLM loads processor from model path)
    ensure_tokenizer_files(args.model, MODEL_ID)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tokenizer=MODEL_ID,  # Always use original tokenizer from HF
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        quantization="modelopt_fp4",
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    print("Running inference...")
    outputs = llm.generate(texts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print("-" * 80)
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
