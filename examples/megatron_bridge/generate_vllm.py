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
"""Sanity-check generation for a unified HF checkpoint, using vLLM.

vLLM auto-detects the ModelOpt quantization from the `hf_quant_config.json`, so no extra quant flags are needed.

Usage:
    python generate_vllm.py --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --trust_remote_code
"""

import argparse
import os

# Disable the FlashInfer MoE kernels (FP8/NVFP4), which JIT-compile and autotune on the first
# run and can hang or error. Falls back to a stable path -- ideal for a quick sanity check.
# Set these before importing vllm; `setdefault` lets a shell-provided value override.
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "0")

import vllm

DEFAULT_PROMPTS = [
    "Hello!",
    "Born in California, Soyer trained as a",
    "The capital of France is",
    "Q: What is 2 + 2?\nA:",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path to the exported unified HF checkpoint."
    )
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
    )
    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    for out in llm.generate(DEFAULT_PROMPTS, sampling):
        print(f"\nPrompt:    {out.prompt!r}")
        print(f"Generated: {out.outputs[0].text!r}")


if __name__ == "__main__":
    main()
