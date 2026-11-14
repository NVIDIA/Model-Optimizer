# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Wan 2.2 inference with Sol-Attn sparse attention.

Sol-Attn (https://arxiv.org/abs/2607.24027) is training-free: it routes high-scoring KV
blocks to exact attention during online softmax and approximates the rest, so no
calibration is needed. Only self-attention (attn1) is sparsified; the first/last
``--skip-first-last`` blocks stay dense.

Requires the ``sol_attn`` kernel package (see README.md).

Usage::

    python wan22_sol_attn.py --tau 1.0 --prompt "A cat playing piano" --output out.mp4
"""

import argparse

import torch
from diffusers.utils import export_to_video
from wan22_skip_softmax import (
    DEFAULT_MODEL_PATH,
    DEFAULT_NEGATIVE_PROMPT,
    _get_num_blocks,
    build_pipeline,
)

import modelopt.torch.sparsity.attention_sparsity as mtsa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan 2.2 video generation with Sol-Attn")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames (4k+1)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0, help="14B only")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Dense inference")
    parser.add_argument(
        "--tau", type=float, default=1.0, help="Higher values select fewer exact KV blocks"
    )
    parser.add_argument("--thresh-type", choices=["diag", "exact"], default="diag")
    parser.add_argument("--skip-first-last", type=int, default=2, help="Dense first/last blocks")
    return parser.parse_args()


def build_sparse_config(args: argparse.Namespace, num_blocks: int) -> dict:
    """Sparsify self-attention with Sol-Attn and keep the first/last blocks dense."""
    sparse_cfg: dict = {
        "*.attn1*": {
            "method": "sol_attn",
            "tau": args.tau,
            "thresh_type": args.thresh_type,
        },
        "default": {"enable": False},
    }
    for i in range(args.skip_first_last):
        sparse_cfg[f"*blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*blocks.{num_blocks - 1 - i}.attn*"] = {"enable": False}
    return {"sparse_cfg": sparse_cfg}


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args.model_path)
    transformers = [t for t in (pipe.transformer, getattr(pipe, "transformer_2", None)) if t]

    if not args.baseline:
        for transformer in transformers:
            config = build_sparse_config(args, _get_num_blocks(transformer))
            mtsa.sparsify(transformer, config)

    pipe_kwargs: dict = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
    }
    if len(transformers) > 1:
        pipe_kwargs["guidance_scale_2"] = args.guidance_scale_2
    output = pipe(**pipe_kwargs)
    export_to_video(output.frames[0], args.output, fps=16)
    print(f"Saved to {args.output}")

    if not args.baseline:
        for transformer in transformers:
            mtsa.print_sparse_attention_summary(transformer)


if __name__ == "__main__":
    main()
