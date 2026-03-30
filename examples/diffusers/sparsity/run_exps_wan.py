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

"""Wan 2.2 skip-softmax experiments: baseline + sparsity sweep.

Supports both 5B (1 transformer, 30 blocks) and 14B (2 transformers, 40 blocks each).

Usage::
    # 5B baseline
    python run_exps_wan.py --model-id Wan-AI/Wan2.2-TI2V-5B-Diffusers --experiment baseline

    # 5B sparse 50%
    python run_exps_wan.py --model-id Wan-AI/Wan2.2-TI2V-5B-Diffusers --experiment sparse_50

    # 14B sparse 50%
    python run_exps_wan.py --model-id /path/to/Wan2.2-T2V-A14B-Diffusers --experiment sparse_50
"""

import argparse
import os
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


DEFAULT_PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.2 skip-softmax experiment runner")
    parser.add_argument("--model-id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiment_outputs/skip_softmax/wan")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "baseline",
            "sparse_25", "sparse_50", "sparse_75",
            "sparse_llm_25", "sparse_llm_50", "sparse_llm_75",
        ],
    )
    parser.add_argument("--skip-first-last", type=int, default=3,
                        help="Number of first/last self-attn layers to exclude from sparsity")
    parser.add_argument("--calib-steps", type=int, default=35)
    parser.add_argument("--calib-frames", type=int, default=81,
                        help="Frame count for calibration (same resolution as inference)")
    return parser.parse_args()


def build_pipeline(model_id):
    print(f"Loading Wan pipeline from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def generate_video(pipe, args, output_path):
    torch.cuda.synchronize()
    start = time.time()

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]

    torch.cuda.synchronize()
    elapsed = time.time() - start

    export_to_video(output, output_path, fps=16)
    return elapsed


def build_calibration_forward_loop(pipe, args):
    """Build forward loop for calibration using same resolution/settings as inference."""
    def forward_loop(model):
        print(f"  Calibration: {args.calib_frames} frames, {args.calib_steps} steps...")
        pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.calib_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.calib_steps,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )
    return forward_loop


def _build_sparse_cfg(num_blocks, skip_first_last):
    """Build sparse config for a Wan transformer, excluding first/last N self-attn layers."""
    sparse_cfg = {
        "*.attn1": {
            "method": "triton_skip_softmax_diffusion",
            "br": 128,
            "bc": 128,
            "backend": "triton",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
        },
        "*.attn2": {"enable": False},
        "default": {"enable": False},
    }

    for i in range(skip_first_last):
        sparse_cfg[f"*blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*blocks.{num_blocks - 1 - i}.attn*"] = {"enable": False}

    return sparse_cfg


def run_baseline(pipe, args, output_dir):
    print("=" * 60)
    print("EXPERIMENT: Baseline (no sparsity)")
    print("=" * 60)
    output_path = os.path.join(output_dir, "baseline.mp4")
    elapsed = generate_video(pipe, args, output_path)
    print(f"Baseline: {elapsed:.1f}s -> {output_path}")
    return elapsed


def run_sparse(pipe, args, output_dir, target_sparsity):
    import modelopt.torch.sparsity.attention_sparsity as mtsa

    pct = int(target_sparsity * 100)
    print("=" * 60)
    print(f"EXPERIMENT: {pct}% sparsity (Triton + skip-softmax)")
    print("=" * 60)

    # Wrap transformer(s) in a ModuleList so we can sparsify in one call
    transformers = [pipe.transformer]
    if pipe.transformer_2 is not None:
        transformers.append(pipe.transformer_2)

    wrapper = torch.nn.ModuleList(transformers)
    num_blocks = len(transformers[0].blocks)
    num_transformers = len(transformers)
    print(f"Sparsifying {num_transformers} transformer(s), {num_blocks} blocks each, "
          f"skip first/last {args.skip_first_last}")

    sparse_cfg = _build_sparse_cfg(num_blocks, args.skip_first_last)
    sparse_cfg["calibration"] = {"target_sparse_ratio": {"prefill": target_sparsity}}

    forward_loop = build_calibration_forward_loop(pipe, args)
    mtsa.sparsify(wrapper, {"sparse_cfg": sparse_cfg}, forward_loop=forward_loop)

    output_path = os.path.join(output_dir, f"sparse_{pct}pct.mp4")
    elapsed = generate_video(pipe, args, output_path)
    print(f"{pct}% sparsity: {elapsed:.1f}s -> {output_path}")
    return elapsed


def _build_sparse_cfg_llm(num_blocks, skip_first_last):
    """Build sparse config using the LLM flash_skip_softmax method."""
    sparse_cfg = {
        "*.attn1": {
            "method": "flash_skip_softmax",
            "br": 128,
            "bc": 128,
            "backend": "pytorch",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
            "thresholds": {"prefill": [1e-3]},
        },
        "*.attn2": {"enable": False},
        "default": {"enable": False},
    }

    for i in range(skip_first_last):
        sparse_cfg[f"*blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*blocks.{num_blocks - 1 - i}.attn*"] = {"enable": False}

    return sparse_cfg


def run_sparse_llm(pipe, args, output_dir, target_sparsity):
    """Sparse using LLM flash_skip_softmax method (F.softmax patching)."""
    import modelopt.torch.sparsity.attention_sparsity as mtsa

    pct = int(target_sparsity * 100)
    print("=" * 60)
    print(f"EXPERIMENT: {pct}% sparsity (LLM flash_skip_softmax)")
    print("=" * 60)

    transformers = [pipe.transformer]
    if pipe.transformer_2 is not None:
        transformers.append(pipe.transformer_2)

    wrapper = torch.nn.ModuleList(transformers)
    num_blocks = len(transformers[0].blocks)
    num_transformers = len(transformers)
    print(f"Sparsifying {num_transformers} transformer(s), {num_blocks} blocks each, "
          f"skip first/last {args.skip_first_last}")

    sparse_cfg = _build_sparse_cfg_llm(num_blocks, args.skip_first_last)
    sparse_cfg["calibration"] = {"target_sparse_ratio": {"prefill": target_sparsity}}

    forward_loop = build_calibration_forward_loop(pipe, args)
    mtsa.sparsify(wrapper, {"sparse_cfg": sparse_cfg}, forward_loop=forward_loop)

    output_path = os.path.join(output_dir, f"sparse_llm_{pct}pct.mp4")
    elapsed = generate_video(pipe, args, output_path)
    print(f"{pct}% sparsity (LLM): {elapsed:.1f}s -> {output_path}")
    return elapsed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipe = build_pipeline(args.model_id)

    experiment = args.experiment
    if experiment == "baseline":
        run_baseline(pipe, args, args.output_dir)
    elif experiment.startswith("sparse_llm_"):
        pct = int(experiment.split("_")[2])
        run_sparse_llm(pipe, args, args.output_dir, pct / 100.0)
    elif experiment.startswith("sparse_"):
        pct = int(experiment.split("_")[1])
        run_sparse(pipe, args, args.output_dir, pct / 100.0)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print("\nDone!")


if __name__ == "__main__":
    main()
