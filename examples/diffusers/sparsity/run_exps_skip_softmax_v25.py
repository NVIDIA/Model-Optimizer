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

"""LTX-2 skip-softmax experiments: baselines + sparsity sweep.

Runs 6 experiments on separate GPUs:
  GPU 0: Baseline (original attention, no Triton)
  GPU 1: Triton baseline (Triton FA kernel, no sparsity)
  GPU 2: 5% sparsity
  GPU 3: 25% sparsity
  GPU 4: 50% sparsity
  GPU 5: 75% sparsity

Usage::
    python run_experiments.py --prompt "A cat playing piano"
"""

import argparse
import functools
import os
import sys
import time

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_2_STAGE_HEIGHT,
    DEFAULT_2_STAGE_WIDTH,
    DEFAULT_AUDIO_GUIDER_PARAMS,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_VIDEO_GUIDER_PARAMS,
)
from ltx_pipelines.utils.media_io import encode_video

# ---- Model paths ----
CHECKPOINT_PATH = "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-dev.safetensors"
DISTILLED_LORA_PATH = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-distilled-lora-384.safetensors"
)
SPATIAL_UPSAMPLER_PATH = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
)
GEMMA_ROOT = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/gemma-3-12b-it-qat-q4_0-unquantized"
)

DEFAULT_PROMPT = "A cat playing piano"
DEFAULT_NUM_FRAMES = 121
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1280


def parse_args():
    parser = argparse.ArgumentParser(description="LTX-2 skip-softmax experiment runner")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument(
        "--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of frames"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="experiment_outputs", help="Output directory"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "baseline",
            "triton_baseline",
            "sparse_5",
            "sparse_25",
            "sparse_50",
            "sparse_75",
            "sparse_v25_25",
            "sparse_v25_50",
            "sparse_v25_75",
            "sparse_v252_25",
            "sparse_v252_50",
            "sparse_v252_75",
            "sparse_v252dbg_25",
            "sparse_v252dbg_50",
            "sparse_v252dbg_75",
            "sparse_v252dbg2_25",
            "sparse_v252dbg2_50",
            "sparse_v252dbg2_75",
            "sparse_v252dbg4_25",
            "sparse_v252dbg4_50",
            "sparse_v252dbg4_75",
            "sparse_v252dbg5_25",
            "sparse_v252dbg5_50",
            "sparse_v252dbg5_75",
            "sparse_v252dbg6_25",
            "sparse_v252dbg6_50",
            "sparse_v252dbg6_75",
            "sparse_v252dbg7_25",
            "sparse_v252dbg7_50",
            "sparse_v252dbg7_75",
            "sparse_v252dbg8_25",
            "sparse_v252dbg8_50",
            "sparse_v252dbg8_75",
            "sparse_v252dbg3_25",
            "sparse_v252dbg3_50",
            "sparse_v252dbg3_75",
            "sparse_v252dbg9_75",
            "sparse_v252dbg10_75",
            "sparse_v252dbg11_75",
            "sparse_v252dbg12_75",
            "sparse_v252dbg13_75",
        ],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--skip-first-last",
        type=int,
        default=2,
        help="Number of first/last layers to exclude from sparsity",
    )
    parser.add_argument(
        "--calib-steps", type=int, default=35, help="Calibration denoising steps"
    )
    parser.add_argument(
        "--calib-frames", type=int, default=81, help="Calibration frame count"
    )
    return parser.parse_args()


def _patch_vae_requires_grad(pipeline):
    for ledger_attr in ("stage_1_model_ledger", "stage_2_model_ledger"):
        ledger = getattr(pipeline, ledger_attr, None)
        if ledger is None:
            continue
        for loader_name in ("video_decoder", "audio_decoder"):
            orig_loader = getattr(ledger, loader_name, None)
            if orig_loader is None:
                continue

            def _make_patched(fn):
                @functools.wraps(fn)
                def patched():
                    model = fn()
                    model.requires_grad_(False)
                    return model

                return patched

            setattr(ledger, loader_name, _make_patched(orig_loader))


def build_pipeline():
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora=[
            LoraPathStrengthAndSDOps(DISTILLED_LORA_PATH, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)
        ],
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=[],
    )
    _patch_vae_requires_grad(pipeline)
    return pipeline


def generate_video(pipeline, args, output_path):
    """Run video generation and return wall-clock time."""
    tiling_config = TilingConfig.default()

    # Warmup
    torch.cuda.synchronize()
    start = time.time()

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=args.seed,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_frames=args.num_frames,
        frame_rate=DEFAULT_FRAME_RATE,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
        audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
        images=[],
        tiling_config=tiling_config,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    encode_video(
        video=video,
        fps=DEFAULT_FRAME_RATE,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=get_video_chunks_number(args.num_frames, tiling_config),
    )

    return elapsed


def load_calib_prompts(calib_size=1):
    from datasets import load_dataset

    dataset = load_dataset("nkp37/OpenVid-1M")
    return list(dataset["train"]["caption"][:calib_size])


def build_calibration_forward_loop(pipeline, num_steps=10, num_frames=81):
    """Build calibration forward loop.

    Uses DEFAULT_2_STAGE resolution (1024x1536) for calibration to get
    larger seq_k and better gap statistics. The gap/log(seq_k) normalization
    ensures the calibrated threshold transfers to inference resolution.
    """
    calib_prompts = load_calib_prompts(1)
    tiling_config = TilingConfig.default()

    def forward_loop(model):
        for i, prompt in enumerate(calib_prompts):
            print(f"  Calibration [{i + 1}/{len(calib_prompts)}]: {prompt[:60]}...")
            pipeline(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=DEFAULT_SEED,
                height=DEFAULT_2_STAGE_HEIGHT,
                width=DEFAULT_2_STAGE_WIDTH,
                num_frames=num_frames,
                frame_rate=DEFAULT_FRAME_RATE,
                num_inference_steps=num_steps,
                video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
                audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
                images=[],
                tiling_config=tiling_config,
            )

    return forward_loop


def run_baseline(pipeline, args, output_dir):
    """Baseline: original attention, no Triton, no sparsity."""
    print("=" * 60)
    print("EXPERIMENT: Baseline (original attention)")
    print("=" * 60)
    output_path = os.path.join(output_dir, "baseline.mp4")
    elapsed = generate_video(pipeline, args, output_path)
    print(f"Baseline: {elapsed:.1f}s → {output_path}")
    return elapsed


def run_triton_baseline(pipeline, args, output_dir):
    """Triton baseline: Triton FA kernel, no sparsity (threshold=0)."""
    import modelopt.torch.sparsity.attention_sparsity as mtsa

    print("=" * 60)
    print("EXPERIMENT: Triton FA baseline (no sparsity)")
    print("=" * 60)

    transformer = pipeline.stage_1_model_ledger.transformer()
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    # Use triton_skip_softmax_diffusion with no calibration → threshold stays None → no skipping
    # But the Triton kernel is still used for attention computation
    sparse_cfg = {
        "*.attn1": {
            "method": "triton_skip_softmax_diffusion",
            "br": 128,
            "bc": 128,
            "backend": "triton",
            "is_causal": False,
            "collect_stats": False,
            "enable": True,
        },
        "*.attn2": {"enable": False},
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg})

    output_path = os.path.join(output_dir, "triton_baseline.mp4")
    elapsed = generate_video(pipeline, args, output_path)
    print(f"Triton baseline: {elapsed:.1f}s → {output_path}")
    return elapsed


def run_sparse(pipeline, args, output_dir, target_sparsity):
    """Sparse: Triton FA kernel + skip-softmax at given sparsity."""
    import modelopt.torch.sparsity.attention_sparsity as mtsa

    pct = int(target_sparsity * 100)
    print("=" * 60)
    print(f"EXPERIMENT: {pct}% sparsity (Triton + skip-softmax)")
    print("=" * 60)

    transformer = pipeline.stage_1_model_ledger.transformer()
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    sparse_cfg = {
        "calibration": {"target_sparse_ratio": {"prefill": target_sparsity}},
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
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    # Exclude first/last layers
    for i in range(args.skip_first_last):
        sparse_cfg[f"*transformer_blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*transformer_blocks.{47 - i}.attn*"] = {"enable": False}

    forward_loop = build_calibration_forward_loop(
        pipeline, num_steps=args.calib_steps, num_frames=args.calib_frames
    )

    mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg}, forward_loop=forward_loop)

    output_path = os.path.join(output_dir, f"sparse_{pct}pct.mp4")
    elapsed = generate_video(pipeline, args, output_path)
    print(f"{pct}% sparsity: {elapsed:.1f}s → {output_path}")
    return elapsed


# Old per-row percentile thresholds (before per-tile min-gap fix).
# Cleared to force fresh calibration with the corrected per-tile calibration.
# FIXED_THRESHOLDS = {
#     75: 0.054199,  # p25 from old per-row calibration
#     50: 0.234375,  # p50 from old per-row calibration
#     25: 0.535156,  # p75 from old per-row calibration
# }
FIXED_THRESHOLDS = {}


def run_sparse_v25(
    pipeline, args, output_dir, target_sparsity, version="v25", fixed_threshold=None
):
    """Sparse V2.5+: Triton FA kernel + skip-softmax + pool-K + precomputed v_mean."""
    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

    pct = int(target_sparsity * 100)
    label = version.upper().replace("V", "V.")
    print("=" * 60)
    print(f"EXPERIMENT: {pct}% sparsity {label} (Triton + pool-K + fresh v_mean)")
    print("=" * 60)

    transformer = pipeline.stage_1_model_ledger.transformer()
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    # Use fixed threshold if available, otherwise calibrate
    use_fixed = fixed_threshold is not None or pct in FIXED_THRESHOLDS
    threshold = fixed_threshold if fixed_threshold is not None else FIXED_THRESHOLDS.get(pct)

    sparse_cfg = {
        "*.attn1": {
            "method": "triton_skip_softmax_diffusion",
            "br": 128,
            "bc": 128,
            "backend": "triton",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
            "enable_v25": True,
        },
        "*.attn2": {"enable": False},
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    # Exclude first/last layers
    for i in range(args.skip_first_last):
        sparse_cfg[f"*transformer_blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*transformer_blocks.{47 - i}.attn*"] = {"enable": False}

    if use_fixed and threshold is not None:
        # Bypass calibration: sparsify without forward_loop, then set threshold manually
        print(f"Using fixed threshold={threshold} (skipping calibration)")
        mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg})
        for module in transformer.modules():
            if isinstance(module, SparseAttentionModule) and module.is_enabled:
                module._sparse_method_instance.skip_softmax_threshold = threshold
    else:
        sparse_cfg["calibration"] = {"target_sparse_ratio": {"prefill": target_sparsity}}
        forward_loop = build_calibration_forward_loop(
            pipeline, num_steps=args.calib_steps, num_frames=args.calib_frames
        )
        mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg}, forward_loop=forward_loop)

    output_path = os.path.join(output_dir, f"sparse_{version}_{pct}pct.mp4")
    elapsed = generate_video(pipeline, args, output_path)
    print(f"{pct}% sparsity {label}: {elapsed:.1f}s → {output_path}")
    return elapsed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building LTX-2 pipeline...")
    pipeline = build_pipeline()

    experiment = args.experiment

    if experiment == "baseline":
        run_baseline(pipeline, args, args.output_dir)
    elif experiment == "triton_baseline":
        run_triton_baseline(pipeline, args, args.output_dir)
    elif experiment == "sparse_v252dbg9_75":
        run_sparse_v25(
            pipeline, args, args.output_dir, 0.75, version="v252dbg9", fixed_threshold=0.054199
        )
    elif experiment == "sparse_v252dbg10_75":
        run_sparse_v25(
            pipeline, args, args.output_dir, 0.75, version="v252dbg10", fixed_threshold=0.054199
        )
    elif experiment == "sparse_v252dbg11_75":
        run_sparse_v25(
            pipeline, args, args.output_dir, 0.75, version="v252dbg11", fixed_threshold=0.054199
        )
    elif experiment == "sparse_v252dbg12_75":
        run_sparse_v25(
            pipeline, args, args.output_dir, 0.75, version="v252dbg12", fixed_threshold=0.054199
        )
    elif experiment == "sparse_v252dbg13_75":
        run_sparse_v25(
            pipeline, args, args.output_dir, 0.75, version="v252dbg13", fixed_threshold=0.054199
        )
    elif experiment.startswith("sparse_v252dbg8_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg8")
    elif experiment.startswith("sparse_v252dbg7_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg7")
    elif experiment.startswith("sparse_v252dbg6_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg6")
    elif experiment.startswith("sparse_v252dbg5_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg5")
    elif experiment.startswith("sparse_v252dbg4_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg4")
    elif experiment.startswith("sparse_v252dbg3_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg3")
    elif experiment.startswith("sparse_v252dbg2_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg2")
    elif experiment.startswith("sparse_v252dbg_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252dbg")
    elif experiment.startswith("sparse_v252_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0, version="v252")
    elif experiment.startswith("sparse_v25_"):
        pct = int(experiment.split("_")[2])
        run_sparse_v25(pipeline, args, args.output_dir, pct / 100.0)
    elif experiment.startswith("sparse_"):
        pct = int(experiment.split("_")[1])
        run_sparse(pipeline, args, args.output_dir, pct / 100.0)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print("\nDone!")


if __name__ == "__main__":
    main()
