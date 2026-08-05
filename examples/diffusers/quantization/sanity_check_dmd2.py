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

"""Restore a quantized DMD2 Qwen-Image student and run one few-step inference.

Confirms the round trip of the new ``qwen-image-dmd2`` quantization flow:

  1. Load the base Qwen-Image pipeline with the consolidated student swapped in
     (via the same :class:`PipelineManager` path quantize.py uses) -- this brings
     the original (unquantized) weights.
  2. Reapply the weight-free quantization checkpoint saved by ``quantize.py``
     (``save_quantizer_state`` -> ``transformer.pt``) via
     ``restore_quantizer_state``, which re-applies the quantizer recipe **and the
     calibrated amax** buffers on top of the loaded weights.
  3. Run a single few-step DMD inference (with VAE decode) and assert the image
     is finite and non-constant.

This deliberately reuses :class:`PipelineManager` and
:func:`qwen_image_dmd2_sampler.dmd2_sample` so the inference path is identical to
calibration's (minus the VAE decode, which is enabled here).

Usage::

    python sanity_check_dmd2.py \\
        --quantized-ckpt   ./qwen_dmd2_fp8/transformer.pt \\
        --student-path     /.../epoch_4_step_17999/model/consolidated \\
        --base-pipeline-path /.../models/Qwen-Image \\
        --output-png       ./qwen_dmd2_fp8/sanity.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from models_utils import ModelType
from pipeline_manager import PipelineManager
from quantize_config import ModelConfig
from qwen_image_dmd2_sampler import dmd2_sample
from utils import restore_quantizer_state

import modelopt.torch.quantization as mtq

logger = logging.getLogger("sanity_check_dmd2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantized-ckpt",
        required=True,
        help="Path to the quantized checkpoint saved by quantize.py (e.g. .../transformer.pt).",
    )
    parser.add_argument(
        "--student-path",
        required=True,
        help="Consolidated DMD2 student dir (provides architecture + base weights to restore into).",
    )
    parser.add_argument(
        "--base-pipeline-path",
        default="Qwen/Qwen-Image",
        help="Base Qwen-Image dir/HF id for the VAE / text-encoder / tokenizer / scheduler.",
    )
    parser.add_argument("--ema-path", default=None, help="Optional EMA shadow overlaid on load.")
    parser.add_argument("--output-png", default="./qwen_dmd2_sanity.png")
    parser.add_argument("--prompt", default="a small red cube on a white table")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    # Few-step sampler knobs (defaults match the canonical 4-step shift=3 student).
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument(
        "--t-list",
        default=None,
        help="Comma-separated schedule incl. trailing 0.0, e.g. '1.0,0.9,0.75,0.5,0.0'.",
    )
    parser.add_argument("--sample-type", default="ode", choices=["ode", "sde"])
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # 1. Build the base pipeline with the student swapped in (unquantized).
    extra_params: dict[str, object] = {
        "student_path": args.student_path,
        "base_pipeline_path": args.base_pipeline_path,
        "sample_steps": args.sample_steps,
        "sample_type": args.sample_type,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
    }
    if args.ema_path:
        extra_params["ema_path"] = args.ema_path
    if args.t_list:
        extra_params["t_list"] = args.t_list

    model_config = ModelConfig(
        model_type=ModelType.QWEN_IMAGE_DMD2,
        model_dtype={"default": torch.bfloat16},
        backbone=["transformer"],
        extra_params=extra_params,
    )
    pm = PipelineManager(model_config, logger)
    pipe = pm.create_pipeline()

    # 2. Restore the quantized architecture + calibrated amax into the student.
    logger.info(
        "Restoring quantizer state (amax + recipe) from %s onto the loaded student",
        args.quantized_ckpt,
    )
    restore_quantizer_state(pipe.transformer, args.quantized_ckpt)
    mtq.print_quant_summary(pipe.transformer)
    pm.setup_device()

    # 3. One few-step inference (with VAE decode).
    gen = torch.Generator(device=pipe.transformer.device).manual_seed(args.seed)
    images = dmd2_sample(pipe, [args.prompt], decode=True, generator=gen, **pm.dmd_sampler_cfg)
    image = images[0]

    import numpy as np

    arr = np.asarray(image)
    stats = {
        "prompt": args.prompt,
        "quantized_ckpt": args.quantized_ckpt,
        "schedule": pm.dmd_sampler_cfg["schedule"],
        "image_shape": list(arr.shape),
        "image_dtype": str(arr.dtype),
        "image_min": float(arr.min()),
        "image_max": float(arr.max()),
        "image_mean": float(arr.mean()),
        "image_std": float(arr.std()),
        "is_finite": bool(np.isfinite(arr).all()),
        "is_not_constant": bool(arr.std() > 0),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_png)), exist_ok=True)
    image.save(args.output_png)
    with open(args.output_png.replace(".png", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    if not stats["is_finite"]:
        logger.error("Sanity check FAILED: image contains non-finite values.")
        sys.exit(1)
    if not stats["is_not_constant"]:
        logger.error("Sanity check FAILED: image is constant (std == 0).")
        sys.exit(1)
    logger.info(
        "Sanity check PASSED: restored quantized student produced a finite image -> %s",
        args.output_png,
    )


if __name__ == "__main__":
    main()
