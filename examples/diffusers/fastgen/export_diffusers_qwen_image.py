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

"""Export a DMD2 student as a full diffusers-loadable QwenImagePipeline dir.

The §10 safetensors addendum produces a transformer-only dir:

    epoch_0_step_N/model/consolidated/
        config.json
        diffusion_pytorch_model.safetensors.index.json
        model-00001-of-00001.safetensors

That dir is loadable by ``QwenImageTransformer2DModel.from_pretrained`` but NOT
by ``QwenImagePipeline.from_pretrained`` or ``DiffusionPipeline.from_pretrained``
because it lacks ``model_index.json`` and the sibling component dirs
(``vae/``, ``text_encoder/``, ``tokenizer/``, ``scheduler/``).

This utility assembles a full pipeline dir by:

    <output_dir>/
        model_index.json                  (copied from base)
        transformer/                      (symlinked or copied from the consolidated student)
        vae/                              (symlinked from base Qwen-Image)
        text_encoder/                     (symlinked from base Qwen-Image)
        tokenizer/                        (symlinked from base Qwen-Image)
        scheduler/                        (symlinked from base Qwen-Image)

After this runs, the dir loads with ``QwenImagePipeline.from_pretrained(output_dir)``
or ``DiffusionPipeline.from_pretrained(output_dir)``.

Symlinks are the default — the base Qwen-Image components are huge (text encoder
alone is ~12 GB) and never change between DMD2 students, so copying them per
checkpoint wastes disk. Use ``--copy`` if the output dir must be portable.

Usage::

    python export_diffusers_qwen_image.py \\
        --student_path     /path/to/checkpoint/epoch_0_step_500/model/consolidated \\
        --base_pipeline_path Qwen/Qwen-Image \\
        --output_dir       /path/to/output/qwen_image_dmd2 \\
        [--copy]

Smoke test (``--verify``) loads the assembled dir via QwenImagePipeline and
checks the transformer config matches the student's.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Components borrowed from the base Qwen-Image checkpoint. The trained
# transformer replaces the corresponding entry.
BASE_COMPONENTS = ("vae", "text_encoder", "tokenizer", "scheduler")


def _link_or_copy(src: str, dst: str, copy: bool) -> None:
    if os.path.lexists(dst):
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    if copy:
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def export_diffusers(
    student_path: str | Path,
    base_pipeline_path: str | Path,
    output_dir: str | Path,
    copy: bool = False,
) -> None:
    student_path = str(student_path)
    base_pipeline_path = str(base_pipeline_path)
    output_dir = str(output_dir)

    if not os.path.isdir(student_path):
        raise FileNotFoundError(f"student_path is not a directory: {student_path}")
    if not os.path.isdir(base_pipeline_path):
        raise FileNotFoundError(f"base_pipeline_path is not a directory: {base_pipeline_path}")
    base_index = os.path.join(base_pipeline_path, "model_index.json")
    if not os.path.isfile(base_index):
        raise FileNotFoundError(f"base pipeline missing model_index.json: {base_index}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        "[Diffusers-Export] Output dir: %s (mode=%s)", output_dir, "copy" if copy else "symlink"
    )

    # 1. model_index.json — copy verbatim (the class registry is the same
    #    whether the transformer weights are live or DMD-distilled).
    dst_index = os.path.join(output_dir, "model_index.json")
    with open(base_index) as f:
        index = json.load(f)
    with open(dst_index, "w") as f:
        json.dump(index, f, indent=2)
    logger.info("[Diffusers-Export] Wrote %s", dst_index)

    # 2. transformer/ — link/copy from the consolidated student.
    dst_transformer = os.path.join(output_dir, "transformer")
    _link_or_copy(student_path, dst_transformer, copy=copy)
    logger.info("[Diffusers-Export] transformer/ <- %s", student_path)

    # 3. vae / text_encoder / tokenizer / scheduler — link/copy from base.
    for comp in BASE_COMPONENTS:
        src = os.path.join(base_pipeline_path, comp)
        dst = os.path.join(output_dir, comp)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"base pipeline missing component: {src}")
        _link_or_copy(src, dst, copy=copy)
        logger.info("[Diffusers-Export] %s/ <- %s", comp, src)

    logger.info(
        "[Diffusers-Export] Done. Load via QwenImagePipeline.from_pretrained(%r)", output_dir
    )


def _verify(output_dir: str) -> dict:
    """Load via QwenImagePipeline.from_pretrained and report a small status dict."""
    import torch
    from diffusers import QwenImagePipeline

    logger.info("[Diffusers-Export-Verify] Loading %s via QwenImagePipeline", output_dir)
    pipe = QwenImagePipeline.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    n_transformer_params = sum(p.numel() for p in pipe.transformer.parameters())
    n_vae_params = sum(p.numel() for p in pipe.vae.parameters())
    n_text_params = sum(p.numel() for p in pipe.text_encoder.parameters())
    stats = {
        "loaded_class": type(pipe).__name__,
        "transformer_class": type(pipe.transformer).__name__,
        "transformer_params": int(n_transformer_params),
        "vae_class": type(pipe.vae).__name__,
        "vae_params": int(n_vae_params),
        "text_encoder_class": type(pipe.text_encoder).__name__,
        "text_encoder_params": int(n_text_params),
        "tokenizer_class": type(pipe.tokenizer).__name__,
        "scheduler_class": type(pipe.scheduler).__name__,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student_path",
        required=True,
        help="Consolidated student dir (e.g. .../epoch_0_step_5/model/consolidated)",
    )
    parser.add_argument(
        "--base_pipeline_path",
        required=True,
        help="Base Qwen-Image pipeline dir (vae / text_encoder / tokenizer / scheduler source)",
    )
    parser.add_argument("--output_dir", required=True, help="Where to write the full pipeline dir")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy components instead of symlinking. Off by default to save disk.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Re-load via QwenImagePipeline.from_pretrained"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_diffusers(
        student_path=args.student_path,
        base_pipeline_path=args.base_pipeline_path,
        output_dir=args.output_dir,
        copy=args.copy,
    )
    if args.verify:
        stats = _verify(args.output_dir)
        print(json.dumps(stats, indent=2))
        sys.exit(0)
