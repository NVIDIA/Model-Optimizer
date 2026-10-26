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

"""Prepare a standard Diffusers Qwen-Image artifact with PDD output heads."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from diffusers import QwenImageTransformer2DModel
from huggingface_hub import snapshot_download

_THIS_DIR = Path(__file__).resolve().parent
_FASTGEN_DIR = _THIS_DIR.parent
_REPO_ROOT = _FASTGEN_DIR.parents[2]
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modelopt.torch.fastgen import PDDConfig  # noqa: E402
from modelopt.torch.fastgen.plugins.qwen_image_pdd import convert_qwen_image_to_pdd  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/diffusers/fastgen/pdd/configs/qwen_image.yaml"),
    )
    parser.add_argument("--model-source", help="HF model ID or full local Diffusers snapshot")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve_source(model_source: str, revision: str | None) -> Path:
    local = Path(model_source).expanduser()
    if local.is_dir():
        return local.resolve()
    return Path(snapshot_download(model_source, revision=revision)).resolve()


def _link_base_pipeline(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    for child in source.iterdir():
        if child.name == "transformer":
            continue
        os.symlink(child.resolve(), output / child.name, target_is_directory=child.is_dir())


def main() -> None:
    args = _parse_args()
    raw = yaml.safe_load(args.config.read_text())
    pdd_config = PDDConfig.model_validate(raw["pdd"])
    model_config = raw["model"]
    model_source = args.model_source or model_config["teacher_model_name_or_path"]
    source = _resolve_source(model_source, model_config.get("teacher_revision"))

    transformer = QwenImageTransformer2DModel.from_pretrained(
        source,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_out_channels = transformer.out_channels
    convert_qwen_image_to_pdd(transformer, pdd_config)
    transformer.register_to_config(out_channels=base_out_channels * pdd_config.grid_size)

    output = args.output_dir.expanduser().resolve()
    _link_base_pipeline(source, output)
    transformer.save_pretrained(output / "transformer", safe_serialization=True)


if __name__ == "__main__":
    main()
