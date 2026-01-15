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

import argparse
import os
from pathlib import Path
from typing import Any

from nemo.collections import llm

from modelopt.torch.puzzletron.export.MCore.llama_nemotron import (
    PuzzletronLlamaNemotronModel,
    PuzzletronNemotronModelConfig,
)


def convert_model(
    hf_model_path_local: str, output_path_nemo_local: str, overwrite: bool = False
) -> Any:
    """Convert a Puzzletron HuggingFace model to NeMo format.

    Args:
        hf_model_path_local: Path to the input Puzzletron HuggingFace model directory
        output_path_nemo_local: Path where the converted Puzzletron NeMo model will be saved
        overwrite: Whether to overwrite existing output directory
    """

    model = PuzzletronLlamaNemotronModel(config=PuzzletronNemotronModelConfig)
    # NOTE: API call to import_ckpt is here: https://github.com/NVIDIA-NeMo/NeMo/blob/294ddff187f68c055d87ffe9400e65975b38693d/nemo/collections/llm/api.py#L888
    print(
        f"calling import_ckpt with model: {model}, "
        f"source: {hf_model_path_local}, "
        f"output_path: {output_path_nemo_local}, "
        f"overwrite: {overwrite}"
    )
    nemo2_path = llm.import_ckpt(
        model=model,
        source="hf://" + hf_model_path_local,
        output_path=Path(output_path_nemo_local),
        overwrite=overwrite,
    )

    print(f"Model saved to {nemo2_path}")
    return nemo2_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Puzzletron HuggingFace model to NeMo format"
    )
    parser.add_argument(
        "--input-ckpt-path",
        "-i",
        type=str,
        required=True,
        help="Path to the input Puzzletron HuggingFace model directory",
    )
    parser.add_argument(
        "--output-ckpt-path",
        "-o",
        type=str,
        required=True,
        help="Path where the converted Puzzletron NeMo model will be saved",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing output directory (default: False)",
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.input_ckpt_path):
        raise FileNotFoundError(f"Input model path does not exist: {args.input_ckpt_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_ckpt_path), exist_ok=True)

    print(f"Converting model from {args.input_ckpt_path} to {args.output_ckpt_path}")
    convert_model(args.input_ckpt_path, args.output_ckpt_path, args.overwrite)


if __name__ == "__main__":
    main()
