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

from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import copy_deci_lm_hf_code


def convert_model(
    nemo_model_path_local: str, output_path_hf_local: str, overwrite: bool = False
) -> Any:
    """Convert a NeMo model to HuggingFace format.

    Args:
        nemo_model_path_local: Path to the input NeMo model file (.nemo)
        output_path_hf_local: Path where the converted HuggingFace model will be saved
        overwrite: Whether to overwrite existing output directory
    """

    # NOTE: API call to export_ckpt is here: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/llm/api.py#L987
    print(
        f"calling export_ckpt with path: {nemo_model_path_local}, "
        f"target: hf, output_path: {output_path_hf_local}, "
        f"target_model_name: PuzzletronLlamaNemotronModel, "
        f"overwrite: {overwrite}"
    )

    hf_path = llm.export_ckpt(
        path=nemo_model_path_local,
        target="hf",
        output_path=Path(output_path_hf_local),
        target_model_name="PuzzletronLlamaNemotronModel",
        overwrite=overwrite,
    )

    copy_deci_lm_hf_code(hf_path)

    print(f"Model saved to {hf_path}")
    return hf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NeMo model to HuggingFace format")
    parser.add_argument(
        "--input-ckpt-path",
        "-i",
        type=str,
        required=True,
        help="Path to the input NeMo model checkpoint",
    )
    parser.add_argument(
        "--output-ckpt-path",
        "-o",
        type=str,
        required=True,
        help="Path where the converted Puzzletron HuggingFace model will be saved",
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
