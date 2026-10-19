# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility script to print a sample of hidden states extracted from a dataset."""

import argparse
import random
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a sample of hidden states from a dataset."
        "This script will crawl the provided directory for hidden state files,"
        " and print a small number of samples."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the base input directory containing hidden states."
        "Alternatively, this can be a path to a specific `.pt` file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to print per split. If input_path is a file, this is ignored. "
        "Defaults to 1.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Iterate through the input directory and find all hidden state files
    if args.input_path.is_file():
        all_files = [args.input_path]
    else:
        all_files = list(args.input_path.glob("*.pt"))

    sampled_files = (
        random.sample(all_files, args.num_samples)
        if len(all_files) > args.num_samples
        else all_files
    )

    for i, file in enumerate(sampled_files):
        data = torch.load(file)
        expected_keys = [
            "input_ids",
            "hidden_states",
            "aux_hidden_states",
            "conversation_id",
        ]
        if set(expected_keys) != set(data.keys()):
            print(f"File {file} does not contain all expected keys: {expected_keys}")
            print(f"  Found keys: {list(data.keys())}")
            continue
        print(f"Sample {i + 1}: {file.name}")
        for key in ["input_ids", "hidden_states", "aux_hidden_states"]:
            print(f"{key}: {data[key].shape} {data[key].dtype} {data[key].device}")
        print(f"conversation_id: {data['conversation_id']}")
        input_ids_list = data["input_ids"].tolist()
        hidden_states = data["hidden_states"]
        print(f"Sample of input_ids (first 10 tokens): \n{input_ids_list[:10]}")
        print(f"Sample of input_ids (last 10 tokens): \n{input_ids_list[-10:]}")
        print(f"Sample of hidden_states (first 10 positions): \n{hidden_states[:10]}")

    print(f"\n\nDone. Found: {len(all_files)} files in total.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
