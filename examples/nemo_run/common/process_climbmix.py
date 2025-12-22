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

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from modelopt.torch.utils.plugins import megatron_preprocess_data

SUBSET_IDX = [
    *[0, 1, 6, 10, 11],
    *[12, 13, 14, 21, 24],
    *[33, 35, 38, 40, 48],
    *[49, 52, 66, 70, 76],
    *[83, 88, 91, 94, 99],
]  # 25% of total dataset


def get_args():
    parser = argparse.ArgumentParser(description="Process ClimbMix dataset")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Path to the directory to store the processed dataset",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="Tokenizer to use for preprocessing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # create raw and processed directories
    raw_dir = Path(args.output_dir) / "climbmix_raw"
    proc_dir = Path(args.output_dir) / "climbmix_proc"

    # only download the subset of the data
    subset_filenames = [f"part_{i}.jsonl" for i in SUBSET_IDX]

    # download raw data
    snapshot_download(
        repo_id="OptimalScale/ClimbMix",
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=subset_filenames,
    )

    # preprocess (tokenize)
    print("Tokenizing ClimbMix dataset...")
    input_paths = [raw_dir / name for name in subset_filenames]
    megatron_preprocess_data(
        input_paths,
        output_dir=proc_dir,
        tokenizer_name_or_path=args.tokenizer,
        append_eod=True,
        max_sequence_length=32000,
        workers=8,
        log_interval=10000,
    )
