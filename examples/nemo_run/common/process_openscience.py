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
import os
from pathlib import Path

from datasets import load_dataset


def get_parser():
    parser = argparse.ArgumentParser(description="Process nvidia/OpenScience dataset")
    parser.add_argument("--output-dir", type=str, default=".")
    return parser


def convert_row_oai(row: dict):
    return {
        "messages": [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def process_subset(raw_dir, proc_dir):
    ds = load_dataset(raw_dir)
    ds = ds.map(convert_row_oai, remove_columns=["input", "output"])

    split_ds = ds["train"].train_test_split(test_size=0.1)
    split_ds["train"].to_json(os.path.join(proc_dir, "training.jsonl"))
    split_ds["test"].to_json(os.path.join(proc_dir, "validation.jsonl"))


if __name__ == "__main__":
    args = get_parser().parse_args()
    raw_dir = f"{args.output_dir}/openscience_raw"
    proc_dir = f"{args.output_dir}/openscience_proc"

    if not os.path.exists(raw_dir):
        q235_subset = load_dataset("nvidia/OpenScience", data_files="OS-Q3-235B-4.jsonl")
        q235_subset.save_to_disk(raw_dir)

    if not os.path.exists(proc_dir):
        Path(proc_dir).mkdir(exist_ok=True)
        print("Processing OpenScience dataset")
        process_subset(raw_dir, proc_dir)
    else:
        print(f"Processed OpenScience dataset exists in: {proc_dir}, skipped processing")
