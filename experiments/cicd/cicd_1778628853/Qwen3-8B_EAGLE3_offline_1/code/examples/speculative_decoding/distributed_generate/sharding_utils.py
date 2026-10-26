#!/bin/bash
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
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=False, default=None)
parser.add_argument("--input_dir", type=str, required=False, default=None)
parser.add_argument("--output_dir", type=str, required=False, default=None)
parser.add_argument("--output_path", type=str, required=False, default=None)
parser.add_argument("--max_lines_per_shard", type=int, default=10000)
parser.add_argument("--combine", action="store_true", help="Undo the sharding process")
args = parser.parse_args()


def shard_jsonl_file(input_path: str, output_dir: str, max_lines: int = 10000) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, encoding="utf-8") as infile:
        shard_idx = 0
        line_count = 0
        outfile = open(
            os.path.join(output_dir, f"train-{shard_idx:05d}-{shard_idx:05d}.jsonl"),
            "w",
            encoding="utf-8",
        )
        try:
            for line in infile:
                if line_count >= max_lines:
                    outfile.close()
                    shard_idx += 1
                    line_count = 0
                    outfile = open(
                        os.path.join(output_dir, f"train-{shard_idx:05d}-{shard_idx:05d}.jsonl"),
                        "w",
                        encoding="utf-8",
                    )
                outfile.write(line)
                line_count += 1
        finally:
            outfile.close()


def combine_jsonl_files(input_dir: str, output_path: str) -> None:
    files = os.listdir(input_dir)
    files = [f for f in files if f.endswith(".jsonl")]
    files.sort()
    with open(output_path, "w", encoding="utf-8") as outfile:
        for file in files:
            with open(os.path.join(input_dir, file), encoding="utf-8") as infile:
                for line in infile:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("finished", False):
                        continue
                    data.pop("conversation_id", None)
                    if data:
                        outfile.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    if args.combine:
        if args.input_dir is None or args.output_path is None:
            raise ValueError("input_dir and output_path are required when combining shards")
        combine_jsonl_files(args.input_dir, args.output_path)
    else:
        if args.input_path is None or args.output_dir is None:
            raise ValueError("input_path and output_dir are required when sharding")
        shard_jsonl_file(args.input_path, args.output_dir, args.max_lines_per_shard)
