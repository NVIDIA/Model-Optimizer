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

"""
Script to process LIMA (Less is More for Alignment) dataset from https://huggingface.co/datasets/GAIR/lima

Before running this script, go to the HuggingFace link to get access to the dataset, then set your HF_TOKEN in your
environment variable e.g. `export HF_TOKEN=<your-hf-token`. You can find your HuggingFace token in your account
settings.
"""

import json
import subprocess
from pathlib import Path


def download_hf_dataset(dataset_name: str, output_dir: str | None = None):
    """Download a dataset from HuggingFace Hub using huggingface-cli."""
    cmd = ["huggingface-cli", "download", dataset_name, "--repo-type", "dataset"]

    if output_dir:
        cmd.extend(["--local-dir", output_dir])

    subprocess.run(cmd, check=True)
    print(f"Successfully downloaded dataset: {dataset_name}")


def process_jsonl(jsonl_path, output_path):
    with open(output_path, "w") as output_fp, open(jsonl_path) as input_fp:
        for line in input_fp:
            row = json.loads(line)
            oai_row = []
            for i, turn in enumerate(row["conversations"]):
                if i % 2 == 0:
                    oai_row.append({"role": "user", "content": turn})
                else:
                    oai_row.append({"role": "assistant", "content": turn})
            output_fp.write(json.dumps({"messages": oai_row}))
            output_fp.write("\n")


if __name__ == "__main__":
    raw_path = "lima_raw_data"
    proc_path = "lima_processed"

    if not Path(raw_path).exists():
        dataset_path = download_hf_dataset("GAIR/lima", raw_path)

    Path(proc_path).mkdir(exist_ok=False)
    process_jsonl(f"{raw_path}/train.jsonl", f"{proc_path}/training.jsonl")
    process_jsonl(f"{raw_path}/test.jsonl", f"{proc_path}/validation.jsonl")
