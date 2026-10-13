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

from metrics.imagereward import compute_image_reward_metrics
from metrics.multimodal import compute_clip, compute_clip_iqa
from utils import load_json_file


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data file.")
    parser.add_argument(
        "--metrics",
        nargs="+",  # or nargs="*"
        default=["imagereward"],
        choices=["imagereward", "clip-iqa", "clip"],
        help="Model IDs to run.",
    )

    return parser.parse_args()


def main(args):
    data = load_json_file(Path(args.data_path))
    results = []
    if "imagereward" in args.metrics:
        results.append(compute_image_reward_metrics(data))

    if "clip-iqa" in args.metrics:
        results.append(compute_clip_iqa(data))

    if "clip" in args.metrics:
        results.append(compute_clip(data))

    if not results:
        raise NotImplementedError(
            "No recognized metrics were provided. Available: 'imagereward', 'clip-iqa', 'clip'"
        )
    print(results)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
