# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import get_args

from specdec_bench import datasets
from specdec_bench.datasets.speed import config_type

datasets_available = {
    "speed": datasets.SPEEDBench,
}


def prepare_data(args: argparse.Namespace) -> None:
    """Prepare and save benchmark data to disk.

    Calls the dataset's ``prepare_data`` classmethod which downloads and
    resolves all external data references, then saves the fully-resolved
    result as a parquet file so that subsequent benchmark runs can load
    directly from disk without re-downloading.

    Args:
        args: Parsed CLI arguments containing dataset type, config,
            output directory, and optional filtering parameters.
    """
    configs = get_args(config_type) if args.config == "all" else [args.config]

    dataset_cls = datasets_available[args.dataset]

    for config in configs:
        print(f"Preparing config '{config}' ...")

        output_path = dataset_cls.prepare_data(
            output_dir=args.output_dir / args.dataset / config,
            config_name=config,
        )

        print(f"  -> Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare benchmark datasets for specdec_bench.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="speed",
        choices=list(datasets_available.keys()),
        help="Dataset to prepare (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=[*list(get_args(config_type)), "all"],
        help='SPEED-Bench configuration to prepare. Use "all" to prepare all configs. (default: %(default)s)',
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/"),
        help="Directory to save the prepared dataset files (default: %(default)s)",
    )

    args = parser.parse_args()
    prepare_data(args)
