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

"""Prepare a token-sized Megatron data blend from a YAML configuration."""

import argparse
from pathlib import Path
from typing import Any, cast

import yaml

__all__ = ["load_config", "main"]


def load_config(path: Path) -> dict[str, Any]:
    """Load a data-blend YAML configuration as a dictionary.

    For example, this YAML::

        tokenizer: /models/Qwen3-8B
        output_dir: /datasets/qwen3-blend
        target_tokens: 1000000
        sources:
          - hf_dataset: nvidia/Nemotron-Pretraining-SFT-v1
            config: Nemotron-SFT-General
            split: train
            content_field: text
            weight: 60
          - hf_dataset: nvidia/Nemotron-SFT-Competitive-Programming-v2
            files:
              - data/competitive_programming_python_00.jsonl
            content_field: messages
            weight: 40

    returns a dictionary with ``tokenizer``, ``output_dir``, ``target_tokens``, and
    ``sources`` keys. Each source has ``hf_dataset``, ``content_field``, and ``weight``;
    it uses ``split`` with an optional ``config``, or selects repository ``files``.
    """
    with path.open(encoding="utf-8") as stream:
        return cast("dict[str, Any]", yaml.safe_load(stream))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the blend YAML file")
    return parser


def main() -> None:
    """Prepare a data blend from the supplied configuration."""
    parser = _build_parser()
    args = parser.parse_args()
    load_config(args.config)
    print(f"Parsed configuration from {args.config}. Data preparation is not implemented yet.")


if __name__ == "__main__":
    main()
