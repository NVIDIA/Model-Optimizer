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

"""Prepare a weighted Megatron data blend from a YAML configuration."""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import yaml

from .megatron_preprocess_data import megatron_preprocess_data

__all__ = ["prepare_megatron_data_blend"]


def load_config(path: Path) -> dict[str, Any]:
    """Load a data-blend YAML configuration as a dictionary.

    For example, this YAML::

        tokenizer: /models/Qwen3-8B
        output_dir: /datasets/qwen3-blend
        target_tokens: 1000000  # Optional; omit to prepare every source in full.
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

    returns a dictionary with ``tokenizer``, ``output_dir``, and ``sources`` keys, plus
    optional ``target_tokens``. Each source has ``hf_dataset``, ``content_field``, and
    ``weight``; it uses ``split`` with optional ``config`` and ``max_samples``, or
    selects repository ``files``.
    """
    with path.open(encoding="utf-8") as stream:
        return cast("dict[str, Any]", yaml.safe_load(stream))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the blend YAML file")
    return parser


def _write_data_blend(path: Path, blend: list[tuple[float, str]]) -> None:
    """Write the weighted Megatron dataset paths."""
    content = "\n".join(f"{weight:g} {prefix}" for weight, prefix in blend) + "\n"
    path.write_text(content, encoding="utf-8")


def _copy_config(source: Path, destination: Path) -> None:
    """Copy the input configuration alongside the generated blend."""
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def _prepare_sources(
    sources: list[dict[str, Any]],
    output_dir: Path,
    tokenizer: str,
    total_tokens: int | None,
) -> list[tuple[float, str]]:
    """Tokenize all sources and return their weighted output paths."""
    workers = min(32, os.cpu_count() or 1)
    blend: list[tuple[float, str]] = []  # (weight, shared .bin/.idx path without extension)
    allocated_tokens = 0
    # Weights are relative, not required to sum to 100 (matching data_blend.txt semantics).
    weight_sum = sum(float(source["weight"]) for source in sources)

    for index, source in enumerate(sources):
        weight = float(source["weight"])
        if total_tokens is None:
            source_tokens = None
        elif index == len(sources) - 1:
            source_tokens = total_tokens - allocated_tokens
        else:
            source_tokens = round(total_tokens * weight / weight_sum)
            allocated_tokens += source_tokens

        dataset = source["hf_dataset"]
        source_dir = output_dir / f"{index:02d}_{dataset.replace('/', '--')}"
        content_field = source["content_field"]
        input_args: dict[str, Any]
        if "files" in source:
            raw_dir = output_dir.parent / "raw" / dataset.replace("/", "--")
            paths = [
                huggingface_hub.hf_hub_download(
                    repo_id=dataset,
                    filename=file,
                    repo_type="dataset",
                    local_dir=raw_dir,
                )
                for file in source["files"]
            ]
            input_args = {"jsonl_paths": paths}
        else:
            input_args = {
                "hf_dataset": dataset,
                "hf_name": source.get("config"),
                "hf_split": source["split"],
                "hf_max_samples_per_split": source.get("max_samples"),
                "hf_streaming": True,
            }

        # Each prefix is the path shared by a tokenized Megatron .bin/.idx file pair.
        prefixes = megatron_preprocess_data(
            **input_args,
            output_dir=source_dir,
            tokenizer_name_or_path=tokenizer,
            json_keys=content_field,
            # Plain text lacks chat-template boundary tokens, so terminate each document with EOS.
            append_eod=content_field == "text",
            # Join lines in text documents by replacing each newline with a space.
            strip_newlines=content_field == "text",
            reasoning_content="inline" if content_field == "messages" else "strip",
            # Guard against pathological records by capping each tokenized document at 256K tokens.
            max_sequence_length=256_000,
            max_tokens=source_tokens,
            workers=workers,
        )
        prefix_weight = weight / len(prefixes)
        blend.extend((prefix_weight, prefix) for prefix in prefixes)

    return blend


def prepare_megatron_data_blend(config_path: Path) -> list[tuple[float, str]]:
    """Download and tokenize the configured weighted data sources."""
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target_tokens = config.get("target_tokens")
    total_tokens = None if target_tokens is None else int(target_tokens)
    tokenizer = str(config["tokenizer"])

    blend = _prepare_sources(config["sources"], output_dir, tokenizer, total_tokens)
    _write_data_blend(output_dir / "data_blend.txt", blend)
    _copy_config(config_path, output_dir / "config.yaml")
    return blend


def main() -> None:
    """Prepare a data blend from the supplied configuration."""
    parser = _build_parser()
    args = parser.parse_args()
    blend = prepare_megatron_data_blend(args.config)
    print(f"Prepared {len(blend)} data paths. See data_blend.txt and config.yaml in the output.")


if __name__ == "__main__":
    main()
