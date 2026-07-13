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

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest
import yaml

from modelopt.torch.utils.plugins.prepare_megatron_data_blend import main


def _setup_test(
    tiny_tokenizer_path: str, tmp_path: Path, monkeypatch, target_tokens: int | None
) -> tuple[Path, Path, Mock]:
    output_dir = tmp_path / "tokenized"
    config = {
        "tokenizer": tiny_tokenizer_path,
        "output_dir": str(output_dir),
        "sources": [
            {
                "hf_dataset": "nanotron/minipile_100_samples",
                "split": "train",
                "max_samples": 100,
                "content_field": "text",
                "weight": 60,
            },
            {
                "hf_dataset": "nvidia/Nemotron-SFT-Competitive-Programming-v2",
                "files": ["data/competitive_programming_python_00.jsonl"],
                "content_field": "messages",
                "weight": 40,
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_yaml = yaml.safe_dump(config)
    if target_tokens is not None:
        config_yaml += f"target_tokens: {target_tokens:_}\n"
    config_path.write_text(config_yaml, encoding="utf-8")
    jsonl_path = tmp_path / "competitive_programming_python_00.jsonl"
    conversation = {
        "messages": [
            {"role": "user", "content": "Write a Python function that adds two integers."},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
        ]
    }
    jsonl_path.write_text(
        "".join(json.dumps(conversation) + "\n" for _ in range(20)), encoding="utf-8"
    )
    download = Mock(return_value=str(jsonl_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        sys, "argv", ["prepare_megatron_data_blend.py", "--config", str(config_path)]
    )
    return output_dir, config_path, download


@pytest.mark.parametrize("target_tokens", [1_000, None], ids=["token-budget", "all-data"])
def test_prepare_megatron_data_blend_with_split_and_files_sources(
    tiny_tokenizer_path: str, tmp_path: Path, monkeypatch, target_tokens: int | None
):
    output_dir, config_path, download = _setup_test(
        tiny_tokenizer_path, tmp_path, monkeypatch, target_tokens
    )

    # Run in-process so the CLI entry point uses the mocked NVIDIA download.
    main()

    download.assert_called_once_with(
        repo_id="nvidia/Nemotron-SFT-Competitive-Programming-v2",
        filename="data/competitive_programming_python_00.jsonl",
        repo_type="dataset",
        local_dir=tmp_path / "raw/nvidia--Nemotron-SFT-Competitive-Programming-v2",
    )
    blend = [
        line.split(maxsplit=1)
        for line in (output_dir / "data_blend.txt").read_text(encoding="utf-8").splitlines()
    ]
    assert [weight for weight, _ in blend] == ["60", "40"]
    for _, prefix in blend:
        assert Path(prefix + ".bin").exists()
        assert Path(prefix + ".idx").exists()
    token_suffixes = ["_tokens600", "_tokens400"] if target_tokens is not None else ["", ""]
    # HF split prefixes use {dataset}_{config}_{split}_{field}_max{samples}.
    assert [Path(prefix).name for _, prefix in blend] == [
        f"nanotron--minipile_100_samples_default_train_text_max100{token_suffixes[0]}",
        f"competitive_programming_python_00_messages{token_suffixes[1]}",
    ]
    assert (output_dir / "config.yaml").read_bytes() == config_path.read_bytes()
