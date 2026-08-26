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

import gzip
import json
from pathlib import Path

import pytest

from modelopt.torch.utils.dataset_utils import download_hf_dataset_as_jsonl
from modelopt.torch.utils.plugins.megatron_preprocess_data import megatron_preprocess_data


def test_megatron_preprocess_data_with_jsonl_path(tmp_path):
    input_jsonl = download_hf_dataset_as_jsonl("nanotron/minipile_100_samples", tmp_path / "raw")
    assert len(input_jsonl) == 1, "Expected 1 JSONL file"
    input_jsonl = Path(input_jsonl[0])

    assert input_jsonl.stat().st_size > 0, "Input JSONL file should not be empty"

    with open(input_jsonl, encoding="utf-8") as f:
        first_line = f.readline().strip()
        first_item = json.loads(first_line)
        assert "text" in first_item, "Each JSONL item should have a 'text' field"
        assert isinstance(first_item["text"], str), "Text field should be a string"

    prefixes = megatron_preprocess_data(
        jsonl_paths=input_jsonl,
        output_dir=tmp_path,
        tokenizer_name_or_path="gpt2",
        json_keys=["text"],
        workers=1,
    )

    assert prefixes == [str(tmp_path / f"{input_jsonl.stem}_text")], (
        f"Expected prefix based on JSONL stem, got {prefixes}"
    )
    assert Path(prefixes[0] + ".bin").exists(), f"Expected binary file {prefixes[0]}.bin"
    assert Path(prefixes[0] + ".idx").exists(), f"Expected index file {prefixes[0]}.idx"
    assert Path(prefixes[0] + ".bin").stat().st_size > 0, "Binary file should not be empty"
    assert Path(prefixes[0] + ".idx").stat().st_size > 0, "Index file should not be empty"


@pytest.mark.parametrize(
    ("hf_dataset", "hf_split", "json_keys"),
    [
        ("nanotron/minipile_100_samples", "train", ["text"]),
    ],
)
def test_megatron_preprocess_data_with_hf_dataset(tmp_path, hf_dataset, hf_split, json_keys):
    prefixes = megatron_preprocess_data(
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        hf_max_samples_per_split=10,
        output_dir=tmp_path,
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        json_keys=json_keys,
        append_eod=True,
        max_sequence_length=32,
        workers=4,
    )

    jsonl_files = sorted(tmp_path.glob("**/*.jsonl"))
    assert len(jsonl_files) == 0, (
        f"No intermediate .jsonl files should be written, found: {jsonl_files}"
    )

    assert len(prefixes) > 0, "Expected at least one output prefix"
    assert len(prefixes) == len(json_keys), f"Expected one prefix per json_key, got {prefixes}"
    for prefix in prefixes:
        assert "_max10" in prefix, (
            f"Expected '_max10' in prefix when hf_max_samples_per_split=10, got {prefix}"
        )
    for prefix in prefixes:
        assert Path(prefix + ".bin").exists(), f"Expected binary file {prefix}.bin"
        assert Path(prefix + ".idx").exists(), f"Expected index file {prefix}.idx"
        assert Path(prefix + ".bin").stat().st_size > 0, f"{prefix}.bin should not be empty"
        assert Path(prefix + ".idx").stat().st_size > 0, f"{prefix}.idx should not be empty"


def test_megatron_preprocess_data_reasoning_content(tmp_path):
    sample = {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to add 2 and 2 together. The result is 4.",
            },
        ]
    }
    jsonl_path = tmp_path / "test_reasoning.jsonl"
    jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    for mode in ("strip", "inline"):
        out_dir = tmp_path / mode
        out_dir.mkdir()
        prefixes = megatron_preprocess_data(
            jsonl_paths=jsonl_path,
            output_dir=out_dir,
            tokenizer_name_or_path="Qwen/Qwen3-0.6B",
            json_keys=["messages"],
            workers=1,
            reasoning_content=mode,
        )
        assert len(prefixes) == 1, f"[{mode}] Expected 1 prefix, got {prefixes}"
        assert prefixes[0] == str(out_dir / "test_reasoning_messages"), (
            f"[{mode}] Unexpected prefix: {prefixes[0]}"
        )
        assert Path(prefixes[0] + ".bin").exists(), f"[{mode}] Expected {prefixes[0]}.bin"
        assert Path(prefixes[0] + ".idx").exists(), f"[{mode}] Expected {prefixes[0]}.idx"
        assert Path(prefixes[0] + ".bin").stat().st_size > 0, (
            f"[{mode}] Binary output should not be empty"
        )

    strip_size = Path(str(tmp_path / "strip" / "test_reasoning_messages") + ".bin").stat().st_size
    inline_size = Path(str(tmp_path / "inline" / "test_reasoning_messages") + ".bin").stat().st_size
    assert inline_size > strip_size, (
        f"inline ({inline_size} bytes) should produce a larger binary than strip ({strip_size} bytes)"
    )


def test_megatron_preprocess_data_with_gzip_input(tmp_path):
    gz_path = tmp_path / "data.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"Line one.\nLine two.\nSample {i}."}) + "\n")

    prefixes = megatron_preprocess_data(
        jsonl_paths=gz_path,
        output_dir=tmp_path,
        tokenizer_name_or_path="gpt2",
        json_keys=["text"],
        workers=1,
        strip_newlines=True,
    )

    # .jsonl.gz → stem should be "data", not "data.jsonl"
    assert prefixes == [str(tmp_path / "data_text")], f"Unexpected prefix: {prefixes}"
    assert Path(prefixes[0] + ".bin").stat().st_size > 0


def test_megatron_preprocess_data_hf_streaming_warning(tmp_path):
    # hf_streaming without hf_max_samples_per_split should warn and fall back to non-streaming
    with pytest.warns(UserWarning, match="hf_streaming"):
        megatron_preprocess_data(
            hf_dataset="nanotron/minipile_100_samples",
            hf_split="train",
            hf_streaming=True,
            output_dir=tmp_path,
            tokenizer_name_or_path="gpt2",
            json_keys=["text"],
            workers=1,
        )
