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

import json
import os
from pathlib import Path

from datasets import load_dataset

from modelopt.torch.utils.plugins.megatron_preprocess_data import megatron_preprocess_data


def download_and_prepare_minipile_dataset(output_dir: Path) -> Path:
    """Download the nanotron/minipile_100_samples dataset and convert to JSONL format.

    Args:
        output_dir: Directory to save the JSONL file

    Returns:
        Path to the created JSONL file
    """
    dataset = load_dataset("nanotron/minipile_100_samples", split="train")

    jsonl_file = output_dir / "minipile_100_samples.jsonl"

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in dataset:
            json_obj = {"text": item["text"]}
            f.write(json.dumps(json_obj) + "\n")

    return jsonl_file


def test_megatron_preprocess_data_with_minipile_jsonl(tmp_path):
    """Test megatron_preprocess_data with nanotron/minipile_100_samples dataset.

    This test:
    1. Downloads the HuggingFace dataset "nanotron/minipile_100_samples"
    2. Converts it to JSONL format
    3. Calls megatron_preprocess_data with jsonl_paths
    4. Verifies that output files are created
    """
    input_jsonl = download_and_prepare_minipile_dataset(tmp_path)

    assert input_jsonl.exists(), "Input JSONL file should exist"
    assert input_jsonl.stat().st_size > 0, "Input JSONL file should not be empty"

    with open(input_jsonl, encoding="utf-8") as f:
        first_line = f.readline().strip()
        first_item = json.loads(first_line)
        assert "text" in first_item, "Each JSONL item should have a 'text' field"
        assert isinstance(first_item["text"], str), "Text field should be a string"

    megatron_preprocess_data(
        jsonl_paths=input_jsonl,
        output_dir=tmp_path,
        tokenizer_name_or_path="gpt2",
        json_keys=["text"],
        workers=1,
    )

    output_prefix = tmp_path / "minipile_100_samples"
    expected_bin_file = f"{output_prefix}_text_document.bin"
    expected_idx_file = f"{output_prefix}_text_document.idx"

    assert os.path.exists(expected_bin_file), (
        f"Expected binary file {expected_bin_file} should exist"
    )
    assert os.path.exists(expected_idx_file), (
        f"Expected index file {expected_idx_file} should exist"
    )

    assert os.path.getsize(expected_bin_file) > 0, "Binary file should not be empty"
    assert os.path.getsize(expected_idx_file) > 0, "Index file should not be empty"


def test_megatron_preprocess_data_with_hf_dataset(tmp_path):
    """Test megatron_preprocess_data with dataset download, --append_eod and --max_sequence_length.

    Downloads nanotron/minipile_100_samples train split from Hugging Face and tokenizes it.
    """
    megatron_preprocess_data(
        hf_dataset="nanotron/minipile_100_samples",
        hf_split="train",
        output_dir=tmp_path,
        tokenizer_name_or_path="gpt2",
        json_keys=["text"],
        append_eod=True,
        max_sequence_length=512,
        workers=4,
    )

    bin_files = sorted(tmp_path.glob("*.bin"))
    idx_files = sorted(tmp_path.glob("*.idx"))

    assert len(bin_files) > 0, f"Expected .bin files in {tmp_path}, found none"
    assert len(idx_files) > 0, f"Expected .idx files in {tmp_path}, found none"

    for f in bin_files + idx_files:
        assert f.stat().st_size > 0, f"{f.name} should not be empty"
