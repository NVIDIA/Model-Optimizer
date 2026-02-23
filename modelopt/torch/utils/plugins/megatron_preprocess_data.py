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

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Processing large data to tokenize for pretraining.

Usage to tokenize one or more JSONL files:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths path/to/input/data1.jsonl path/to/input/data2.jsonl ... \
    --json_keys text \
    --output_dir /path/to/tokenized/Qwen3/ \
    --tokenizer Qwen/Qwen3-0.6B
```

Usage to tokenize all JSONL files in a directory:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --input_dir /path/to/input/data/ \
    --json_keys text \
    --output_dir /path/to/tokenized/Qwen3/ \
    --tokenizer Qwen/Qwen3-0.6B
```

Usage to download and tokenize a dataset from Hugging Face Hub:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Pretraining-Dataset-sample \
    --hf_name Nemotron-SFT-Code \
    --hf_split train \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir /path/to/tokenized/Qwen3/
```

NOTE: If you skip --hf_name, it will download and tokenize all subsets for the dataset.
If you skip --hf_split, it will download and tokenize all splits for the subset.
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from warnings import warn

import requests
from datasets import load_dataset
from huggingface_hub.utils import build_hf_headers
from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer

from modelopt.torch.utils import num2hrb

__all__ = ["megatron_preprocess_data"]


class _Encoder:
    tokenizer: AutoTokenizer = None

    def __init__(
        self,
        tokenizer_name_or_path: str,
        json_keys: list[str],
        append_eod: bool,
        max_sequence_length: int | None,
    ):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.json_keys = json_keys
        self.append_eod = append_eod
        self.max_sequence_length = max_sequence_length
        self.max_document_length = (
            max_sequence_length * 8 if max_sequence_length is not None else None
        )
        print(f"Setting max document length: {self.max_document_length}")

    def initializer(self):
        # Use Encoder class as a container for global data
        _Encoder.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def encode(self, json_line: str):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        doc_len = 0
        enc_len = 0
        for key in self.json_keys:
            text = data[key]

            # Truncate text by character length if specified
            doc_len += len(text)
            if self.max_document_length is not None:
                text = text[: self.max_document_length]
                # print(f"Document truncated from {original_length} to {self.max_document_length} characters")

            # Tokenize the entire text as one document
            encoded = _Encoder.tokenizer.encode(text)

            enc_len += len(encoded)
            if self.max_sequence_length is not None:
                encoded = encoded[: self.max_sequence_length]
                # print(f"Sequence truncated from {original_length} to {self.max_sequence_length} tokens")

            if len(encoded) > 0 and self.append_eod:
                encoded.append(_Encoder.tokenizer.eos_token_id)

            ids[key] = encoded
            lens[key] = [len(encoded)] if len(encoded) > 0 else []
        return ids, lens, (doc_len, enc_len)


class _Partition:
    def __init__(self, vocab_size: int, json_keys: list[str], log_interval: int, workers: int):
        self.vocab_size = vocab_size
        self.json_keys = json_keys
        self.log_interval = log_interval
        self.workers = workers

    def _print_processing_stats(
        self, count: int, total_doc_len: int, total_enc_len: int, *, force_print: bool = False
    ):
        if count % self.log_interval == 0 or force_print:
            print(
                f"\tProcessed {num2hrb(count)} docs = {num2hrb(total_doc_len)} chars = {num2hrb(total_enc_len)} tokens"
            )

    def process_json_file(
        self, input_file_name: str | Path, output_dir: str | Path, encoder: _Encoder
    ):
        output_prefix = Path(output_dir) / Path(input_file_name).stem

        print(f"\nOpening {input_file_name}")
        fin = open(input_file_name, encoding="utf-8")

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"

        for key in self.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix, key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, level)
            if Path(output_bin_files[key]).exists() and Path(output_idx_files[key]).exists():
                continue
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(self.vocab_size),
            )

        if not builders:
            print(f"\t[SKIP] Output files corresponding to {input_file_name} already exist")
            return 0

        total_doc_len, total_enc_len, final_enc_len = 0, 0, 0
        for i, (doc, sentence_lens, (doc_len, enc_len)) in enumerate(encoded_docs, start=1):
            total_doc_len += doc_len
            total_enc_len += enc_len
            final_enc_len += sum(sentence_lens[key])
            for key in doc:
                builders[key].add_document(doc[key], sentence_lens[key])
            self._print_processing_stats(i, total_doc_len, total_enc_len)
        self._print_processing_stats(i, total_doc_len, total_enc_len, force_print=True)

        fin.close()
        for key in builders:
            builders[key].finalize(output_idx_files[key])

        return final_enc_len


def _download_hf_dataset(
    dataset: str,
    output_dir: str | Path,
    json_keys: list[str],
    name: str | None = None,
    split: str | None = "train",
    max_samples_per_split: int | None = None,
) -> list[str]:
    """Download a Hugging Face dataset and save as JSONL files.

    Returns:
        List of paths to downloaded JSONL files.
    """
    print(f"Downloading dataset {dataset} from Hugging Face")
    jsonl_paths: list[str] = []

    try:
        response = requests.get(
            f"https://datasets-server.huggingface.co/splits?dataset={dataset}",
            headers=build_hf_headers(),
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch dataset splits for {dataset}: {e}") from e

    response_json = response.json()
    print(f"\nFound {len(response_json['splits'])} total splits for {dataset}:")
    for entry in response_json["splits"]:
        print(f"\t{entry}")

    splits_to_process = []
    for entry in response_json["splits"]:
        if name is not None and name != entry.get("config", None):
            continue
        if split is not None and split != entry["split"]:
            continue
        splits_to_process.append(entry)

    print(f"\nFound {len(splits_to_process)} splits to process:")
    for entry in splits_to_process:
        print(f"\t{entry}")

    for entry in splits_to_process:
        skip_processing = False
        path = entry["dataset"]
        name = entry.get("config", None)
        split = entry["split"]
        if max_samples_per_split is not None:
            split = f"{split}[:{max_samples_per_split}]"
        jsonl_file_path = f"{output_dir}/raw/{path.replace('/', '--')}_{name}_{split}.jsonl"

        print(f"\nLoading HF dataset {path=}, {name=}, {split=}")
        if os.path.exists(jsonl_file_path):
            jsonl_paths.append(jsonl_file_path)
            print(f"\t[SKIP] Raw dataset {jsonl_file_path} already exists")
            continue
        ds = load_dataset(path=path, name=name, split=split)

        for key in json_keys:
            if key not in ds.features:
                warn(f"[SKIP] {key=} not found in {ds.features=}")
                skip_processing = True
                break

        if skip_processing:
            continue

        print(f"Saving raw dataset to {jsonl_file_path}")
        ds.to_json(jsonl_file_path)
        jsonl_paths.append(jsonl_file_path)

    print(f"\n\nTokenizing JSONL paths: {jsonl_paths}\n")
    return jsonl_paths


def megatron_preprocess_data(
    *,
    input_dir: str | Path | None = None,
    jsonl_paths: str | Path | list[str] | list[Path] | None = None,
    # Hugging Face Hub dataset arguments
    hf_dataset: str | None = None,
    hf_name: str | None = None,
    hf_split: str | None = "train",
    hf_max_samples_per_split: int | None = None,
    # Other arguments
    output_dir: str | Path,
    tokenizer_name_or_path: str,
    json_keys: list[str] = ["text"],
    append_eod: bool = False,
    max_sequence_length: int | None = None,
    workers: int = 1,
    log_interval: int = 1000,
):
    """Process large data for pretraining.

    Exactly one of ``input_dir``, ``jsonl_paths``, or ``hf_dataset`` must be provided.

    Args:
        input_dir (str | Path, optional): Directory containing JSONL files to tokenize.
        jsonl_paths (str | Path | list, optional): One or more paths to JSONL files.
        hf_dataset (str, optional): Hugging Face Hub dataset name or path to download and tokenize.
        hf_name (str, optional): Hugging Face Hub dataset subset name. Downloads all subsets if None.
        hf_split (str, optional): Hugging Face Hub dataset split. Defaults to "train".
        hf_max_samples_per_split (int, optional): Maximum number of samples to download per split from Hugging Face Hub.
            Skip to download all samples.
        output_dir (str | Path): Path to directory to save binary output files.
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer to use.
        json_keys (list, optional): List of keys to extract from json. Defaults to ["text"].
        append_eod (bool, optional): Append an <eod> token to the end of a document. Defaults to False.
        max_sequence_length (int, optional): Maximum tokenized sequence length. Defaults to None.
        workers (int, optional): Number of worker processes to launch. Defaults to 1.
        log_interval (int, optional): Interval between progress updates. Defaults to 100000.
    """
    num_sources = sum(x is not None for x in (input_dir, jsonl_paths, hf_dataset))
    if num_sources != 1:
        raise ValueError(
            "Exactly one of `input_dir`, `jsonl_paths`, or `hf_dataset` must be provided."
        )

    if hf_dataset is not None:
        jsonl_paths = _download_hf_dataset(
            hf_dataset,
            output_dir,
            json_keys,
            name=hf_name,
            split=hf_split,
            max_samples_per_split=hf_max_samples_per_split,
        )

    if input_dir is not None:
        file_names = sorted(Path(input_dir).glob("*.jsonl"))
        if not file_names:
            raise ValueError(f"No JSONL files found in input directory: {input_dir}")
    elif isinstance(jsonl_paths, (str, Path)):
        file_names = [jsonl_paths]  # type: ignore[list-item]
    else:
        file_names = list(jsonl_paths)  # type: ignore[arg-type]

    Path(output_dir).mkdir(exist_ok=True)
    vocab_size = AutoTokenizer.from_pretrained(tokenizer_name_or_path).vocab_size

    encoder = _Encoder(tokenizer_name_or_path, json_keys, append_eod, max_sequence_length)
    partition = _Partition(vocab_size, json_keys, log_interval, workers)

    final_enc_len = 0
    for name in file_names:
        num_tokens = partition.process_json_file(name, output_dir, encoder)
        final_enc_len += num_tokens

    print(f"\n\n>>> Total number of tokens currently processed: {num2hrb(final_enc_len)}")


def main():
    """Sample main function to process large data for pretraining."""
    parser = argparse.ArgumentParser(prog="megatron_preprocess_data")
    # Dataset arguments (pre-downloaded .jsonl files or download from Hugging Face Hub)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Directory containing JSONL files")
    group.add_argument(
        "--jsonl_paths", nargs="+", type=str, help="One or more paths to JSONL files"
    )
    group.add_argument(
        "--hf_dataset",
        type=str,
        help="Hugging Face Hub dataset path to download and tokenize",
    )
    parser.add_argument(
        "--hf_name",
        type=str,
        default=None,
        help="Hugging Face Hub dataset subset name. Skip to download and tokenize all subsets for the dataset.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Hugging Face Hub dataset split. Skip to download and tokenize all splits for the subset.",
    )
    parser.add_argument(
        "--hf_max_samples_per_split",
        type=int,
        default=None,
        help="Maximum number of samples to download per split from Hugging Face Hub. Skip to download all samples.",
    )
    # Other arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name or path")
    parser.add_argument("--json_keys", nargs="+", default=["text"], help="JSON keys to tokenize")
    parser.add_argument("--append_eod", action="store_true", help="Append <eod> token")
    parser.add_argument(
        "--max_sequence_length", type=int, default=None, help="Maximum sequence length"
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--log_interval", type=int, default=100000, help="Log interval")
    args = parser.parse_args()

    print("\n==================== Arguments ====================")
    for k, v in args.__dict__.items():
        print(f"{k:<35} {v}")
    print("===================================================\n")

    megatron_preprocess_data(
        input_dir=args.input_dir,
        jsonl_paths=args.jsonl_paths,
        hf_dataset=args.hf_dataset,
        hf_name=args.hf_name,
        hf_split=args.hf_split,
        hf_max_samples_per_split=args.hf_max_samples_per_split,
        output_dir=args.output_dir,
        tokenizer_name_or_path=args.tokenizer,
        json_keys=args.json_keys,
        append_eod=args.append_eod,
        max_sequence_length=args.max_sequence_length,
        workers=args.workers,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
