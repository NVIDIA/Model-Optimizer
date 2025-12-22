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

Usage:

```python
from modelopt.torch.utils.plugins import megatron_preprocess_data

megatron_preprocess_data(
    input_path="path/to/input/data",
    output_dir="path/to/output/dir",
    tokenizer_name_or_path="hf_model_name",
    json_keys=["name of json key(s) to tokenize"],
)
```
"""

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import requests
from datasets import load_dataset
from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer


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

    def _print_processing_stats(self, count: int, total_doc_len: int, total_enc_len: int):
        if count % self.log_interval == 0:
            print(
                f"Processed {count} documents, {total_doc_len} chars, {total_enc_len} tokens",
                file=sys.stderr,
            )

    def process_json_file(
        self, input_file_name: str | Path, output_dir: str | Path, encoder: _Encoder
    ):
        output_prefix = Path(output_dir) / Path(input_file_name).stem

        print("Opening", input_file_name)
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
            print(f"Output files corresponding to {input_file_name} already exist, skipping")
            return 0

        total_doc_len, total_enc_len, final_enc_len = 0, 0, 0
        for i, (doc, sentence_lens, (doc_len, enc_len)) in enumerate(encoded_docs, start=1):
            total_doc_len += doc_len
            total_enc_len += enc_len
            final_enc_len += sum(sentence_lens[key])
            for key in doc:
                builders[key].add_document(doc[key], sentence_lens[key])
            self._print_processing_stats(i, total_doc_len, total_enc_len)

        fin.close()
        for key in builders:
            builders[key].finalize(output_idx_files[key])

        return final_enc_len


def megatron_preprocess_data(
    input_path: str | Path | list[str] | list[Path],
    output_dir: str | Path,
    tokenizer_name_or_path: str,
    json_keys: list[str] = ["text"],
    append_eod: bool = False,
    max_sequence_length: int | None = None,
    workers: int = 1,
    log_interval: int = 1000,
):
    """Process large data for pretraining.

    Args:
        input_path (str | Path | list): Path to file or directory
            containing input JSONL files, or list of paths to JSONL files
        output_dir (str | Path): Path to directory to save binary output files
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer to use
        json_keys (list, optional): List of keys to extract from json. Defaults to ["text"]
        append_eod (bool, optional): Append an <eod> token to the end of a document. Defaults to False
        max_sequence_length (int, optional): Maximum tokenized sequence length. Defaults to None
        workers (int, optional): Number of worker processes to launch. Defaults to 1
        log_interval (int, optional): Interval between progress updates. Defaults to 1000
    """
    if isinstance(input_path, list):
        file_names = input_path
    elif Path(input_path).is_file():
        file_names = [input_path]
    else:
        file_names = sorted(Path(input_path).glob("*.jsonl"))
        if not file_names:
            raise ValueError(f"No JSONL files found in input path: {input_path}")

    Path(output_dir).mkdir(exist_ok=True)
    vocab_size = AutoTokenizer.from_pretrained(tokenizer_name_or_path).vocab_size

    encoder = _Encoder(tokenizer_name_or_path, json_keys, append_eod, max_sequence_length)
    partition = _Partition(vocab_size, json_keys, log_interval, workers)

    final_enc_len = 0
    for name in file_names:
        num_tokens = partition.process_json_file(name, output_dir, encoder)
        final_enc_len += num_tokens

    print(f">>> Total number of tokens: {final_enc_len}")


def main():
    """Sample main function to process large data for pretraining.

    Example usage:

    >>> python megatron_preprocess_data.py \
            --dataset "nvidia/Nemotron-Pretraining-Dataset-sample" \
            --tokenizer "meta-llama/Llama-3.2-1B-Instruct" \
            --output_dir "./processed_data"
    """
    parser = argparse.ArgumentParser(prog="megatron_preprocess_data")
    parser.add_argument("--input_path", type=str, default=None, help="Input path.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nvidia/Nemotron-Pretraining-Dataset-sample",
        help="Hugging Face Hub dataset name or path",
    )
    parser.add_argument("--subset", type=str, default=None, help="Hugging Face Hub dataset subset")
    parser.add_argument("--split", type=str, default="train", help="Hugging Face Hub dataset split")
    parser.add_argument(
        "--output_dir", type=str, default="./processed_data", help="Output directory"
    )
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name or path")
    parser.add_argument("--json_keys", nargs="+", default=["text"], help="JSON keys to tokenize")
    parser.add_argument("--append_eod", action="store_true", help="Append <eod> token")
    parser.add_argument(
        "--max_sequence_length", type=int, default=None, help="Maximum sequence length"
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval")
    args = parser.parse_args()

    if args.input_path is None:
        args.input_path = []

        try:
            response = requests.get(
                f"https://datasets-server.huggingface.co/splits?dataset={args.dataset}",
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch dataset splits for {args.dataset}: {e}")
            return

        for entry in response.json()["splits"]:
            skip_processing = False
            name = entry["dataset"]
            subset = entry.get("config", None)
            split = entry["split"]

            if args.subset is not None and args.subset != subset:
                skip_processing = True
            if args.split is not None and args.split != split:
                skip_processing = True

            print(f"Loading dataset {name} with subset {subset} and split {split}")
            dataset = load_dataset(name, subset, split=split)

            for key in args.json_keys:
                if key not in dataset.features:
                    print(f"Key {key} not found in dataset features. Skipping...")
                    skip_processing = True
                    break

            if skip_processing:
                continue

            json_file_path = args.output_dir + "/" + name + "_" + subset + "_" + split + ".jsonl"
            dataset.to_json(json_file_path)
            args.input_path += [json_file_path]

    megatron_preprocess_data(
        input_path=args.input_path,
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
