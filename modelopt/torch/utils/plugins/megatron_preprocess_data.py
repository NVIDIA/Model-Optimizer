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

"""Processing large data pretraining and post-training datasets to tokenize for usage in megatron pretraining scripts.

We apply chat_template to the data if the JSON key is a list of message dicts (e.g. Nemotron-Post-Training-Dataset-v2)
so that we can tokenize the data for usage in megatron pretraining scripts.

Usage to tokenize one or more JSONL files (pretraining, ``text`` key):

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

Usage to tokenize a post-training dataset with ``messages`` key (chat format):

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths path/to/sft_data.jsonl \
    --json_keys messages \
    --output_dir /path/to/tokenized/Qwen3/ \
    --tokenizer Qwen/Qwen3-0.6B
```

When the value for a JSON key is a list of message dicts (e.g.
``[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]``),
``tokenizer.apply_chat_template`` is automatically used to render the conversation
into a single text string before tokenization.

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
from pathlib import Path

from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer

from modelopt.torch.utils import num2hrb
from modelopt.torch.utils.dataset_utils import download_hf_dataset_as_jsonl

__all__ = ["megatron_preprocess_data"]


class _Encoder:
    tokenizer: AutoTokenizer = None
    _chat_template_logged: set[str] = set()

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
            value = data[key]

            if isinstance(value, list):
                if key not in _Encoder._chat_template_logged:
                    _Encoder._chat_template_logged.add(key)
                    print(f"Applying chat_template to '{key}' key")
                kwargs = {}
                tools = data.get("tools")
                if tools:
                    kwargs["tools"] = tools
                text = _Encoder.tokenizer.apply_chat_template(value, tokenize=False, **kwargs)
            else:
                text = value

            # Truncate text by character length if specified
            if self.max_document_length is not None:
                original_length = len(text)
                text = text[: self.max_document_length]
                if original_length != len(text):
                    print(f"Document truncated from {original_length} to {len(text)} characters")
            doc_len += len(text)

            # Tokenize the entire text as one document
            encoded = _Encoder.tokenizer.encode(text)

            if self.max_sequence_length is not None:
                encoded = encoded[: self.max_sequence_length]
                # print(f"Sequence truncated from {original_length} to {self.max_sequence_length} tokens")
            enc_len += len(encoded)

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
                f"\tProcessed {num2hrb(count)} docs = {num2hrb(total_doc_len)} chars = {num2hrb(total_enc_len)} tokens",
                flush=True,
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
    json_keys: str | list[str] = ["text"],
    append_eod: bool = False,
    max_sequence_length: int | None = None,
    workers: int = 1,
    log_interval: int = 100000,
):
    """Process large data for pretraining.

    Exactly one of ``input_dir``, ``jsonl_paths``, or ``hf_dataset`` must be provided.

    Args:
        input_dir: Directory containing JSONL files to tokenize.
        jsonl_paths: One or more paths to JSONL files.
        hf_dataset: Hugging Face Hub dataset name or path to download and tokenize.
        hf_name: Hugging Face Hub dataset subset name. Downloads all subsets if None.
        hf_split: Hugging Face Hub dataset split. Defaults to "train".
        hf_max_samples_per_split: Maximum number of samples to download per split from Hugging Face Hub.
            Skip to download all samples.
        output_dir: Path to directory to save binary output files.
        tokenizer_name_or_path: Name or path of the Hugging Face tokenizer to use.
        json_keys: Key or list of keys to extract from json. Defaults to ["text"].
        append_eod: Append an <eod> token to the end of a document. Defaults to False.
        max_sequence_length: Maximum tokenized sequence length. Defaults to None.
        workers: Number of worker processes to launch. Defaults to 1.
        log_interval: Interval between progress updates. Defaults to 100000.
    """
    if isinstance(json_keys, str):
        json_keys = [json_keys]
    num_sources = sum(x is not None for x in (input_dir, jsonl_paths, hf_dataset))
    if num_sources != 1:
        raise ValueError(
            "Exactly one of `input_dir`, `jsonl_paths`, or `hf_dataset` must be provided."
        )

    if hf_dataset is not None:
        jsonl_paths = download_hf_dataset_as_jsonl(
            hf_dataset,
            f"{output_dir}/raw",
            json_keys,
            name=hf_name,
            split=hf_split,
            max_samples_per_split=hf_max_samples_per_split,
            num_proc=workers,
        )
        print(f"\n\nTokenizing downloaded JSONL files: {jsonl_paths}\n")

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

    print(f"\n\n>>> Total number of tokens currently processed: {num2hrb(final_enc_len)}\nDone!")


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
