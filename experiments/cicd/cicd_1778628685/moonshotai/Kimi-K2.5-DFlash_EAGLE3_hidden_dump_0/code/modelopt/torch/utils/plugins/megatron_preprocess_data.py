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

"""Tokenize pretraining and post-training datasets into Megatron's binary indexed format.

When the value for a JSON key is a list of message dicts, ``tokenizer.apply_chat_template``
is automatically used (``add_special_tokens=False`` to avoid a duplicate BOS). Plain-text
values are tokenized directly.

**Tokenize JSONL files — pretraining text** (use ``--append_eod`` so Megatron knows document
boundaries when concatenating sequences; use ``--strip_newlines`` for prose/web text):

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths path/to/data1.jsonl path/to/data2.jsonl \
    --json_keys text \
    --output_dir /path/to/tokenized/Qwen3/ \
    --tokenizer Qwen/Qwen3-0.6B \
    --append_eod \
    --strip_newlines
```

**Tokenize JSONL files — post-training chat data** (omit ``--append_eod``; chat template adds EOS):

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths path/to/sft_data.jsonl \
    --json_keys messages \
    --output_dir /path/to/tokenized/Qwen3/ \
    --tokenizer Qwen/Qwen3-0.6B
```

Pass ``--input_dir /path/to/dir`` to tokenize all ``.jsonl`` / ``.jsonl.gz`` files in a directory.

**Download and tokenize from Hugging Face Hub:**

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Pretraining-SFT-v1 \
    --hf_name Nemotron-SFT-General \
    --hf_split train \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir /path/to/tokenized/Qwen3/ \
    --append_eod \
    --strip_newlines
```

Omit ``--hf_name`` to process all subsets; omit ``--hf_split`` for all splits. When ``--hf_max_samples_per_split``
is set, the dataset is automatically shuffled to avoid biased sampling from the prefix.

**Large datasets — streaming mode** (only consumed rows downloaded, no disk cache):

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-CC-v2.1 \
    --hf_name High-Quality \
    --hf_max_samples_per_split 5000000 \
    --hf_streaming \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir /path/to/tokenized/Qwen3/ \
    --append_eod \
    --strip_newlines
```

Note: streaming does not cache to disk, so re-runs re-download. For full-dataset streaming
without a sample cap this is slower than non-streaming mode, but it avoids Arrow schema
compatibility issues with complex nested message types.
"""

import argparse
import gzip
import json
import multiprocessing
import time
import warnings
from pathlib import Path

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer

from modelopt.torch.utils import num2hrb

__all__ = ["megatron_preprocess_data"]


def _is_main_or_first_worker() -> bool:
    """Return True only for the main process or the first pool worker.

    ``multiprocessing.current_process()._identity`` is ``()`` in the main process
    and ``(N,)`` in the N-th pool worker.  Gating noisy prints on this prevents
    the same message from appearing once per worker when using many workers.
    """
    identity = multiprocessing.current_process()._identity
    return not identity or identity[0] == 1


class _Encoder:
    tokenizer: AutoTokenizer = None
    _chat_template_logged: set[str] = set()

    def __init__(
        self,
        tokenizer_name_or_path: str,
        json_keys: list[str],
        append_eod: bool,
        max_sequence_length: int | None,
        reasoning_content: str = "strip",
        strip_newlines: bool = False,
    ):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.json_keys = json_keys
        self.append_eod = append_eod
        self.max_sequence_length = max_sequence_length
        self.max_document_length = (
            max_sequence_length * 8 if max_sequence_length is not None else None
        )
        self.reasoning_content = reasoning_content
        self.strip_newlines = strip_newlines
        print(f"Setting max document length: {self.max_document_length}")
        print(f"reasoning_content mode: {self.reasoning_content}")

    def initializer(self):
        # Use Encoder class as a container for global data
        _Encoder.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        # Suppress "Token indices sequence length is longer than model_max_length" warnings.
        # model_max_length is the model's inference limit and irrelevant here — we are only
        # converting text to token IDs, not running a forward pass. Megatron splits documents
        # into fixed-length training sequences at training time.
        _Encoder.tokenizer.model_max_length = int(1e30)

    def _process_messages(self, messages: list[dict]) -> list[dict]:
        """Handle reasoning_content field in v3 Nemotron datasets.

        Nemotron Post-Training v3 datasets include a ``reasoning_content`` field
        in assistant messages alongside ``content``. Behaviour is controlled by
        ``self.reasoning_content``:

        - ``"strip"``  (default): remove the field before calling apply_chat_template.
          Safe for any tokenizer; reasoning traces are discarded.
        - ``"inline"``: prepend ``<think>…</think>`` to ``content`` then strip the
          field. Preserves the reasoning trace in a tokenizer-agnostic way.
        - ``"native"``: pass messages unchanged; the tokenizer's chat template must
          handle ``reasoning_content`` itself (e.g. Qwen3).
        """
        if self.reasoning_content == "native":
            return messages
        processed = []
        for msg in messages:
            if "reasoning_content" not in msg:
                processed.append(msg)
                continue
            msg = dict(msg)  # shallow copy — don't mutate the original
            rc = msg.pop("reasoning_content")
            if self.reasoning_content == "inline" and rc:
                msg["content"] = f"<think>\n{rc}\n</think>\n{msg.get('content', '')}"
            processed.append(msg)
        return processed

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
                    if _is_main_or_first_worker():
                        print(f"Applying chat_template to '{key}' key")
                kwargs = {}
                tools = data.get("tools")
                if tools:
                    kwargs["tools"] = tools
                value = self._process_messages(value)
                try:
                    text = _Encoder.tokenizer.apply_chat_template(value, tokenize=False, **kwargs)
                except Exception as e:
                    print(
                        f"apply_chat_template failed: {e}\nData:\n{json.dumps(data, indent=2, default=str)}",
                        flush=True,
                    )
                    raise
                # chat template already embeds all special tokens; don't add BOS again
                add_special_tokens = False
            else:
                text = value.replace("\n", " ") if self.strip_newlines else value
                add_special_tokens = True

            # Truncate text by character length if specified
            if self.max_document_length is not None:
                original_length = len(text)
                text = text[: self.max_document_length]
                if original_length != len(text) and _is_main_or_first_worker():
                    print(f"Document truncated from {original_length} to {len(text)} characters")
            doc_len += len(text)

            # Tokenize the entire text as one document
            encoded = _Encoder.tokenizer.encode(text, add_special_tokens=add_special_tokens)

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
        self,
        count: int,
        total_doc_len: int,
        total_enc_len: int,
        start_time: float,
        *,
        force_print: bool = False,
    ):
        if count % self.log_interval == 0 or force_print:
            elapsed = (time.time() - start_time) / 60
            print(
                f"\tProcessed {num2hrb(count)} docs = {num2hrb(total_doc_len)} chars"
                f" = {num2hrb(total_enc_len)} tokens ({elapsed:.1f}mins)",
                flush=True,
            )

    def process_json_file(
        self, input_file_name: str | Path, output_dir: str | Path, encoder: _Encoder
    ) -> tuple[int, list[str]]:
        input_path = Path(input_file_name)
        stem = input_path.stem if input_path.suffix != ".gz" else Path(input_path.stem).stem
        output_prefix = Path(output_dir) / stem
        prefixes = [f"{output_prefix}_{key}" for key in self.json_keys]

        print(f"\nOpening {input_file_name}")
        if input_path.suffix == ".gz":
            fin = gzip.open(input_path, "rt", encoding="utf-8")
        else:
            fin = open(input_path, encoding="utf-8")

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.json_keys:
            output_bin_files[key] = f"{output_prefix}_{key}.bin"
            output_idx_files[key] = f"{output_prefix}_{key}.idx"
            if Path(output_bin_files[key]).exists() and Path(output_idx_files[key]).exists():
                continue
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(self.vocab_size),
            )

        if not builders:
            print(f"\t[SKIP] Output files corresponding to {input_file_name} already exist")
            return 0, prefixes

        start_time = time.time()
        total_doc_len, total_enc_len, final_enc_len = 0, 0, 0
        for i, (doc, sentence_lens, (doc_len, enc_len)) in enumerate(encoded_docs, start=1):
            total_doc_len += doc_len
            total_enc_len += enc_len
            final_enc_len += sum(sum(sentence_lens[k]) for k in sentence_lens)
            for key in doc:
                builders[key].add_document(doc[key], sentence_lens[key])
            self._print_processing_stats(i, total_doc_len, total_enc_len, start_time)
        self._print_processing_stats(i, total_doc_len, total_enc_len, start_time, force_print=True)

        fin.close()
        for key in builders:
            builders[key].finalize(output_idx_files[key])

        return final_enc_len, prefixes

    @staticmethod
    def _iter_hf_as_json(ds):
        """Yield each row of a HF Dataset as a JSON string, matching JSONL line format."""
        for row in ds:
            yield json.dumps(dict(row))

    def process_hf_split(
        self,
        output_dir: str | Path,
        encoder: "_Encoder",
        dataset_name: str,
        config: str | None,
        split: str,
        max_samples: int | None = None,
        streaming: bool = False,
    ) -> tuple[int, list[str]]:
        """Load a HF dataset split and tokenize directly without writing an intermediate JSONL.

        When ``streaming=True``, only consumed rows are downloaded — useful for large pretraining
        datasets where downloading the full split is impractical. Note that streaming mode does not
        cache to disk, so re-runs will re-download the data.

        When ``max_samples`` is set, the dataset is shuffled to avoid sampling from a biased prefix of the dataset.
        """
        print(f"\nLoading HF dataset {dataset_name=}, {config=}, {split=}, {streaming=}")
        ds = load_dataset(path=dataset_name, name=config, split=split, streaming=streaming)
        if max_samples is not None:
            # Shuffle first so the selected subset is random, not a biased prefix.
            # Non-streaming: global index shuffle (memory-mapped, efficient) then .select(N).
            # Streaming: buffer shuffle (approximate) then .take(N).
            ds = ds.shuffle(seed=42)
            if streaming:
                ds = ds.take(max_samples)
            else:
                ds = ds.select(range(max_samples))

        # features are available from dataset metadata without downloading data
        features = ds.features if ds.features is not None else {}
        if features:
            for key in self.json_keys:
                if key not in features:
                    raise KeyError(
                        f"{key=} not found in dataset features. Available: {list(features)}"
                    )

        safe_name = dataset_name.replace("/", "--")
        sample_tag = f"_max{max_samples}" if max_samples is not None else ""
        output_prefix = Path(output_dir) / f"{safe_name}_{config}_{split}"

        prefixes = []
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in self.json_keys:
            prefix = f"{output_prefix}_{key}{sample_tag}"
            prefixes.append(prefix)
            output_bin_files[key] = f"{prefix}.bin"
            output_idx_files[key] = f"{prefix}.idx"
            if Path(output_bin_files[key]).exists() and Path(output_idx_files[key]).exists():
                continue
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(self.vocab_size),
            )

        if not builders:
            print(f"\t[SKIP] Output files for {dataset_name} {config}/{split} already exist")
            return 0, prefixes

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, self._iter_hf_as_json(ds), 32)

        start_time = time.time()
        total_doc_len, total_enc_len, final_enc_len = 0, 0, 0
        i = 0
        for i, (doc, sentence_lens, (doc_len, enc_len)) in enumerate(encoded_docs, start=1):
            total_doc_len += doc_len
            total_enc_len += enc_len
            final_enc_len += sum(sum(sentence_lens[key]) for key in sentence_lens)
            for key in doc:
                builders[key].add_document(doc[key], sentence_lens[key])
            self._print_processing_stats(i, total_doc_len, total_enc_len, start_time)

        if i:
            self._print_processing_stats(
                i, total_doc_len, total_enc_len, start_time, force_print=True
            )

        pool.close()
        pool.join()
        for key in builders:
            builders[key].finalize(output_idx_files[key])

        return final_enc_len, prefixes


def _enumerate_hf_splits(
    dataset_name: str,
    name: str | None,
    split: str | None,
) -> list[tuple[str | None, str]]:
    """Return (config, split) pairs to process for a HuggingFace dataset."""
    configs: list[str | None]
    if name is not None:
        configs = [name]
    else:
        try:
            configs = get_dataset_config_names(dataset_name)
        except (FileNotFoundError, ConnectionError) as e:
            print(f"[WARN] Could not find configs for dataset '{dataset_name}': {e}")
            configs = [None]

    result: list[tuple[str | None, str]] = []
    for config in configs:
        if split is not None:
            result.append((config, split))
        else:
            result.extend((config, s) for s in get_dataset_split_names(dataset_name, config))
    return result


def megatron_preprocess_data(
    *,
    input_dir: str | Path | None = None,
    jsonl_paths: str | Path | list[str] | list[Path] | None = None,
    # Hugging Face Hub dataset arguments
    hf_dataset: str | None = None,
    hf_name: str | None = None,
    hf_split: str | None = None,
    hf_max_samples_per_split: int | None = None,
    hf_streaming: bool = False,
    # Other arguments
    output_dir: str | Path,
    tokenizer_name_or_path: str,
    json_keys: str | list[str] = ["text"],
    append_eod: bool = False,
    max_sequence_length: int | None = None,
    workers: int = 1,
    log_interval: int = 100000,
    reasoning_content: str = "strip",
    strip_newlines: bool = False,
):
    """Process large data for pretraining.

    Exactly one of ``input_dir``, ``jsonl_paths``, or ``hf_dataset`` must be provided.

    Args:
        input_dir: Directory containing JSONL files to tokenize.
        jsonl_paths: One or more paths to JSONL files.
        hf_dataset: Hugging Face Hub dataset name or path to download and tokenize.
        hf_name: Hugging Face Hub dataset subset name. Downloads all subsets if None.
        hf_split: Hugging Face Hub dataset split. Defaults to None (all splits).
        hf_max_samples_per_split: Maximum number of rows to consume per split.
        hf_streaming: Load HuggingFace datasets in streaming mode. Only consumed rows are
            downloaded — useful for very large pretraining datasets or datasets with complex
            nested message schemas that cause Arrow type-cast errors in non-streaming mode.
            Note: streaming does not cache to disk, so re-runs re-download. Defaults to False.
        output_dir: Path to directory to save binary output files.
        tokenizer_name_or_path: Name or path of the Hugging Face tokenizer to use.
        json_keys: Key or list of keys to extract from json. Defaults to ["text"].
        append_eod: Append an <eod> token to the end of a document. Defaults to False.
        max_sequence_length: Maximum tokenized sequence length. Defaults to None.
        workers: Number of worker processes to launch. Defaults to 1.
        log_interval: Interval between progress updates. Defaults to 100000.
        reasoning_content: How to handle the ``reasoning_content`` field present in many
            Nemotron Post-Training v3 datasets. One of:
            ``"strip"`` (default) — remove before applying chat template (safe for any tokenizer);
            ``"inline"`` — wrap in ``<think>…</think>`` and prepend to ``content``;
            ``"native"`` — pass unchanged, requires the tokenizer chat template to handle it.
        strip_newlines: Replace newlines with spaces in plain-text values before tokenization.
            Defaults to False (newlines are preserved, matching prior behaviour). Has no effect
            on chat-template encoded values.

    Returns:
        List of output file prefixes (one per json_key per split/file, without ``.bin``/``.idx``
        extension) that can be used directly to build weighted ``data_paths`` argument in megatron training scripts.
    """
    if isinstance(json_keys, str):
        json_keys = [json_keys]
    num_sources = sum(x is not None for x in (input_dir, jsonl_paths, hf_dataset))
    if num_sources != 1:
        raise ValueError(
            "Exactly one of `input_dir`, `jsonl_paths`, or `hf_dataset` must be provided."
        )
    if hf_streaming and hf_max_samples_per_split is None and _is_main_or_first_worker():
        warnings.warn(
            "--hf_streaming is set but --hf_max_samples_per_split is not. "
            "Streaming without a sample cap re-downloads the full dataset on every run with no "
            "disk cache, which is slower than the cached non-streaming path.",
            stacklevel=2,
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vocab_size = AutoTokenizer.from_pretrained(tokenizer_name_or_path).vocab_size

    encoder = _Encoder(
        tokenizer_name_or_path,
        json_keys,
        append_eod,
        max_sequence_length,
        reasoning_content,
        strip_newlines,
    )
    partition = _Partition(vocab_size, json_keys, log_interval, workers)

    final_enc_len = 0
    all_prefixes: list[str] = []
    overall_start = time.time()

    if hf_dataset is not None:
        for config, split in _enumerate_hf_splits(hf_dataset, hf_name, hf_split):
            enc_len, prefixes = partition.process_hf_split(
                output_dir,
                encoder,
                hf_dataset,
                config,
                split,
                hf_max_samples_per_split,
                hf_streaming,
            )
            final_enc_len += enc_len
            all_prefixes.extend(prefixes)
    else:
        if input_dir is not None:
            file_names = sorted(
                [*Path(input_dir).glob("*.jsonl"), *Path(input_dir).glob("*.jsonl.gz")]
            )
            if not file_names:
                raise ValueError(f"No JSONL files found in input directory: {input_dir}")
        elif isinstance(jsonl_paths, (str, Path)):
            file_names = [jsonl_paths]  # type: ignore[list-item]
        else:
            file_names = list(jsonl_paths)  # type: ignore[arg-type]

        for name in file_names:
            enc_len, prefixes = partition.process_json_file(name, output_dir, encoder)
            final_enc_len += enc_len
            all_prefixes.extend(prefixes)

    elapsed = (time.time() - overall_start) / 60
    print(
        f"\n\n>>> Total number of tokens currently processed: {num2hrb(final_enc_len)}"
        f" (time: {elapsed:.1f}mins)"
    )
    print(
        "\n>>> Output prefixes (Use to build weighted data_paths / blend in megatron training scripts):"
    )
    for prefix in all_prefixes:
        print(f"\t{prefix}")
    print("Done!")
    return all_prefixes


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
        default=None,
        help="Hugging Face Hub dataset split. Skip to download and tokenize all splits for the subset.",
    )
    parser.add_argument(
        "--hf_max_samples_per_split",
        type=int,
        default=None,
        help="Maximum number of rows to consume per split.",
    )
    parser.add_argument(
        "--hf_streaming",
        action="store_true",
        help=(
            "Load HuggingFace datasets in streaming mode. Only consumed rows are downloaded — "
            "useful for very large pretraining datasets (e.g. Nemotron-CC-v2.1). "
            "Note: streaming does not cache to disk, so re-runs will re-download the data."
        ),
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
    parser.add_argument(
        "--reasoning_content",
        type=str,
        default="strip",
        choices=["strip", "inline", "native"],
        help=(
            "How to handle the reasoning_content field in Nemotron v3 datasets. "
            "'strip': discard (default, safe for any tokenizer). "
            "'inline': wrap in <think>...</think> and prepend to content. "
            "'native': pass unchanged (requires tokenizer chat template support, e.g. Qwen3)."
        ),
    )
    parser.add_argument(
        "--strip_newlines",
        action="store_true",
        help=(
            "Replace newlines with spaces in plain-text values (non-coding pretraining data) before tokenization. "
            "Has no effect on chat-template encoded values."
        ),
    )
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
        hf_streaming=args.hf_streaming,
        output_dir=args.output_dir,
        tokenizer_name_or_path=args.tokenizer,
        json_keys=args.json_keys,
        append_eod=args.append_eod,
        max_sequence_length=args.max_sequence_length,
        workers=args.workers,
        log_interval=args.log_interval,
        reasoning_content=args.reasoning_content,
        strip_newlines=args.strip_newlines,
    )


if __name__ == "__main__":
    main()
