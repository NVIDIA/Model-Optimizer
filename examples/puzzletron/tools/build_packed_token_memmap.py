#!/usr/bin/env python3
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

"""Build a deterministic fixed-sequence token memmap with disjoint CPU workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from modelopt.torch.puzzletron.utils.data.dataset import render_messages_to_text


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    worker = int(task["worker"])
    workers = int(task["workers"])
    start = int(task["start"])
    stop = int(task["stop"])
    seq_length = int(task["seq_length"])
    loaded = load_from_disk(task["dataset_path"])
    if isinstance(loaded, DatasetDict):
        loaded = loaded[task["split"]]
    loaded = loaded.shuffle(seed=int(task["shuffle_seed"]))
    shard = loaded.shard(num_shards=workers, index=worker, contiguous=True)
    tokenizer = AutoTokenizer.from_pretrained(
        task["tokenizer_path"],
        trust_remote_code=bool(task["trust_remote_code"]),
    )
    eos = int(tokenizer.eos_token_id)

    output = np.memmap(
        task["output"],
        dtype=np.uint32,
        mode="r+",
        shape=(int(task["num_samples"]), seq_length + 1),
    )
    progress_dir = Path(task["progress_dir"])
    buffer: list[int] = []
    consumed = 0
    tokenized: list[list[int]] = []
    row_index = start

    def consume(sequences) -> None:
        nonlocal buffer, consumed, row_index
        for sequence in sequences:
            if row_index == stop:
                break
            buffer.extend(sequence[:200_000])
            buffer.append(eos)
            while row_index < stop and len(buffer) - consumed >= seq_length + 1:
                output[row_index, :] = np.asarray(
                    buffer[consumed : consumed + seq_length + 1],
                    dtype=np.uint32,
                )
                consumed += seq_length
                row_index += 1
                if consumed >= 8 * seq_length:
                    buffer = buffer[consumed:]
                    consumed = 0
                if (row_index - start) % 16 == 0 or row_index == stop:
                    _atomic_json(
                        progress_dir / f"worker_{worker:04d}.json",
                        {
                            "worker": worker,
                            "rows_complete": row_index - start,
                            "rows_total": stop - start,
                        },
                    )

    batch_size = int(task["tokenize_batch_size"])
    for offset in range(0, len(shard), batch_size):
        batch = shard[offset : offset + batch_size][task["content_field"]]
        texts = [render_messages_to_text(messages, tokenizer) for messages in batch]
        sequences = tokenizer(texts, truncation=False)["input_ids"]
        tokenized.extend(sequences)
        consume(sequences)
        if row_index == stop:
            break
    if not tokenized:
        raise RuntimeError(f"worker {worker} received an empty dataset shard")
    while row_index < stop:
        consume(tokenized)
        if len(buffer) - consumed < seq_length + 1 and not any(tokenized):
            raise RuntimeError(f"worker {worker} cannot make progress packing tokens")
    output.flush()
    return {"worker": worker, "rows": stop - start, "examples": len(tokenized)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--content-field", default="messages")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--tokenize-batch-size", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=444)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    for option, value in (
        ("--num-samples", args.num_samples),
        ("--seq-length", args.seq_length),
        ("--workers", args.workers),
        ("--tokenize-batch-size", args.tokenize_batch_size),
    ):
        if value <= 0:
            parser.error(f"{option} must be positive")

    output = args.output.resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    expected_bytes = args.num_samples * (args.seq_length + 1) * np.dtype(np.uint32).itemsize
    workers = min(int(args.workers), int(args.num_samples))
    metadata = {
        "status": "complete",
        "version": 1,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "split": args.split,
        "content_field": args.content_field,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "shuffle_seed": args.shuffle_seed,
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": "uint32",
        "bytes": expected_bytes,
        "workers": workers,
    }
    try:
        existing_metadata = json.loads(metadata_path.read_text())
        reusable = (
            existing_metadata == metadata
            and output.is_file()
            and output.stat().st_size == expected_bytes
        )
    except (OSError, ValueError):
        reusable = False
    if reusable:
        print(json.dumps({"status": "reused", "output": str(output), **metadata}, indent=2))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    progress_dir = output.parent / f".{output.name}.progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.truncate(expected_bytes)

    tasks = []
    for worker in range(workers):
        start = args.num_samples * worker // workers
        stop = args.num_samples * (worker + 1) // workers
        tasks.append(
            {
                **vars(args),
                "output": str(output),
                "worker": worker,
                "workers": workers,
                "start": start,
                "stop": stop,
                "progress_dir": str(progress_dir),
            }
        )
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))
        for result in results:
            print(json.dumps(result, sort_keys=True), flush=True)
    if sum(int(result["rows"]) for result in results) != args.num_samples:
        raise RuntimeError("packed-token workers did not cover every requested row")
    _atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
