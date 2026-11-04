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

"""Prepare a unique, conversation-disjoint native AutoModel packed corpus."""

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

_PARTITIONS = ("calibration", "validation", "diagnostic", "train")
_FIELDS = ("input_ids", "document_ids", "position_ids", "labels")


def _partition(identity, seed):
    value = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16], 16) % 10000
    return (
        "calibration"
        if value < 400
        else "validation"
        if value < 700
        else "diagnostic"
        if value < 1000
        else "train"
    )


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_packed_corpus(
    conversations,
    output,
    *,
    sequence_length,
    pack_counts,
    seed=42,
    padding_idx=0,
    source_metadata=None,
    calibration_sequence_length=None,
):
    """Pack each source token once, retaining native document/position boundaries.

    ``conversations`` yields (stable conversation ID, token IDs). Long documents
    may continue in the next pack; their token intervals never repeat and every
    fragment of a conversation belongs to the same split. Split assignment does
    not depend on the requested sizes, so longer training budgets share a prefix.
    """
    from nemo_automodel.components.datasets.llm.neat_packing import _build_packed_sample

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Packed corpus output must be a new directory.")
    if (
        sequence_length <= 0
        or set(pack_counts) != set(_PARTITIONS)
        or any(count <= 0 for count in pack_counts.values())
    ):
        raise ValueError("Provide a positive sequence length and positive counts for every split.")
    split_lengths = dict.fromkeys(pack_counts, sequence_length)
    if calibration_sequence_length is not None:
        if not 0 < calibration_sequence_length <= sequence_length:
            raise ValueError(
                "Calibration context must be positive and no longer than training context."
            )
        split_lengths["calibration"] = calibration_sequence_length
    pending = output.parent / ("." + output.name + "-" + uuid.uuid4().hex)
    pending.mkdir(parents=True)
    arrays, streams, written, fragments, samples, lengths = {}, {}, {}, {}, {}, {}
    seen = set()
    duplicates = 0
    try:
        for split, count in pack_counts.items():
            directory = pending / split
            directory.mkdir()
            arrays[split] = {
                field: np.lib.format.open_memmap(
                    directory / f"{field}.npy",
                    mode="w+",
                    dtype=np.int32,
                    shape=(count, split_lengths[split]),
                )
                for field in _FIELDS
            }
            streams[split] = (directory / "provenance.jsonl").open("w")
            written[split], fragments[split], samples[split], lengths[split] = 0, [], [], 0
        for identity, tokens in conversations:
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            split = _partition(identity, seed)
            if written[split] == pack_counts[split] or not tokens:
                continue
            start = 0
            while start < len(tokens) and written[split] < pack_counts[split]:
                stop = min(len(tokens), start + split_lengths[split] - lengths[split])
                chunk = tokens[start:stop]
                labels = tokens[start + 1 : stop + 1]
                if len(labels) < len(chunk):
                    labels = [*labels, -100]
                samples[split].append({"input_ids": chunk, "labels": labels})
                fragments[split].append({"conversation_id": identity, "start": start, "stop": stop})
                lengths[split] += len(chunk)
                start = stop
                if lengths[split] == split_lengths[split]:
                    pack = _build_packed_sample(samples[split], split_lengths[split], padding_idx)
                    values = {**pack, "document_ids": pack["attention_mask"]}
                    index = written[split]
                    for field in _FIELDS:
                        arrays[split][field][index] = values[field].numpy()
                    streams[split].write(
                        json.dumps({"pack": index, "fragments": fragments[split]}) + "\n"
                    )
                    written[split] += 1
                    if written[split] == 1 or written[split] % 128 == 0:
                        print(
                            json.dumps({"packing": written.copy(), "active_split": split}),
                            flush=True,
                        )
                    fragments[split], samples[split], lengths[split] = [], [], 0
            if written == pack_counts:
                break
        if written != pack_counts:
            raise ValueError(
                f"Unique source data exhausted: prepared {written}, requested {pack_counts}; supply more conversations."
            )
        for split in pack_counts:
            for array in arrays[split].values():
                array.flush()
                array._mmap.close()
            streams[split].close()
        manifest = {
            "version": 1,
            "packing": "nemo_neat_document_ids",
            "sequence_length": sequence_length,
            "seed": seed,
            "source": source_metadata or {},
            "duplicates_skipped": duplicates,
            "partitions": {
                split: {
                    "packs": count,
                    "sequence_length": split_lengths[split],
                    "valid_tokens": count * split_lengths[split],
                    "files": {path.name: _hash(path) for path in (pending / split).iterdir()},
                }
                for split, count in pack_counts.items()
            },
        }
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(pending, output)
        return manifest
    except BaseException:
        for stream in streams.values():
            stream.close()
        for fields in arrays.values():
            for array in fields.values():
                if not array._mmap.closed:
                    array._mmap.close()
        shutil.rmtree(pending)
        raise


class PackedBatchSequence(Sequence):
    """Deterministic update batches, shared over CP/TP/EP and sliced only over DP."""

    def __init__(
        self,
        directory,
        partition,
        *,
        global_batch_size=8,
        dp_rank=0,
        dp_size=1,
        device="cpu",
        sequence_length=None,
    ):
        """Open immutable packed rows without sampler padding or epoch repetition."""
        root = Path(directory)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["version"] != 1 or self.manifest["packing"] != "nemo_neat_document_ids":
            raise ValueError("Unsupported packed-corpus contract.")
        self.fingerprint = _hash(root / "manifest.json")
        self.size = self.manifest["partitions"][partition]["packs"]
        self.global_batch_size = global_batch_size
        self.local_batch, remainder = divmod(global_batch_size, dp_size)
        if (
            remainder
            or not self.local_batch
            or self.size % global_batch_size
            or not 0 <= dp_rank < dp_size
        ):
            raise ValueError(
                "Batch/split counts must divide exactly over sample-DP lanes; rows are never repeated."
            )
        self.dp_rank, self.device = dp_rank, device
        prepared_length = self.manifest["partitions"][partition].get(
            "sequence_length", self.manifest["sequence_length"]
        )
        self.sequence_length = sequence_length or prepared_length
        if not 0 < self.sequence_length <= prepared_length:
            raise ValueError("Requested context exceeds the prepared corpus.")
        self.arrays = {
            field: np.load(root / partition / f"{field}.npy", mmap_mode="r") for field in _FIELDS
        }

    def __len__(self):
        return self.size // self.global_batch_size

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.global_batch_size + self.dp_rank * self.local_batch
        stop = start + self.local_batch
        tensors = {
            field: torch.tensor(
                array[start:stop, : self.sequence_length].copy(),
                dtype=torch.long,
                device=self.device,
            )
            for field, array in self.arrays.items()
        }
        docs = tensors.pop("document_ids")
        return {
            **tensors,
            "attention_mask": docs,
            "_packed_seq_ids": docs,
            "valid_mask": docs > 0,
            "sample_ids": torch.arange(start, stop, device=self.device),
        }


def _conversations(args, tokenizer, revision, *, native_chat_tokenization=False):
    from datasets import load_dataset

    from modelopt.torch.utils.dataset_utils import prepare_messages_for_chat_template

    streams = [
        iter(
            load_dataset(args.dataset, split=split, revision=revision, streaming=True).shuffle(
                seed=args.seed + index, buffer_size=10000
            )
        )
        for index, split in enumerate(args.splits)
    ]
    count = 0
    while streams and count < args.max_conversations:
        remaining = []
        for stream in streams:
            if count >= args.max_conversations:
                break
            try:
                row = next(stream)
            except StopIteration:
                continue
            remaining.append(stream)
            count += 1
            if count == 1 or count % 10000 == 0:
                print(json.dumps({"source_conversations": count}), flush=True)
            messages = prepare_messages_for_chat_template(
                row["messages"], reasoning_content="native"
            )
            tools = row.get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                **({"tools": tools} if tools else {}),
            )
            identity = hashlib.sha256(text.encode()).hexdigest()
            # Custom tokenizers may distinguish structural tokens from user text
            # inside their native chat encoder. Preserve that segmentation.
            tokens = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    **({"tools": tools} if tools else {}),
                )
                if native_chat_tokenization
                else tokenizer.encode(text, add_special_tokens=False)
            )
            yield identity, tokens
        streams = remaining


def main():
    """Resolve source revisions and prepare the GPTQ/repair/held-out splits once."""
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    from modelopt.torch.utils.dataset_utils import SUPPORTED_DATASET_CONFIG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument(
        "--dataset",
        default=SUPPORTED_DATASET_CONFIG["nemotron-post-training-dataset-v2"]["config"]["path"],
    )
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--splits", nargs="+", default=["stem", "chat", "math", "code"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--calibration-sequence-length", type=int)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--calibration-packs", type=int, default=128)
    parser.add_argument("--validation-packs", type=int, default=128)
    parser.add_argument("--diagnostic-packs", type=int, default=128)
    parser.add_argument("--max-conversations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
    tokenizer_revision = (
        HfApi().model_info(args.tokenizer, revision=args.tokenizer_revision).sha
        if not Path(args.tokenizer).is_dir()
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=tokenizer_revision, trust_remote_code=args.trust_remote_code
    )
    tokenizer_files = {}
    if tokenizer_revision is None:
        directory = Path(args.tokenizer)
        files = {
            *directory.glob("tokenizer*"),
            *directory.glob("*vocab*"),
            *directory.glob("merges.txt"),
            *directory.glob("special_tokens_map.json"),
            *directory.glob("*.jinja"),
            *directory.glob("chat_templates/*.jinja"),
        }
        tokenizer_files = {
            str(path.relative_to(directory)): _hash(path)
            for path in sorted(files)
            if path.is_file()
        }
    manifest = prepare_packed_corpus(
        _conversations(args, tokenizer, revision),
        args.output,
        sequence_length=args.sequence_length,
        pack_counts={
            "train": args.updates * args.global_batch_size,
            "calibration": args.calibration_packs,
            "validation": args.validation_packs,
            "diagnostic": args.diagnostic_packs,
        },
        seed=args.seed,
        padding_idx=tokenizer.pad_token_id or 0,
        source_metadata={
            "dataset": args.dataset,
            "revision": revision,
            "splits": args.splits,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_files": tokenizer_files,
        },
        calibration_sequence_length=args.calibration_sequence_length,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
