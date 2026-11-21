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

"""Write and read teacher targets as Megatron-LM offline KD V3 corpus sources.

Megatron-LM's offline KD V3 format (``docs/user-guide/features/offline_logits_distillation_v3.md``,
NVIDIA/Megatron-LM#9352) stores each sample's exact inputs next to the teacher's sparse top-k
log-probs, so a student trains on the cache alone. A corpus source is a directory of ``chunk-*``
directories, each holding an Arrow IPC part and an ``index.json`` written last, which publishes the
chunk.

The first section adapts Hugging Face rows and teacher configs to the format and reads sources
back. The second mirrors Megatron-LM's writer, which ModelOpt cannot import; each of its
definitions names the Megatron-LM code whose output it reproduces byte for byte.
"""

import base64
import glob
import hashlib
import json
import os
from collections import OrderedDict

import numpy as np
import pyarrow as pa
import torch

__all__ = [
    "CorpusSource",
    "chunk_dir",
    "corpus_settings",
    "publish_chunk",
    "strip_row",
    "topk_targets",
]


# ModelOpt side: Hugging Face rows in and out, and the settings the format leaves to each producer.


class CorpusSource:
    """Random access to the samples of one corpus source, e.g. a ``dump_teacher_v3.py`` split.

    The published chunks must share one settings dict, written for ``tokenizer``, and hold exactly
    ``num_samples`` samples without overlapping. A block is the unit of compression, so the last few
    decoded blocks are kept.
    """

    def __init__(self, source: str, tokenizer, cached_blocks: int = 4):
        """Index the published chunks of ``source``."""
        indexes = []
        for path in sorted(glob.glob(os.path.join(source, "chunk-*", "index.json"))):
            with open(path) as f:
                indexes.append((os.path.dirname(path), json.load(f)))
        if not indexes:
            raise FileNotFoundError(f"{source} has no published chunks.")
        self.settings = indexes[0][1]["settings"]
        contract = self.settings["contract"]
        if contract["tokenizer"]["sha256"] != _tokenizer_fingerprint(tokenizer):
            raise ValueError(f"{source} was written with another tokenizer.")
        self._blocks, rows, spans = [], [0], []
        for directory, index in indexes:
            if index["settings"] != self.settings:
                raise ValueError(f"{directory} has other settings than the rest of {source}.")
            producer = index.get("producer", {})
            if "start_iteration" in producer and "end_iteration" in producer:
                spans.append((producer["start_iteration"], producer["end_iteration"], directory))
            schemas = [
                pa.ipc.read_schema(pa.py_buffer(base64.b64decode(s))) for s in index["schemas"]
            ]
            for part in index["parts"]:
                path = os.path.join(directory, part["path"])
                for block in part["blocks"]:
                    self._blocks.append(
                        (path, schemas[part["schema"]], block["offset"], block["length"])
                    )
                    rows.append(block["rows"])
        for (_, end, first), (start, _, second) in zip(sorted(spans), sorted(spans)[1:]):
            if start < end:
                raise ValueError(f"{first} and {second} overlap.")
        self._starts = np.cumsum(rows)
        if self._starts[-1] != contract["num_samples"]:
            raise ValueError(
                f"{source} holds {self._starts[-1]} of its {contract['num_samples']} samples; "
                "finish writing it first."
            )
        self._cached_blocks = cached_blocks
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getitem__(self, i: int) -> dict:
        """Sample ``i`` as numpy arrays; ``indices`` joins the split index columns into int64."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        block = int(np.searchsorted(self._starts, i, side="right")) - 1
        batch = self._read_block(block)
        sample = {}
        for name, column in zip(batch.schema.names, batch.columns):
            values = column[i - int(self._starts[block])].values
            if pa.types.is_fixed_size_list(values.type):
                width = values.type.list_size
                values = values.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)
            else:
                values = values.to_numpy(zero_copy_only=False)
            sample[name] = values.copy()  # Arrow's buffers are read-only.
        indices = sample.pop("indices_low").astype(np.int64)
        if "indices_high" in sample:
            indices |= sample.pop("indices_high").astype(np.int64) << 16
        sample["indices"] = indices
        return sample

    def _read_block(self, block: int) -> pa.RecordBatch:
        if block in self._cache:
            self._cache.move_to_end(block)
            return self._cache[block]
        path, schema, offset, length = self._blocks[block]
        with open(path, "rb") as f:
            f.seek(offset)
            message = pa.ipc.read_message(pa.py_buffer(f.read(length)))
        batch = pa.ipc.read_record_batch(message, schema)
        self._cache[block] = batch
        if len(self._cache) > self._cached_blocks:
            self._cache.popitem(last=False)
        return batch


def strip_row(input_ids: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> dict:
    """Convert an unpadded Hugging Face row (``labels`` aligned with ``input_ids``) to stored inputs.

    Position ``t`` predicts token ``t + 1``: its label is the next token, and it carries loss when
    that token's Hugging Face label is not ``ignore_index``. Trailing positions without loss are
    dropped, since no kept position attends to them. ``positions`` lists the scored positions.
    """
    weights = torch.zeros(input_ids.numel(), dtype=torch.bool)
    weights[:-1] = labels[1:] != ignore_index
    positions = weights.nonzero().flatten()
    length = int(positions[-1]) + 1 if positions.numel() else 0
    return {
        "tokens": input_ids[:length],
        "labels": input_ids[1 : length + 1],
        "position_ids": torch.arange(length),
        "loss_mask": weights[:length],
        "segments": [length] if length else [],
        "positions": positions,
    }


def corpus_settings(
    tokenizer, teacher_config, num_samples: int, *, recorded: dict, provenance: dict
) -> dict:
    """Return the metadata shared by every file of one corpus source.

    Samples are unpacked rows of one segment each, declared ``isolated_segments`` as the format
    asks of producers outside Megatron-LM; a Megatron-LM ``--sft`` student reads them.
    ``num_samples`` is the source's row count.
    """
    # The layout of Megatron-LM's v3_runtime.py _dump_settings() for a corpus source.
    return {
        "format_version": FORMAT_VERSION,
        "contract": {
            "tokenizer": {"sha256": _tokenizer_fingerprint(tokenizer)},
            "attention_mode": "isolated_segments",
            "prefix_stable": _prefix_stable(teacher_config),
            "features": [],
            "num_samples": num_samples,
        },
        "recorded": recorded,
        "provenance": provenance,
    }


def _prefix_stable(config) -> bool:
    """Whether a retained prefix keeps the teacher's distributions (no length-dependent RoPE)."""
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    layers = [value for value in rope.values() if isinstance(value, dict)] or [rope]
    return not any(p.get("rope_type", p.get("type")) in ("dynamic", "longrope") for p in layers)


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------
# Mirrored from Megatron-LM: megatron/training/distillation/ of NVIDIA/Megatron-LM#9352 at 9757fcc6,
# which ModelOpt cannot import. Each definition names the code it mirrors; when the format changes,
# update it and recheck that the files stay byte-identical with Megatron-LM's writer.
# ------------------------------------------------------------------------------------------------

# format/contract.py
FORMAT_VERSION = 3
LOGPROB_SENTINEL = -1000.0
# format/corpus.py
CHUNK_VERSION = 1
# format/arrow.py
META_KEY = b"offline_kd"
INPUT_COLUMNS = ("tokens", "labels", "position_ids", "loss_mask", "segments")
BLOCK_BYTES = 32 << 20  # DEFAULT_BLOCK_BYTES
_MAGIC_BYTES = 8  # "ARROW1" plus padding at the start of an IPC file.
# v3_storage.py part_name(0): a chunk's only part.
PART_NAME = "part-000000.arrow"


def chunk_dir(source: str, start: int, end: int) -> str:
    """Return the directory of the ``source`` chunk that holds rows ``[start, end)``."""
    # v3_storage.py flush_dir(CHUNK, start, end), which counts teacher iterations instead.
    return os.path.join(source, f"chunk-{start:012d}-{end:012d}")


def topk_targets(logits: torch.Tensor, top_k: int, vocab_size: int):
    """Return ``(values, indices, tail)``: top-k T=1 log-probs and the log of the remaining mass.

    Logits beyond the tokenizer's ``vocab_size`` (output-layer padding) are dropped before the
    softmax, as in Megatron. The float32 tail must be kept before casting values to the save dtype.
    """
    values, indices = logits[:, :vocab_size].float().log_softmax(-1).topk(top_k, dim=-1)
    # format/contract.py tail_logprob(); these rows hold no sentinel entries to mask.
    tail = torch.log(-torch.expm1(values.logsumexp(-1).clamp(max=0.0)))
    return values, indices, tail.clamp(min=LOGPROB_SENTINEL)


def publish_chunk(
    source: str,
    samples: list[dict],
    settings: dict,
    *,
    start: int,
    vocab_size: int,
    block_bytes: int = BLOCK_BYTES,
) -> str:
    """Write ``samples``, rows ``[start, start + len(samples))`` of ``source``, as one chunk.

    Each sample holds the :func:`strip_row` inputs and one teacher row per scored position:
    float16 ``values`` and integer ``indices`` of shape ``[rows, K]`` and the float32 ``tail``.
    The Arrow part is written first, then ``index.json``. Returns the chunk directory.
    """
    if any(s["values"].shape[0] != int(s["loss_mask"].sum()) for s in samples):
        raise ValueError("Each sample needs one teacher row per scored position")
    end = start + len(samples)
    data, part = _write_part(_record_batch(samples, vocab_size), settings, block_bytes)
    # corpus_saver.py CorpusLogitsSaver._publish_flush(), after share_schemas() of one part.
    chunk = {
        "chunk_version": CHUNK_VERSION,
        # Megatron-LM's planner finds overlapping chunks only through these iteration fields,
        # so each row is an iteration of a global batch of one.
        "producer": {
            "name": "modelopt",
            "global_batch_size": 1,
            "start_iteration": start,
            "end_iteration": end,
        },
        "settings": settings,
        "schemas": [part.pop("schema")],
        "parts": [{"path": PART_NAME, "schema": 0, **part}],
    }
    directory = chunk_dir(source, start, end)
    os.makedirs(directory, exist_ok=True)
    _write_atomic(os.path.join(directory, PART_NAME), data)
    # format/corpus.py publish_chunk()
    index = json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()
    _write_atomic(os.path.join(directory, "index.json"), index)
    return directory


# format/contract.py tokenizer_fingerprint(), as hf_tokenizer_identity() calls it.
def _tokenizer_fingerprint(tokenizer) -> str:
    vocab = sorted(((str(t), int(i)) for t, i in tokenizer.get_vocab().items()), key=lambda x: x[1])
    canonical = {"vocab": vocab, "size": len(tokenizer)}
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


# format/arrow.py _lists()
def _lists(lengths: list[int], flat: np.ndarray, kind: pa.DataType) -> pa.Array:
    offsets = pa.array(np.concatenate(([0], np.cumsum(lengths))), pa.int64())
    return pa.LargeListArray.from_arrays(offsets, pa.array(flat, kind))


# format/arrow.py _row_lists()
def _rows(lengths: list[int], rows: np.ndarray, kind: pa.DataType) -> pa.Array:
    inner = pa.FixedSizeListArray.from_arrays(pa.array(rows.reshape(-1), kind), rows.shape[1])
    offsets = pa.array(np.concatenate(([0], np.cumsum(lengths))), pa.int64())
    return pa.LargeListArray.from_arrays(offsets, inner)


# format/arrow.py samples_batch(), with format/contract.py split_indices().
def _record_batch(samples: list[dict], vocab_size: int) -> pa.RecordBatch:
    """One row per sample, input columns first, in the order Megatron's readers expect."""
    lengths = [s["tokens"].numel() for s in samples]
    rows = [s["values"].shape[0] for s in samples]

    def flat(key, dtype):
        return torch.cat([s[key] for s in samples]).numpy().astype(dtype, copy=False)

    indices = flat("indices", np.int32)  # Not int64: this is the chunk's largest array.
    if indices.size and (indices.min() < 0 or indices.max() >= vocab_size):
        raise ValueError(f"Teacher indices must be token ids below vocab_size={vocab_size}")
    columns = {
        **{key: _lists(lengths, flat(key, np.int32), pa.int32()) for key in INPUT_COLUMNS[:3]},
        "loss_mask": _lists(lengths, flat("loss_mask", np.bool_), pa.bool_()),
        "segments": _lists(
            [len(s["segments"]) for s in samples],
            np.array([n for s in samples for n in s["segments"]], dtype=np.int32),
            pa.int32(),
        ),
        "values": _rows(rows, flat("values", np.float16), pa.float16()),
        "indices_low": _rows(rows, (indices & 0xFFFF).astype(np.uint16), pa.uint16()),
    }
    if vocab_size > 1 << 16:  # Ids above 65535 need bits 16-23.
        columns["indices_high"] = _rows(rows, (indices >> 16 & 0xFF).astype(np.uint8), pa.uint8())
    columns["tail"] = _lists(rows, flat("tail", np.float32), pa.float32())
    return pa.record_batch(columns)


# format/arrow.py write_file()
def _write_part(batch: pa.RecordBatch, settings: dict, block_bytes: int) -> tuple[bytes, dict]:
    """Serialize ``batch`` into ``(data file bytes, part index)`` with bounded blocks."""
    options = pa.ipc.IpcWriteOptions(compression=pa.Codec("zstd", compression_level=3))
    columns = base64.b64encode(batch.schema.serialize().to_pybytes()).decode()
    schema = batch.schema.with_metadata({META_KEY: json.dumps(settings, sort_keys=True).encode()})
    sink = pa.BufferOutputStream()
    spans, start = [], 0
    with pa.ipc.new_file(sink, schema, options=options) as writer:
        for rows in _block_rows(batch, block_bytes):
            block = pa.RecordBatch.from_arrays(batch.slice(start, rows).columns, schema=schema)
            begin = sink.tell()
            writer.write_batch(block)
            spans.append((rows, begin, sink.tell(), block))
            start += rows
    data = sink.getvalue().to_pybytes()
    blocks = [_block_entry(data, *span, options) for span in spans]
    return data, {"schema": columns, "size": len(data), "blocks": blocks}


# format/arrow.py _row_bytes() and _block_rows()
def _block_rows(batch: pa.RecordBatch, block_bytes: int) -> list[int]:
    """Greedy row counts per block: at least one row, at most ``block_bytes`` uncompressed."""
    sizes = np.zeros(batch.num_rows, dtype=np.int64)
    for column in batch.columns:
        item = column.type.value_type
        fixed = pa.types.is_fixed_size_list(item)
        width, base = (item.list_size, item.value_type) if fixed else (1, item)
        lengths = column.value_lengths().to_numpy(zero_copy_only=False)
        sizes += lengths * width * max(1, base.bit_width // 8)
    counts, rows, size = [], 0, 0
    for nbytes in sizes.tolist():
        if rows and size + nbytes > block_bytes:
            counts.append(rows)
            rows, size = 0, 0
        rows += 1
        size += nbytes
    return [*counts, rows] if rows else counts


# format/arrow.py write_file(): its loop that locates each block in the file.
def _block_entry(data: bytes, rows: int, begin: int, end: int, block, options) -> dict:
    """Locate a block's record batch message and its input-column prefix in the file."""
    reader = pa.BufferReader(pa.py_buffer(data[begin:end]))
    if begin == 0:
        reader.seek(_MAGIC_BYTES)
    while True:  # The first block's span also holds the schema message.
        offset = reader.tell()
        message = pa.ipc.read_message(reader)
        if message.type == "record batch":
            break
    length = end - begin - offset
    return {
        "rows": rows,
        "offset": begin + offset,
        "length": length,
        "input_length": length - message.body.size + _input_body_size(block, options),
        "supervised": len(block.column("values").flatten()),
    }


# format/arrow.py _input_body_size()
def _input_body_size(block: pa.RecordBatch, options) -> int:
    """Body bytes of the input columns alone, which prefix the full block body."""
    inputs = block.select(list(INPUT_COLUMNS))
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, inputs.schema, options=options) as writer:
        writer.write_batch(inputs)
    reader = pa.BufferReader(sink.getvalue())
    pa.ipc.read_message(reader)  # Schema.
    return pa.ipc.read_message(reader).body.size
