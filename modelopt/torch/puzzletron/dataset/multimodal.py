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

"""Native AutoModel multimodal dataset adapters for Puzzletron."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .batch import (
    DataLayout,
    Modality,
    PackedSequenceMetadata,
    PuzzletronBatch,
    _canonical_packed_ids,
)

INTERSYN_SINGLE_DATASET = "finyorko/single_turn"
INTERSYN_SINGLE_REVISION = "59acf6773cd2bf5b485f37296e862216aa99516b"
INTERSYN_SINGLE_SPLIT = "train"
INTERSYN_MULTI_DATASET = "finyorko/multi-turn"
INTERSYN_MULTI_REVISION = "728987eff962e9df6f3da5ad5824a4accdd62bd2"
INTERSYN_MULTI_SPLIT = "multi"

__all__ = [
    "INTERSYN_MULTI_DATASET",
    "INTERSYN_MULTI_REVISION",
    "INTERSYN_MULTI_SPLIT",
    "INTERSYN_SINGLE_DATASET",
    "INTERSYN_SINGLE_REVISION",
    "INTERSYN_SINGLE_SPLIT",
    "batch_from_automodel",
    "load_materialized_conversation_subset",
    "load_materialized_intersyn_subset",
    "load_materialized_conversation_dataset",
    "materialize_intersyn_subset",
    "materialize_normalized_conversation_samples",
    "materialize_normalized_intersyn_samples",
    "normalize_nemotron_vlm_sample",
    "normalize_intersyn_multi",
    "normalize_intersyn_single",
]


def _pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
        if value.get("src"):
            raise ValueError(
                "remote dataset-viewer image URLs must be downloaded before subset materialization"
            )
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(f"unsupported InterSyn image value: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def materialize_normalized_conversation_samples(
    samples: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    acquisition: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Stream normalized conversations into an atomically published offline subset."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    samples_path = output_dir / "samples.json"
    if manifest_path.is_file() and samples_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if acquisition is not None and manifest.get("acquisition") != dict(acquisition):
            raise ValueError(
                f"existing materialization at {output_dir} does not match requested acquisition"
            )
        return manifest
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise ValueError(
                f"dataset destination exists without a complete manifest: {output_dir}"
            )
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}-",
            dir=str(output_dir.parent),
        )
    )
    images_dir = transaction / "images"
    images_dir.mkdir()
    image_manifest: list[dict[str, Any]] = []
    revisions: set[tuple[str, str]] = set()
    sample_count = 0
    try:
        with (transaction / "samples.json").open("w", encoding="utf-8") as stream:
            stream.write("[\n")
            for sample_index, raw_sample in enumerate(samples):
                sample = deepcopy(dict(raw_sample))
                source = dict(sample.get("source") or {})
                revisions.add(
                    (
                        str(source.get("dataset", "")),
                        str(source.get("revision", "")),
                    )
                )
                row_id = str(source.get("row_id", sample_index))
                image_index = 0
                for message in sample.get("conversation", []):
                    for item in message.get("content", []):
                        if item.get("type") != "image":
                            continue
                        image = _pil_image(item.get("image")).convert("RGB")
                        try:
                            width, height = image.size
                            relative = (
                                Path("images")
                                / f"{sample_index:04d}_{image_index:02d}.png"
                            )
                            absolute = transaction / relative
                            temporary = absolute.with_suffix(".png.tmp")
                            image.save(temporary, format="PNG")
                            temporary.replace(absolute)
                        finally:
                            image.close()
                        digest = hashlib.sha256(absolute.read_bytes()).hexdigest()
                        item["image"] = relative.as_posix()
                        image_manifest.append(
                            {
                                "sample_index": sample_index,
                                "row_id": row_id,
                                "image_index": image_index,
                                "path": relative.as_posix(),
                                "sha256": digest,
                                "width": width,
                                "height": height,
                            }
                        )
                        image_index += 1
                if image_index != int(sample.get("image_count", image_index)):
                    raise ValueError(
                        f"sample {row_id!r} declared {sample.get('image_count')} "
                        f"images but serialized {image_index}"
                    )
                if sample_index:
                    stream.write(",\n")
                json.dump(sample, stream, sort_keys=True)
                sample_count += 1
            stream.write("\n]\n")
        if expected_count is not None and sample_count != int(expected_count):
            raise RuntimeError(
                f"materialized {sample_count}/{int(expected_count)} expected rows"
            )
        manifest = {
            "version": 1,
            "sample_count": sample_count,
            "image_count": len(image_manifest),
            "sources": [
                {"dataset": dataset, "revision": revision}
                for dataset, revision in sorted(revisions)
            ],
            "images": image_manifest,
        }
        if acquisition is not None:
            manifest["acquisition"] = dict(acquisition)
        if diagnostics:
            manifest["diagnostics"] = dict(diagnostics)
        _write_json_atomic(transaction / "manifest.json", manifest)
        os.replace(transaction, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(transaction, ignore_errors=True)
        raise


def materialize_normalized_intersyn_samples(
    samples: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compatibility wrapper for the original InterSyn materializer."""

    return materialize_normalized_conversation_samples(samples, output_dir)


def load_materialized_conversation_subset(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load a materialized conversation subset without contacting Hugging Face."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    samples = json.loads((output_dir / "samples.json").read_text())
    if len(samples) != int(manifest["sample_count"]):
        raise ValueError("materialized InterSyn sample count does not match manifest")
    for sample in samples:
        for message in sample.get("conversation", []):
            for item in message.get("content", []):
                if item.get("type") == "image":
                    image_path = output_dir / item["image"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    item["image"] = str(image_path.resolve())
    return samples


def load_materialized_intersyn_subset(output_dir: str | Path) -> list[dict[str, Any]]:
    """Compatibility wrapper for the original InterSyn loader."""

    return load_materialized_conversation_subset(output_dir)


class _ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Mapping[str, Any]]):
        self.samples = [dict(sample) for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return deepcopy(self.samples[index])


def _source_balanced_order(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin immutable sources while preserving each source's row order."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        copied = dict(sample)
        source = copied.get("source") or {}
        identity = (str(source.get("dataset", "unknown")), str(source.get("revision", "")))
        groups.setdefault(identity, []).append(copied)
    ordered: list[dict[str, Any]] = []
    max_group_size = max((len(group) for group in groups.values()), default=0)
    for row_index in range(max_group_size):
        for group in groups.values():
            if row_index < len(group):
                ordered.append(group[row_index])
    return ordered


def load_materialized_conversation_dataset(
    path_or_dataset: str | Path,
    *,
    num_samples: int | None = None,
    seq_length: int | None = None,
    pretokenize: bool | None = None,
    truncate: bool | None = None,
    inject_fake_images: bool | None = None,
    max_length: int | None = None,
    **unknown: Any,
):
    """Hydra-friendly AutoModel VLM dataset factory for the offline subset.

    AutoModel passes the complete ``dataset`` config to its dataset target,
    including processor/collator controls that it consumes after construction.
    Accept those declared controls explicitly so config composition stays
    transparent, but reject misspelled or otherwise unknown fields.
    """
    if unknown:
        raise TypeError(f"Unsupported materialized-conversation dataset options: {sorted(unknown)}")
    del seq_length, pretokenize, truncate, inject_fake_images, max_length
    samples = _source_balanced_order(load_materialized_conversation_subset(path_or_dataset))
    if num_samples is not None:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_samples > len(samples):
            raise ValueError(
                f"requested num_samples={num_samples}, but the materialized subset "
                f"contains only {len(samples)} rows"
            )
        samples = samples[:num_samples]
    return _ConversationDataset(samples)


def _select_valid_rows(rows, normalize, count: int) -> list[dict[str, Any]]:
    selected = []
    failures = []
    for row_index, row in enumerate(rows):
        try:
            sample = normalize(row)
            for message in sample["conversation"]:
                for item in message["content"]:
                    if item.get("type") == "image":
                        image = _pil_image(item.get("image"))
                        image.load()
            selected.append(sample)
        except Exception as exc:  # Invalid source rows are expected and recorded.
            failures.append(
                {
                    "row_index": row_index,
                    "row_id": str(row.get("id", "")) if isinstance(row, Mapping) else "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only found {len(selected)}/{count} valid rows; first failures={failures[:3]}"
        )
    return selected


def materialize_intersyn_subset(
    output_dir: str | Path,
    *,
    rows_per_source: int = 8,
    dataset_loader=None,
) -> dict[str, Any]:
    """Download once, validate, and cache the pinned 8+8 InterSyn acceptance set."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    samples_path = output_dir / "samples.json"
    if manifest_path.is_file() and samples_path.is_file():
        return json.loads(manifest_path.read_text())
    if rows_per_source <= 0:
        raise ValueError("rows_per_source must be positive")
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    single_rows = dataset_loader(
        INTERSYN_SINGLE_DATASET,
        split=INTERSYN_SINGLE_SPLIT,
        revision=INTERSYN_SINGLE_REVISION,
        streaming=True,
    )
    multi_rows = dataset_loader(
        INTERSYN_MULTI_DATASET,
        split=INTERSYN_MULTI_SPLIT,
        revision=INTERSYN_MULTI_REVISION,
        streaming=True,
    )
    samples = [
        *_select_valid_rows(single_rows, normalize_intersyn_single, rows_per_source),
        *_select_valid_rows(multi_rows, normalize_intersyn_multi, rows_per_source),
    ]
    manifest = materialize_normalized_intersyn_samples(samples, output_dir)
    if manifest["sample_count"] != rows_per_source * 2:
        raise RuntimeError("InterSyn materialization produced an unexpected sample count")
    return manifest


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"InterSyn row {row.get('id', '<unknown>')!r} has no non-empty {key!r}")
    return value.strip()


def _source(dataset: str, revision: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "revision": revision,
        "row_id": str(row.get("id", "")),
        "topic": str(row.get("topic", "")),
    }


def _turn(image: Any, human: str, assistant: str) -> list[dict[str, Any]]:
    if image is None:
        raise ValueError("InterSyn multimodal turn has no image")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant}],
        },
    ]


def normalize_intersyn_single(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one real InterSyn single-turn row to AutoModel conversation form."""
    conversation = _turn(
        row.get("image"),
        _required_text(row, "human"),
        _required_text(row, "gpt"),
    )
    return {
        "conversation": conversation,
        "image_count": 1,
        "source": _source(INTERSYN_SINGLE_DATASET, INTERSYN_SINGLE_REVISION, row),
        "captions": [str(row.get("caption", ""))],
    }


def normalize_intersyn_multi(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the five-column-group InterSyn multi-turn schema without reordering turns."""
    conversation: list[dict[str, Any]] = []
    captions: list[str] = []
    image_count = 0
    for turn_idx in range(1, 6):
        human = row.get(f"human{turn_idx}")
        assistant = row.get(f"gpt{turn_idx}")
        image = row.get(f"image{turn_idx}")
        if human in (None, "") and assistant in (None, "") and image is None:
            continue
        conversation.extend(
            _turn(
                image,
                _required_text(row, f"human{turn_idx}"),
                _required_text(row, f"gpt{turn_idx}"),
            )
        )
        captions.append(str(row.get(f"caption{turn_idx}", "")))
        image_count += 1
    if image_count < 2:
        raise ValueError(
            f"InterSyn multi-turn row {row.get('id', '<unknown>')!r} must contain at least two images"
        )
    return {
        "conversation": conversation,
        "image_count": image_count,
        "source": _source(INTERSYN_MULTI_DATASET, INTERSYN_MULTI_REVISION, row),
        "captions": captions,
    }


def normalize_nemotron_vlm_sample(
    row: Mapping[str, Any],
    *,
    subset: str,
    revision: str,
) -> dict[str, Any]:
    """Preserve a complete Nemotron conversation while attaching its matched image."""

    messages = row.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("Nemotron-VLM row has no message sequence")
    image = row.get("image")
    if image is None:
        raise ValueError("Nemotron-VLM row has no matched image")
    conversation = []
    image_items = []
    for raw_message in deepcopy(list(messages)):
        if not isinstance(raw_message, Mapping):
            raise TypeError("Nemotron-VLM messages must be mappings")
        message = dict(raw_message)
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            conversation.append(message)
            continue
        if not isinstance(content, Sequence):
            raise TypeError("Nemotron-VLM message content must be text or a sequence")
        normalized_content = []
        for item in content:
            if isinstance(item, str):
                normalized_content.append({"type": "text", "text": item})
                continue
            if not isinstance(item, Mapping):
                raise TypeError("Nemotron-VLM content items must be strings or mappings")
            item = dict(item)
            if item.get("type") == "video":
                raise ValueError("Nemotron-VLM video rows are not supported")
            if item.get("type") == "image":
                image_items.append(item)
            normalized_content.append(item)
        message["content"] = normalized_content
        conversation.append(message)
    if len(image_items) != 1:
        raise ValueError(
            f"Nemotron-VLM rows must reference exactly one image, found {len(image_items)}"
        )
    image_item = image_items[0]
    for key in ("images", "path", "image_url", "url", "value", "data"):
        image_item.pop(key, None)
    image_item["image"] = image
    if not any(message.get("role") == "assistant" for message in conversation):
        raise ValueError("Nemotron-VLM row has no assistant response")
    return {
        "conversation": conversation,
        "image_count": 1,
        "source": {
            "dataset": "nvidia/Nemotron-VLM-Dataset-v2",
            "revision": revision,
            "subset": str(subset),
            "row_id": str(row.get("id", "")),
        },
    }


def _media_counts(collated: Mapping[str, Any], batch_size: int) -> torch.Tensor | None:
    for key in ("n_images_per_sample", "num_images_per_sample", "media_counts"):
        value = collated.get(key)
        if isinstance(value, torch.Tensor):
            counts = value.to(dtype=torch.int32).reshape(-1)
            if counts.numel() != batch_size:
                raise ValueError(f"{key} has {counts.numel()} entries for batch size {batch_size}")
            return counts
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            counts = torch.tensor(list(value), dtype=torch.int32)
            if counts.numel() != batch_size:
                raise ValueError(f"{key} has {counts.numel()} entries for batch size {batch_size}")
            return counts
    grid = collated.get("image_grid_thw")
    if isinstance(grid, torch.Tensor):
        if batch_size == 1:
            return torch.tensor([grid.shape[0]], dtype=torch.int32, device=grid.device)
        if grid.shape[0] == batch_size:
            return torch.ones(batch_size, dtype=torch.int32, device=grid.device)
        raise ValueError(
            "multimodal batches with multiple images require n_images_per_sample metadata"
        )
    return None


def batch_from_automodel(
    collated: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
    source_metadata: Mapping[str, Any],
    layout: DataLayout | str,
) -> PuzzletronBatch:
    """Normalize an AutoModel VLM/text collator result into ``PuzzletronBatch``."""
    if not isinstance(collated.get("input_ids"), torch.Tensor):
        raise ValueError("AutoModel collator output has no tensor input_ids")
    input_ids = collated["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    batch_size, seq_len = input_ids.shape
    labels = collated.get("labels", collated.get("targets"))
    if labels is not None and labels.ndim == 1:
        labels = labels.unsqueeze(0)
    raw_attention_mask = collated.get("attention_mask")
    attention_mask = raw_attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    elif attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    loss_mask = collated.get("loss_mask")
    packed_seq_ids = collated.get("_packed_seq_ids")
    padding_mask = collated.get("padding_mask")
    if isinstance(packed_seq_ids, torch.Tensor):
        if tuple(packed_seq_ids.shape) != (batch_size, seq_len):
            raise ValueError("_packed_seq_ids must match input_ids [batch, sequence]")
        valid_tokens = packed_seq_ids.ne(0)
    elif isinstance(raw_attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        valid_tokens = attention_mask.bool()
    elif (
        isinstance(padding_mask, torch.Tensor)
        and tuple(padding_mask.shape) == (batch_size, seq_len)
    ):
        valid_tokens = ~padding_mask.bool()
    else:
        valid_tokens = torch.ones_like(input_ids, dtype=torch.bool)
    ce_mask = (
        loss_mask.bool()
        if isinstance(loss_mask, torch.Tensor)
        else (labels != -100 if isinstance(labels, torch.Tensor) else valid_tokens)
    )
    if ce_mask.ndim == 1:
        ce_mask = ce_mask.unsqueeze(0)
    ce_mask = ce_mask & valid_tokens
    if isinstance(labels, torch.Tensor):
        labels = labels.clone().masked_fill(~valid_tokens, -100)
    hidden_mask = valid_tokens

    reserved = {
        "labels",
        "targets",
        "loss_mask",
        "n_images_per_sample",
        "num_images_per_sample",
        "media_counts",
        "_packed_seq_ids",
    }
    model_kwargs = {key: value for key, value in collated.items() if key not in reserved}
    model_kwargs["input_ids"] = input_ids
    model_kwargs["attention_mask"] = attention_mask

    sequence_ids = None
    cu = collated.get("cu_seqlens")
    if isinstance(packed_seq_ids, torch.Tensor):
        # Neat packing uses 1-based IDs independently in every packed row and
        # zero for padding. Canonical IDs must instead be globally unique so
        # per-original-sample activation/KD reductions never merge rows.
        raw_ids = packed_seq_ids.to(dtype=torch.long).masked_fill(packed_seq_ids.eq(0), -1)
        sequence_ids, cu, offsets, max_seqlen = _canonical_packed_ids(raw_ids)
    elif isinstance(cu, torch.Tensor):
        cu = cu.reshape(-1).to(dtype=torch.int32)
        offsets = tuple((int(start), int(stop)) for start, stop in zip(cu[:-1], cu[1:]))
        max_seqlen = max((stop - start for start, stop in offsets), default=0)
        sequence_ids = collated.get("seq_idx")
        if sequence_ids is None:
            if batch_size != 1:
                raise ValueError(
                    "cu_seqlens without seq_idx is only unambiguous for one packed row"
                )
            if int(cu[0]) != 0 or int(cu[-1]) != seq_len:
                raise ValueError("one-row packed cu_seqlens must span the full input sequence")
            sequence_ids = torch.empty((1, seq_len), dtype=torch.long, device=input_ids.device)
            for sequence_id, (start, stop) in enumerate(offsets):
                sequence_ids[0, start:stop] = sequence_id
    else:
        cu = None
        offsets = ()
        max_seqlen = None
    counts = _media_counts(collated, int(batch_size))
    media_offsets = None
    if counts is not None:
        media_offsets = torch.cat(
            (
                torch.zeros(1, dtype=counts.dtype, device=counts.device),
                counts.cumsum(0),
            )
        )
    if sequence_ids is None:
        sequence_ids = collated.get("seq_idx")
    sequence = PackedSequenceMetadata(
        global_cu_seqlens=cu,
        max_seqlen=max_seqlen,
        seq_ids=sequence_ids,
        sample_offsets=offsets,
        media_counts=counts,
        media_offsets=media_offsets,
    )
    modality = (
        Modality.MULTIMODAL if counts is not None and int(counts.sum()) > 0 else Modality.TEXT
    )
    return PuzzletronBatch(
        model_kwargs=model_kwargs,
        labels=labels,
        ce_mask=ce_mask,
        kd_mask=ce_mask.clone(),
        hidden_mask=hidden_mask,
        sequence=sequence,
        sample_ids=tuple(sample_ids),
        source_metadata=source_metadata,
        modality=modality,
        layout=DataLayout(layout),
    )
