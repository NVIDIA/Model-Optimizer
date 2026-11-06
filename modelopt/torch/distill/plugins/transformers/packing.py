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

"""Document-isolated forwards for recurrent, pooled and custom HF architectures."""

from contextlib import contextmanager
from contextvars import ContextVar
from itertools import pairwise

import torch

__all__ = ["batch_documents", "document_batches", "empty_document_batch"]

_BATCH_DOCUMENTS = ContextVar("modelopt_batch_documents", default=True)


@contextmanager
def document_batching(enabled):
    token = _BATCH_DOCUMENTS.set(enabled)
    try:
        yield
    finally:
        _BATCH_DOCUMENTS.reset(token)


def batch_documents(documents, token_budget, *, max_documents=8):
    """Coalesce independent text rows without exceeding the original packed token budget."""
    supported = {
        "input_ids",
        "position_ids",
        "attention_mask",
        "valid_mask",
        "_packed_seq_ids",
        "labels",
        "loss_mask",
        "sample_ids",
        "document_ids",
    }
    if not _BATCH_DOCUMENTS.get() or any(
        set(document) - supported
        or any(not isinstance(value, torch.Tensor) for value in document.values())
        or (document.get("attention_mask") is not None and document["attention_mask"].ndim != 2)
        for document in documents
    ):
        return documents
    result, group, width = [], [], 0

    def flush():
        if len(group) == 1:
            result.append(group[0])
            return
        batch = {}
        for key in group[0]:
            values = [document[key] for document in group]
            if any(not isinstance(value, torch.Tensor) for value in values):
                raise ValueError("Batched document metadata must be tensors.")
            axis = 1 if key == "position_ids" and values[0].ndim == 3 else 0
            if key not in ("sample_ids", "document_ids") or values[0].ndim > 1:
                fill = -100 if key == "labels" else 0
                values = [
                    torch.nn.functional.pad(value, (0, width - value.shape[-1]), value=fill)
                    for value in values
                ]
            batch[key] = torch.cat(values, dim=axis)
        attention = batch.get("attention_mask")
        batch["attention_mask"] = batch["valid_mask"].bool() & (
            attention.bool() if attention is not None else True
        )
        result.append(batch)

    for document in documents:
        next_width = max(width, document["input_ids"].shape[-1])
        if group and (
            len(group) >= max_documents
            or next_width * (len(group) + 1) > token_budget
            or set(document) != set(group[0])
        ):
            flush()
            group, width = [], 0
        group.append(document)
        width = max(width, document["input_ids"].shape[-1])
    if group:
        flush()
    return result


def document_batches(batch):
    """Split compact packed text into independent documents, keeping token weights."""
    documents = batch.get("_packed_seq_ids")
    if documents is None:
        return [batch]
    tokens = batch.get("input_ids")
    if not isinstance(documents, torch.Tensor) or not isinstance(tokens, torch.Tensor):
        raise ValueError("Document-isolated packing requires tensor input_ids and document IDs.")
    if documents.ndim != 2 or documents.shape != tokens.shape or bool((documents < 0).any()):
        raise ValueError("Packed document IDs must be nonnegative and match token IDs.")
    if bool(documents.eq(1).all()):
        return [batch]
    for key in ("pixel_values", "pixel_values_videos", "input_features", "audio_values"):
        if batch.get(key) is not None:
            raise ValueError(
                "Packed multimodal payloads require a family-owned document_batches implementation."
            )
    batch_size, length = tokens.shape
    result = []
    for row in range(batch_size):
        ids = documents[row]
        boundaries = [0, *(ids[1:] != ids[:-1]).nonzero().flatten().add(1).tolist(), length]
        seen = set()
        for start, end in pairwise(boundaries):
            identity = int(ids[start])
            if identity == 0:
                continue
            if identity in seen:
                raise ValueError("Each packed document must occupy one contiguous token span.")
            seen.add(identity)
            document = {}
            for key, value in batch.items():
                if key == "attention_mask" and isinstance(value, torch.Tensor) and value.ndim == 4:
                    if (
                        value.shape[0] not in (1, batch_size)
                        or value.shape[-2] not in (1, length)
                        or value.shape[-1] != length
                    ):
                        raise ValueError(
                            "Packed 4D masks must broadcast over the batch and query axes."
                        )
                    mask_row = row if value.shape[0] == batch_size else 0
                    query = slice(None) if value.shape[-2] == 1 else slice(start, end)
                    value = value[mask_row : mask_row + 1, :, query, start:end]
                elif (
                    key == "position_ids"
                    and isinstance(value, torch.Tensor)
                    and value.ndim == 3
                    and value.shape[1:] == tokens.shape
                ):
                    value = value[:, row : row + 1, start:end]
                elif (
                    isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size
                ):
                    if value.ndim >= 2 and value.shape[1] == length:
                        value = value[row : row + 1, start:end]
                    else:
                        value = value[row : row + 1]
                document[key] = value
            document["_packed_seq_ids"] = torch.ones_like(tokens[row : row + 1, start:end])
            document.setdefault("valid_mask", document["_packed_seq_ids"].bool())
            result.append(document)
    return result


def empty_document_batch(batch):
    """Execute a zero-valid token when another distributed rank has more documents."""
    tokens = batch["input_ids"][:1, :1]
    return {
        "input_ids": tokens,
        "position_ids": torch.zeros_like(tokens),
        "_packed_seq_ids": torch.ones_like(tokens),
        "valid_mask": torch.zeros_like(tokens, dtype=torch.bool),
    }
