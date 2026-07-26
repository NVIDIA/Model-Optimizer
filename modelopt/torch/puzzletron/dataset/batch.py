# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, topology-aware Puzzletron model input batches."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import torch

__all__ = ["DataLayout", "Modality", "PackedSequenceMetadata", "PuzzletronBatch"]


class DataLayout(str, Enum):
    FIXED = "fixed"
    PADDED_VARLEN = "padded_varlen"
    PACKED_VARLEN = "packed_varlen"


class Modality(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class PackedSequenceMetadata:
    """Sequence and media boundaries retained across distributed partitions."""

    global_cu_seqlens: torch.Tensor | None = None
    local_cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    seq_ids: torch.Tensor | None = None
    sample_offsets: tuple[tuple[int, int], ...] = ()
    media_counts: torch.Tensor | None = None
    media_offsets: torch.Tensor | None = None
    cp_rank: int = 0
    cp_size: int = 1


_SEQUENCE_MODEL_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "cache_position",
    "seq_idx",
    "mm_token_type_ids",
    "padding_mask",
}
_IMAGE_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "image_position_ids",
}


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(raw)
    return digest.hexdigest()


def _identity_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_digest(value),
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _identity_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _slice_batch_tensor(value: torch.Tensor, start: int, stop: int, batch_size: int) -> torch.Tensor:
    if value.ndim >= 2 and value.shape[0] == 3 and value.shape[1] == batch_size:
        return value[:, start:stop]
    if value.ndim and value.shape[0] == batch_size:
        return value[start:stop]
    return value


def _pad_batch_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    pad_rows: int,
    fill_value: int | float | bool = 0,
) -> torch.Tensor:
    if value.ndim >= 2 and value.shape[0] == 3 and value.shape[1] == batch_size:
        padding = value.new_full(
            (value.shape[0], pad_rows, *value.shape[2:]), fill_value
        )
        return torch.cat((value, padding), dim=1)
    if value.ndim and value.shape[0] == batch_size:
        padding = value.new_full((pad_rows, *value.shape[1:]), fill_value)
        return torch.cat((value, padding), dim=0)
    return value


def _move(value: Any, *args, **kwargs) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(*args, **kwargs)
    if isinstance(value, Mapping):
        return {key: _move(item, *args, **kwargs) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, *args, **kwargs) for item in value)
    if isinstance(value, list):
        return [_move(item, *args, **kwargs) for item in value]
    return value


def _canonical_packed_ids(
    sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...], int]:
    """Renumber every valid contiguous segment across packed rows globally.

    ``sequence_ids`` uses ``-1`` for padding. Input IDs may restart in each
    packed row, so relying on their numeric values would merge unrelated source
    examples in activation/KD reductions. The returned IDs are unique across
    the entire batch and the cumulative lengths are over valid tokens only.
    """
    if sequence_ids.ndim != 2:
        raise ValueError(
            f"packed sequence_ids must be [batch, sequence], got {tuple(sequence_ids.shape)}"
        )
    canonical = torch.full_like(sequence_ids, -1, dtype=torch.long)
    lengths: list[int] = []
    next_id = 0
    for row_index in range(sequence_ids.shape[0]):
        row = sequence_ids[row_index]
        start = 0
        while start < row.numel():
            value = int(row[start].item())
            stop = start + 1
            while stop < row.numel() and int(row[stop].item()) == value:
                stop += 1
            if value >= 0:
                canonical[row_index, start:stop] = next_id
                lengths.append(stop - start)
                next_id += 1
            start = stop
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu = torch.tensor(cumulative, dtype=torch.int32, device=sequence_ids.device)
    offsets = tuple((start, stop) for start, stop in zip(cumulative[:-1], cumulative[1:]))
    return canonical, cu, offsets, max(lengths, default=0)


@dataclass(frozen=True)
class PuzzletronBatch:
    """The only model-input contract shared by Puzzletron stages.

    The batch remains backend-neutral: descriptor/stage adapters decide which
    validated ``model_kwargs`` are consumed by a particular model forward.
    """

    model_kwargs: Mapping[str, Any]
    labels: torch.Tensor | None = None
    ce_mask: torch.Tensor | None = None
    kd_mask: torch.Tensor | None = None
    hidden_mask: torch.Tensor | None = None
    sequence: PackedSequenceMetadata = field(default_factory=PackedSequenceMetadata)
    sample_ids: tuple[str, ...] = ()
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    modality: Modality = Modality.TEXT
    layout: DataLayout = DataLayout.FIXED
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", dict(self.model_kwargs))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        object.__setattr__(self, "sample_ids", tuple(str(item) for item in self.sample_ids))
        object.__setattr__(self, "modality", Modality(self.modality))
        object.__setattr__(self, "layout", DataLayout(self.layout))
        self._validate()
        identity = {
            "model_kwargs": self.model_kwargs,
            "labels": self.labels,
            "ce_mask": self.ce_mask,
            "kd_mask": self.kd_mask,
            "hidden_mask": self.hidden_mask,
            "sequence": self.sequence,
            "sample_ids": self.sample_ids,
            "source_metadata": self.source_metadata,
            "modality": self.modality,
            "layout": self.layout,
        }
        encoded = json.dumps(_identity_value(identity), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "fingerprint", hashlib.sha256(encoded.encode()).hexdigest())

    @property
    def input_ids(self) -> torch.Tensor:
        value = self.model_kwargs.get("input_ids")
        if not isinstance(value, torch.Tensor):
            raise ValueError("PuzzletronBatch requires tensor model_kwargs['input_ids']")
        return value

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[-1])

    def _validate(self) -> None:
        input_ids = self.model_kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("PuzzletronBatch input_ids must have shape [batch, sequence]")
        batch_size, seq_len = map(int, input_ids.shape)
        if self.sample_ids and len(self.sample_ids) != batch_size:
            raise ValueError(
                f"sample_ids length {len(self.sample_ids)} does not match batch size {batch_size}"
            )
        for name, tensor in (
            ("labels", self.labels),
            ("ce_mask", self.ce_mask),
            ("kd_mask", self.kd_mask),
            ("hidden_mask", self.hidden_mask),
        ):
            if tensor is not None and tuple(tensor.shape) != (batch_size, seq_len):
                raise ValueError(f"{name} must have shape {(batch_size, seq_len)}, got {tuple(tensor.shape)}")
        if self.hidden_mask is not None:
            valid = self.hidden_mask.bool()
            for name, mask in (("ce_mask", self.ce_mask), ("kd_mask", self.kd_mask)):
                if mask is not None and bool((mask.bool() & ~valid).any()):
                    raise ValueError(f"{name} must be a subset of hidden_mask")
            if self.labels is not None and bool(self.labels[~valid].ne(-100).any()):
                raise ValueError("labels outside hidden_mask must be -100")
        for key in _SEQUENCE_MODEL_KEYS - {"position_ids"}:
            value = self.model_kwargs.get(key)
            if isinstance(value, torch.Tensor) and value.shape[-1] != seq_len:
                raise ValueError(f"model_kwargs[{key!r}] has sequence length {value.shape[-1]}, expected {seq_len}")
        position_ids = self.model_kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            valid = tuple(position_ids.shape) == (batch_size, seq_len) or (
                position_ids.ndim == 3
                and position_ids.shape[1] == batch_size
                and position_ids.shape[2] == seq_len
            )
            if not valid:
                raise ValueError(
                    "position_ids must be [batch, sequence] or [rope_dims, batch, sequence], "
                    f"got {tuple(position_ids.shape)}"
                )

        metadata = self.sequence
        if metadata.global_cu_seqlens is not None:
            cu = metadata.global_cu_seqlens.detach().cpu().to(torch.long)
            if self.layout is not DataLayout.PACKED_VARLEN:
                raise ValueError("global_cu_seqlens requires layout=packed_varlen")
            if cu.ndim != 1 or cu.numel() < 2 or int(cu[0]) != 0:
                raise ValueError("global_cu_seqlens must be one-dimensional and start at 0")
            valid_tokens = (
                int(metadata.seq_ids.ge(0).sum().item())
                if metadata.seq_ids is not None
                else (
                    int(self.hidden_mask.sum().item())
                    if self.hidden_mask is not None
                    else batch_size * seq_len
                )
            )
            if metadata.cp_size == 1 and int(cu[-1]) != valid_tokens:
                raise ValueError(
                    "global_cu_seqlens must end at the number of valid packed tokens "
                    f"({valid_tokens}), got {int(cu[-1])}"
                )
            if metadata.cp_size > 1:
                local_cu = metadata.local_cu_seqlens
                if local_cu is None or int(local_cu[-1]) != valid_tokens:
                    raise ValueError(
                        "CP-partitioned packed batches require local_cu_seqlens ending at "
                        f"the local valid-token count ({valid_tokens})"
                    )
            if not bool(torch.all(cu[1:] > cu[:-1])):
                raise ValueError("global_cu_seqlens must be strictly increasing")
            offsets = tuple((int(start), int(stop)) for start, stop in zip(cu[:-1], cu[1:]))
            if metadata.sample_offsets and tuple(metadata.sample_offsets) != offsets:
                raise ValueError(
                    f"sample_offsets {metadata.sample_offsets} do not match global_cu_seqlens {offsets}"
                )
            lengths_cu = (
                metadata.local_cu_seqlens.detach().cpu().to(torch.long)
                if metadata.cp_size > 1 and metadata.local_cu_seqlens is not None
                else cu
            )
            max_len = int((lengths_cu[1:] - lengths_cu[:-1]).max().item())
            if metadata.max_seqlen is not None and int(metadata.max_seqlen) != max_len:
                raise ValueError(f"max_seqlen={metadata.max_seqlen} does not match packed maximum {max_len}")
        if metadata.seq_ids is not None and tuple(metadata.seq_ids.shape) != (batch_size, seq_len):
            raise ValueError("seq_ids must have the same [batch, sequence] shape as input_ids")
        if metadata.seq_ids is not None and self.layout is DataLayout.PACKED_VARLEN:
            valid_ids = metadata.seq_ids[metadata.seq_ids >= 0].detach().cpu().to(torch.long)
            if self.hidden_mask is not None and not torch.equal(
                metadata.seq_ids.ge(0).to(self.hidden_mask.device),
                self.hidden_mask.bool(),
            ):
                raise ValueError("packed seq_ids padding must match hidden_mask")
            if valid_ids.numel() and metadata.cp_size == 1:
                unique = torch.unique(valid_ids, sorted=True)
                expected = torch.arange(unique.numel(), dtype=torch.long)
                if not torch.equal(unique, expected):
                    raise ValueError("packed seq_ids must be globally contiguous from zero")
                if metadata.global_cu_seqlens is not None:
                    counts = torch.bincount(valid_ids, minlength=unique.numel())
                    lengths = (
                        metadata.global_cu_seqlens.detach().cpu().to(torch.long)[1:]
                        - metadata.global_cu_seqlens.detach().cpu().to(torch.long)[:-1]
                    )
                    if not torch.equal(counts, lengths):
                        raise ValueError(
                            "packed seq_ids token counts do not match global_cu_seqlens"
                        )
        if metadata.media_counts is not None:
            counts = metadata.media_counts.detach().cpu().to(torch.long).reshape(-1)
            if counts.numel() != batch_size:
                raise ValueError("media_counts must contain one entry per batch row")
            expected = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
            if metadata.media_offsets is not None and not torch.equal(
                metadata.media_offsets.detach().cpu().to(torch.long), expected
            ):
                raise ValueError("media_offsets must be the cumulative sum of media_counts")

    def replace_model_kwargs(self, **updates: Any) -> "PuzzletronBatch":
        kwargs = dict(self.model_kwargs)
        kwargs.update(updates)
        return replace(self, model_kwargs=kwargs)

    def to(self, *args, **kwargs) -> "PuzzletronBatch":
        sequence = replace(
            self.sequence,
            global_cu_seqlens=_move(self.sequence.global_cu_seqlens, *args, **kwargs),
            local_cu_seqlens=_move(self.sequence.local_cu_seqlens, *args, **kwargs),
            seq_ids=_move(self.sequence.seq_ids, *args, **kwargs),
            media_counts=_move(self.sequence.media_counts, *args, **kwargs),
            media_offsets=_move(self.sequence.media_offsets, *args, **kwargs),
        )
        return replace(
            self,
            model_kwargs=_move(self.model_kwargs, *args, **kwargs),
            labels=_move(self.labels, *args, **kwargs),
            ce_mask=_move(self.ce_mask, *args, **kwargs),
            kd_mask=_move(self.kd_mask, *args, **kwargs),
            hidden_mask=_move(self.hidden_mask, *args, **kwargs),
            sequence=sequence,
        )

    def cp_partition(
        self,
        token_indices: torch.Tensor,
        *,
        cp_rank: int,
        cp_size: int,
    ) -> "PuzzletronBatch":
        if cp_size < 1 or not 0 <= cp_rank < cp_size:
            raise ValueError(f"invalid CP rank {cp_rank}/{cp_size}")
        indices = token_indices.to(device=self.input_ids.device, dtype=torch.long).reshape(-1)

        def slice_sequence(value: Any, key: str | None = None) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            if key == "position_ids" and value.ndim == 3:
                return value.index_select(2, indices.to(value.device))
            if key in _SEQUENCE_MODEL_KEYS and value.shape[-1] == self.sequence_length:
                return value.index_select(-1, indices.to(value.device))
            return value

        model_kwargs = {
            key: slice_sequence(value, key) for key, value in self.model_kwargs.items()
        }
        labels = None if self.labels is None else self.labels.index_select(-1, indices)
        ce_mask = None if self.ce_mask is None else self.ce_mask.index_select(-1, indices)
        kd_mask = None if self.kd_mask is None else self.kd_mask.index_select(-1, indices)
        hidden_mask = (
            None if self.hidden_mask is None else self.hidden_mask.index_select(-1, indices)
        )
        metadata = self.sequence
        local_cu = None
        max_seqlen = metadata.max_seqlen
        seq_ids = metadata.seq_ids
        if seq_ids is not None:
            seq_ids = seq_ids.index_select(-1, indices.to(seq_ids.device))
        if metadata.global_cu_seqlens is not None:
            num_samples = int(metadata.global_cu_seqlens.numel() - 1)
            if seq_ids is None:
                raise ValueError("CP-partitioned packed batches require seq_ids")
            counts = [int(seq_ids.eq(sample_id).sum().item()) for sample_id in range(num_samples)]
            counts = [count for count in counts if count > 0]
            local_cu = torch.tensor(
                [0, *torch.tensor(counts, dtype=torch.int64).cumsum(0).tolist()],
                dtype=metadata.global_cu_seqlens.dtype,
                device=metadata.global_cu_seqlens.device,
            )
            max_seqlen = max(counts, default=0)
        sequence = replace(
            metadata,
            local_cu_seqlens=local_cu,
            max_seqlen=max_seqlen,
            seq_ids=seq_ids,
            sample_offsets=metadata.sample_offsets,
            cp_rank=int(cp_rank),
            cp_size=int(cp_size),
        )
        return replace(
            self,
            model_kwargs=model_kwargs,
            labels=labels,
            ce_mask=ce_mask,
            kd_mask=kd_mask,
            hidden_mask=hidden_mask,
            sequence=sequence,
        )

    def _media_slice(self, start: int, stop: int) -> tuple[int, int]:
        counts = self.sequence.media_counts
        if counts is None:
            return 0, 0
        cumulative = torch.cat(
            (torch.zeros(1, dtype=torch.long), counts.detach().cpu().to(torch.long).cumsum(0))
        )
        return int(cumulative[start]), int(cumulative[stop])

    def dp_slice(self, *, dp_rank: int, dp_size: int) -> "PuzzletronBatch":
        """Return one disjoint contiguous DP shard without duplicating samples/media."""
        if dp_size < 1 or not 0 <= dp_rank < dp_size:
            raise ValueError(f"invalid DP rank {dp_rank}/{dp_size}")
        if dp_size == 1:
            return self
        if (
            self.layout is DataLayout.PACKED_VARLEN
            and self.batch_size == 1
            and self.sequence.global_cu_seqlens is not None
            and self.sequence.seq_ids is not None
        ):
            return self._dp_slice_single_packed_row(dp_rank=dp_rank, dp_size=dp_size)
        if self.batch_size % dp_size:
            raise ValueError(
                f"batch size {self.batch_size} must be divisible by dp_size={dp_size}; "
                "shard packed rows in the dataset sampler instead of duplicating them"
            )
        return self.pp_microbatches(dp_size)[dp_rank]

    def _dp_slice_single_packed_row(
        self, *, dp_rank: int, dp_size: int
    ) -> "PuzzletronBatch":
        """Partition one packed text row by complete source samples.

        Legacy text packing materializes ``micro_batch_size`` source samples as
        one row. Splitting that row by tokens would change the MiniTron metric
        because each DP rank would square a partial sample mean. Instead, give
        every rank a contiguous set of complete packed samples and rebuild its
        local packing metadata.
        """
        counts = self.sequence.media_counts
        if counts is not None and int(counts.sum().item()) > 0:
            raise ValueError(
                "single-row multimodal packs must be sharded by the AutoModel sampler"
            )
        cu = self.sequence.global_cu_seqlens
        assert cu is not None
        cu_long = cu.detach().cpu().to(torch.long)
        num_samples = int(cu_long.numel() - 1)
        if num_samples < dp_size:
            raise ValueError(
                f"packed row contains {num_samples} samples, fewer than dp_size={dp_size}"
            )
        samples_per_rank, remainder = divmod(num_samples, dp_size)
        sample_start = dp_rank * samples_per_rank + min(dp_rank, remainder)
        sample_stop = sample_start + samples_per_rank + int(dp_rank < remainder)
        token_start = int(cu_long[sample_start])
        token_stop = int(cu_long[sample_stop])

        local_cu = (
            cu[sample_start : sample_stop + 1] - cu[sample_start]
        ).clone()

        def slice_tokens(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            if value.ndim == 3 and value.shape[1] == 1 and value.shape[2] == self.sequence_length:
                return value[:, :, token_start:token_stop]
            if value.ndim >= 2 and value.shape[0] == 1 and value.shape[-1] == self.sequence_length:
                return value[..., token_start:token_stop]
            if value.ndim == 1 and value.shape[0] == self.sequence_length:
                return value[token_start:token_stop]
            return value

        model_kwargs = {
            key: (local_cu if key == "cu_seqlens" else slice_tokens(value))
            for key, value in self.model_kwargs.items()
        }
        local_ids = self.sequence.seq_ids[:, token_start:token_stop].to(torch.long)
        local_ids = local_ids - sample_start
        offsets = tuple(
            (int(start), int(stop)) for start, stop in zip(local_cu[:-1], local_cu[1:])
        )
        sequence = replace(
            self.sequence,
            global_cu_seqlens=local_cu,
            local_cu_seqlens=None,
            max_seqlen=max((stop - start for start, stop in offsets), default=0),
            seq_ids=local_ids,
            sample_offsets=offsets,
            media_counts=None,
            media_offsets=None,
            cp_rank=0,
            cp_size=1,
        )
        shard_name = self.sample_ids[0] if self.sample_ids else "packed-row"
        source_metadata = {
            **self.source_metadata,
            "dp_packed_sample_range": [sample_start, sample_stop],
            "dp_rank": dp_rank,
            "dp_size": dp_size,
        }
        return replace(
            self,
            model_kwargs=model_kwargs,
            labels=None if self.labels is None else self.labels[:, token_start:token_stop],
            ce_mask=None if self.ce_mask is None else self.ce_mask[:, token_start:token_stop],
            kd_mask=None if self.kd_mask is None else self.kd_mask[:, token_start:token_stop],
            hidden_mask=(
                None
                if self.hidden_mask is None
                else self.hidden_mask[:, token_start:token_stop]
            ),
            sequence=sequence,
            sample_ids=(f"{shard_name}:dp-{dp_rank}",),
            source_metadata=source_metadata,
        )

    def pp_microbatches(self, n_microbatches: int) -> tuple["PuzzletronBatch", ...]:
        if n_microbatches < 1:
            raise ValueError("n_microbatches must be at least one")
        chunk = math.ceil(self.batch_size / n_microbatches)
        result = []
        for start in range(0, self.batch_size, chunk):
            stop = min(start + chunk, self.batch_size)
            media_start, media_stop = self._media_slice(start, stop)
            model_kwargs: dict[str, Any] = {}
            image_grid = self.model_kwargs.get("image_grid_thw")
            for key, value in self.model_kwargs.items():
                if not isinstance(value, torch.Tensor):
                    model_kwargs[key] = value
                    continue
                if key == "image_grid_thw":
                    model_kwargs[key] = value[media_start:media_stop]
                elif key == "pixel_values" and self.sequence.media_counts is not None:
                    total_media = int(self.sequence.media_counts.sum().item())
                    if value.shape[0] == total_media:
                        model_kwargs[key] = value[media_start:media_stop]
                    elif isinstance(image_grid, torch.Tensor):
                        patch_counts = image_grid.detach().cpu().to(torch.long).prod(dim=1)
                        patch_offsets = torch.cat(
                            (torch.zeros(1, dtype=torch.long), patch_counts.cumsum(0))
                        )
                        model_kwargs[key] = value[
                            int(patch_offsets[media_start]) : int(patch_offsets[media_stop])
                        ]
                    else:
                        raise ValueError("cannot align flat pixel_values without image_grid_thw")
                elif key in _IMAGE_KEYS and self.sequence.media_counts is not None:
                    model_kwargs[key] = value[media_start:media_stop]
                else:
                    model_kwargs[key] = _slice_batch_tensor(value, start, stop, self.batch_size)

            counts = self.sequence.media_counts
            if counts is not None:
                counts = counts[start:stop]
                offsets = torch.cat(
                    (
                        torch.zeros(1, dtype=counts.dtype, device=counts.device),
                        counts.cumsum(0),
                    )
                )
            else:
                offsets = None
            seq_ids = self.sequence.seq_ids
            if seq_ids is not None:
                seq_ids = seq_ids[start:stop]
            if self.layout is DataLayout.PACKED_VARLEN and seq_ids is not None:
                seq_ids, global_cu, sample_offsets, max_seqlen = _canonical_packed_ids(seq_ids)
                sequence = replace(
                    self.sequence,
                    global_cu_seqlens=global_cu,
                    local_cu_seqlens=None,
                    max_seqlen=max_seqlen,
                    seq_ids=seq_ids,
                    sample_offsets=sample_offsets,
                    media_counts=counts,
                    media_offsets=offsets,
                    cp_rank=0,
                    cp_size=1,
                )
            else:
                sequence = replace(
                    self.sequence,
                    seq_ids=seq_ids,
                    media_counts=counts,
                    media_offsets=offsets,
                )
            result.append(
                replace(
                    self,
                    model_kwargs=model_kwargs,
                    labels=None if self.labels is None else self.labels[start:stop],
                    ce_mask=None if self.ce_mask is None else self.ce_mask[start:stop],
                    kd_mask=None if self.kd_mask is None else self.kd_mask[start:stop],
                    hidden_mask=(
                        None if self.hidden_mask is None else self.hidden_mask[start:stop]
                    ),
                    sequence=sequence,
                    sample_ids=self.sample_ids[start:stop],
                )
            )
        return tuple(result)

    def pad_batch_to_multiple(self, multiple: int) -> "PuzzletronBatch":
        """Append fully masked rows for a static PP schedule without copying media."""

        if multiple < 1:
            raise ValueError("batch multiple must be at least one")
        pad_rows = (-self.batch_size) % int(multiple)
        if pad_rows == 0:
            return self

        model_kwargs: dict[str, Any] = {}
        for key, value in self.model_kwargs.items():
            if not isinstance(value, torch.Tensor) or key in _IMAGE_KEYS or key == "cu_seqlens":
                model_kwargs[key] = value
                continue
            fill_value = -1 if key == "seq_idx" else 0
            model_kwargs[key] = _pad_batch_tensor(
                value,
                batch_size=self.batch_size,
                pad_rows=pad_rows,
                fill_value=fill_value,
            )

        def pad_rows_of(value: torch.Tensor | None, fill_value):
            if value is None:
                return None
            return _pad_batch_tensor(
                value,
                batch_size=self.batch_size,
                pad_rows=pad_rows,
                fill_value=fill_value,
            )

        counts = self.sequence.media_counts
        if counts is not None:
            counts = torch.cat((counts, counts.new_zeros(pad_rows)))
            offsets = torch.cat(
                (
                    torch.zeros(1, dtype=counts.dtype, device=counts.device),
                    counts.cumsum(0),
                )
            )
        else:
            offsets = self.sequence.media_offsets
        sequence = replace(
            self.sequence,
            seq_ids=pad_rows_of(self.sequence.seq_ids, -1),
            media_counts=counts,
            media_offsets=offsets,
        )
        return replace(
            self,
            model_kwargs=model_kwargs,
            labels=pad_rows_of(self.labels, -100),
            ce_mask=pad_rows_of(self.ce_mask, False),
            kd_mask=pad_rows_of(self.kd_mask, False),
            hidden_mask=pad_rows_of(self.hidden_mask, False),
            sequence=sequence,
            sample_ids=(
                *self.sample_ids,
                *(f"__pp_padding_{index}" for index in range(pad_rows)),
            ),
        )
