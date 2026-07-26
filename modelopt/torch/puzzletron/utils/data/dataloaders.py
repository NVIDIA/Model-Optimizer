# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DataLoader utilities for language model training and validation."""

import hashlib
import importlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

import datasets
import torch
import torch.distributed
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ...tools.logger import mprint
from .dataset import ConstantLengthDataset
from .packed_memmap import PackedTokenMemmapDataset

__all__ = [
    "create_train_dataloader",
    "create_validation_dataloader",
    "create_padded_tensor",
    "prepare_validation_dataloader",
    "prepare_automodel_text_validation_dataloader",
    "prepare_multimodal_validation_dataloader",
]


def _cfg_get(args, key: str, default=None):
    try:
        return args.get(key, default)
    except Exception:
        return getattr(args, key, default)


def _resolve_load_dataset_fn(load_dataset_fn):
    if load_dataset_fn is None:
        return load_from_disk_fn
    if callable(load_dataset_fn):
        return load_dataset_fn
    if isinstance(load_dataset_fn, str):
        aliases = {
            "disk": load_from_disk_fn,
            "load_from_disk": load_from_disk_fn,
            "load_from_disk_fn": load_from_disk_fn,
            "streaming": load_streaming_fn,
            "load_streaming": load_streaming_fn,
            "load_streaming_fn": load_streaming_fn,
        }
        if load_dataset_fn in aliases:
            return aliases[load_dataset_fn]
        module_name, _, object_name = load_dataset_fn.rpartition(".")
        if module_name and object_name:
            value = getattr(importlib.import_module(module_name), object_name)
            if callable(value):
                return value
    raise TypeError(f"Unsupported load_dataset_fn value: {load_dataset_fn!r}")


def prepare_validation_dataloader(
    args,
    tokenizer: PreTrainedTokenizerBase | None = None,
    *,
    data_layout: str | None = None,
):
    """Build the shared deterministic validation loader without a model backend dependency."""
    if tokenizer is None:
        tokenizer_name = _cfg_get(args, "tokenizer_name", None)
        model_name_or_path = _cfg_get(
            args,
            "model_name_or_path",
            _cfg_get(args, "teacher_dir", None),
        )
        if tokenizer_name is None and model_name_or_path is None:
            raise ValueError("validation data requires tokenizer_name or model_name_or_path")
        trust_remote_code = _cfg_get(args, "trust_remote_code", False)
        descriptor_name = _cfg_get(args, "descriptor", None)
        if not trust_remote_code and descriptor_name:
            from ...anymodel.model_descriptor import ModelDescriptorFactory

            descriptor = ModelDescriptorFactory.get(descriptor_name)
            trust_remote_code = descriptor.requires_trust_remote_code()
        started = time.monotonic()
        mprint(
            "prepare_validation_dataloader: loading tokenizer from "
            f"{tokenizer_name or model_name_or_path} "
            f"(trust_remote_code={trust_remote_code})"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        mprint(
            "prepare_validation_dataloader: tokenizer ready "
            f"({type(tokenizer).__name__}, {time.monotonic() - started:.1f}s)"
        )

    if data_layout in {"padded_varlen", "packed_varlen"}:
        return prepare_automodel_text_validation_dataloader(
            args,
            tokenizer=tokenizer,
            data_layout=data_layout,
        )

    packed_path = _cfg_get(args, "packed_token_cache_path", None)
    if packed_path:
        dataset = PackedTokenMemmapDataset(
            packed_path,
            limit=_cfg_get(args, "eval_samples", None),
        )
        return DataLoader(
            dataset,
            batch_size=_cfg_get(args, "micro_batch_size", 1),
            pin_memory=True,
            collate_fn=collate_fn_with_none_support,
        )

    loader = create_validation_dataloader(
        accelerator=None,
        seed=_cfg_get(args, "seed", 42),
        tokenizer=tokenizer,
        block_size=_cfg_get(args, "block_size", 4096),
        dataset=_cfg_get(args, "dataset_path", _cfg_get(args, "dataset", None)),
        content_field=_cfg_get(args, "data_column", "messages"),
        fim_rate=_cfg_get(args, "fim_rate", 0.0),
        fim_spm_rate=_cfg_get(args, "fim_spm_rate", 0.0),
        micro_batch_size=_cfg_get(args, "micro_batch_size", 1),
        eval_samples=_cfg_get(args, "eval_samples", None),
        dataset_name=_cfg_get(args, "val_dataset_name", "__auto__"),
        source_datasets_to_discard=_cfg_get(args, "source_datasets_to_discard", tuple()),
        bos_rate=_cfg_get(args, "bos_rate", 1.0),
        varlen=(
            data_layout == "packed_varlen"
            if data_layout is not None
            else _cfg_get(args, "varlen", True)
        ),
        shuffle_seed=_cfg_get(args, "shuffle_seed", None),
        load_dataset_fn=_resolve_load_dataset_fn(_cfg_get(args, "load_dataset_fn", None)),
        realized_cache_dir=_cfg_get(args, "realized_dataset_cache_dir", None),
    )
    mprint(
        "prepare_validation_dataloader: ready "
        f"(len={len(loader) if hasattr(loader, '__len__') else 'unknown'})"
    )
    return loader


def prepare_automodel_text_validation_dataloader(
    args,
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_layout: str,
):
    """Build AutoModel-native padded or neat-packed text validation data."""

    from ...distillation.dataset import make_puzzletron_chat_dataset

    dataset_path = _cfg_get(args, "dataset_path", _cfg_get(args, "dataset", None))
    if not dataset_path:
        raise ValueError("variable-length text data requires dataset_path or dataset")
    split = str(_cfg_get(args, "val_dataset_name", "validation"))
    eval_samples = _cfg_get(args, "eval_samples", None)
    max_length = int(_cfg_get(args, "block_size", 4096))
    dataset = make_puzzletron_chat_dataset(
        tokenizer=tokenizer,
        dataset_path=str(dataset_path),
        split=split,
        num_samples=eval_samples if data_layout == "padded_varlen" else None,
        seq_length=max_length,
        seed=int(_cfg_get(args, "seed", 42)),
    )
    if data_layout == "packed_varlen":
        from nemo_automodel.components.datasets.llm.neat_packing import neat_pack_dataset
        from nemo_automodel.components.datasets.utils import neat_packed_collater
        from nemo_automodel.components.models.common.packing import configure_packing

        dataset = neat_pack_dataset(
            dataset,
            split=split,
            pack_size=max_length,
            max_packs=eval_samples,
            padding_idx=getattr(tokenizer, "pad_token_id", 0) or 0,
            drop_long_samples=True,
        )
        configure_packing(attn_implementation="flash_attention_2")
        collate_fn = partial(
            neat_packed_collater,
            attn_implementation="flash_attention_2",
        )
    else:
        from nemo_automodel.components.datasets.utils import default_collater

        collate_fn = default_collater
    return DataLoader(
        dataset,
        batch_size=int(_cfg_get(args, "micro_batch_size", 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def prepare_multimodal_validation_dataloader(
    args: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path,
    data_layout: str,
):
    """Build the native AutoModel processor/collator validation path for VLM batches."""

    from transformers import AutoProcessor

    from ...dataset import load_materialized_conversation_dataset

    data_path = _cfg_get(args, "path", _cfg_get(args, "dataset_path", None))
    if not data_path:
        raise ValueError("multimodal width-slice data requires data.path or dataset_path")
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        trust_remote_code=bool(_cfg_get(args, "trust_remote_code", True)),
    )
    dataset = load_materialized_conversation_dataset(
        data_path,
        num_samples=_cfg_get(args, "eval_samples", None),
    )
    from nemo_automodel.components.datasets.vlm.collate_fns import (
        neat_packed_vlm_collater,
        pad_collate_fn,
    )
    from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

    max_length = int(_cfg_get(args, "max_sample_length", _cfg_get(args, "block_size", 4096)))
    tokenized = PreTokenizedDatasetWrapper(
        dataset,
        processor,
        max_length=max_length,
        truncate=False,
        inject_fake_images=False,
    )
    if data_layout == "packed_varlen":
        from nemo_automodel.components.datasets.vlm.neat_packing_vlm import neat_pack_dataset_vlm

        packing = _cfg_get(args, "packing", {}) or {}
        pack_size = int(
            packing.get("pack_size", _cfg_get(args, "pack_size", max_length))
        )
        tokenized = neat_pack_dataset_vlm(
            tokenized,
            pack_size=pack_size,
            padding_idx=getattr(processor.tokenizer, "pad_token_id", 0) or 0,
            drop_long_samples=True,
            max_packs=_cfg_get(args, "eval_samples", None),
            ds_raw=dataset,
            packing_ratio=float(
                packing.get(
                    "packing_ratio",
                    _cfg_get(args, "packing_ratio", 1.0),
                )
            ),
            processor=processor,
        )
        collate_fn = partial(
            neat_packed_vlm_collater,
            padding_idx=getattr(processor.tokenizer, "pad_token_id", 0) or 0,
            max_length=pack_size,
            attn_implementation="flash_attention_2",
        )
    else:
        collate_fn = partial(pad_collate_fn, processor=processor, max_length=max_length)
    return DataLoader(
        tokenized,
        batch_size=int(_cfg_get(args, "micro_batch_size", 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def collate_none_fn(
    batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None
):
    return None


collate_fn_map_with_none_support = {**default_collate_fn_map, type(None): collate_none_fn}
collate_fn_with_none_support = partial(collate, collate_fn_map=collate_fn_map_with_none_support)


class LoadDatasetFn(Protocol):
    def __call__(
        self, dataset_path: str, content_field: str, keep_in_memory: bool = False
    ) -> Mapping[str, Dataset]: ...


def load_from_disk_fn(
    dataset_path: str, content_field: str, keep_in_memory: bool = False
) -> Mapping[str, Dataset]:
    return datasets.load_from_disk(dataset_path, keep_in_memory=keep_in_memory)


def load_streaming_fn(
    dataset_path: str, content_field: str, keep_in_memory: bool = False
) -> Mapping[str, Dataset]:
    dataset = datasets.load_dataset(
        dataset_path,
        streaming=True,
        features=datasets.Features(
            {
                content_field: datasets.Value(dtype="string"),
            }
        ),
        keep_in_memory=keep_in_memory,
    )

    return dataset


def create_train_dataloader(
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    dataset_path: str | Mapping[str, Dataset],
    content_field: str,
    fim_rate: float,
    fim_spm_rate: float,
    micro_batch_size: int,
    load_dataset_fn: LoadDatasetFn = load_from_disk_fn,
    dataset_name: str = "train",
    keep_in_memory: bool = False,
    shuffle_seed: int | None = None,
    source_datasets_to_discard: Sequence[str] = (),
    bos_rate: float = 1.0,
    num_workers: int = 0,
    packed_token_cache_path: str | Path | None = None,
) -> DataLoader:
    """Create an infinite training DataLoader over ConstantLengthDataset."""
    # ConstantLengthDataset.__iter__ does not consult torch.utils.data.get_worker_info()
    # to shard work across DataLoader workers, so num_workers > 0 would have every
    # worker iterate the full dataset and emit duplicate samples. Reject explicitly
    # until ConstantLengthDataset gains worker-aware iteration; the guard can then
    # be removed.
    if num_workers > 0:
        raise ValueError(
            f"create_train_dataloader: num_workers={num_workers} is not supported "
            f"because ConstantLengthDataset.__iter__ does not shard via "
            f"torch.utils.data.get_worker_info(). Use num_workers=0 (the default) "
            f"or add worker-aware sharding to ConstantLengthDataset.__iter__."
        )

    if packed_token_cache_path:
        return DataLoader(
            PackedTokenMemmapDataset(
                packed_token_cache_path,
                sequence_length=block_size,
            ),
            batch_size=micro_batch_size,
            pin_memory=True,
            num_workers=0,
        )

    if isinstance(dataset_path, str):
        dataset = load_dataset_fn(dataset_path, content_field, keep_in_memory)
    else:
        dataset = dataset_path

    train_data = dataset[dataset_name]
    if shuffle_seed is not None:
        # `keep_in_memory` is only valid on map-style HF Datasets; streaming
        # `IterableDataset.shuffle()` only accepts `seed` (and an optional
        # `buffer_size`). Branch on the dataset type so streaming users
        # (`load_from_disk: false`) don't crash on this call.
        if isinstance(train_data, datasets.IterableDataset):
            train_data = train_data.shuffle(seed=shuffle_seed)
        else:
            train_data = train_data.shuffle(seed=shuffle_seed, keep_in_memory=keep_in_memory)

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=block_size,
        content_field=content_field,
        fim_rate=fim_rate,
        fim_spm_rate=fim_spm_rate,
        seed=seed,
        source_datasets_to_discard=source_datasets_to_discard,
        bos_rate=bos_rate,
    )

    return DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )


def _realized_cache_path(
    realized_cache_dir: str | Path | None,
    *,
    dataset: str | Mapping[str, Dataset],
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    micro_batch_size: int,
    varlen: bool,
    content_field: str,
    fim_rate: float,
    fim_spm_rate: float,
    seed: int,
    shuffle_seed: int | None,
    source_datasets_to_discard: Sequence[str],
    bos_rate: float,
    eval_samples: int | None,
    load_dataset_fn: LoadDatasetFn,
) -> Path | None:
    """Content-addressed path for a realized validation set, or ``None`` to disable caching.

    Caching is only possible when ``dataset`` is a path string (an already-loaded
    dataset object can't be hashed reliably). The key captures every input that
    changes the realized (tokenized + packed) examples, so a different block_size,
    eval_samples, split, seed, tokenizer, or dataset produces a different file and
    never reuses a stale one.
    """
    if realized_cache_dir is None or not isinstance(dataset, str):
        return None
    # For non-varlen datasets the realized cache is a list of individual fixed-length
    # examples; ``micro_batch_size`` only changes the DataLoader batching after the
    # cache is loaded. Varlen packing bakes the microbatch into each packed sequence,
    # so it must remain part of the key there.
    realized_micro_batch_size = micro_batch_size if varlen else 1
    key = {
        # Version 2 persists packed sample boundaries (cu_seqlens). Caches from
        # the legacy boundary-free format must never be reused by canonical DP/CP.
        "packing_contract_version": 2,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer)),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "block_size": block_size,
        "micro_batch_size": realized_micro_batch_size,
        "varlen": varlen,
        "content_field": content_field,
        "fim_rate": fim_rate,
        "fim_spm_rate": fim_spm_rate,
        "seed": seed,
        "shuffle_seed": shuffle_seed,
        "source_datasets_to_discard": list(source_datasets_to_discard or ()),
        "bos_rate": bos_rate,
        "eval_samples": eval_samples,
        "loader": getattr(load_dataset_fn, "__name__", str(load_dataset_fn)),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return Path(realized_cache_dir) / f"realized_val-{digest}.pt"


def create_validation_dataloader(
    accelerator: Accelerator | None,
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    dataset: str | Mapping[str, Dataset],
    content_field: str,
    fim_rate: float,
    fim_spm_rate: float,
    micro_batch_size: int,
    eval_samples: int | None = None,
    load_dataset_fn: LoadDatasetFn = load_from_disk_fn,
    dataset_name: str = "__auto__",
    keep_in_memory: bool = False,
    source_datasets_to_discard: Sequence[str] = (),
    bos_rate: float = 1.0,
    varlen: bool = True,
    shuffle_seed: int | None = None,
    realized_cache_dir: str | Path | None = None,
    packed_token_cache_path: str | Path | None = None,
):
    if packed_token_cache_path:
        return DataLoader(
            PackedTokenMemmapDataset(
                packed_token_cache_path,
                limit=eval_samples,
                sequence_length=block_size,
            ),
            batch_size=micro_batch_size,
            pin_memory=True,
            collate_fn=collate_fn_with_none_support,
        )

    if accelerator is None:
        accelerator = Printer()

    if accelerator.is_main_process:
        t0 = time.monotonic()
        # Content-addressed on-disk cache of the realized (tokenized + packed)
        # validation set. Realizing it is single-process and expensive at long
        # block_size, so a cache hit skips both the raw load and the packing. The
        # key is GPU-count independent, so a file written by a 1-GPU run is reused
        # as-is by a multi-GPU run.
        cache_path = _realized_cache_path(
            realized_cache_dir,
            dataset=dataset,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            block_size=block_size,
            micro_batch_size=micro_batch_size,
            varlen=varlen,
            content_field=content_field,
            fim_rate=fim_rate,
            fim_spm_rate=fim_spm_rate,
            seed=seed,
            shuffle_seed=shuffle_seed,
            source_datasets_to_discard=source_datasets_to_discard,
            bos_rate=bos_rate,
            eval_samples=eval_samples,
            load_dataset_fn=load_dataset_fn,
        )
        mprint(f"Validation cache path: {cache_path}")
        if cache_path is not None and cache_path.exists():
            mprint(f"Loading realized validation dataset from cache: {cache_path}")
            val_offloaded_dataset = torch.load(cache_path, weights_only=False)
        else:
            if isinstance(dataset, str):
                mprint(f"Loading validation dataset from disk: {dataset}")
                load_t0 = time.monotonic()
                dataset = load_dataset_fn(dataset, content_field, keep_in_memory)
                mprint(f"Loaded validation dataset ({time.monotonic() - load_t0:.1f}s)")

            if isinstance(dataset, datasets.Dataset | torch.utils.data.Dataset):
                valid_data = dataset
                mprint(
                    "#### Path to specific dataset was given (not DatasetDict), taking it as-is ####"
                )
            else:
                assert isinstance(dataset, datasets.DatasetDict)
                if dataset_name == "__auto__":
                    val_split_options = []
                    for val_key_prefix in ("val", "test"):
                        if len(val_split_options) == 0:
                            val_split_options = [
                                split
                                for split in dataset  # DatasetDict is dict-like and supports direct iteration
                                if split.lower().startswith(val_key_prefix)
                            ]
                    assert len(val_split_options) == 1, (
                        f"Expected exactly one validation split, got {val_split_options=} ({dataset.keys()=})"
                    )
                    val_split = val_split_options[0]
                    mprint(f"Inferred validation split automatically: '{val_split}'")
                else:
                    val_split = dataset_name
                    mprint(f"Validation split explicitly chosen: '{val_split}'")
                valid_data = dataset[val_split]

            if shuffle_seed is not None:
                mprint(f"Shuffling with {shuffle_seed=}")
                valid_data = valid_data.shuffle(seed=shuffle_seed)

            valid_dataset = ConstantLengthDataset(
                tokenizer,
                valid_data,
                infinite=False,
                seq_length=block_size * micro_batch_size if varlen else block_size,
                content_field=content_field,
                fim_rate=fim_rate,
                fim_spm_rate=fim_spm_rate,
                seed=seed,
                source_datasets_to_discard=source_datasets_to_discard,
                bos_rate=bos_rate,
                return_cu_seqlens=varlen,
                seqlen_cap=block_size if varlen else None,
            )
            if varlen and eval_samples is not None:
                eval_samples = eval_samples // micro_batch_size
            mprint(
                "Realizing validation dataset in memory "
                f"(realized_samples={eval_samples}, varlen={varlen})"
            )
            realize_t0 = time.monotonic()
            val_offloaded_dataset = realize_dataset_in_memory(valid_dataset, eval_samples)
            mprint(f"Realized validation dataset ({time.monotonic() - realize_t0:.1f}s)")

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Each process needs its own temporary path: independent RPC workers
                # can realize the same deterministic cache concurrently.  A shared
                # ``.tmp`` name lets one worker rename another worker's file.
                tmp_path = cache_path.with_name(
                    f"{cache_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
                )
                mprint(f"Saving realized validation dataset cache: {cache_path}")
                try:
                    torch.save(val_offloaded_dataset, tmp_path)
                    tmp_path.replace(cache_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                mprint(f"Saved realized validation dataset to cache: {cache_path}")

        valid_data_len = len(val_offloaded_dataset)
        mprint(f"num validation examples = {valid_data_len} ({time.monotonic() - t0:.1f}s)")
    else:
        val_offloaded_dataset = None

    if not isinstance(accelerator, Printer):
        obj_list = [val_offloaded_dataset]
        torch.distributed.broadcast_object_list(obj_list)
        val_offloaded_dataset = obj_list[0]

    # let accelerate prepare to handle distributed sampling
    val_dataloader = DataLoader(
        val_offloaded_dataset,
        batch_size=1 if varlen else micro_batch_size,
        pin_memory=True,
        collate_fn=collate_fn_with_none_support,
    )

    return val_dataloader


def realize_dataset_in_memory(dataset: IterableDataset, eval_samples: int | None) -> list[dict]:
    tqdm_desc = f"realize_dataset_in_memory({eval_samples=})"
    if eval_samples is None:
        offloaded_dataset = list(tqdm(dataset, desc=tqdm_desc))
    else:
        val_iter = iter(dataset)
        offloaded_dataset = [next(val_iter) for _ in tqdm(range(eval_samples), desc=tqdm_desc)]
    return offloaded_dataset


TensorT = TypeVar("TensorT", bound=torch.Tensor)


@torch.no_grad()
def create_padded_tensor(
    tensor: TensorT, desired_shape: Sequence[int], padding_value: float = 0
) -> TensorT:
    if tensor.shape == torch.Size(desired_shape):
        return tensor

    padded_tensor = torch.full(
        desired_shape, fill_value=padding_value, dtype=tensor.dtype, device=tensor.device
    )
    indices = torch.where(torch.ones_like(tensor, dtype=torch.bool))
    padded_tensor[indices] = tensor.view(-1)
    return padded_tensor


class Printer:
    is_main_process = True
    process_index = None

    @staticmethod
    def print(*args, **kwargs) -> None:
        print(*args, **kwargs)
