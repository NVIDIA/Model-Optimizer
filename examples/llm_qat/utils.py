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

import gc
import hashlib
import os
import tempfile
import types
from contextlib import contextmanager
from functools import partial

import datasets
import torch
import transformers
from peft import LoraConfig, TaskType
from transformers import default_data_collator

IGNORE_INDEX = -100


@contextmanager
def main_process_first():
    """Context manager to run code on the main process first."""
    if not torch.distributed.is_initialized():
        yield
        return

    rank = torch.distributed.get_rank()
    if rank == 0:
        yield
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        yield
    torch.distributed.barrier()


def _normalize_to_messages(sample: dict) -> dict:
    """Convert Daring-Anteater conversations format to standard messages format."""
    return {
        "messages": [
            {"role": turn["from"].lower(), "content": turn["value"]}
            for turn in sample.get("conversations", [])
        ]
    }


def _is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank_world() -> tuple[int, int]:
    if not _is_dist_initialized():
        return 0, 1
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _role_to_text(role: str) -> str:
    role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
    return role_map.get(role.lower(), role.capitalize())


def _resolve_jsonl_files(dataset_path: str) -> list[str]:
    """Resolve comma-separated paths (files and/or directories) to a list of JSONL files."""
    parts = [p.strip() for p in dataset_path.split(",") if p.strip()]
    collected = []
    for p in parts:
        if os.path.isfile(p):
            collected.append(p)
        elif os.path.isdir(p):
            jsonl_files = sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".jsonl"))
            if not jsonl_files:
                import warnings

                warnings.warn(f"No .jsonl files found in directory: {p}")
            collected.extend(jsonl_files)
        else:
            raise ValueError(f"Dataset path does not exist: {p}")
    if not collected:
        raise ValueError(f"No .jsonl files resolved from: {dataset_path}")
    return collected


def _build_cache_path(
    dataset: str, dataset_cache_path: str, max_length: int, train_size: int, eval_size: int
) -> str:
    if dataset_cache_path:
        return dataset_cache_path
    cache_key = hashlib.sha1(
        f"{dataset}|{max_length}|{train_size}|{eval_size}".encode()
    ).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"llm_qat_tokenized_{cache_key}")


def _is_non_empty_dir(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


def _make_tokenize_fn(tokenizer: transformers.PreTrainedTokenizer, max_length: int):
    def process_and_tokenize(sample):
        messages = sample.get("messages") or []
        all_input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        all_labels = [IGNORE_INDEX] if tokenizer.bos_token_id is not None else []

        for message in messages:
            role = _role_to_text(str(message.get("role", "")))
            content = str(message.get("content", ""))
            text = f"{role}: {content}\n"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            labels = input_ids if role == "Assistant" else [IGNORE_INDEX] * len(input_ids)
            all_input_ids.extend(input_ids)
            all_labels.extend(labels)
            if len(all_input_ids) > max_length:
                break

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            all_input_ids.append(eos_token_id)
            all_labels.append(IGNORE_INDEX)
        all_attention_mask = [1] * len(all_input_ids)

        cur_seq_length = len(all_input_ids)
        if cur_seq_length < max_length:
            pad_token = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
            )
            if pad_token is None:
                raise ValueError("Tokenizer must provide either pad_token_id or eos_token_id")
            all_input_ids += [pad_token] * (max_length - cur_seq_length)
            all_attention_mask += [0] * (max_length - cur_seq_length)
            all_labels += [IGNORE_INDEX] * (max_length - cur_seq_length)

        return {
            "input_ids": all_input_ids[:max_length],
            "attention_mask": all_attention_mask[:max_length],
            "labels": all_labels[:max_length],
        }

    return process_and_tokenize


_dataset_cache: dict = {}


def _load_cached_dataset(
    dataset_id: str,
    load_raw_fn,
    tokenizer: transformers.PreTrainedTokenizer,
    split: str,
    max_length: int,
    train_size: int = 0,
    eval_size: int = 0,
    dataset_cache_path: str = "",
    tokenize_fn_factory=None,
) -> datasets.Dataset:
    """Load, tokenize, split and cache a dataset in messages format.

    Args:
        dataset_id: Identifier for cache key (e.g. "Daring-Anteater" or a file path).
        load_raw_fn: Callable returning an HF Dataset with a ``messages`` column.
        split: "train" or "test".
        tokenize_fn_factory: Optional callable(tokenizer, max_length) -> tokenize_fn.
            Defaults to ``_make_tokenize_fn`` (chat/messages format).
    """
    cache_path = _build_cache_path(
        dataset_id, dataset_cache_path, max_length, train_size, eval_size
    )

    if cache_path in _dataset_cache:
        return _dataset_cache[cache_path][split]

    if _is_non_empty_dir(cache_path):
        _dataset_cache[cache_path] = datasets.load_from_disk(cache_path)
        return _dataset_cache[cache_path][split]

    rank, world_size = _dist_rank_world()
    if rank == 0:
        os.makedirs(cache_path, exist_ok=True)
    if _is_dist_initialized():
        torch.distributed.barrier()

    raw_dataset = load_raw_fn()
    eval_size = min(2000, len(raw_dataset)) if eval_size == 0 else eval_size
    if eval_size <= 0:
        raise ValueError("eval_size must be > 0")
    train_size = len(raw_dataset) - eval_size if train_size == 0 else train_size
    if train_size <= 0 or train_size + eval_size > len(raw_dataset):
        raise ValueError("not enough data for train-eval split")

    selected = raw_dataset.shuffle(seed=42).select(range(train_size + eval_size))
    split_dataset = selected.train_test_split(test_size=eval_size, shuffle=True, seed=42)

    if tokenize_fn_factory is None:
        tokenize_fn_factory = _make_tokenize_fn
    tokenize_fn = tokenize_fn_factory(tokenizer, max_length)
    tokenized_parts = {}
    for split_name in ["train", "test"]:
        shard = split_dataset[split_name].shard(num_shards=world_size, index=rank, contiguous=True)
        remove_columns = (
            list(shard.features) if len(shard) > 0 else list(split_dataset[split_name].features)
        )
        tokenized_parts[split_name] = shard.map(
            tokenize_fn,
            remove_columns=remove_columns,
            desc=f"Tokenizing {split_name} split rank {rank}/{world_size}",
        )

    rank_cache_path = os.path.join(cache_path, "rank_parts", f"rank_{rank}")
    os.makedirs(rank_cache_path, exist_ok=True)
    datasets.DatasetDict(tokenized_parts).save_to_disk(rank_cache_path)

    if _is_dist_initialized():
        torch.distributed.barrier()

    if rank == 0:
        merged = {}
        for split_name in ["train", "test"]:
            shard_list = []
            for worker_rank in range(world_size):
                shard_path = os.path.join(cache_path, "rank_parts", f"rank_{worker_rank}")
                shard_ds = datasets.load_from_disk(shard_path)[split_name]
                shard_list.append(shard_ds)
            merged[split_name] = (
                datasets.concatenate_datasets(shard_list) if len(shard_list) > 1 else shard_list[0]
            )
        datasets.DatasetDict(merged).save_to_disk(cache_path)

    if _is_dist_initialized():
        torch.distributed.barrier()
    _dataset_cache[cache_path] = datasets.load_from_disk(cache_path)
    return _dataset_cache[cache_path][split]


def _make_pretrain_tokenize_fn(tokenizer: transformers.PreTrainedTokenizer, max_length: int):
    """Tokenize plain text for causal LM pretraining (all tokens are predicted)."""

    def process_and_tokenize(sample):
        text = sample.get("text", "")
        input_ids = tokenizer.encode(text, add_special_tokens=True)[:max_length]

        cur_len = len(input_ids)
        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_token is None:
            raise ValueError("Tokenizer must provide either pad_token_id or eos_token_id")
        attention_mask = [1] * cur_len + [0] * (max_length - cur_len)
        labels = input_ids + [IGNORE_INDEX] * (max_length - cur_len)
        input_ids = input_ids + [pad_token] * (max_length - cur_len)

        return {
            "input_ids": input_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
            "labels": labels[:max_length],
        }

    return process_and_tokenize


def _load_fineweb_edu_raw(num_samples: int) -> datasets.Dataset:
    """Stream fineweb_edu_100BT and materialize num_samples into an in-memory Dataset."""
    ds_iter = datasets.load_dataset(
        "HuggingFaceFW/fineweb_edu_100BT", split="train", streaming=True
    )
    rows = []
    for i, sample in enumerate(ds_iter):
        if i >= num_samples:
            break
        rows.append({"text": sample["text"]})
    return datasets.Dataset.from_list(rows)


def _load_daring_anteater_raw() -> datasets.Dataset:
    ds = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
    return ds.map(_normalize_to_messages, remove_columns=list(ds.features))


def get_daring_anteater(
    tokenizer: transformers.AutoTokenizer,
    split="train",
    max_length=4096,
    train_size=0,
    eval_size=0,
):
    return _load_cached_dataset(
        "Daring-Anteater",
        _load_daring_anteater_raw,
        tokenizer,
        split,
        max_length,
        train_size,
        eval_size,
    )


def make_supervised_data_module(
    dataset="Daring-Anteater",
    dataset_cache_path: str = "",
    tokenizer: transformers.PreTrainedTokenizer = None,
    train_size: int = 0,
    eval_size: int = 0,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    cache_ready = bool(dataset_cache_path) and _is_non_empty_dir(dataset_cache_path)

    tokenize_fn_factory = None  # default: chat/messages format

    if dataset == "Daring-Anteater":
        load_raw = _load_daring_anteater_raw
        dataset_id = "Daring-Anteater"
    elif dataset == "fineweb_edu":
        total = (train_size or 10000) + (eval_size or 2000)
        load_raw = partial(_load_fineweb_edu_raw, num_samples=total)
        dataset_id = "fineweb_edu"
        tokenize_fn_factory = _make_pretrain_tokenize_fn
    elif os.path.exists(dataset) or cache_ready:
        data_files = _resolve_jsonl_files(dataset) if os.path.exists(dataset) else None

        def load_raw():
            return datasets.load_dataset("json", data_files=data_files, split="train")

        dataset_id = dataset
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    kwargs = {
        "tokenizer": tokenizer,
        "max_length": tokenizer.model_max_length,
        "train_size": train_size,
        "eval_size": eval_size,
        "dataset_cache_path": dataset_cache_path,
        "tokenize_fn_factory": tokenize_fn_factory,
    }
    return {
        "train_dataset": _load_cached_dataset(dataset_id, load_raw, split="train", **kwargs),
        "eval_dataset": _load_cached_dataset(dataset_id, load_raw, split="test", **kwargs),
        "data_collator": default_data_collator,
    }


def get_lora_config():
    return LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def monkey_patch_training_step_to_fix_memory_leak(trainer):
    def new_func(original_f_name, trainer, *args, **kwargs):
        gc.collect()
        return getattr(trainer, original_f_name)(*args, **kwargs)

    for f_name in ["training_step", "prediction_step", "_load_best_model"]:
        setattr(trainer, "_original_" + f_name, getattr(trainer, f_name))
        setattr(
            trainer, f_name, types.MethodType(partial(new_func, "_original_" + f_name), trainer)
        )


def get_metrics_with_perplexity(metrics):
    """Add perplexity to the metrics."""
    if "eval_loss" in metrics:
        metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])))
    return metrics
