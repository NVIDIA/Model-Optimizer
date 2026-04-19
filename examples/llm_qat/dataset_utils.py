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

"""Dataset blend utilities for QAT/QAD training.

Provides YAML-driven dataset blending with:
- Multiple dataset sources with configurable ratios
- Chat tokenization via apply_chat_template (correct assistant-only label masking)
- Pretrain tokenization for plain text datasets
- Distributed rank-aware loading and tokenization with disk caching
- Multi-process tokenization via ``num_proc`` (scales with local GPU count)
- Streaming dataset loading to avoid full downloads

Usage as standalone CLI (pre-tokenize and cache):

    python dataset_utils.py \\
        --dataset_config configs/dataset/blend.yaml \\
        --model_name_or_path Qwen/Qwen3-1.7B

Schema reference: See configs/dataset/README.md
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import datasets
import torch
import yaml
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0, warn_rank_0
from modelopt.torch.utils.distributed import DistributedProcessGroup

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetSourceConfig:
    """Configuration for a single dataset source in a blend.

    See configs/dataset/README.md for full schema.
    """

    hf_path: str
    ratio: float
    split: str | dict[str, float] = ""
    dataset_kwargs: dict = field(default_factory=dict)
    apply_chat_template: bool = True
    chat_key: str = "messages"
    category: str = ""

    def __post_init__(self):
        if not self.split:
            raise ValueError(f"{self.hf_path}: 'split' is required")


@dataclass
class BlendConfig:
    """Top-level configuration for a dataset blend.

    See configs/dataset/README.md for full schema.
    """

    sources: list[DatasetSourceConfig] = field(default_factory=list)
    blend_size: int = 100000
    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "eval": 0.1, "test": 0.1}
    )

    def __post_init__(self):
        total = sum(self.splits.values())
        if total <= 0:
            raise ValueError("Split ratios must sum to > 0")
        self.splits = {k: v / total for k, v in self.splits.items()}
        if self.blend_size <= 0:
            raise ValueError(f"blend_size must be > 0, got {self.blend_size}")


@dataclass
class ParallelConfig:
    """Parallelism strategy for dataset processing.

    Combines distributed rank-level sharding with intra-rank multi-process
    tokenization via ``num_proc``. The ``effective_num_proc`` property auto-scales
    workers per rank based on ``local_world_size`` to avoid CPU over-subscription.
    """

    num_proc: int = 16
    rank: int = 0
    world_size: int = 1

    @property
    def local_world_size(self) -> int:
        """Ranks on this node (from ``LOCAL_WORLD_SIZE`` env var set by torchrun/SLURM)."""
        lws = os.environ.get("LOCAL_WORLD_SIZE")
        if lws:
            return int(lws)
        if self.is_distributed:
            warn_rank_0(
                f"LOCAL_WORLD_SIZE not set in distributed mode. "
                f"Falling back to global world_size={self.world_size} (assumes single node)."
            )
            return self.world_size  # conservative: avoids over-subscription
        return 1

    @property
    def effective_num_proc(self) -> int | None:
        """Workers per rank, scaled by local (per-node) rank count.

        Returns ``None`` when sequential processing is appropriate (``num_proc <= 1``
        after scaling), which tells HF ``datasets.map()`` to use the main process.
        """
        lws = self.local_world_size
        n = max(1, self.num_proc // lws) if lws > 1 else self.num_proc
        return n if n > 1 else None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_blend_config(config_path: str) -> BlendConfig:
    """Parse a dataset blend YAML file into a :class:`BlendConfig`."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    sources = [DatasetSourceConfig(**s) for s in raw.get("sources", [])]
    kwargs: dict = {"sources": sources}
    if "blend_size" in raw:
        kwargs["blend_size"] = raw["blend_size"]
    if "splits" in raw:
        kwargs["splits"] = raw["splits"]
    return BlendConfig(**kwargs)


def _normalize_ratios(sources: list[DatasetSourceConfig]) -> list[float]:
    """Return normalized ratio weights summing to 1.0."""
    total = sum(s.ratio for s in sources)
    if total <= 0:
        raise ValueError("Sum of source ratios must be > 0")
    return [s.ratio / total for s in sources]


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank_world() -> tuple[int, int]:
    if not _is_dist_initialized():
        return 0, 1
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _barrier():
    if _is_dist_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Tokenization functions
# ---------------------------------------------------------------------------


def _supports_chatml_heuristic(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Check if tokenizer uses ChatML format (<|im_start|>/<|im_end|>)."""
    try:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        return tokenizer.unk_token_id not in (im_start, im_end)
    except Exception:
        return False


def _chat_template_has_generation(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Return True if the tokenizer's chat template declares ``{% generation %}``.

    Mirrors the regex check used by HuggingFace transformers when deciding
    whether ``return_assistant_tokens_mask=True`` can produce native masks.
    When ``chat_template`` is a dict, only the ``"default"`` template is
    inspected — that is what ``apply_chat_template`` resolves to when called
    without a template name or tools (see ``PreTrainedTokenizerBase.get_chat_template``).
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return False
    if isinstance(template, dict):
        template = template.get("default")
        if not isinstance(template, str):
            return False
    return bool(re.search(r"\{\%-?\s*generation\s*-?\%\}", template))


def _encode_role(tokenizer: PreTrainedTokenizerBase, role: str) -> list[int]:
    """Encode a role string, returning only the role tokens (no special tokens)."""
    return tokenizer.encode(role, add_special_tokens=False)


def _matches_role(input_ids: list[int], start: int, role_ids: list[int]) -> bool:
    """Check if input_ids[start:start+len(role_ids)] matches role_ids."""
    end = start + len(role_ids)
    return end <= len(input_ids) and input_ids[start:end] == role_ids


def _chatml_assistant_mask(input_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Build assistant mask from <|im_start|>assistant ... <|im_end|> boundaries.

    Marks content tokens inside assistant turns as 1, everything else as 0.
    The role tokens and the newline after them are excluded.
    Handles tokenizers where 'assistant' may be split into multiple tokens.
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_ids = _encode_role(tokenizer, "assistant")
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    masks = [0] * len(input_ids)
    n_role = len(assistant_ids)
    in_assistant = False
    skip_remaining = 0
    skip_newline = False

    for i, tid in enumerate(input_ids):
        if tid == im_start_id:
            in_assistant = _matches_role(input_ids, i + 1, assistant_ids)
            if in_assistant:
                skip_remaining = n_role
            skip_newline = False
            continue
        if tid == im_end_id:
            in_assistant = False
            continue
        if in_assistant:
            if skip_remaining > 0:
                skip_remaining -= 1
                if skip_remaining == 0:
                    skip_newline = True
                continue
            if skip_newline and tid == newline_id:
                skip_newline = False
                continue
            masks[i] = 1

    return masks


_TESTED_MODEL_FAMILIES = (
    "qwen",
    "nemotron",
)


def _check_model_family(tokenizer: PreTrainedTokenizerBase) -> None:
    """Warn once if the model is not from a tested family."""
    model_name = getattr(tokenizer, "name_or_path", "") or ""
    name_lower = model_name.lower()
    if not any(family in name_lower for family in _TESTED_MODEL_FAMILIES):
        warn_rank_0(
            f"Model '{model_name}' is not from a tested model family "
            f"({', '.join(_TESTED_MODEL_FAMILIES)}). "
            f"Assistant token masking may be incorrect. "
            f"Please verify training loss and masked tokens manually."
        )


def make_chat_tokenize_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    chat_key: str = "messages",
):
    """Create a tokenize function for chat datasets using ``apply_chat_template``.

    Uses ``return_assistant_tokens_mask=True`` so that only assistant tokens
    contribute to the training loss. Falls back to heuristic ChatML-based masking
    for models whose chat template lacks ``{% generation %}`` support.

    Tested model families (ChatML format): Qwen2, Qwen2.5, Qwen3, Qwen3.5, Nemotron 3.
    """
    _check_model_family(tokenizer)

    use_native = _chat_template_has_generation(tokenizer)
    if not use_native:
        if not _supports_chatml_heuristic(tokenizer):
            model_name = getattr(tokenizer, "name_or_path", "unknown")
            raise ValueError(
                f"Chat template for '{model_name}' does not support "
                f"{{% generation %}} and does not use ChatML format. "
                f"Use make_pretrain_tokenize_fn instead."
            )
        warn_rank_0(
            "Chat template lacks {% generation %} support. "
            "Using heuristic ChatML-based assistant masking."
        )

    def tokenize(sample):
        messages = sample.get(chat_key)
        if not messages:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            return {
                "input_ids": [pad_id] * max_length,
                "attention_mask": [0] * max_length,
                "labels": [IGNORE_TOKEN_ID] * max_length,
            }

        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=use_native,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        except (ValueError, TypeError, KeyError) as e:
            warn_rank_0(f"Failed to tokenize sample: {e}. Skipping.")
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            return {
                "input_ids": [pad_id] * max_length,
                "attention_mask": [0] * max_length,
                "labels": [IGNORE_TOKEN_ID] * max_length,
            }

        input_ids = result["input_ids"]
        if use_native:
            assistant_masks = result["assistant_masks"]
        else:
            assistant_masks = _chatml_assistant_mask(input_ids, tokenizer)

        labels = [tid if mask else IGNORE_TOKEN_ID for tid, mask in zip(input_ids, assistant_masks)]
        return {
            "input_ids": input_ids,
            "attention_mask": result["attention_mask"],
            "labels": labels,
        }

    return tokenize


def make_pretrain_tokenize_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
):
    """Create a tokenize function for plain text (pretraining-style).

    All non-padding tokens are trainable (labels = input_ids).
    """

    def tokenize(sample):
        text = sample.get("text", "")
        if not text:
            # Try common fallbacks
            text = sample.get("article", "") or sample.get("content", "")

        input_ids = tokenizer.encode(text, add_special_tokens=True)[:max_length]
        cur_len = len(input_ids)

        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_token is None:
            raise ValueError("Tokenizer must provide either pad_token_id or eos_token_id")

        attention_mask = [1] * cur_len + [0] * (max_length - cur_len)
        labels = list(input_ids) + [IGNORE_TOKEN_ID] * (max_length - cur_len)
        input_ids = list(input_ids) + [pad_token] * (max_length - cur_len)

        return {
            "input_ids": input_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
            "labels": labels[:max_length],
        }

    return tokenize


# ---------------------------------------------------------------------------
# Dataset loading (streaming, rank-aware)
# ---------------------------------------------------------------------------


def _parse_split_spec(split_spec: str | dict[str, float]) -> dict[str, float]:
    """Parse a split specification into {split_name: weight} dict.

    Examples:
        "train"          -> {"train": 1.0}
        "code,math,stem" -> {"code": 1.0, "math": 1.0, "stem": 1.0}
        {code: 3, math: 2} -> {"code": 3.0, "math": 2.0}
    """
    if isinstance(split_spec, dict):
        return {k: float(v) for k, v in split_spec.items()}
    parts = [p.strip() for p in str(split_spec).split(",") if p.strip()]
    return dict.fromkeys(parts, 1.0)


def _stream_samples(
    hf_path: str,
    split_name: str,
    num_samples: int,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    dataset_kwargs: dict | None = None,
) -> list[dict]:
    """Stream this rank's portion of ``num_samples`` from a single split.

    When ``world_size > 1``, each rank loads only its shard of the data:
    - Local datasets: O(1) random access via ``select()``
    - Streaming datasets: ``skip(offset).take(per_rank)`` (skips without storing)
    """
    per_rank = num_samples // world_size
    offset = rank * per_rank
    if rank == world_size - 1:
        per_rank = num_samples - offset  # last rank gets remainder

    is_local = os.path.exists(hf_path)
    t0 = time.time()

    if is_local:
        print_rank_0(f"\tLoading local dataset {hf_path}...")
        try:
            ds = datasets.load_from_disk(hf_path)
            if isinstance(ds, datasets.DatasetDict):
                ds = ds[split_name]
            if shuffle:
                ds = ds.shuffle(seed=seed)
            end = min(offset + per_rank, len(ds))
            result = list(ds.select(range(offset, end)))
            print_rank_0(f"\tFetched {len(result)} samples in {time.time() - t0:.1f}s")
            return result
        except Exception as e:
            warn_rank_0(f"Failed to load {hf_path} [{split_name}]: {e}. Skipping this split.")
            return []

    print_rank_0(f"\tStreaming {hf_path} [{split_name}]...")
    load_kwargs: dict = {"split": split_name, "streaming": True}
    load_kwargs.update(dataset_kwargs or {})
    print_rank_0(f"\tFetching {per_rank} samples (rank {rank})...")
    try:
        ds = datasets.load_dataset(hf_path, **load_kwargs)
        if shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        result = list(ds.skip(offset).take(per_rank))
    except Exception as e:
        warn_rank_0(f"Failed to stream {hf_path} [{split_name}]: {e}. Skipping this split.")
        return []
    print_rank_0(f"\tFetched {len(result)} samples in {time.time() - t0:.1f}s")
    return result


def _load_source_samples(
    source: DatasetSourceConfig,
    num_samples: int,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict]:
    """Load raw samples from a single source (all splits combined), rank-aware."""
    split_weights = _parse_split_spec(source.split)
    total_weight = sum(split_weights.values())

    all_samples = []
    remaining = num_samples
    split_items = list(split_weights.items())

    for i, (split_name, weight) in enumerate(split_items):
        if i == len(split_items) - 1:
            n = remaining  # last split gets the remainder
        else:
            n = max(1, round(weight / total_weight * num_samples))
            remaining -= n

        samples = _stream_samples(
            source.hf_path,
            split_name,
            n,
            shuffle,
            shuffle_buffer,
            seed,
            rank,
            world_size,
            dataset_kwargs=source.dataset_kwargs,
        )
        print_rank_0(
            f"  {source.hf_path} [{split_name}]: requested {n}, got {len(samples)}"
            f" (rank {rank}/{world_size})"
        )
        all_samples.extend(samples)

    return all_samples


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

_dataset_cache: dict[str, datasets.DatasetDict] = {}


def _tokenizer_fingerprint(tokenizer: PreTrainedTokenizerBase) -> tuple[str, str]:
    """Return ``(short_name, fingerprint)`` for cache key construction.

    The fingerprint captures class name, vocab size, and special token IDs so that
    tokenizers of the same class but different vocabularies produce distinct caches.
    """
    cls_name = type(tokenizer).__name__
    parts = [
        cls_name,
        f"vocab={tokenizer.vocab_size}",
        f"eos={tokenizer.eos_token_id}",
        f"bos={getattr(tokenizer, 'bos_token_id', None)}",
        f"pad={tokenizer.pad_token_id}",
        f"unk={getattr(tokenizer, 'unk_token_id', None)}",
    ]
    return cls_name, "|".join(parts)


def _build_cache_path(
    config: BlendConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    cache_dir: str,
) -> str:
    """Build a deterministic cache path for the blend config."""
    base = cache_dir if cache_dir else tempfile.gettempdir()

    tok_name, tok_fp = _tokenizer_fingerprint(tokenizer)
    splits_str = ",".join(f"{k}:{v}" for k, v in sorted(config.splits.items()))
    sig = f"{tok_fp}|{max_length}|{config.blend_size}|{splits_str}"
    for s in config.sources:
        sig += f"|{s.hf_path}|{s.ratio}|{s.split}|{s.dataset_kwargs}"
    cache_key = hashlib.sha1(sig.encode()).hexdigest()[:12]

    return os.path.join(
        base,
        f"llm_qat_{tok_name}_blend{config.blend_size}_{cache_key}",
    )


def _is_non_empty_dir(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


def _load_cached_dataset(cache_path: str) -> datasets.DatasetDict | None:
    """Try to load from in-memory or disk cache. Returns ``None`` if not cached."""
    if cache_path in _dataset_cache:
        print_rank_0(f"Using in-memory cached dataset: {cache_path}")
        return _dataset_cache[cache_path]

    if _is_non_empty_dir(cache_path):
        if os.path.exists(os.path.join(cache_path, "dataset_dict.json")):
            print_rank_0(f"Using disk-cached dataset: {cache_path}")
            _dataset_cache[cache_path] = datasets.load_from_disk(cache_path)
            return _dataset_cache[cache_path]

    return None


# ---------------------------------------------------------------------------
# Helper: concatenate dataset parts
# ---------------------------------------------------------------------------

_EMPTY_TOKENIZED = {"input_ids": [], "attention_mask": [], "labels": []}


def _concat_parts(parts: list[datasets.Dataset]) -> datasets.Dataset:
    """Concatenate non-empty dataset parts, returning an empty dataset if all are empty."""
    non_empty = [p for p in parts if len(p) > 0]
    if not non_empty:
        return datasets.Dataset.from_dict(_EMPTY_TOKENIZED)
    if len(non_empty) == 1:
        return non_empty[0]
    return datasets.concatenate_datasets(non_empty)


# ---------------------------------------------------------------------------
# Helper: load all source samples (rank-aware)
# ---------------------------------------------------------------------------


def _load_all_source_samples(
    config: BlendConfig,
    norm_ratios: list[float],
    parallel: ParallelConfig,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int,
) -> tuple[list[list[dict]], list[int]]:
    """Load raw samples from all sources for this rank (flat, no split).

    Returns:
        (per_source_samples, per_source_counts) where
        ``per_source_samples[i]`` is the list of raw dicts for source *i*
        and ``per_source_counts[i] = len(per_source_samples[i])``.
    """
    per_source_samples: list[list[dict]] = []
    per_source_counts: list[int] = []

    print_rank_0(f"Loading {len(config.sources)} sources into blend...")

    num_sources = len(config.sources)
    for idx, (source, norm_ratio) in enumerate(zip(config.sources, norm_ratios), 1):
        source_total = max(1, round(norm_ratio * config.blend_size))

        cat_label = f" [{source.category}]" if source.category else ""
        print_rank_0(
            f"Source [{idx}/{num_sources}]: {source.hf_path}{cat_label}"
            f" (ratio={norm_ratio:.3f}, n={source_total})"
        )

        samples = _load_source_samples(
            source,
            source_total,
            shuffle,
            shuffle_buffer,
            seed,
            parallel.rank,
            parallel.world_size,
        )
        per_source_samples.append(samples)
        per_source_counts.append(len(samples))

    local_total = sum(per_source_counts)
    group = DistributedProcessGroup(group=None)
    global_total = DistributedProcessGroup.get_dist_syncd_obj(
        local_total, group, op=lambda objs: sum(objs)
    )
    print_rank_0(f"Total raw samples across all ranks: {global_total}")
    return per_source_samples, per_source_counts


# ---------------------------------------------------------------------------
# Helper: tokenize a single source split
# ---------------------------------------------------------------------------


def _tokenize_source_split(
    source: DatasetSourceConfig,
    raw_samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    parallel: ParallelConfig,
) -> datasets.Dataset:
    """Tokenize raw samples for a single source and split.

    Data is already rank-specific (loaded by ``_load_all_source_samples``),
    so no sharding is needed here. Uses ``parallel.effective_num_proc`` for
    multi-process tokenization.
    """
    if source.apply_chat_template:
        tokenize_fn = make_chat_tokenize_fn(tokenizer, max_length, chat_key=source.chat_key)
    else:
        tokenize_fn = make_pretrain_tokenize_fn(tokenizer, max_length)

    ds = datasets.Dataset.from_list(raw_samples)
    if len(ds) == 0:
        return datasets.Dataset.from_dict(_EMPTY_TOKENIZED)

    print_rank_0(
        f"\tTokenizing {len(raw_samples)} samples (num_proc={parallel.effective_num_proc})..."
    )
    tokenized = ds.map(
        tokenize_fn,
        remove_columns=list(ds.features),
        num_proc=parallel.effective_num_proc,
        desc=f"Tokenizing {source.hf_path} rank {parallel.rank}/{parallel.world_size}",
    )
    before = len(tokenized)
    tokenized = tokenized.filter(
        lambda x: any(label != IGNORE_TOKEN_ID for label in x["labels"]),
        num_proc=parallel.effective_num_proc,
    )
    dropped = before - len(tokenized)
    if dropped:
        warn_rank_0(
            f"Dropped {dropped}/{before} samples with no valid labels "
            f"from {source.hf_path} (all labels are IGNORE_INDEX after tokenization)."
        )
    return tokenized


# ---------------------------------------------------------------------------
# Helper: merge distributed shards
# ---------------------------------------------------------------------------


def _merge_distributed_shards(
    cache_path: str,
    local_flat: datasets.Dataset,
    parallel: ParallelConfig,
    splits: dict[str, float],
    seed: int = 42,
) -> datasets.DatasetDict:
    """Save per-rank flat data, merge on rank 0, shuffle, and split by ratios.

    Each rank saves its local tokenized data as a flat Dataset. Rank 0 loads
    all shards, concatenates, shuffles deterministically, then splits by the
    configured ratios.
    """
    print_rank_0(f"\tSaving rank {parallel.rank} data to disk...")
    temp_dir = os.path.join(cache_path, "temp")
    rank_path = os.path.join(temp_dir, f"rank_{parallel.rank}")
    os.makedirs(rank_path, exist_ok=True)
    local_flat.save_to_disk(rank_path)

    _barrier()

    if parallel.rank == 0:

        def load_rank(r: int) -> datasets.Dataset:
            return datasets.load_from_disk(os.path.join(temp_dir, f"rank_{r}"))

        print_rank_0(f"\tMerging {parallel.world_size} shards...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, parallel.world_size)) as pool:
            all_shards = list(pool.map(load_rank, range(parallel.world_size)))

        merged = _concat_parts(all_shards)
        merged = merged.shuffle(seed=seed)

        total = len(merged)
        split_datasets = {}
        offset = 0
        split_items = list(splits.items())
        for i, (split_name, ratio) in enumerate(split_items):
            if i == len(split_items) - 1:
                count = total - offset
            else:
                count = round(ratio * total)
            split_datasets[split_name] = merged.select(range(offset, offset + count))
            offset += count

        result = datasets.DatasetDict(split_datasets)
        result.save_to_disk(cache_path)

        shutil.rmtree(temp_dir, ignore_errors=True)
        split_summary = ", ".join(f"{k}={len(v)}" for k, v in result.items())
        print_rank_0(f"Cached blended dataset to {cache_path} ({split_summary})")

    _barrier()

    return datasets.load_from_disk(cache_path)


# ---------------------------------------------------------------------------
# Core: build blended dataset
# ---------------------------------------------------------------------------


def build_blend_dataset(
    config: BlendConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    seed: int = 42,
    cache_dir: str = ".dataset_cache/tokenized",
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
    num_proc: int = 16,
) -> datasets.DatasetDict:
    """Build a blended, tokenized dataset from a :class:`BlendConfig`.

    Returns a ``DatasetDict`` with keys matching ``config.splits``
    (e.g. ``"train"``, ``"eval"``, ``"test"``).
    """
    cache_path = _build_cache_path(config, tokenizer, max_length, cache_dir)

    cached = _load_cached_dataset(cache_path)
    if cached is not None:
        return cached

    rank, world_size = _dist_rank_world()
    parallel = ParallelConfig(num_proc=num_proc, rank=rank, world_size=world_size)

    if rank == 0:
        os.makedirs(cache_path, exist_ok=True)
    _barrier()

    norm_ratios = _normalize_ratios(config.sources)
    per_source_samples, per_source_counts = _load_all_source_samples(
        config, norm_ratios, parallel, shuffle, shuffle_buffer, seed
    )

    print_rank_0(f"Tokenizing {len(config.sources)} sources...")
    tokenized_parts: list[datasets.Dataset] = []
    for source, samples in zip(config.sources, per_source_samples):
        if samples:
            tokenized_parts.append(
                _tokenize_source_split(source, samples, tokenizer, max_length, parallel)
            )

    local_flat = _concat_parts(tokenized_parts)

    print_rank_0("Merging distributed shards...")
    result = _merge_distributed_shards(cache_path, local_flat, parallel, config.splits, seed)
    _dataset_cache[cache_path] = result
    return result


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main():
    import transformers
    from arguments import DataArguments, ModelArguments

    from modelopt.torch.opt.plugins.transformers import ModelOptArgParser

    parser = ModelOptArgParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    config = load_blend_config(data_args.dataset_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=model_args.model_max_length
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = build_blend_dataset(
        config,
        tokenizer,
        model_args.model_max_length,
        seed=data_args.dataset_seed,
        cache_dir=data_args.dataset_cache_dir,
        shuffle=data_args.shuffle,
        shuffle_buffer=data_args.shuffle_buffer,
        num_proc=data_args.num_proc,
    )
    split_summary = ", ".join(f"{k}: {len(v)}" for k, v in ds.items())
    print(f"Built dataset: {split_summary}")


if __name__ == "__main__":
    main()
