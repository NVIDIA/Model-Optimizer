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

"""Utility functions for getting samples and forward loop function for different datasets."""

import copy
import json
import os
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _join_messages_content(sample: dict) -> str:
    return "\n".join(turn["content"] for turn in sample["messages"])


# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_DATASET_CONFIG: dict[str, Any] = {
    "open_code_reasoning": {
        "config": {"path": "nvidia/OpenCodeReasoning", "name": "split_0", "split": ["split_0"]},
        "preprocess": lambda sample: "\n".join([sample["input"], sample["output"]]),
    },
    "open_math_reasoning": {
        "config": {
            "path": "nvidia/OpenMathReasoning",
            "split": ["cot", "tir", "genselect"],
        },
        "preprocess": lambda sample: "\n".join([sample["problem"], sample["generated_solution"]]),
    },
    "llama-nemotron-post-training-dataset": {
        "config": {
            "path": "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "name": "SFT",
            "split": ["code", "math", "science", "chat", "safety"],
        },
        "preprocess": lambda sample: (
            "\n".join(turn["content"] for turn in sample["input"]) + "\n" + sample["output"]
        ),
    },
    "nemotron-post-training-dataset-v2": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v2",
            "split": ["stem", "chat", "math", "code"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-post-training-dataset-v1": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": ["stem", "chat", "math", "code", "tool_calling"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-sft-instruction-following-chat-v2": {
        # Skips ``reasoning_on`` split: heterogeneous messages schema fails streaming cast.
        "config": {
            "path": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
            "split": ["reasoning_off"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-science-v1": {
        "config": {
            "path": "nvidia/Nemotron-Science-v1",
            "split": ["MCQ", "RQA"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-competitive-programming-v1": {
        # Skips ``infinibyte_part0[0|1]``: heterogeneous schema fails streaming cast.
        "config": {
            "path": "nvidia/Nemotron-Competitive-Programming-v1",
            "split": [
                "competitive_coding_cpp_part00",
                "competitive_coding_cpp_part01",
                "competitive_coding_python_part00",
                "competitive_coding_python_part01",
            ],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-sft-agentic-v2": {
        # Only ``search`` streams cleanly: ``interactive_agent`` has a heterogeneous
        # tools schema (string vs list) that breaks pyarrow JSON inference, and
        # ``tool_calling`` contains at least one malformed JSON row in a later shard.
        "config": {
            "path": "nvidia/Nemotron-SFT-Agentic-v2",
            "split": ["search"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-math-v2": {
        "config": {
            "path": "nvidia/Nemotron-Math-v2",
            "split": ["high_part00", "high_part01", "high_part02", "medium", "low"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-sft-swe-v2": {
        # Skips ``openhands_swe`` split: heterogeneous schema fails streaming cast.
        "config": {
            "path": "nvidia/Nemotron-SFT-SWE-v2",
            "split": ["agentless"],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "nemotron-sft-multilingual-v1": {
        "config": {
            "path": "nvidia/Nemotron-SFT-Multilingual-v1",
            "split": [
                "code_de",
                "code_es",
                "code_fr",
                "code_it",
                "code_ja",
                "code_zh",
                "math_de",
                "math_es",
                "math_fr",
                "math_it",
                "math_ja",
                "math_zh",
                "stem_de",
                "stem_es",
                "stem_fr",
                "stem_it",
                "stem_ja",
                "stem_zh",
            ],
        },
        "preprocess": _join_messages_content,
        "chat_key": "messages",
    },
    "magpie": {
        "config": {
            "path": "Magpie-Align/Magpie-Pro-MT-300K-v0.1",
            "split": ["train"],
        },
        "preprocess": lambda sample: "\n".join(turn["value"] for turn in sample["conversations"]),
        "chat_key": "conversations",
    },
    "cnn_dailymail": {
        "config": {"path": "abisee/cnn_dailymail", "name": "3.0.0", "split": ["train"]},
        "preprocess": lambda sample: sample["article"],
    },
    "pile": {
        "config": {"path": "monology/pile-uncopyrighted", "name": "v1.0", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "pg19": {
        "config": {"path": "pg19", "name": "v1.0", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "wikipedia": {
        "config": {"path": "wikipedia", "name": "20220301.en", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "c4": {
        "config": {"path": "c4", "name": "en", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "wikitext": {
        "config": {"path": "wikitext", "name": "wikitext-103-v1", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
}

# Named groups of registered datasets, expanded in ``get_dataset_dataloader``.
# Useful when callers want a single ``--dataset`` token that fans out to several
# entries; per-dataset ``num_samples`` is split evenly across the members.
DATASET_COMBOS: dict[str, list[str]] = {
    "cnn_nemotron_v2_mix": ["cnn_dailymail", "nemotron-post-training-dataset-v2"],
    "nemotron-post-training-v3": [
        "nemotron-sft-instruction-following-chat-v2",
        "nemotron-science-v1",
        "nemotron-competitive-programming-v1",
        "nemotron-sft-agentic-v2",
        "nemotron-math-v2",
        "nemotron-sft-swe-v2",
        "nemotron-sft-multilingual-v1",
    ],
}


def _validate_dataset_combos() -> None:
    """Validate DATASET_COMBOS at import time: fail loud on typos / collisions."""
    overlap = set(DATASET_COMBOS) & set(SUPPORTED_DATASET_CONFIG)
    if overlap:
        raise ValueError(
            f"DATASET_COMBOS keys collide with SUPPORTED_DATASET_CONFIG: {sorted(overlap)}"
        )
    for combo_name, members in DATASET_COMBOS.items():
        if not members:
            raise ValueError(f"DATASET_COMBOS['{combo_name}'] must contain at least one dataset.")
        unknown = [m for m in members if m not in SUPPORTED_DATASET_CONFIG]
        if unknown:
            raise ValueError(
                f"DATASET_COMBOS['{combo_name}'] references unknown datasets: {unknown}"
            )


_validate_dataset_combos()

__all__ = [
    "create_forward_loop",
    "download_hf_dataset_as_jsonl",
    "get_dataset_dataloader",
    "get_dataset_samples",
    "get_jsonl_text_samples",
    "get_max_batch_size",
    "get_supported_datasets",
]


def get_jsonl_text_samples(jsonl_path: str, num_samples: int, key: str = "text") -> list[str]:
    """Load up to ``num_samples`` entries from a JSONL file using the ``text`` field.

    Each non-empty line must be a JSON object containing a ``text`` field.
    """
    if num_samples <= 0:
        return []

    samples: list[str] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if len(samples) >= num_samples:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in JSONL file {jsonl_path} at line {line_idx}: {e}"
                ) from e

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected a JSON object in JSONL file {jsonl_path} at line {line_idx}, "
                    f"got {type(obj)}."
                )

            if key not in obj:
                raise ValueError(
                    f"Missing required field '{key}' in JSONL file {jsonl_path} at line {line_idx}."
                )

            samples.append(str(obj[key]))

    return samples


def _normalize_splits(split: str | list[str]) -> list[str]:
    """Ensure split is always a list."""
    return [split] if isinstance(split, str) else list(split)


def _auto_preprocess_sample(
    sample: dict, dataset_name: str, tokenizer: "PreTrainedTokenizerBase | None" = None
) -> str:
    """Auto-detect dataset format and preprocess a single sample based on column conventions.

    Column detection order (first match wins):
        1. ``messages`` / ``conversations`` -> ``tokenizer.apply_chat_template`` (with ``tools`` if present)
        2. ``prompt`` (+ optional ``completion`` / ``response`` / ``output``) -> concatenate
        3. ``text`` -> use as-is
        4. ``input`` (+ optional ``output``) -> concatenate

    Raises:
        ValueError: If the tokenizer is missing/incompatible for chat-format datasets,
            or if no recognized column is found.
    """

    def _has_non_null_value(key: str) -> bool:
        return sample.get(key) is not None

    chat_key = next((k for k in ("messages", "conversations") if _has_non_null_value(k)), None)
    if chat_key is not None:
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Dataset '{dataset_name}' has a '{chat_key}' column but no tokenizer with "
                "apply_chat_template was provided."
            )
        kwargs: dict[str, Any] = {}
        tools = sample.get("tools")
        if tools is not None:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(sample[chat_key], tokenize=False, **kwargs)

    if _has_non_null_value("prompt"):
        parts = [sample["prompt"]]
        parts.extend(
            sample[k] for k in ("completion", "response", "output") if _has_non_null_value(k)
        )
        return "\n".join(parts)

    if _has_non_null_value("text"):
        return sample["text"]

    if _has_non_null_value("input"):
        parts = [sample["input"]]
        if _has_non_null_value("output"):
            parts.append(sample["output"])
        return "\n".join(parts)

    raise ValueError(
        f"Cannot auto-detect format for dataset '{dataset_name}'. "
        f"Found columns: {list(sample.keys())}. "
        "Expected one of: 'messages', 'conversations', 'prompt', 'text', or 'input'."
    )


def get_dataset_samples(
    dataset_name: str,
    num_samples: int,
    *,
    apply_chat_template: bool = False,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    split: str | list[str] | None = None,
) -> list[str]:
    """Load a portion of a dataset with the dataset name and a given size.

    Supports both registered datasets (in ``SUPPORTED_DATASET_CONFIG``) and arbitrary
    HuggingFace datasets.  Unregistered datasets are auto-detected by column names:
    ``messages``/``conversations`` (chat), ``prompt``, ``text``, or ``input``.

    Args:
        dataset_name: Name or HuggingFace path of the dataset to load, a local directory path,
            or a path to a ``.jsonl`` file.  For local directory paths, the
            predefined config from ``SUPPORTED_DATASET_CONFIG`` is matched if the base folder name
            matches a registered key (e.g. ``/hf-local/abisee/cnn_dailymail`` matches ``cnn_dailymail`` key).
            For ``.jsonl`` paths, the file is first loaded via HuggingFace's ``json``
            builder and routed through the same auto-preprocess path as unregistered HF
            datasets so chat / prompt / text columns are handled consistently with live
            HF datasets.  If that path fails on JSON parsing or PyArrow schema
            unification, it falls back to a line-by-line reader that extracts the
            legacy ``text`` field for backward compatibility.  The fallback is also
            used when the optional ``datasets`` package isn't installed, preserving
            legacy plain-``.jsonl`` workflows in base installations.  Local JSONL
            files only expose the ``train`` split; passing any other ``split`` raises.
        num_samples: Number of samples to load from the dataset.
        apply_chat_template: Whether to apply the chat template to the samples
            (if supported by the dataset).  For unregistered datasets with a
            ``messages`` column, chat template is always applied regardless of
            this flag.
        tokenizer: Tokenizer to use for applying the chat template to the samples.
            No tokenization is done and plain text is still returned.
        split: Override the split(s) to load.  Accepts a single split name or a list.
            If ``None``, uses the splits defined in ``SUPPORTED_DATASET_CONFIG`` for
            registered datasets, or ``["train"]`` for unregistered datasets.

    Returns:
        Samples: The list of samples.
    """
    if dataset_name in DATASET_COMBOS:
        raise ValueError(
            f"'{dataset_name}' is a DATASET_COMBOS entry, not a single dataset. "
            "Use ``get_dataset_dataloader`` to expand combos, or pass one of "
            f"its members: {DATASET_COMBOS[dataset_name]}"
        )

    # Local JSONL: load via HF's ``json`` builder and route through the same
    # auto-preprocess path as unregistered HF datasets so chat / prompt / text
    # columns are handled consistently with a downloaded HF dataset.  Never
    # matches ``SUPPORTED_DATASET_CONFIG``.
    is_jsonl = dataset_name.endswith(".jsonl") and os.path.isfile(dataset_name)
    requested_splits = _normalize_splits(split) if split is not None else None
    if requested_splits is not None and not requested_splits:
        raise ValueError("``split`` must contain at least one split name.")

    # HF's file-based builders only expose ``train`` for the ``data_files`` form
    # we use, so any other split is a caller error.  Surface it up front rather
    # than letting ``load_dataset`` fail and silently dropping into the
    # text-field fallback (which would ignore the requested split).
    if is_jsonl and requested_splits is not None:
        invalid = [s for s in requested_splits if s != "train"]
        if invalid:
            raise ValueError(
                f"Local JSONL files only expose the 'train' split, got {invalid}. "
                "Either omit ``split`` or pass ``split='train'``."
            )

    # Lazy ``datasets`` import: legacy ``.jsonl`` workflows historically didn't
    # require the optional ``datasets`` extra, so keep them working with just
    # the stdlib reader when the package isn't installed.
    try:
        from datasets import load_dataset
    except ImportError:
        if is_jsonl:
            return get_jsonl_text_samples(dataset_name, num_samples, key="text")
        raise

    local_dataset_path = None
    if os.path.exists(dataset_name):  # Local path
        local_dataset_path = dataset_name
        if not is_jsonl:
            # Directory paths may match a registered key via their basename
            # (e.g. /hf-local/abisee/cnn_dailymail -> cnn_dailymail).
            dataset_name = os.path.basename(os.path.normpath(local_dataset_path))

    is_registered = not is_jsonl and dataset_name in SUPPORTED_DATASET_CONFIG

    if is_registered:
        dataset_config = SUPPORTED_DATASET_CONFIG[dataset_name]
        config = dataset_config["config"].copy()
        if local_dataset_path:
            config["path"] = local_dataset_path
        splits = requested_splits if requested_splits is not None else config.pop("split", [None])
        if split is not None:
            config.pop("split", None)

        if apply_chat_template:
            if "chat_key" not in dataset_config:
                warn(
                    f"Dataset {dataset_name} does not support chat template."
                    " Chat template will not be applied."
                )
            elif tokenizer is None:
                raise ValueError("Tokenizer is required when applying chat template.")

        def _preprocess(sample: dict) -> str:
            if apply_chat_template and "chat_key" in dataset_config:
                kwargs: dict[str, Any] = {}
                tools = sample.get("tools")
                if tools is not None:
                    kwargs["tools"] = tools
                return tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    sample[dataset_config["chat_key"]], tokenize=False, **kwargs
                )
            return dataset_config["preprocess"](sample)

    else:
        print(
            f"Dataset '{dataset_name}' is not in SUPPORTED_DATASET_CONFIG. "
            "Auto-detecting format from column names."
        )
        if is_jsonl:
            config = {"path": "json", "data_files": local_dataset_path}
        else:
            config = {"path": local_dataset_path or dataset_name}
        # HF's file-based builders (incl. ``json``) label a string/list ``data_files``
        # as the ``train`` split unconditionally — the filename on disk is ignored.
        # Named splits require a dict ``data_files={"train": ..., "test": ...}``,
        # which we don't expose here.
        splits = requested_splits if requested_splits is not None else ["train"]

        def _preprocess(sample: dict) -> str:
            return _auto_preprocess_sample(sample, dataset_name, tokenizer)

    if not splits:
        raise ValueError("``split`` must contain at least one split name.")

    # Narrow the legacy fallback to JSON-parsing / Arrow schema failures.  Any
    # other error (split-not-found, IO, OOM, ...) should surface to the caller
    # rather than be hidden by the text-field reader.  Imported lazily because
    # the exact module paths vary across versions; an empty tuple is a valid
    # ``except`` target that catches nothing if neither is importable.
    fallback_types: tuple[type[BaseException], ...] = ()
    try:
        from datasets.exceptions import DatasetGenerationError

        fallback_types += (DatasetGenerationError,)
    except ImportError:
        pass
    try:
        from pyarrow.lib import ArrowInvalid

        fallback_types += (ArrowInvalid,)
    except ImportError:
        pass

    # load_dataset does not support a list of splits while streaming, so load each separately.
    print(f"Loading dataset with {config=} and {splits=}")
    try:
        dataset_splits = [load_dataset(streaming=True, **config, split=s) for s in splits]

        num_per_split = [num_samples // len(dataset_splits)] * len(dataset_splits)
        num_per_split[-1] += num_samples - sum(num_per_split)

        samples: list[str] = []
        for dataset, n in zip(dataset_splits, num_per_split):
            for i, sample in enumerate(dataset):
                if i >= n:
                    break
                text = _preprocess(sample)
                if text:
                    samples.append(text)

        return samples
    except fallback_types as e:
        # Backward-compat fallback: legacy callers passed JSONL files whose only usable
        # field is ``text``.  If the HF ``json`` builder fails on schema inference or
        # JSON parsing, fall back to a line-by-line reader that pulls ``text`` directly.
        if not is_jsonl:
            raise
        assert local_dataset_path is not None  # is_jsonl implies the path exists
        try:
            fallback_samples = get_jsonl_text_samples(local_dataset_path, num_samples, key="text")
        except Exception:
            # Fallback can't help either — surface the original HF error.
            raise e from None
        safe_name = Path(local_dataset_path).name
        warn(
            f"Failed to load JSONL file '{safe_name}' via the HF 'json' builder "
            f"({type(e).__name__}); fell back to legacy text-field reader."
        )
        return fallback_samples


class _CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {
            key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def _pack_documents_into_rows(
    samples: list[str], tokenizer: "PreTrainedTokenizerBase", seq_length: int, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global-stream document packing (Megatron-LM pretraining style).

    Concatenate all raw samples into one EOS-separated token stream, then slice
    the stream into uniform-length rows. Rows can (and usually do) start mid-doc —
    this matches the distribution Megatron's blended-dataset pretraining uses with
    ``.bin``/``.idx`` files, so the trained model has seen this pattern extensively.

    Returns ``(input_ids, attention_mask)`` tensors of shape ``(num_rows, seq_length)``.
    Non-final rows are fully real tokens (mask=1 throughout). The final partial row
    (when the stream runs out before reaching ``num_rows``) has mask=1 over the real
    tail and mask=0 over trailing pad.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    has_eos_sep = eos_id is not None
    token_stream: list[int] = []
    for s in samples:
        token_stream.extend(tokenizer.encode(s, add_special_tokens=False))
        if has_eos_sep:
            token_stream.append(eos_id)
        if len(token_stream) >= num_rows * seq_length:
            break

    n_full = min(num_rows, len(token_stream) // seq_length)
    rows_ids: list[list[int]] = [
        token_stream[i * seq_length : (i + 1) * seq_length] for i in range(n_full)
    ]
    rows_masks: list[list[int]] = [[1] * seq_length for _ in range(n_full)]
    # Trailing partial row (if any remain in the num_rows budget).
    if n_full < num_rows and len(token_stream) > n_full * seq_length:
        tail = token_stream[n_full * seq_length :]
        real_len = len(tail)
        tail.extend([pad_id] * (seq_length - real_len))
        rows_ids.append(tail)
        rows_masks.append([1] * real_len + [0] * (seq_length - real_len))

    return (
        torch.tensor(rows_ids, dtype=torch.long),
        torch.tensor(rows_masks, dtype=torch.long),
    )


def get_dataset_dataloader(
    dataset_name: str | list[str] = "cnn_dailymail",
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    batch_size: int = 1,
    num_samples: int | list[int] = 512,
    max_sample_length: int = 512,
    device: torch.device | str | None = None,
    include_labels: bool = False,
    apply_chat_template: bool = False,
    pack: bool = False,
) -> DataLoader:
    """Get a dataloader with the dataset name and tokenizer of the target model.

    Args:
        dataset_name: Name of the dataset to load, a path to a ``.jsonl`` file, or a list
            mixing the two. Each entry is loaded via :func:`get_dataset_samples` and the
            resulting samples are concatenated before tokenization. ``num_samples`` may be
            an ``int`` (applied to a single source) or a list aligned with ``dataset_name``.
        tokenizer: Instance of HuggingFace tokenizer.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset (interpreted as number of *output
            rows* in both ``pack=False`` and ``pack=True`` modes — in packed mode the
            loader oversamples raw text 4x to ensure enough docs to fill all rows).
        max_sample_length: Maximum length of a sample (or per-row length under ``pack=True``).
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader (ignored when
            ``pack=True``).
        apply_chat_template: Whether to apply the chat template to the samples
            (if supported by the dataset).
        pack: If True, use global-stream document packing (Megatron-LM pretraining
            style): all raw samples are concatenated into one EOS-separated token
            stream and sliced into uniform-length rows. Rows can (and usually do)
            start mid-document — this matches the distribution Megatron's blended
            ``.bin``/``.idx`` pretraining uses, so the trained model has seen this
            pattern extensively. Non-final rows are fully real tokens (no pad); only
            the trailing partial row (when the stream runs out before reaching
            ``num_samples`` rows) is padded. Default ``False`` for backwards-compatibility
            with the prior one-doc-per-row tokenize-and-pad behavior; calibration
            callers should pass ``True``.

    Returns:
        An instance of dataloader.
    """
    assert tokenizer is not None, "Please provide a tokenizer."
    # Tokenizer encoding may modify the tokenizer in place, so we need to clone it.
    tokenizer = copy.deepcopy(tokenizer)

    if tokenizer.padding_side != "left":
        warn(
            "Tokenizer with the right padding_side may impact calibration accuracy. Recommend set to left"
        )

    if isinstance(num_samples, int):
        num_samples = [num_samples]

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    assert len(dataset_name) == len(num_samples), (
        "dataset_name and num_samples must be the same length"
    )

    # Reject inputs that include both a combo and one of its member datasets
    # (e.g. ``["cnn_dailymail", "cnn_nemotron_v2_mix"]``), since the combo would sample the
    # plain entry a second time with a smaller per-member quota.
    plain_inputs = {n for n in dataset_name if n not in DATASET_COMBOS}
    for ds_name in dataset_name:
        if ds_name in DATASET_COMBOS:
            overlap = plain_inputs & set(DATASET_COMBOS[ds_name])
            if overlap:
                raise ValueError(
                    f"--dataset includes both combo '{ds_name}' and its "
                    f"member(s) {sorted(overlap)}; remove one to avoid "
                    "double-sampling."
                )

    expanded_names: list[str] = []
    expanded_num_samples: list[int] = []
    for ds_name, n in zip(dataset_name, num_samples):
        if ds_name in DATASET_COMBOS:
            members = DATASET_COMBOS[ds_name]
            base, remainder = divmod(n, len(members))
            for i, member in enumerate(members):
                expanded_names.append(member)
                expanded_num_samples.append(base + (1 if i < remainder else 0))
        else:
            expanded_names.append(ds_name)
            expanded_num_samples.append(n)
    dataset_name, num_samples = expanded_names, expanded_num_samples

    # Sample count semantics:
    # - pack=False: gather exactly `num_sample` raw docs per source, one per output row.
    # - pack=True:  oversample 8x per source to ensure enough raw docs to fill all rows,
    #               since each row greedily packs multiple docs.
    sample_multiplier = 8 if pack else 1
    all_samples = []
    for ds_name, num_sample in zip(dataset_name, num_samples):
        samples = get_dataset_samples(
            ds_name,
            num_sample * sample_multiplier,
            apply_chat_template=apply_chat_template,
            tokenizer=tokenizer,
        )
        all_samples.extend(samples)

    # Multi-source pack=True without shuffling would consume all of oversampled source 1's docs
    # before any of oversampled source 2 are reached
    if pack and len(dataset_name) > 1:
        random.Random(0).shuffle(all_samples)

    if pack:
        total_rows = sum(num_samples)
        input_ids, attention_mask = _pack_documents_into_rows(
            all_samples, tokenizer, max_sample_length, total_rows
        )
        if input_ids.shape[0] < total_rows:
            warn(
                f"pack=True produced {input_ids.shape[0]} rows out of {total_rows} "
                f"requested — raw text exhausted before filling all rows (8x oversample "
                f"of num_samples was insufficient). Increase `num_samples` or shorten "
                f"`max_sample_length`."
            )
        if device:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        tokenized_dataset = _CustomDataset(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    batch_encoded = tokenizer(
        all_samples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sample_length,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    if include_labels:
        # Labels are needed when backward is called in the model.
        # The labels should be a shifted version of the input_ids.
        # However, we should not shift the input_ids here since the labels are shifted by
        # Huggingface models during loss calculation as shown here -
        # https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/llama/modeling_llama.py#L1093-L1095
        batch_encoded["labels"] = torch.where(
            batch_encoded["attention_mask"] > 0.5, batch_encoded["input_ids"], -100
        )
        tokenized_dataset = _CustomDataset(batch_encoded)
    else:
        # Always include attention_mask so the model correctly ignores padding tokens
        # during calibration. Without it, HF models create a full causal mask and
        # padding tokens participate in attention, skewing calibration statistics.
        tokenized_dataset = _CustomDataset(
            {
                "input_ids": batch_encoded["input_ids"],
                "attention_mask": batch_encoded["attention_mask"],
            }
        )

    calib_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def get_supported_datasets() -> list[str]:
    """Retrieves a list of datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

        .. code-block:: python

            from modelopt.torch.utils import get_supported_datasets

            print("Supported datasets:", get_supported_datasets())
    """
    return list(SUPPORTED_DATASET_CONFIG.keys()) + list(DATASET_COMBOS.keys())


@contextmanager
def _disable_use_cache(model: torch.nn.Module) -> Iterator[None]:
    """Set ``model.config.use_cache = False`` for the duration of the block.

    KV caching is unwanted during calibration / memory-probe forward passes:
    it wastes memory, and for hybrid Mamba/attention models (e.g., NemotronH)
    the cache state is mutated in-place and breaks correctness. Setting
    ``use_cache`` unconditionally (rather than only when it was already
    present) also sidesteps configs that never assign the attribute at all
    — e.g., ``Step3p5Config`` from stepfun-ai/Step-3.5-Flash — where forward
    code that reads ``self.config.use_cache`` would otherwise raise
    ``AttributeError``. The prior value is restored on exit if one existed.
    """
    config = getattr(model, "config", None)
    if config is None:
        yield
        return
    had_attr = hasattr(config, "use_cache")
    prev = config.use_cache if had_attr else None
    config.use_cache = False
    try:
        yield
    finally:
        if had_attr:
            config.use_cache = prev
        else:
            with suppress(AttributeError):
                delattr(config, "use_cache")


def get_max_batch_size(
    model: torch.nn.Module,
    max_sample_length: int = 512,
    sample_memory_usage_ratio: float = 1.0,
    sample_input_single_batch: torch.Tensor | None = None,
    enable_grad: bool = False,
):
    """Get the maximum batch size that can be used for the model."""

    def _get_free_gpu_mem():
        min_gpu_free_mem = torch.cuda.get_device_properties(0).total_memory
        max_allocated_mem = 0
        for device in range(torch.cuda.device_count()):
            free_mem = torch.cuda.mem_get_info(device)[0]
            if free_mem < min_gpu_free_mem:
                min_gpu_free_mem = free_mem
                max_allocated_mem = torch.cuda.max_memory_allocated(device)
        return min_gpu_free_mem, max_allocated_mem

    torch.cuda.empty_cache()

    free_mem_before, max_allocated_before = _get_free_gpu_mem()
    is_enc_dec = model_type_is_enc_dec(model)
    infer_method = model.generate if is_enc_dec else model.forward

    if sample_input_single_batch is None:
        sample_input_single_batch = (
            torch.ones([1, max_sample_length], dtype=torch.int32, device=model.device) * 100
        )

    with _disable_use_cache(model):
        # Calculate single batch inference with dummy input.
        with torch.set_grad_enabled(enable_grad):
            infer_method(sample_input_single_batch)
        free_mem_after, max_allocated_after = _get_free_gpu_mem()

        mem_diff_per_data_batch = (
            max(
                (free_mem_before - free_mem_after),
                (max_allocated_after - max_allocated_before),
            )
            * sample_memory_usage_ratio
        )
        if mem_diff_per_data_batch <= 0:  # pragma: no cover - GPU memory probe edge case
            print(  # pragma: no cover
                "Warning: No measurable memory usage found for a single batch. "
                "Falling back to batch_size=1."
            )
            target_data_batch = 1  # pragma: no cover
        else:
            target_data_batch = max(int(free_mem_before / mem_diff_per_data_batch), 1)

        def _expand_to(batch: int) -> torch.Tensor:
            return sample_input_single_batch.expand(
                [
                    batch if index == 0 else dim
                    for index, dim in enumerate(sample_input_single_batch.shape)
                ]
            )

        target_input = _expand_to(target_data_batch)

        # For some models on multi GPU, we observe the memory per batch is not a constant.
        # So we just test the target batch size and make sure we do not go OOM.
        while target_data_batch > 1:
            with torch.set_grad_enabled(enable_grad):
                try:
                    infer_method(target_input)
                    break
                except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU OOM retry path
                    target_data_batch = target_data_batch // 2  # pragma: no cover
                    target_input = _expand_to(target_data_batch)  # pragma: no cover
                    torch.cuda.empty_cache()  # pragma: no cover

    # Regulate the data batch target to be 1, 2, 4, 8, 12, ..., capped at 64
    if target_data_batch < 2:
        return 1
    elif target_data_batch < 4:
        return 2
    elif target_data_batch < 512:
        return target_data_batch // 4 * 4
    else:
        return 512


def _process_batch(
    batch_data, infer_method, max_working_batch_size=None, allowed_non_tensor_keys=None
):
    """Process a batch of data through the model's inference method.

    Args:
        batch_data: Dictionary containing the batch data
        infer_method: Model's inference method (either forward or generate)
        max_working_batch_size: Maximum batch size known to work without OOM
        allowed_non_tensor_keys: Set of key names whose values may be non-tensor types

    Returns:
        The maximum batch size that worked successfully
    """
    allowed_non_tensor_keys = allowed_non_tensor_keys or set()
    assert all(
        torch.is_tensor(data) or data is None or key in allowed_non_tensor_keys
        for key, data in batch_data.items()
    ), f"batch_data values must be tensors or None, except for keys: {allowed_non_tensor_keys}."
    # Get the batch size of current data
    batch_size = batch_data[next(iter(batch_data.keys()))].shape[0]

    # If we know a smaller batch size works, preemptively split
    if max_working_batch_size is not None and batch_size > max_working_batch_size:
        # Split the batch to avoid OOM
        for i in range(0, batch_size, max_working_batch_size):
            end_idx = min(i + max_working_batch_size, batch_size)
            split_data = {}
            for key in batch_data:
                if batch_data[key] is None:
                    split_data[key] = None
                else:
                    split_data[key] = batch_data[key][i:end_idx, ...]

            max_working_batch_size = _process_batch(
                split_data, infer_method, max_working_batch_size, allowed_non_tensor_keys
            )

        return max_working_batch_size

    # Try processing with current batch size
    try:
        infer_method(**batch_data)
        return (
            batch_size
            if max_working_batch_size is None
            else max(batch_size, max_working_batch_size)
        )  # This batch size worked successfully
    except torch.cuda.OutOfMemoryError:
        assert batch_size > 1, (
            "CUDA out of memory error occurred while processing a single sample. "
            "This indicates the model is too large for the available GPU memory. "
            "Consider reducing the model size, using a smaller max_sample_length, "
            "or using a GPU with more memory."
        )

    # Split the batch in half
    mid = (batch_size + 1) // 2
    warn(f"CUDA out of memory with batch size {batch_size}, trying with batch size {mid}")
    split_data_1 = {key: batch_data[key][:mid, ...] for key in batch_data}
    split_data_2 = {key: batch_data[key][mid:, ...] for key in batch_data}

    # Recursively process each half and track max working batch size
    max_working_batch_size = _process_batch(
        split_data_1, infer_method, allowed_non_tensor_keys=allowed_non_tensor_keys
    )
    max_working_batch_size = _process_batch(
        split_data_2, infer_method, max_working_batch_size, allowed_non_tensor_keys
    )

    # Return the minimum of the two (to be conservative)
    return max_working_batch_size


def _forward_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    allowed_non_tensor_keys: set | None = None,
) -> None:
    """Runs forward passes through the model using data from the dataloader.

    Args:
        model: The PyTorch model to run inference on
        dataloader: DataLoader containing the batched input data
        allowed_non_tensor_keys: Set of key names whose values may be non-tensor types
    """
    with _disable_use_cache(model), torch.no_grad():
        is_enc_dec = model_type_is_enc_dec(model)
        infer_method = model.generate if is_enc_dec else model.forward
        max_working_batch_size = None  # Initialize max working batch size as None

        for _, data in enumerate(tqdm(dataloader)):
            # Process batch and update max working batch size
            max_working_batch_size = _process_batch(
                data, infer_method, max_working_batch_size, allowed_non_tensor_keys
            )


def create_forward_loop(
    model: torch.nn.Module | None = None,
    dataset_name: str = "cnn_dailymail",
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    batch_size: int = 1,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: str | None = None,
    include_labels: bool = False,
    dataloader: DataLoader | None = None,
    allowed_non_tensor_keys: set | None = None,
) -> Callable:
    """Creates and returns a forward loop function configured for a specific model, dataset, and tokenizer.

    This function initializes a forward loop function tailored to process batches of data from the specified dataset
    using the given model and tokenizer. The forward loop function, when called, iterates over the dataset, applies the
    tokenizer to prepare the input data, feeds it into the model, and returns the model's predictions.

    Args:
        model: The PyTorch model for inference.
        dataset_name: The name of the dataset to be used. Must be one of the datasets in get_supported_datasets().
        tokenizer: The tokenizer used to preprocess text data into a format suitable
            for the model.
        batch_size: Batch size of the returned dataloader. If 0 is provided, we auto determine the batch_size.
        num_samples: Number of samples from the dataset.
        max_sample_length: Maximum length of a sample.
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader.
        dataloader: If provided, use the provided dataloader instead.
        allowed_non_tensor_keys: Set of key names whose batch values may be non-tensor types.
            Useful when the dataloader yields batches with non-standard fields (e.g., nested
            model outputs).

    Example usage for quantization:

    .. code-block:: python

        import modelopt.torch.quantization as mtq
        from modelopt.torch.utils import create_forward_loop

        # Initialize model and tokenizer
        # ...

        # Create forward loop for calibration
        forward_loop = create_forward_loop(
            model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer
        )

        # Quantize the model with the calibration dataset
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    Returns:
        A forward loop function that can be called with no arguments. When called, this function iterates over
            the dataset specified by `dataset_name`.
    """
    if dataloader is None:
        if batch_size == 0:
            # We let the system to determine the max data batch for each forward.
            batch_size = get_max_batch_size(model, max_sample_length)
            print(f"Update calib batch {batch_size}")

        dataloader = get_dataset_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=num_samples,
            max_sample_length=max_sample_length,
            device=device,
            include_labels=include_labels,
        )

    return lambda model: _forward_loop(model, dataloader, allowed_non_tensor_keys)


def model_type_is_enc_dec(model):
    enc_dec_model_list = ["t5", "bart", "whisper"]
    return any(model_name in model.__class__.__name__.lower() for model_name in enc_dec_model_list)


def download_hf_dataset_as_jsonl(
    dataset_name: str,
    output_dir: str | Path,
    json_keys: str | list[str] = ["text"],
    name: str | None = None,
    split: str | None = None,
    max_samples_per_split: int | None = None,
    num_proc: int | None = None,
) -> list[str]:
    """Download a Hugging Face dataset and save as JSONL files.

    Args:
        dataset_name: Name or HuggingFace path of the dataset to download
        output_dir: Directory to save the JSONL files
        json_keys: Key or list of keys to extract from the dataset. Defaults to ["text"].
        name: Name of the subset to download
        split: Split of the dataset to download. Defaults to None (all splits).
        max_samples_per_split: Maximum number of samples to download per split. Defaults to None.
        num_proc: Number of processes to use for parallel processing. Defaults to None.

    Returns:
        List of paths to downloaded JSONL files.
    """
    from datasets import load_dataset
    from huggingface_hub.utils import build_hf_headers

    print(f"Downloading dataset {dataset_name} from Hugging Face")
    if isinstance(json_keys, str):
        json_keys = [json_keys]
    jsonl_paths: list[str] = []

    try:
        response = requests.get(
            f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}",
            headers=build_hf_headers(),
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch dataset splits for {dataset_name}: {e}") from e

    response_json = response.json()
    print(f"\nFound {len(response_json['splits'])} total splits for {dataset_name}:")
    for entry in response_json["splits"]:
        print(f"\t{entry}")

    splits_to_process = []
    for entry in response_json["splits"]:
        if name is not None and name != entry.get("config", None):
            continue
        if split is not None and split != entry["split"]:
            continue
        splits_to_process.append(entry)

    print(f"\nFound {len(splits_to_process)} splits to process:")
    for entry in splits_to_process:
        print(f"\t{entry}")

    for entry in splits_to_process:
        path = entry["dataset"]
        name = entry.get("config", None)
        split = entry["split"]
        if max_samples_per_split is not None:
            split = f"{split}[:{max_samples_per_split}]"
        jsonl_file_path = f"{output_dir}/{path.replace('/', '--')}_{name}_{split}.jsonl"

        print(f"\nLoading HF dataset {path=}, {name=}, {split=}")
        if os.path.exists(jsonl_file_path):
            jsonl_paths.append(jsonl_file_path)
            print(f"\t[SKIP] Raw dataset {jsonl_file_path} already exists")
            continue
        ds = load_dataset(path=path, name=name, split=split)

        for key in json_keys:
            if key not in ds.features:
                raise KeyError(
                    f"{key=} not found in dataset features. Available: {list(ds.features)}"
                )

        print(f"Saving raw dataset to {jsonl_file_path}")
        ds.to_json(jsonl_file_path, num_proc=num_proc)
        jsonl_paths.append(jsonl_file_path)

    return jsonl_paths
