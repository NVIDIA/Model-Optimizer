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

"""Prepare a weighted Megatron data blend from a YAML configuration."""

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import yaml
from datasets import load_dataset

from modelopt.torch.utils import (
    REASONING_CONTENT_MODES,
    ReasoningContentMode,
    prepare_messages_for_chat_template,
)

from .megatron_preprocess_data import megatron_preprocess_data

__all__ = ["prepare_chat_data_blend", "prepare_megatron_data_blend"]

# A turn of a ``text`` row in the NeMo chat format: ``<extra_id_1>User\n...``. Any marker ends the
# turn, so a trailing delimiter is not kept as content. The role must be followed by a newline, a
# marker or the end, so ``<extra_id_1>Userguide`` is no role at all.
_CHAT_TURN = re.compile(
    r"<extra_id_\d+>(User|Assistant|System)(?:\n|(?=<extra_id_\d+>|\Z))(.*?)(?=<extra_id_\d+>|\Z)",
    re.DOTALL,
)
_CHAT_ROLES = {"User": "user", "Assistant": "assistant", "System": "system"}
_CHAT_MARKER = re.compile(r"<extra_id_\d+>")
_CHAT_TRAILING_MARKER = re.compile(r"<extra_id_\d+>\s*\Z")


def load_config(path: Path) -> dict[str, Any]:
    """Load a data-blend YAML configuration as a dictionary.

    For example, this YAML::

        tokenizer: /models/Qwen3-8B
        output_dir: /datasets/qwen3-blend
        target_tokens: 1000000  # Optional; omit to prepare every source in full.
        sources:
          - hf_dataset: nvidia/Nemotron-Pretraining-SFT-v1
            config: Nemotron-SFT-General
            split: train
            content_field: text
            weight: 60
          - hf_dataset: nvidia/Nemotron-SFT-Competitive-Programming-v2
            files:
              - data/competitive_programming_python_00.jsonl
            content_field: messages
            weight: 40

    returns a dictionary with ``tokenizer``, ``output_dir``, and ``sources`` keys, plus
    optional ``target_tokens``. Each source has ``hf_dataset``, ``content_field``, and
    ``weight``; it uses ``split`` with optional ``config`` and ``max_samples``, or
    selects repository ``files``.

    With ``output_format: chat_jsonl``, the blend is written as chat JSONL instead (see
    :func:`prepare_chat_data_blend`): ``tokenizer`` and ``target_tokens`` are not used, and the
    optional ``target_samples``, ``shuffle``, ``seed`` and ``reasoning_content`` keys apply.
    """
    with path.open(encoding="utf-8") as stream:
        return cast("dict[str, Any]", yaml.safe_load(stream))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the blend YAML file")
    return parser


def _write_data_blend(path: Path, blend: list[tuple[float, str]]) -> None:
    """Write the weighted Megatron dataset paths."""
    content = "\n".join(f"{weight:g} {prefix}" for weight, prefix in blend) + "\n"
    path.write_text(content, encoding="utf-8")


def _copy_config(source: Path, destination: Path) -> None:
    """Copy the input configuration alongside the generated blend."""
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def _download_files(source: dict[str, Any], output_dir: Path) -> list[str]:
    """Download the repository ``files`` of a source and return their local paths."""
    dataset = source["hf_dataset"]
    raw_dir = output_dir.parent / "raw" / dataset.replace("/", "--")
    return [
        huggingface_hub.hf_hub_download(
            repo_id=dataset, filename=file, repo_type="dataset", local_dir=raw_dir
        )
        for file in source["files"]
    ]


def _prepare_sources(
    sources: list[dict[str, Any]],
    output_dir: Path,
    tokenizer: str,
    total_tokens: int | None,
) -> list[tuple[float, str]]:
    """Tokenize all sources and return their weighted output paths."""
    workers = min(32, os.cpu_count() or 1)
    blend: list[tuple[float, str]] = []  # (weight, shared .bin/.idx path without extension)
    allocated_tokens = 0
    # Weights are relative, not required to sum to 100 (matching data_blend.txt semantics).
    weight_sum = sum(float(source["weight"]) for source in sources)

    for index, source in enumerate(sources):
        weight = float(source["weight"])
        if total_tokens is None:
            source_tokens = None
        elif index == len(sources) - 1:
            source_tokens = total_tokens - allocated_tokens
        else:
            source_tokens = round(total_tokens * weight / weight_sum)
            allocated_tokens += source_tokens

        dataset = source["hf_dataset"]
        source_dir = output_dir / f"{index:02d}_{dataset.replace('/', '--')}"
        content_field = source["content_field"]
        input_args: dict[str, Any]
        if "files" in source:
            input_args = {"jsonl_paths": _download_files(source, output_dir)}
        else:
            input_args = {
                "hf_dataset": dataset,
                "hf_name": source.get("config"),
                "hf_split": source["split"],
                "hf_max_samples_per_split": source.get("max_samples"),
                "hf_streaming": True,
            }

        # Each prefix is the path shared by a tokenized Megatron .bin/.idx file pair.
        prefixes = megatron_preprocess_data(
            **input_args,
            output_dir=source_dir,
            tokenizer_name_or_path=tokenizer,
            json_keys=content_field,
            # Plain text lacks chat-template boundary tokens, so terminate each document with EOS.
            append_eod=content_field == "text",
            # Join lines in text documents by replacing each newline with a space.
            strip_newlines=content_field == "text",
            reasoning_content="inline" if content_field == "messages" else "strip",
            # Guard against pathological records by capping each tokenized document at 256K tokens.
            max_sequence_length=256_000,
            max_tokens=source_tokens,
            workers=workers,
        )
        prefix_weight = weight / len(prefixes)
        blend.extend((prefix_weight, prefix) for prefix in prefixes)

    return blend


def _to_chat(
    row: dict[str, Any], content_field: str, reasoning_content: ReasoningContentMode = "native"
) -> dict[str, Any] | None:
    """Convert a source row to ``{"messages": [...]}``, or ``None`` if it has nothing to train on.

    ``messages`` go through :func:`prepare_messages_for_chat_template` with ``reasoning_content``,
    and non-empty ``tools`` and ``system`` fields are kept for the chat path. A ``text`` row in the NeMo chat
    format (``<extra_id_1>User`` / ``Assistant`` markers) is split into turns; any other ``text``
    row becomes one assistant turn, so all of it is trained.
    """
    if content_field == "messages":
        messages = row.get("messages")
        if not messages:
            return None
        messages = prepare_messages_for_chat_template(messages, reasoning_content=reasoning_content)
        chat: dict[str, Any] = {"messages": messages}
        # As in the tokenizing mode, the chat template renders the tool definitions into the prompt;
        # Megatron-Bridge turns a top-level system prompt into the leading system message.
        for key in ("tools", "system"):
            if row.get(key):
                chat[key] = row[key]
    else:
        text = row.get("text") or ""
        if not text:
            return None
        if "<extra_id_" not in text:
            return {"messages": [{"role": "assistant", "content": text}]}
        matches = _CHAT_TURN.findall(text)
        # Markers that parse to no turns make the row malformed, not plain text: falling through
        # would train the raw <extra_id_*> delimiters as assistant content. Text before the first
        # marker, a marker without a known role (a Tool turn, or a SteerLM label line) or a known turn
        # left empty would silently cut it out of the conversation, so drop the row. Only a bare
        # delimiter at the very end needs no role, and only the System block, which real rows leave
        # empty, may be empty.
        roleless = len(_CHAT_MARKER.findall(text)) - len(matches)
        if (
            not _CHAT_MARKER.match(text.lstrip())
            or roleless > (1 if _CHAT_TRAILING_MARKER.search(text) else 0)
            or any(not content.strip() and role != "System" for role, content in matches)
        ):
            return None
        turns = [
            {"role": _CHAT_ROLES[role], "content": content.strip()}
            for role, content in matches
            if content.strip()
        ]
        chat = {"messages": turns}
    # Nothing in a row without an assistant turn is trained under assistant-only loss.
    if not any(turn.get("role") == "assistant" for turn in chat["messages"]):
        return None
    return chat


def _iter_rows(
    source: dict[str, Any], output_dir: Path, shuffle: bool, seed: int
) -> Iterator[dict[str, Any]]:
    """Yield the rows of a source: its ``files`` in order, or its Hugging Face split, streamed."""
    if "files" in source:
        for path in _download_files(source, output_dir):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError as e:
                        # As in the tokenizing reader: some Nemotron files carry a malformed record.
                        print(f"[WARN] Skipping malformed JSONL record in {path}: {e}", flush=True)
        return
    dataset = load_dataset(
        source["hf_dataset"], source.get("config"), split=source["split"], streaming=True
    )
    yield from dataset.shuffle(seed=seed) if shuffle else dataset


def _take_chat_rows(
    source: dict[str, Any],
    output_dir: Path,
    want: int,
    shuffle: bool,
    seed: int,
    reasoning_content: ReasoningContentMode,
) -> list[dict[str, Any]]:
    """Return the first ``want`` usable chat rows of a source.

    Fails if the source cannot be read or runs out of usable rows, so the blend cannot change
    silently; gives up after ``10 * want + 100`` rejected rows, a floor that keeps a small budget from
    misjudging a source by its first few rows.
    """
    # Megatron-Bridge's chat SFT path consumes this output; keep exactly the rows it accepts. Imported
    # here, not at module scope, because the tokenizing mode does not need Megatron-Bridge installed.
    from megatron.bridge.data.conversation_processing import is_text_only_chat_example

    name = f"{source['hf_dataset']} {source.get('split') or source.get('files')}"
    rows: list[dict[str, Any]] = []
    rejected = 0
    max_rejected = 10 * want + 100
    try:
        for row in _iter_rows(source, output_dir, shuffle, seed):
            chat = _to_chat(row, source["content_field"], reasoning_content)
            # Judge the source row's top-level media keys too, which the conversion does not keep.
            if chat is None or not is_text_only_chat_example({**row, **chat}):
                rejected += 1
                if rejected > max_rejected:
                    break
                continue
            rows.append(chat)
            if len(rows) >= want:
                break
    except Exception as e:  # e.g. a gated split, or a schema the streaming reader cannot cast
        raise RuntimeError(f"{name}: failed to read or convert") from e
    if len(rows) < want:
        gave_up = rejected > max_rejected
        reason = f"gave up after {rejected} rejected rows" if gave_up else "source exhausted"
        raise RuntimeError(f"{name}: only {len(rows)}/{want} usable rows ({reason})")
    print(f"{name}: {len(rows)} rows ({rejected} rejected)", flush=True)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` atomically, so an interrupted write never leaves a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    """Read a cached part, or return ``None`` if it is missing or unreadable."""
    try:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except (OSError, ValueError):
        return None


def _part_path(parts_dir: Path, source: dict[str, Any], settings: dict[str, Any]) -> Path:
    """Return the cache file of a source's selected rows, keyed by every input that selects them.

    A part holds the first usable rows of the source, so it serves any budget it covers.
    """
    selection = {key: source.get(key) for key in ("hf_dataset", "config", "split", "files")}
    selection.update(settings, content_field=source["content_field"])
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:16]
    label = source.get("split") or Path(source["files"][0]).name.split(".")[0]
    return parts_dir / f"{source['hf_dataset'].replace('/', '--')}_{label}_{digest}.jsonl"


def _chat_budgets(config: dict[str, Any]) -> list[int]:
    """Return each source's row budget; validate the inputs."""
    weights = [float(source["weight"]) for source in config["sources"]]
    if not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"Source weights must be finite, nonnegative and not all zero: {weights}.")
    # Required: the selected rows are held in memory to be mixed, which a whole source may not fit.
    target = config.get("target_samples")
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError(
            f"target_samples is required and must be a positive integer, got {target!r}."
        )
    return [round(target * w / sum(weights)) for w in weights]


def prepare_chat_data_blend(config_path: Path) -> dict[str, int]:
    """Write a weighted chat blend as ``train.jsonl`` / ``validation.jsonl``; return their sizes.

    For Megatron-Bridge's chat SFT path (``distill.py --sft --sft_hf_dataset``), which renders each
    conversation with the model's chat template and can restrict the loss to assistant tokens.
    Each source contributes ``round(target_samples * weight / sum(weights))`` rows (a source rounded
    to zero rows is skipped) that the chat path accepts as text-only chat. ``content_field`` must be
    ``messages`` or ``text``; ``messages`` keep ``reasoning_content`` for the chat template by
    default (``reasoning_content: native``; set ``inline`` for a template that ignores the field, or
    ``strip``). Hugging Face splits are buffer-shuffled with ``seed`` unless ``shuffle: false``;
    ``files`` (``.jsonl`` or ``.jsonl.gz``) are read from the beginning, skipping malformed records.
    The rows are mixed with ``seed`` and ``min(1000, n // 20)`` of them are held out for validation.
    Each source is cached under ``<output_dir>/parts``, keyed by the settings that select its rows,
    so an interrupted run resumes.
    """
    config = load_config(config_path)
    if config.get("output_format") != "chat_jsonl":
        raise ValueError(
            "prepare_chat_data_blend writes output_format: chat_jsonl, "
            f"got {config.get('output_format')!r}."
        )
    ignored = {"tokenizer", "target_tokens"} & config.keys()
    if ignored:
        raise ValueError(f"output_format: chat_jsonl does not use {sorted(ignored)}.")
    if any("max_samples" in source for source in config["sources"]):
        raise ValueError("output_format: chat_jsonl does not use max_samples; set target_samples.")
    fields = {source["content_field"] for source in config["sources"]} - {"messages", "text"}
    if fields:
        raise ValueError(
            f"output_format: chat_jsonl reads content_field messages or text: {fields}."
        )
    reasoning_content = config.get("reasoning_content", "native")
    if reasoning_content not in REASONING_CONTENT_MODES:
        raise ValueError(
            f"reasoning_content must be one of {sorted(REASONING_CONTENT_MODES)}, "
            f"got {reasoning_content!r}."
        )
    budgets = _chat_budgets(config)
    output_dir = Path(config["output_dir"])
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    shuffle = bool(config.get("shuffle", True))
    seed = int(config.get("seed", 1234))
    # Everything besides the source itself that decides which rows a part holds.
    settings = {"shuffle": shuffle, "reasoning_content": reasoning_content}
    if shuffle:
        settings["seed"] = seed

    rows: list[dict[str, Any]] = []
    for source, want in zip(config["sources"], budgets):
        if want == 0:
            print(f"{source['hf_dataset']}: weight rounds to 0 rows, skipped", flush=True)
            continue
        part = _part_path(parts_dir, source, settings)
        got = _read_jsonl(part)
        if got is None or len(got) < want:
            got = _take_chat_rows(source, output_dir, want, shuffle, seed, reasoning_content)
            _write_jsonl(part, got)
        rows.extend(got[:want])

    random.Random(seed).shuffle(rows)
    num_validation = min(1000, len(rows) // 20)
    splits = {"validation": rows[:num_validation], "train": rows[num_validation:]}
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError(f"Too few rows ({len(rows)}) for a train/validation split.")
    for name, split_rows in splits.items():
        _write_jsonl(output_dir / f"{name}.jsonl", split_rows)
    _copy_config(config_path, output_dir / "config.yaml")
    return {name: len(split_rows) for name, split_rows in splits.items()}


def prepare_megatron_data_blend(config_path: Path) -> list[tuple[float, str]]:
    """Download and tokenize the configured weighted data sources."""
    config = load_config(config_path)
    output_format = config.get("output_format", "megatron")
    if output_format != "megatron":
        raise ValueError(
            f"Unknown output_format {output_format!r} for a tokenized blend; use megatron, or "
            "chat_jsonl with prepare_chat_data_blend."
        )
    chat_keys = {"target_samples", "shuffle", "seed", "reasoning_content"} & config.keys()
    if chat_keys:
        raise ValueError(f"Tokenized blends do not use {sorted(chat_keys)}.")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target_tokens = config.get("target_tokens")
    total_tokens = None if target_tokens is None else int(target_tokens)
    tokenizer = str(config["tokenizer"])

    blend = _prepare_sources(config["sources"], output_dir, tokenizer, total_tokens)
    _write_data_blend(output_dir / "data_blend.txt", blend)
    _copy_config(config_path, output_dir / "config.yaml")
    return blend


def main() -> None:
    """Prepare a data blend from the supplied configuration."""
    parser = _build_parser()
    args = parser.parse_args()
    # Each entry point rejects a config written for the other.
    if load_config(args.config).get("output_format") == "chat_jsonl":
        sizes = prepare_chat_data_blend(args.config)
        print(f"Prepared chat JSONL: {sizes}. See train.jsonl, validation.jsonl and config.yaml.")
        return
    blend = prepare_megatron_data_blend(args.config)
    print(f"Prepared {len(blend)} data paths. See data_blend.txt and config.yaml in the output.")


if __name__ == "__main__":
    main()
