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
import json
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

from .megatron_preprocess_data import megatron_preprocess_data

__all__ = ["prepare_chat_data_blend", "prepare_megatron_data_blend"]

# A turn of a ``text`` row in the NeMo chat format: ``<extra_id_1>User\n...``. Any marker ends the
# turn, so a trailing delimiter or a label marker is not kept as content.
_CHAT_TURN = re.compile(
    r"<extra_id_\d+>(User|Assistant|System)\n?(.*?)(?=<extra_id_\d+>|\Z)", re.DOTALL
)
_CHAT_ROLES = {"User": "user", "Assistant": "assistant", "System": "system"}


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
    optional ``target_samples``, ``shuffle`` and ``seed`` keys apply.
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


def _to_chat(row: dict[str, Any], content_field: str) -> dict[str, Any] | None:
    """Convert a source row to ``{"messages": [...]}``, or ``None`` if it has no usable content.

    A ``text`` row in the NeMo chat format (``<extra_id_1>User`` / ``Assistant`` markers) is split
    into turns; any other ``text`` row becomes one assistant turn, so all of it is trained.
    """
    if content_field == "messages":
        return {"messages": row["messages"]} if row.get("messages") else None
    text = row.get("text") or ""
    if not text:
        return None
    if "<extra_id_" in text:
        turns = [
            {"role": _CHAT_ROLES[role], "content": content.strip()}
            for role, content in _CHAT_TURN.findall(text)
            if content.strip()
        ]
        # Markers that parse to no turns make the row malformed, not plain text: falling through
        # would train the raw <extra_id_*> delimiters as assistant content.
        return {"messages": turns} if turns else None
    return {"messages": [{"role": "assistant", "content": text}]}


def _iter_rows(source: dict[str, Any], output_dir: Path, shuffle: bool) -> Iterator[dict[str, Any]]:
    """Yield the rows of a source: its ``files`` in order, or its Hugging Face split, streamed."""
    if "files" in source:
        for path in _download_files(source, output_dir):
            with open(path, encoding="utf-8") as f:
                yield from (json.loads(line) for line in f if line.strip())
        return
    dataset = load_dataset(
        source["hf_dataset"], source.get("config"), split=source["split"], streaming=True
    )
    yield from dataset.shuffle(seed=42) if shuffle else dataset


def _take_chat_rows(
    source: dict[str, Any], output_dir: Path, want: int | None, shuffle: bool
) -> list[dict[str, Any]]:
    """Return the first ``want`` usable chat rows of a source (all of them if ``want`` is None).

    Fails if the source cannot be read or runs out of usable rows, so the blend cannot change
    silently; gives up after ``10 * want`` rejected rows.
    """
    # Megatron-Bridge's chat SFT path consumes this output; keep exactly the rows it accepts.
    from megatron.bridge.data.conversation_processing import is_text_only_chat_example

    name = f"{source['hf_dataset']} {source.get('split') or source.get('files')}"
    rows: list[dict[str, Any]] = []
    rejected = 0
    try:
        for row in _iter_rows(source, output_dir, shuffle):
            chat = _to_chat(row, source["content_field"])
            if chat is None or not is_text_only_chat_example(chat):
                rejected += 1
                if want is not None and rejected > 10 * want:
                    break
                continue
            rows.append(chat)
            if want is not None and len(rows) >= want:
                break
    except Exception as e:  # e.g. a gated split, or a schema the streaming reader cannot cast
        raise RuntimeError(f"{name}: failed to load") from e
    if want is not None and len(rows) < want:
        raise RuntimeError(f"{name}: only {len(rows)}/{want} usable rows")
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


def prepare_chat_data_blend(config_path: Path) -> dict[str, int]:
    """Write a weighted chat blend as ``train.jsonl`` / ``validation.jsonl``; return their sizes.

    For Megatron-Bridge's chat SFT path (``distill.py --sft --sft_hf_dataset``), which renders each
    conversation with the model's chat template and can restrict the loss to assistant tokens.
    Each source contributes ``round(target_samples * weight / sum(weights))`` rows (every usable
    row if ``target_samples`` is omitted) that the chat path accepts as text-only chat. Hugging Face
    splits are buffer-shuffled unless ``shuffle: false``; ``files`` are read from the beginning.
    The rows are mixed with ``seed`` and ``min(1000, n // 20)`` of them are held out for validation.
    Each source is cached under ``<output_dir>/parts``, so an interrupted run resumes.
    """
    config = load_config(config_path)
    ignored = {"tokenizer", "target_tokens"} & config.keys()
    if ignored:
        raise ValueError(f"output_format: chat_jsonl does not use {sorted(ignored)}.")
    if any("max_samples" in source for source in config["sources"]):
        raise ValueError("output_format: chat_jsonl does not use max_samples; set target_samples.")
    output_dir = Path(config["output_dir"])
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    target = config.get("target_samples")
    shuffle = bool(config.get("shuffle", True))
    weight_sum = sum(float(source["weight"]) for source in config["sources"])

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(config["sources"]):
        want = (
            None
            if target is None
            else max(1, round(int(target) * float(source["weight"]) / weight_sum))
        )
        label = source.get("split") or "files"
        dataset = source["hf_dataset"].replace("/", "--")
        part = parts_dir / f"{index:02d}_{dataset}_{source.get('config')}_{label}.jsonl"
        got = _read_jsonl(part)
        if got is None or (want is not None and len(got) < want):
            got = _take_chat_rows(source, output_dir, want, shuffle)
            _write_jsonl(part, got)
        rows.extend(got[:want])

    random.Random(int(config.get("seed", 1234))).shuffle(rows)
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
    chat_keys = {"target_samples", "shuffle", "seed"} & config.keys()
    if config.get("output_format", "megatron") != "megatron" or chat_keys:
        raise ValueError(
            "Tokenized blends do not use output_format: chat_jsonl, target_samples, shuffle or "
            "seed; run prepare_chat_data_blend for chat JSONL."
        )
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
    output_format = load_config(args.config).get("output_format", "megatron")
    if output_format == "chat_jsonl":
        sizes = prepare_chat_data_blend(args.config)
        print(f"Prepared chat JSONL: {sizes}. See train.jsonl, validation.jsonl and config.yaml.")
        return
    if output_format != "megatron":
        raise ValueError(f"Unknown output_format {output_format!r}; use megatron or chat_jsonl.")
    blend = prepare_megatron_data_blend(args.config)
    print(f"Prepared {len(blend)} data paths. See data_blend.txt and config.yaml in the output.")


if __name__ == "__main__":
    main()
