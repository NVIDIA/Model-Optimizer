#!/usr/bin/env python3
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

"""Merge DFlash training sources with conservative, multimodal-safe deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from array import array
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

WORD_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+(?:['\u2019][^\W_]+)*",
    flags=re.UNICODE,
)
OUTPUT_PREFIX_PATTERN = re.compile(r"^output-(\d+)-")


@dataclass(frozen=True)
class Source:
    """An input JSONL file or directory and the root for its relative media paths."""

    name: str
    input_path: Path
    media_root: Path | None
    files: tuple[Path, ...]


@dataclass
class Partition:
    """Source files assigned to one independent merge worker."""

    files: dict[str, list[Path]] = field(default_factory=lambda: defaultdict(list))


class MediaResolutionError(ValueError):
    """Raised when a multimodal record cannot be converted to absolute media paths."""


def parse_source(value: str) -> tuple[str, Path]:
    """Parse a ``NAME=PATH`` command-line source argument."""

    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("sources must use NAME=PATH")
    return name, Path(path)


def parse_media_root(value: str) -> tuple[str, Path]:
    """Parse a ``NAME=PATH`` command-line media-root argument."""

    return parse_source(value)


def source_files(input_path: Path) -> Iterator[Path]:
    """Yield JSONL input files in deterministic order."""

    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Expected a .jsonl file: {input_path}")
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing input source: {input_path}")
    files = sorted(input_path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {input_path}")
    yield from files


def listed_source_files(file_list: Path) -> tuple[Path, ...]:
    """Read an explicit, ordered list of JSONL input files."""

    if not file_list.is_file():
        raise FileNotFoundError(f"Missing source-file list: {file_list}")
    files = tuple(
        Path(line.strip())
        for line in file_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not files:
        raise ValueError(f"Source-file list is empty: {file_list}")
    for path in files:
        if path.suffix != ".jsonl" or not path.is_file():
            raise FileNotFoundError(f"Invalid JSONL file listed in {file_list}: {path}")
    return files


def resolve_media_path(value: str, media_root: Path | None, resolved_paths: dict[str, str]) -> str:
    """Return a checked absolute media path for an image or video content part."""

    cache_key = f"{media_root}\0{value}"
    if cache_key in resolved_paths:
        return resolved_paths[cache_key]
    path = Path(value)
    if not path.is_absolute():
        if media_root is None:
            raise MediaResolutionError(f"Relative media path without a media root: {value}")
        path = media_root / path
    path = path.resolve()
    if not path.is_file():
        raise MediaResolutionError(f"Missing media file: {path}")
    resolved_paths[cache_key] = str(path)
    return resolved_paths[cache_key]


def has_text(content: str | list[dict]) -> bool:
    """Return whether a message content value contains non-empty text."""

    if isinstance(content, str):
        return bool(content.strip())
    return any(part.get("type") == "text" and str(part.get("text", "")).strip() for part in content)


def normalize_messages(
    record: dict, media_root: Path | None, resolved_paths: dict[str, str]
) -> list[dict] | None:
    """Validate a record and resolve its image/video references."""

    messages = record.get("messages") or record.get("conversations")
    if not isinstance(messages, list) or record.get("generation_error"):
        return None

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, (str, list)):
            return None
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"text", "image", "video"}:
                return None
            normalized_part = dict(part)
            part_type = normalized_part["type"]
            if part_type == "text":
                if not isinstance(normalized_part.get("text"), str):
                    return None
            else:
                media_value = normalized_part.get(part_type)
                if not isinstance(media_value, str):
                    return None
                normalized_part[part_type] = resolve_media_path(
                    media_value, media_root, resolved_paths
                )
            parts.append(normalized_part)
        normalized.append({"role": role, "content": parts})

    if not any(
        message["role"] == "user" and has_text(message["content"]) for message in normalized
    ):
        return None
    if not any(
        message["role"] == "assistant" and has_text(message["content"]) for message in normalized
    ):
        return None
    return normalized


def canonical_json(value: object) -> bytes:
    """Serialize identity values stably and without whitespace."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def candidate_key(messages: list[dict]) -> bytes:
    """Hash the fixed context used only to find likely duplicate candidates."""

    context = [message for message in messages if message["role"] != "assistant"]
    return hashlib.sha256(canonical_json(context)).digest()


def message_words(messages: list[dict]) -> array:
    """Return sorted, multiplicity-preserving hashes of all textual message words."""

    words: list[int] = []
    for message in messages:
        content = message["content"]
        text_values = (
            [content]
            if isinstance(content, str)
            else [part["text"] for part in content if part["type"] == "text"]
        )
        for text in text_values:
            words.extend(
                hash(word) & ((1 << 64) - 1) for word in WORD_PATTERN.findall(text.casefold())
            )
    return array("Q", sorted(words))


def has_required_overlap(first: array, second: array, threshold: float) -> bool:
    """Check mutual multiset-word overlap with a two-pointer scan."""

    if not first or not second:
        return False
    minimum = int(threshold * len(first) + 0.999999999)
    if len(second) < minimum or len(second) * threshold > len(first):
        return False

    first_index = 0
    second_index = 0
    shared = 0
    while first_index < len(first) and second_index < len(second):
        first_word = first[first_index]
        second_word = second[second_index]
        if first_word == second_word:
            shared += 1
            first_index += 1
            second_index += 1
        elif first_word < second_word:
            first_index += 1
        else:
            second_index += 1
    return shared >= minimum and shared >= int(threshold * len(second) + 0.999999999)


def parse_args() -> argparse.Namespace:
    """Parse merge configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="JSONL file or directory to include; may be specified multiple times.",
    )
    parser.add_argument(
        "--media-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Root used to resolve relative image/video paths for a source.",
    )
    parser.add_argument(
        "--source-file-list",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Ordered JSONL file list to use instead of recursively scanning a source directory.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL destination.")
    parser.add_argument(
        "--word-overlap",
        type=float,
        default=0.90,
        help="Mutual multiset-word overlap required to remove a candidate (default: 0.90).",
    )
    parser.add_argument(
        "--cache-contexts",
        type=int,
        default=25_000,
        help="Maximum recently seen prompt/media contexts retained for fuzzy comparisons.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Merge-worker count. Output files with the same synthetic shard prefix stay together; "
            "deduplication is local to each worker when this is greater than one."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output after success."
    )
    parser.add_argument(
        "--keep-work-dir", action="store_true", help="Keep parallel worker manifests and outputs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print parallel worker assignments without merging."
    )
    return parser.parse_args()


def load_sources(args: argparse.Namespace) -> list[Source]:
    """Validate source arguments and materialize their ordered JSONL files."""

    media_roots = dict(parse_media_root(value) for value in args.media_root)
    file_lists = dict(parse_source(value) for value in args.source_file_list)
    source_arguments = list(map(parse_source, args.source))
    source_names = {name for name, _ in source_arguments}
    unknown_file_lists = set(file_lists) - source_names
    if unknown_file_lists:
        raise ValueError(f"--source-file-list names are not sources: {sorted(unknown_file_lists)}")
    sources = [
        Source(
            name,
            input_path,
            media_roots.get(name),
            listed_source_files(file_lists[name])
            if name in file_lists
            else tuple(source_files(input_path)),
        )
        for name, input_path in source_arguments
    ]
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Each --source name must be unique")
    for source in sources:
        if source.media_root is not None and not source.media_root.is_dir():
            raise FileNotFoundError(f"Missing media root for {source.name}: {source.media_root}")

    return sources


def merge_sources(
    sources: list[Source], output: Path, word_overlap: float, cache_contexts: int
) -> None:
    """Merge source records into one atomic DFlash-training JSONL file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    candidates: OrderedDict[bytes, list[array]] = OrderedDict()
    written = Counter()
    skipped = Counter()
    processed = 0

    try:
        with temporary.open("x", encoding="utf-8") as destination:
            for source in sources:
                resolved_paths: dict[str, str] = {}
                for path in source.files:
                    with path.open(encoding="utf-8") as input_file:
                        for line_number, line in enumerate(input_file, start=1):
                            if not line.strip():
                                continue
                            processed += 1
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as error:
                                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                            try:
                                messages = normalize_messages(
                                    record, source.media_root, resolved_paths
                                )
                            except MediaResolutionError:
                                skipped["missing_media"] += 1
                                continue
                            if messages is None:
                                skipped["invalid_or_failed"] += 1
                                continue

                            key = candidate_key(messages)
                            words = message_words(messages)
                            retained = candidates.get(key, [])
                            if any(
                                has_required_overlap(words, previous, word_overlap)
                                for previous in retained
                            ):
                                skipped["near_duplicate"] += 1
                                continue

                            candidates.setdefault(key, []).append(words)
                            candidates.move_to_end(key)
                            while len(candidates) > cache_contexts:
                                candidates.popitem(last=False)

                            json.dump(
                                {
                                    "id": record.get("id", record.get("conversation_id")),
                                    "dataset": record.get("dataset", source.name),
                                    "messages": messages,
                                },
                                destination,
                                ensure_ascii=False,
                            )
                            destination.write("\n")
                            written[source.name] += 1
                            if processed % 100_000 == 0:
                                print(
                                    f"Processed {processed:,}; wrote {sum(written.values()):,}; "
                                    f"skipped {dict(skipped)}",
                                    flush=True,
                                )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"Wrote {sum(written.values()):,} conversations to {output}")
    print(f"By source: {dict(written)}; skipped: {dict(skipped)}")


def assign_source_files(partitions: list[Partition], source: Source) -> None:
    """Assign synthetic shard variants together and distribute other JSONL files evenly."""

    if source.input_path.is_file():
        partitions[0].files[source.name].extend(source.files)
        return

    grouped_by_prefix: dict[int, list[Path]] = defaultdict(list)
    ungrouped = []
    for path in source.files:
        match = OUTPUT_PREFIX_PATTERN.match(path.name)
        if match is None:
            ungrouped.append(path)
        else:
            grouped_by_prefix[int(match.group(1))].append(path)

    for prefix, files in sorted(grouped_by_prefix.items()):
        partitions[prefix % len(partitions)].files[source.name].extend(files)
    for index, path in enumerate(ungrouped):
        partitions[index % len(partitions)].files[source.name].append(path)


def write_manifest(path: Path, files: list[Path]) -> None:
    """Write one worker's ordered source-file manifest."""

    path.write_text("".join(f"{file}\n" for file in files), encoding="utf-8")


def worker_command(
    worker: int,
    partition: Partition,
    source_by_name: dict[str, Source],
    args: argparse.Namespace,
    manifests_dir: Path,
    parts_dir: Path,
) -> list[str]:
    """Build a single-worker invocation of this script for one file partition."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(parts_dir / f"part-{worker:02d}.jsonl"),
        "--word-overlap",
        str(args.word_overlap),
        "--cache-contexts",
        str(args.cache_contexts),
    ]
    for source_name, files in sorted(partition.files.items()):
        source = source_by_name[source_name]
        manifest = manifests_dir / f"{source_name}-{worker:02d}.txt"
        write_manifest(manifest, files)
        command.extend(("--source", f"{source.name}={source.input_path}"))
        command.extend(("--source-file-list", f"{source.name}={manifest}"))
        if source.media_root is not None:
            command.extend(("--media-root", f"{source.name}={source.media_root}"))
    return command


def concatenate_parts(parts_dir: Path, output: Path, overwrite: bool) -> None:
    """Join successful worker outputs into one atomic final JSONL file."""

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as destination:
            for part in sorted(parts_dir.glob("part-*.jsonl")):
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def merge_in_parallel(sources: list[Source], args: argparse.Namespace) -> None:
    """Merge source files concurrently while keeping synthetic shard variants together."""

    partitions = [Partition() for _ in range(args.jobs)]
    for source in sources:
        assign_source_files(partitions, source)
    for worker, partition in enumerate(partitions):
        counts = {source: len(files) for source, files in sorted(partition.files.items())}
        print(f"Worker {worker:02d}: {counts}")
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.output.with_name(f".{args.output.name}.parallel-{os.getpid()}")
    manifests_dir = work_dir / "manifests"
    parts_dir = work_dir / "parts"
    manifests_dir.mkdir(parents=True)
    parts_dir.mkdir()
    source_by_name = {source.name: source for source in sources}
    commands = [
        (worker, worker_command(worker, partition, source_by_name, args, manifests_dir, parts_dir))
        for worker, partition in enumerate(partitions)
        if partition.files
    ]

    processes = []
    merged = False
    try:
        for worker, command in commands:
            print(f"Starting worker {worker:02d}", flush=True)
            processes.append(subprocess.Popen(command))
        failed_workers = [
            worker for (worker, _), process in zip(commands, processes) if process.wait()
        ]
        if failed_workers:
            raise RuntimeError(f"Merge workers failed: {failed_workers}")
        concatenate_parts(parts_dir, args.output, args.overwrite)
        merged = True
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        if merged and not args.keep_work_dir:
            shutil.rmtree(work_dir)

    print(f"Wrote parallel merge to {args.output}")


def main() -> None:
    """Merge DFlash sources with one or more independent workers."""

    args = parse_args()
    if not args.source:
        raise ValueError("At least one --source is required")
    if not 0 < args.word_overlap <= 1:
        raise ValueError("--word-overlap must be in (0, 1]")
    if args.cache_contexts <= 0:
        raise ValueError("--cache-contexts must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists (pass --overwrite to replace it): {args.output}"
        )

    sources = load_sources(args)
    if args.jobs == 1:
        if args.dry_run:
            for source in sources:
                print(f"Source {source.name}: {len(source.files)} file(s)")
            return
        merge_sources(sources, args.output, args.word_overlap, args.cache_contexts)
    else:
        merge_in_parallel(sources, args)


if __name__ == "__main__":
    main()
