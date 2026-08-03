#!/usr/bin/env python3
"""Build the Cosmos3 Nano DFlash dataset by deduplicating output-prefix groups in parallel."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


PREFIX_PATTERN = re.compile(r"^output-(\d+)-")


@dataclass
class Partition:
    """Input files assigned to one independent merge worker."""

    files: dict[str, list[Path]] = field(default_factory=lambda: defaultdict(list))


def parse_args() -> argparse.Namespace:
    """Parse the recipe-specific parallel-merge configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pai-output", required=True, type=Path)
    parser.add_argument("--vqa-output", required=True, type=Path)
    parser.add_argument("--plain-text-input", required=True, type=Path)
    parser.add_argument("--multilingual-output", required=True, type=Path)
    parser.add_argument("--pai-root", required=True, type=Path)
    parser.add_argument("--vqa-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--word-overlap", type=float, default=0.90)
    parser.add_argument("--cache-contexts", type=int, default=25_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prefix_groups(directory: Path) -> dict[str, list[Path]]:
    """Return synthetic-output groups keyed by their first numeric prefix."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Missing synthetic-output directory: {directory}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(directory.glob("output-*.jsonl")):
        match = PREFIX_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Cannot determine output prefix for: {path}")
        groups[match.group(1)].append(path)
    if not groups:
        raise FileNotFoundError(f"No synthetic outputs found in: {directory}")
    return groups


def assign_groups(partitions: list[Partition], source_name: str, groups: dict[str, list[Path]]) -> None:
    """Assign each whole prefix group to one worker by stable numeric prefix."""

    for prefix, files in groups.items():
        partitions[int(prefix) % len(partitions)].files[source_name].extend(files)


def write_manifest(path: Path, files: list[Path]) -> None:
    """Write one worker's ordered source-file manifest."""

    path.write_text("".join(f"{file}\n" for file in files), encoding="utf-8")


def worker_command(
    worker: int,
    partition: Partition,
    args: argparse.Namespace,
    manifests_dir: Path,
    parts_dir: Path,
) -> list[str]:
    """Build the existing single-worker merger command for one partition."""

    merger = Path(__file__).with_name("merge_dflash_datasets.py")
    command = [
        sys.executable,
        str(merger),
        "--output",
        str(parts_dir / f"part-{worker:02d}.jsonl"),
        "--word-overlap",
        str(args.word_overlap),
        "--cache-contexts",
        str(args.cache_contexts),
    ]
    source_paths = {
        "pai_understanding": args.pai_output,
        "vqa_v2": args.vqa_output,
        "plain_text": args.plain_text_input,
        "specdec_multilingual_prompt": args.multilingual_output,
    }
    for source_name, files in sorted(partition.files.items()):
        manifest = manifests_dir / f"{source_name}-{worker:02d}.txt"
        write_manifest(manifest, files)
        command.extend(("--source", f"{source_name}={source_paths[source_name]}"))
        command.extend(("--source-file-list", f"{source_name}={manifest}"))
    if "pai_understanding" in partition.files:
        command.extend(("--media-root", f"pai_understanding={args.pai_root}"))
    if "vqa_v2" in partition.files:
        command.extend(("--media-root", f"vqa_v2={args.vqa_root}"))
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


def main() -> None:
    """Run prefix-isolated merge workers and combine their completed shard files."""

    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if not 0 < args.word_overlap <= 1:
        raise ValueError("--word-overlap must be in (0, 1]")
    if args.cache_contexts <= 0:
        raise ValueError("--cache-contexts must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {args.output}")
    if not args.plain_text_input.is_file():
        raise FileNotFoundError(f"Missing plain-text input: {args.plain_text_input}")
    if not args.pai_root.is_dir() or not args.vqa_root.is_dir():
        raise FileNotFoundError("Missing PAI or VQA media root")

    partitions = [Partition() for _ in range(args.jobs)]
    assign_groups(partitions, "pai_understanding", prefix_groups(args.pai_output))
    assign_groups(partitions, "vqa_v2", prefix_groups(args.vqa_output))
    assign_groups(partitions, "specdec_multilingual_prompt", prefix_groups(args.multilingual_output))
    partitions[0].files["plain_text"].append(args.plain_text_input)

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
    commands = [
        (worker, worker_command(worker, partition, args, manifests_dir, parts_dir))
        for worker, partition in enumerate(partitions)
        if partition.files
    ]

    processes = []
    try:
        for worker, command in commands:
            print(f"Starting worker {worker:02d}", flush=True)
            processes.append(subprocess.Popen(command))
        failed_workers = [
            worker
            for (worker, _), process in zip(commands, processes)
            if process.wait()
        ]
        if failed_workers:
            raise RuntimeError(f"Merge workers failed: {failed_workers}")
        concatenate_parts(parts_dir, args.output, args.overwrite)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        if args.output.exists() and not args.keep_work_dir:
            shutil.rmtree(work_dir)

    print(f"Wrote parallel merge to {args.output}")


if __name__ == "__main__":
    main()
