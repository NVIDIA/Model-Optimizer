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
"""Create synthetic-generation shards from supported benchmark layouts.

Supported adapters:
  * pai_understanding: PAI-Bench-U parquet metadata plus a local video tree.
  * vqa_v2: VQA v2 question/annotation JSON plus COCO image directories.
  * specdec_multilingual_prompt: OpenAI-style text prompts from NVIDIA's dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


THINKING_SUFFIX = (
    "\n\nAnswer using the following format:\n\n<think>\nYour reasoning grounded in the visual input.\n"
    "</think>\n\nWrite the final answer immediately after the </think> tag."
)

VQA_SPLITS = {
    "train": (
        "train2014",
        "COCO_train2014",
        "v2_OpenEnded_mscoco_train2014_questions.json",
        "v2_mscoco_train2014_annotations.json",
    ),
    "val": (
        "val2014",
        "COCO_val2014",
        "v2_OpenEnded_mscoco_val2014_questions.json",
        "v2_mscoco_val2014_annotations.json",
    ),
    "test": ("test2015", "COCO_test2015", "v2_OpenEnded_mscoco_test2015_questions.json", None),
}


def prompt(question: str, style: str, options: dict[str, str] | None = None) -> str:
    text = question.strip()
    if options:
        choices = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
        text += f"\n\nOptions:\n{choices}"
    if style == "plain":
        return text
    if options:
        text += "\n\nSelect the single best option based only on the visual input."
    return text + THINKING_SUFFIX


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def find_json(root: Path, filename: str) -> Path:
    candidates = (root / filename, root / "Questions" / filename, root / "Annotations" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} below {root}")


def image_path(
    image_root: Path, data_subtype: str, prefix: str, image_id: int
) -> tuple[Path, str] | None:
    filename = f"{prefix}_{image_id:012d}.jpg"
    for candidate in (image_root / data_subtype / filename, image_root / filename):
        if candidate.is_file():
            return candidate, str(candidate.relative_to(image_root))
    return None


def media_path_within_root(media_root: Path, relative_path: str) -> Path | None:
    """Resolve a media reference only when it remains below its declared root."""

    media_root = media_root.resolve()
    candidate = (media_root / relative_path).resolve()
    try:
        candidate.relative_to(media_root)
    except ValueError:
        return None
    return candidate


def normalize_options(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(answer)
        for key, answer in value.items()
        if answer is not None and str(answer).strip()
    }


def make_mosaic(video: Path, output_dir: Path, num_frames: int) -> str | None:
    """Use ffmpeg only: preparation stays independent of host OpenCV builds."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("--media_mode image_mosaic requires ffmpeg on the preparation host.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("--media_mode image_mosaic requires Pillow.") from exc
    digest = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:16]
    frame_dir = output_dir / "media_cache" / digest
    mosaic_path = frame_dir / "mosaic.png"
    if mosaic_path.is_file():
        return str(mosaic_path.relative_to(output_dir))
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_dir.glob("frame_*.png"):
        frame.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={max(1, min(num_frames, 32))}",
            "-frames:v",
            str(num_frames),
            str(frame_dir / "frame_%05d.png"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))[:num_frames]
    if not frames:
        return None
    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new("RGB", (columns * 224, rows * 224), (0, 0, 0))
    for index, frame in enumerate(frames):
        image = Image.open(frame).convert("RGB").resize((224, 224))
        row, column = divmod(index, columns)
        canvas.paste(image, (column * 224, row * 224))
    canvas.save(mosaic_path, "PNG")
    return str(mosaic_path.relative_to(output_dir))


def download_pai_bench(dataset_dir: Path, repo_id: str, force_download: bool) -> None:
    """Materialize the complete PAI-Bench-U Hugging Face dataset, including videos."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "--download requires huggingface_hub. Install it in the preparation environment."
        ) from exc
    dataset_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_dir),
        force_download=force_download,
    )


def pai_records(args: argparse.Namespace, output_dir: Path) -> Iterator[dict[str, Any]]:
    if not args.dataset_dir:
        raise ValueError("--dataset_dir is required with --dataset pai_understanding")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if args.download:
        download_pai_bench(dataset_dir, args.repo_id, args.force_download)
    media_root = Path(args.media_root or dataset_dir).expanduser().resolve()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("PAI-Bench-U preparation requires datasets and pyarrow.") from exc
    parquet_files = sorted((dataset_dir / "data").glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data/test-*.parquet found below {dataset_dir}")
    rows = load_dataset(
        "parquet", data_files={"test": [str(path) for path in parquet_files]}, split="test"
    )
    for row_index, row in enumerate(rows):
        category = row.get("category") or ""
        if args.category and category != args.category:
            continue
        question, relative_video = row.get("question"), row.get("video_path")
        if (
            not isinstance(question, str)
            or not isinstance(relative_video, str)
            or not relative_video.strip()
        ):
            continue
        relative_video = relative_video.strip()
        video = media_path_within_root(media_root, relative_video)
        if video is None:
            yield {"_missing_media": f"outside media root: {relative_video}"}
            continue
        if not video.is_file():
            yield {"_missing_media": str(video)}
            continue
        options = normalize_options(row.get("index2ans"))
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt(question, args.prompt_style, options)}
        ]
        media: dict[str, str] = {"video_path": relative_video}
        if args.media_mode == "image_mosaic":
            image = make_mosaic(video, output_dir, args.num_frames)
            if not image:
                yield {"_missing_media": str(video)}
                continue
            user_content.append({"type": "image", "image": image})
            media.update({"image_path": image, "source_video_path": relative_video})
        else:
            user_content.append({"type": "video", "video": relative_video, "fps": 4})
        yield {
            "id": f"pai_understanding_{row_index:05d}",
            "dataset": "pai_understanding",
            "messages": [{"role": "user", "content": user_content}],
            "prompt": prompt(question, args.prompt_style, options),
            "question": question,
            "reference_answer": row.get("answer"),
            "index2ans": options,
            "category": category,
            "subcategory": row.get("subcategory") or "",
            **media,
        }


def vqa_records(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if not args.vqa_root or not args.image_root:
        raise ValueError("--vqa_root and --image_root are required with --dataset vqa_v2")
    vqa_root = Path(args.vqa_root).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    for split in [value.strip() for value in args.vqa_splits.split(",") if value.strip()]:
        if split not in VQA_SPLITS:
            raise ValueError(f"Unsupported VQA split {split!r}; choose from {sorted(VQA_SPLITS)}")
        subtype, prefix, question_name, annotation_name = VQA_SPLITS[split]
        questions = load_json(find_json(vqa_root, question_name)).get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"questions is not a list for split={split}")
        annotations: dict[int, dict[str, Any]] = {}
        if annotation_name:
            annotation_rows = load_json(find_json(vqa_root, annotation_name)).get("annotations")
            if not isinstance(annotation_rows, list):
                raise ValueError(f"annotations is not a list for split={split}")
            annotations = {
                item["question_id"]: item
                for item in annotation_rows
                if isinstance(item, dict) and isinstance(item.get("question_id"), int)
            }
        elif not args.include_unannotated_vqa:
            raise ValueError(
                f"VQA split={split} has no public annotations; pass --include_unannotated_vqa to use it."
            )
        for question_row in questions:
            if not isinstance(question_row, dict):
                continue
            question_id, image_id, question = (
                question_row.get("question_id"),
                question_row.get("image_id"),
                question_row.get("question"),
            )
            if (
                not isinstance(question_id, int)
                or not isinstance(image_id, int)
                or not isinstance(question, str)
            ):
                continue
            resolved = image_path(image_root, subtype, prefix, image_id)
            if not resolved:
                yield {"_missing_media": f"{subtype}/{prefix}_{image_id:012d}.jpg"}
                continue
            _, relative_image = resolved
            annotation = annotations.get(question_id, {})
            counts = Counter(
                item.get("answer", "").strip()
                for item in (annotation.get("answers") or [])
                if isinstance(item, dict)
                and isinstance(item.get("answer"), str)
                and item["answer"].strip()
            )
            record: dict[str, Any] = {
                "id": f"vqa_v2_{question_id}",
                "dataset": "vqa_v2",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt(question, args.prompt_style)},
                            {"type": "image", "image": relative_image},
                        ],
                    }
                ],
                "prompt": prompt(question, args.prompt_style),
                "question": question,
                "image_path": relative_image,
                "question_id": question_id,
                "image_id": image_id,
                "source_split": split,
            }
            if annotation:
                record.update(
                    {
                        "reference_answer": annotation.get("multiple_choice_answer"),
                        "question_type": annotation.get("question_type"),
                        "answer_type": annotation.get("answer_type"),
                        "answer_counts": dict(counts),
                    }
                )
                if args.include_all_answers:
                    record["answers"] = annotation.get("answers")
            yield record


def text_records(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if not args.text_data:
        raise ValueError("--text_data is required with --dataset specdec_multilingual_prompt")
    path = Path(args.text_data).expanduser().resolve()
    if path.is_file():
        files = [path]
    else:
        # The NVIDIA repository includes sample-*.jsonl convenience subsets.
        # Prefer default.jsonl for a directory input so a full run neither
        # misses most of the data nor duplicates the sampled records.
        default_file = path / "default.jsonl"
        files = [default_file] if default_file.is_file() else sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL input file found at {path}")

    record_index = 0
    for file_path in files:
        with file_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    source = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file_path}:{line_number}") from exc
                messages = source.get("messages") or source.get("conversations")
                if not isinstance(messages, list):
                    continue
                conversations: list[dict[str, str]] = []
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role") or message.get("from")
                    content = message.get("content") or message.get("value")
                    if role in ("human", "user"):
                        role = "user"
                    elif role in ("gpt", "assistant"):
                        role = "assistant"
                    elif role in ("system", "developer"):
                        role = "system"
                    else:
                        continue
                    if not isinstance(content, str) or not content.strip():
                        continue
                    conversations.append({"role": role, "content": content})
                if not any(message["role"] == "user" for message in conversations):
                    continue
                yield {
                    "id": source.get("id", f"specdec_multilingual_prompt_{record_index:07d}"),
                    "dataset": "nvidia/Speculative-Decoding-Multilingual-Prompt-v2",
                    "source_file": file_path.name,
                    "conversations": conversations,
                }
                record_index += 1


def commit_staged_output(staging_dir: Path, output_dir: Path) -> None:
    """Replace generated shards only after the complete staged build succeeds."""

    for shard in output_dir.glob("train-*.jsonl"):
        shard.unlink()
    shutil.rmtree(output_dir / "media_cache", ignore_errors=True)
    for path in staging_dir.iterdir():
        path.replace(output_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("pai_understanding", "vqa_v2", "specdec_multilingual_prompt"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_dir", help="PAI-Bench-U metadata root")
    parser.add_argument("--media_root", help="PAI-Bench-U video root")
    parser.add_argument(
        "--download",
        action="store_true",
        help="For pai_understanding, snapshot the complete Hugging Face dataset into --dataset_dir before sharding.",
    )
    parser.add_argument("--repo_id", default="shi-labs/physical-ai-bench-understanding")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--category", help="Optional exact PAI-Bench-U category")
    parser.add_argument("--vqa_root", help="VQA v2 questions and annotations root")
    parser.add_argument("--image_root", help="VQA v2 COCO image root")
    parser.add_argument("--vqa_splits", default="train", help="Comma-separated VQA v2 splits")
    parser.add_argument("--include_unannotated_vqa", action="store_true")
    parser.add_argument("--include_all_answers", action="store_true")
    parser.add_argument(
        "--text_data",
        help="JSONL file or directory for specdec_multilingual_prompt; rows must contain messages or conversations.",
    )
    parser.add_argument("--media_mode", choices=("native", "image_mosaic"), default="native")
    parser.add_argument("--prompt_style", choices=("plain", "thinking"), default="thinking")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_lines_per_shard", type=int, default=128)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        help="Deterministically shuffle PAI or VQA records before applying --start_index/--num_samples.",
    )
    parser.add_argument("--missing_media", choices=("error", "skip"), default="error")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_lines_per_shard <= 0 or args.num_frames <= 0 or args.start_index < 0:
        raise ValueError(
            "max_lines_per_shard/num_frames must be positive and start_index must be non-negative"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = list(output_dir.glob("train-*.jsonl"))
    if (existing_shards or (output_dir / "media_cache").exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already contains generated shards: {output_dir} (pass --overwrite to replace them)."
        )

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    output = None
    written = missing = source_index = shard_index = lines = 0
    try:
        if args.dataset == "pai_understanding":
            iterator = pai_records(args, staging_dir)
        elif args.dataset == "vqa_v2":
            iterator = vqa_records(args)
        else:
            iterator = text_records(args)
        if args.shuffle_seed is not None:
            if args.dataset == "specdec_multilingual_prompt":
                raise ValueError(
                    "--shuffle_seed is not supported for the full text dataset; it would materialize every prompt."
                )
            shuffled_records = list(iterator)
            random.Random(args.shuffle_seed).shuffle(shuffled_records)
            iterator = iter(shuffled_records)

        for record in iterator:
            if "_missing_media" in record:
                missing += 1
                print(f"Missing media: {record['_missing_media']}")
                continue
            if source_index < args.start_index:
                source_index += 1
                continue
            source_index += 1
            if args.num_samples is not None and written >= args.num_samples:
                break
            if output is None or lines >= args.max_lines_per_shard:
                if output:
                    output.close()
                    shard_index += 1
                output = (staging_dir / f"train-{shard_index:05d}-{shard_index:05d}.jsonl").open(
                    "w", encoding="utf-8"
                )
                lines = 0
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            lines += 1
        if output:
            output.close()
            output = None
        if missing and args.missing_media == "error":
            raise FileNotFoundError(
                f"{missing} referenced media files were missing; rerun with --missing_media skip "
                "for an explicit partial dataset."
            )
        if not written:
            raise RuntimeError("No usable records were written")
        commit_staged_output(staging_dir, output_dir)
    finally:
        if output:
            output.close()
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Wrote {written} {args.dataset} records to {shard_index + 1} shard(s) at {output_dir}")


if __name__ == "__main__":
    main()
