# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Precompute Qwen-Image-Edit-2511 DMD2 training caches.

Two local input layouts are supported:

* SpatialEdit-style WebDataset roots containing ``*.tar`` shards.  A sample contains
  ``<key>.json`` and indexed images such as ``<key>.0.jpg`` / ``<key>.1.jpg``.  Images are
  ordered by index; the last image is the target and every preceding image is a reference.
* A JSONL manifest with ``target``, ``conditioning`` (a path or path list), and ``prompt``.
  The data-tooling aliases ``generated_image`` / ``reference_image`` (targets) and
  ``conditioning_images`` (sources) are also accepted, including ``{archive, member}``
  descriptors. Relative paths are resolved against the manifest directory. ``id``,
  ``negative_prompt``, and ``metadata`` are optional.

The output follows ``BaseMultiresolutionDataset``'s sharded ``metadata.json`` layout.  Each
``.pt`` record contains a sampled target ``latent``, a list of deterministic
``conditioning_latents``, and positive/negative multimodal embeddings and masks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import tarfile
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

logger = logging.getLogger(__name__)

_IMAGE_MEMBER_RE = re.compile(
    r"^(?P<key>.+)\.(?P<index>\d+)\.(?:jpe?g|png|webp)$",
    flags=re.IGNORECASE,
)
_IMAGE_MARKER_RE = re.compile(r"(?:<image>\s*)+", flags=re.IGNORECASE)


@dataclass
class EditSample:
    """One loaded edit pair.  Only one instance is retained while streaming input shards."""

    sample_id: str
    target_image: Image.Image
    conditioning_images: list[Image.Image]
    prompt: str
    negative_prompt: str
    target_path: str
    conditioning_paths: list[str]
    source_metadata: dict[str, Any] | None = None


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_tar_rgb(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Image.Image:
    fileobj = archive.extractfile(member)
    if fileobj is None:
        raise OSError(f"Could not read {member.name!r} from {archive.name!r}")
    with Image.open(io.BytesIO(fileobj.read())) as image:
        return image.convert("RGB")


def _strip_image_markers(text: str) -> str:
    """Remove dataset placeholders because EditPlus inserts its own visual tokens."""

    return _IMAGE_MARKER_RE.sub("", text).strip()


def _conversation_text(payload: dict[str, Any], key: str) -> str | None:
    conversation = payload.get(key)
    if not isinstance(conversation, list):
        return None
    for message in conversation:
        if isinstance(message, dict) and message.get("from") in {"human", "user"}:
            value = message.get("value") or message.get("text")
            if isinstance(value, str) and value.strip():
                return value
    return None


def extract_spatialedit_prompt(payload: dict[str, Any], variant: str = "human") -> str:
    """Extract a usable instruction across SpatialEdit's three shard families."""

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if variant == "human":
        candidates = (
            metadata.get("instruction_human"),
            _conversation_text(payload, "conversations_human"),
            metadata.get("instruction"),
            _conversation_text(payload, "conversations"),
        )
    elif variant == "raw":
        candidates = (
            metadata.get("instruction"),
            _conversation_text(payload, "conversations"),
            metadata.get("instruction_human"),
            _conversation_text(payload, "conversations_human"),
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant!r}")
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            prompt = _strip_image_markers(candidate)
            if prompt:
                return prompt
    raise ValueError("SpatialEdit sample has no non-empty edit instruction")


def _spatialedit_sample_id(payload: dict[str, Any], fallback: str) -> str:
    metadata = payload.get("metadata")
    candidates = (
        payload.get("SAMPLE_ID"),
        payload.get("id"),
        metadata.get("id") if isinstance(metadata, dict) else None,
        fallback,
    )
    return str(next(value for value in candidates if value is not None and str(value)))


def _spatialedit_metadata(
    payload: dict[str, Any],
    tar_path: Path,
    sample_key: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": "SpatialEdit-500K",
        "tar_path": str(tar_path.resolve()),
        "webdataset_key": sample_key,
    }
    for key in ("metadata", "meta", "data_type", "multi_image", "only_text"):
        if key in payload:
            result[key] = payload[key]
    return result


def iter_spatialedit_samples(
    root: Path,
    *,
    negative_prompt: str = " ",
    prompt_variant: str = "human",
    shard_rank: int = 0,
    shard_world: int = 1,
) -> Iterator[EditSample]:
    """Stream native SpatialEdit WebDataset pairs without extracting shards to disk."""

    tar_paths = sorted(path for path in root.rglob("*.tar") if path.is_file())
    if not tar_paths:
        raise FileNotFoundError(f"No .tar shards found under {root}")
    selected = tar_paths[shard_rank::shard_world]
    logger.info(
        "SpatialEdit input: %d/%d tar shards assigned to rank %d",
        len(selected),
        len(tar_paths),
        shard_rank,
    )

    for tar_path in selected:
        try:
            with tarfile.open(tar_path, mode="r:*") as archive:
                grouped: dict[str, dict[str, Any]] = {}
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    if member.name.lower().endswith(".json"):
                        key = member.name[: -len(".json")]
                        grouped.setdefault(key, {})["json"] = member
                        continue
                    match = _IMAGE_MEMBER_RE.match(member.name)
                    if match:
                        group = grouped.setdefault(match.group("key"), {})
                        group.setdefault("images", {})[int(match.group("index"))] = member

                for sample_key in sorted(grouped):
                    members = grouped[sample_key]
                    image_members = members.get("images", {})
                    if "json" not in members or len(image_members) < 2:
                        logger.warning(
                            "Skipping incomplete WebDataset sample %s::%s (json=%s, images=%d)",
                            tar_path,
                            sample_key,
                            "json" in members,
                            len(image_members),
                        )
                        continue
                    json_file = archive.extractfile(members["json"])
                    if json_file is None:
                        raise OSError(f"Could not read metadata for {tar_path}::{sample_key}")
                    payload = json.loads(json_file.read())
                    if not isinstance(payload, dict):
                        raise ValueError(f"Metadata for {tar_path}::{sample_key} is not an object")

                    ordered = sorted(image_members.items())
                    conditioning = [_load_tar_rgb(archive, member) for _, member in ordered[:-1]]
                    _, target_member = ordered[-1]
                    target = _load_tar_rgb(archive, target_member)
                    display_prefix = f"{tar_path.resolve()}::"
                    yield EditSample(
                        sample_id=_spatialedit_sample_id(payload, sample_key),
                        target_image=target,
                        conditioning_images=conditioning,
                        prompt=extract_spatialedit_prompt(payload, prompt_variant),
                        negative_prompt=negative_prompt,
                        target_path=f"{display_prefix}{target_member.name}",
                        conditioning_paths=[
                            f"{display_prefix}{member.name}" for _, member in ordered[:-1]
                        ],
                        source_metadata=_spatialedit_metadata(payload, tar_path, sample_key),
                    )
        except Exception:
            logger.error("Failed while reading WebDataset shard %s", tar_path)
            logger.debug(traceback.format_exc())
            raise


def _manifest_value(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _resolve_manifest_path(value: Any, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {field!r} must be a non-empty local path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {field} image does not exist: {path}")
    return path


class _ArchiveImageLoader:
    """Small LRU of open tar files for archive/member JSONL descriptors."""

    def __init__(self, max_open_archives: int = 8) -> None:
        self.max_open_archives = max_open_archives
        self._archives: OrderedDict[Path, tarfile.TarFile] = OrderedDict()

    def _archive(self, path: Path) -> tarfile.TarFile:
        archive = self._archives.pop(path, None)
        if archive is None:
            archive = tarfile.open(path, mode="r:*")
        self._archives[path] = archive
        while len(self._archives) > self.max_open_archives:
            _, stale = self._archives.popitem(last=False)
            stale.close()
        return archive

    def load(self, archive_path: Path, member_name: str) -> Image.Image:
        archive = self._archive(archive_path)
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Archive member does not exist: {archive_path}::{member_name}"
            ) from exc
        return _load_tar_rgb(archive, member)

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()


def _load_manifest_image(
    value: Any,
    base_dir: Path,
    field: str,
    archive_loader: _ArchiveImageLoader,
) -> tuple[Image.Image, str]:
    """Load a local path or ``{archive, member}`` image descriptor."""

    if isinstance(value, str):
        path = _resolve_manifest_path(value, base_dir, field)
        return _load_rgb(path), str(path)
    if not isinstance(value, dict):
        raise ValueError(
            f"Manifest field {field!r} must be a path or an {{archive, member}} object"
        )
    if "path" in value:
        path = _resolve_manifest_path(value["path"], base_dir, field)
        return _load_rgb(path), str(path)

    archive_path = _resolve_manifest_path(value.get("archive"), base_dir, f"{field}.archive")
    member = value.get("member")
    if not isinstance(member, str) or not member:
        raise ValueError(f"Manifest field {field!r}.member must be a non-empty string")
    return archive_loader.load(archive_path, member), f"{archive_path}::{member}"


def iter_jsonl_samples(
    manifest: Path,
    *,
    negative_prompt: str = " ",
    shard_rank: int = 0,
    shard_world: int = 1,
) -> Iterator[EditSample]:
    """Stream generic local edit records from a JSONL manifest."""

    base_dir = manifest.resolve().parent
    archive_loader = _ArchiveImageLoader()
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Manifest line {line_index + 1} is not a JSON object")
                shard_value = record.get("archive_index")
                if shard_value is None:
                    shard_value = line_index
                try:
                    assigned_rank = int(shard_value) % shard_world
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Manifest line {line_index + 1} archive_index must be an integer"
                    ) from exc
                if assigned_rank != shard_rank:
                    continue

                target_value = _manifest_value(
                    record,
                    (
                        "target",
                        "target_image",
                        "generated_image",
                        "output_image",
                        "output",
                        "reference_image",
                    ),
                )
                conditioning_value = _manifest_value(
                    record,
                    (
                        "conditioning",
                        "conditioning_images",
                        "reference_images",
                        "source_images",
                        "source",
                        "input",
                    ),
                )
                prompt_value = _manifest_value(
                    record, ("prompt", "instruction", "edit_instruction")
                )
                if isinstance(conditioning_value, (str, dict)):
                    conditioning_value = [conditioning_value]
                if not isinstance(conditioning_value, list) or not conditioning_value:
                    raise ValueError(
                        f"Manifest line {line_index + 1} must provide one or more "
                        "conditioning images"
                    )
                if not isinstance(prompt_value, str) or not prompt_value.strip():
                    raise ValueError(f"Manifest line {line_index + 1} has no edit prompt")

                target_image, target_path = _load_manifest_image(
                    target_value,
                    base_dir,
                    "target",
                    archive_loader,
                )
                loaded_conditioning = [
                    _load_manifest_image(value, base_dir, "conditioning", archive_loader)
                    for value in conditioning_value
                ]
                per_sample_negative = record.get("negative_prompt", negative_prompt)
                if not isinstance(per_sample_negative, str):
                    raise ValueError(
                        f"Manifest line {line_index + 1} negative_prompt must be a string"
                    )
                sample_id = next(
                    str(value)
                    for value in (
                        record.get("id"),
                        record.get("sample_id"),
                        record.get("source_id"),
                        record.get("key"),
                        line_index,
                    )
                    if value is not None and str(value)
                )
                yield EditSample(
                    sample_id=sample_id,
                    target_image=target_image,
                    conditioning_images=[image for image, _ in loaded_conditioning],
                    prompt=_strip_image_markers(prompt_value),
                    negative_prompt=per_sample_negative,
                    target_path=target_path,
                    conditioning_paths=[path for _, path in loaded_conditioning],
                    source_metadata=record.get("metadata"),
                )
    finally:
        archive_loader.close()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


class MetadataShardWriter:
    """Incrementally write metadata shards so 500K samples need not stay in memory."""

    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
        shard_rank: int,
        shard_world: int,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_rank = shard_rank
        self.shard_world = shard_world
        self.buffer: list[dict[str, Any]] = []
        self.shards: list[str] = []
        self.total_items = 0

    def add(self, item: dict[str, Any]) -> None:
        self.buffer.append(item)
        self.total_items += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        rank_prefix = f"r{self.shard_rank:02d}_" if self.shard_world > 1 else ""
        filename = f"metadata_shard_{rank_prefix}s{len(self.shards):04d}.json"
        _write_json_atomic(self.output_dir / filename, self.buffer)
        self.shards.append(filename)
        self.buffer = []

    def finish(self, **config: Any) -> Path:
        self.flush()
        if not self.shards:
            raise RuntimeError("No valid samples were preprocessed; metadata was not written")
        index_name = (
            f"metadata_r{self.shard_rank:02d}.json" if self.shard_world > 1 else "metadata.json"
        )
        payload = {
            "processor": "qwen_image_edit",
            "model_type": "qwen_image_edit",
            "total_items": self.total_items,
            "num_shards": len(self.shards),
            "shard_size": self.shard_size,
            "shards": self.shards,
            **config,
        }
        if self.shard_world > 1:
            payload.update(shard_rank=self.shard_rank, shard_world=self.shard_world)
        index_path = self.output_dir / index_name
        _write_json_atomic(index_path, payload)
        return index_path


def _cache_identity(
    sample: EditSample,
    model_name: str,
    resolution: tuple[int, int],
    conditioning_max_pixels: int,
) -> str:
    fields = (
        model_name,
        sample.sample_id,
        sample.target_path,
        *sample.conditioning_paths,
        sample.prompt,
        sample.negative_prompt,
        f"{resolution[0]}x{resolution[1]}",
        str(conditioning_max_pixels),
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def _stable_sample_seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def preprocess_samples(
    samples: Iterable[EditSample],
    *,
    output_dir: Path,
    model_name: str,
    device: str,
    max_pixels: int,
    conditioning_max_pixels: int,
    metadata_shard_size: int,
    shard_rank: int,
    shard_world: int,
    verify: bool,
    overwrite: bool,
    fail_fast: bool,
    limit: int | None,
    seed: int,
    log_every: int,
) -> Path:
    """Encode a stream of edit samples and return the generated metadata index path."""

    import torch
    from nemo_automodel.components.datasets.diffusion.multi_tier_bucketing import (
        MultiTierBucketCalculator,
    )
    from preprocess.processors import QwenImageEditProcessor

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = QwenImageEditProcessor()
    models = processor.load_models(model_name, device)
    calculator = MultiTierBucketCalculator(quantization=64, max_pixels=max_pixels)
    writer = MetadataShardWriter(
        output_dir,
        metadata_shard_size,
        shard_rank,
        shard_world,
    )

    attempted = 0
    failures = 0
    for sample in samples:
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        try:
            original_width, original_height = sample.target_image.size
            bucket = calculator.get_bucket_for_image(original_width, original_height)
            target_width, target_height = bucket["resolution"]
            resolution = (target_width, target_height)
            cache_hash = _cache_identity(
                sample,
                model_name,
                resolution,
                conditioning_max_pixels,
            )
            cache_subdir = output_dir / f"{target_width}x{target_height}"
            cache_subdir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_subdir / f"{cache_hash}.pt"

            if overwrite or not cache_file.is_file():
                resized_target, crop_offset = calculator.resize_and_crop(
                    sample.target_image,
                    target_width,
                    target_height,
                    crop_mode="center",
                )
                sample_seed = _stable_sample_seed(sample.sample_id, seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                target_tensor = processor.preprocess_image(resized_target)
                latent = processor.encode_image(target_tensor, models, device)
                if verify and not processor.verify_latent(latent, models, device):
                    raise ValueError("target latent verification failed")

                conditioning_latents = processor.encode_conditioning_images(
                    sample.conditioning_images,
                    models,
                    device,
                    max_pixels=conditioning_max_pixels,
                )
                text_encodings = processor.encode_edit_prompts(
                    sample.prompt,
                    sample.negative_prompt,
                    sample.conditioning_images,
                    models,
                    device,
                )
                cache_metadata = {
                    "conditioning_latents": conditioning_latents,
                    "original_resolution": (original_width, original_height),
                    "bucket_resolution": resolution,
                    "crop_offset": crop_offset,
                    "prompt": sample.prompt,
                    "negative_prompt": sample.negative_prompt,
                    "image_path": sample.target_path,
                    "conditioning_image_paths": sample.conditioning_paths,
                    "conditioning_resolutions": [
                        tuple(image.size) for image in sample.conditioning_images
                    ],
                    "bucket_id": bucket["id"],
                    "aspect_ratio": bucket["aspect_ratio"],
                    "sample_id": sample.sample_id,
                    "source_metadata": sample.source_metadata,
                }
                cache = processor.get_cache_data(latent, text_encodings, cache_metadata)
                temporary = cache_file.with_name(f".{cache_file.name}.tmp-{os.getpid()}")
                torch.save(cache, temporary)
                os.replace(temporary, cache_file)
            else:
                crop_offset = (0, 0)

            writer.add(
                {
                    "cache_file": str(cache_file.resolve()),
                    "image_path": sample.target_path,
                    "conditioning_image_paths": sample.conditioning_paths,
                    "conditioning_resolutions": [
                        list(image.size) for image in sample.conditioning_images
                    ],
                    "sample_id": sample.sample_id,
                    "bucket_resolution": [target_width, target_height],
                    "original_resolution": [original_width, original_height],
                    "prompt": sample.prompt,
                    "bucket_id": bucket["id"],
                    "aspect_ratio": bucket["aspect_ratio"],
                    "pixels": target_width * target_height,
                    "model_type": processor.model_type,
                }
            )
            if log_every > 0 and attempted % log_every == 0:
                logger.info(
                    "Processed %d samples (%d failures, %d cached records)",
                    attempted,
                    failures,
                    writer.total_items,
                )
        except Exception as exc:
            failures += 1
            logger.error("Failed sample %s: %s", sample.sample_id, exc)
            logger.debug(traceback.format_exc())
            if fail_fast:
                raise

    index_path = writer.finish(
        model_name=model_name,
        max_pixels=max_pixels,
        conditioning_max_pixels=conditioning_max_pixels,
        attempted_items=attempted,
        failed_items=failures,
        negative_prompt_is_per_sample=True,
    )
    logger.info(
        "Finished preprocessing: %d records, %d failures; metadata=%s",
        writer.total_items,
        failures,
        index_path,
    )
    return index_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--webdataset-root",
        "--input-dir",
        dest="webdataset_root",
        type=Path,
        help="SpatialEdit-style root recursively containing WebDataset .tar shards",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="Local JSONL with target, conditioning, and prompt fields",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda or cuda:<gpu-id> when --gpu-id is provided.",
    )
    parser.add_argument("--gpu-id", type=int, help="GPU index used when --device is omitted")
    parser.add_argument("--max-pixels", type=_positive_int, default=1024 * 1024)
    parser.add_argument(
        "--conditioning-max-pixels",
        type=_positive_int,
        default=1024 * 1024,
        help="Per-reference VAE pixel budget (the official EditPlus default is 1024^2)",
    )
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument(
        "--prompt-variant",
        choices=("human", "raw"),
        default="human",
        help="SpatialEdit instruction variant; ignored for JSONL manifests",
    )
    parser.add_argument("--metadata-shard-size", type=_positive_int, default=10_000)
    parser.add_argument("--shard-rank", "--shard-idx", dest="shard_rank", type=int, default=0)
    parser.add_argument(
        "--shard-world",
        "--shard-count",
        dest="shard_world",
        type=_positive_int,
        default=1,
    )
    parser.add_argument("--limit", "--max-samples", dest="limit", type=_positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_rank < args.shard_world:
        raise ValueError(
            f"shard_rank must satisfy 0 <= rank < world; got {args.shard_rank}/{args.shard_world}"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    device = args.device
    if device is None:
        device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"

    if args.webdataset_root is not None:
        samples = iter_spatialedit_samples(
            args.webdataset_root,
            negative_prompt=args.negative_prompt,
            prompt_variant=args.prompt_variant,
            shard_rank=args.shard_rank,
            shard_world=args.shard_world,
        )
    else:
        samples = iter_jsonl_samples(
            args.manifest,
            negative_prompt=args.negative_prompt,
            shard_rank=args.shard_rank,
            shard_world=args.shard_world,
        )

    return preprocess_samples(
        samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=device,
        max_pixels=args.max_pixels,
        conditioning_max_pixels=args.conditioning_max_pixels,
        metadata_shard_size=args.metadata_shard_size,
        shard_rank=args.shard_rank,
        shard_world=args.shard_world,
        verify=args.verify,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        limit=args.limit,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
