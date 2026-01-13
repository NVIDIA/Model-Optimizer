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

"""Nemotron VLM dataset utilities.

This module contains the Nemotron-VLM-Dataset-v2 specific logic:
- Subsets can store images in `media/shard_*.tar` (images only)
- Prompts/messages live in `<subset>/<subset>.jsonl` and reference the image filename (e.g. `292180.png`)

We join the tar images with the JSONL messages by the shared filename and yield samples compatible with our
VLM calibration pipeline.
"""

from __future__ import annotations

import functools
import json
import os
import random
import tarfile
from io import BytesIO
from typing import Any

import torch

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@functools.lru_cache(maxsize=8)
def list_repo_files_cached(repo_id: str, repo_type: str = "dataset") -> list[str]:
    """List files in a HuggingFace repo (cached).

    Args:
        repo_id: HF repo id (e.g., a dataset repo).
        repo_type: HF repo type, usually "dataset" here.
    """
    from huggingface_hub import list_repo_files

    return list_repo_files(repo_id=repo_id, repo_type=repo_type)


def extract_first_image_from_messages(messages: Any) -> Any:
    """Best-effort extraction of an image reference from Nemotron-style `messages`."""
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image"):
                continue
            if "image" in part:
                return part["image"]
            # fallback
            for key in ("images", "path", "image_url", "url", "value", "data"):
                if key in part:
                    return part[key]
    return None


class NemotronTarPlusJsonlIterable(torch.utils.data.IterableDataset):
    """Join Nemotron VLM `media/shard_*.tar` (images-only) with `<subset>/<subset>.jsonl` (messages)."""

    def __init__(
        self,
        repo_id: str,
        subsets: list[str],
        shard_paths: list[str],
        num_samples: int,
        seed: int,
        shuffle_buffer_size: int,
        max_shards: int | None,
    ):
        """Create an iterable dataset for Nemotron-VLM-Dataset-v2.

        Args:
            repo_id: Dataset repo id, e.g. "nvidia/Nemotron-VLM-Dataset-v2".
            subsets: Subset names to draw from (e.g., "sparsetables").
            shard_paths: Tar shard paths under `<subset>/media/`.
            num_samples: Total number of samples to yield.
            seed: RNG seed for sampling.
            shuffle_buffer_size: Unused for now (kept for API compatibility).
            max_shards: Max number of shards to use per subset (limits downloads).
        """
        super().__init__()
        self.repo_id = repo_id
        self.subsets = subsets
        self.shard_paths = shard_paths
        self.num_samples = num_samples
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_shards = max_shards

    def __iter__(self):
        from huggingface_hub import hf_hub_download
        from PIL import Image

        rng = random.Random(self.seed)

        # Partition shards by subset.
        shards_by_subset: dict[str, list[str]] = {s: [] for s in self.subsets}
        for p in self.shard_paths:
            subset = p.split("/", 1)[0]
            if subset in shards_by_subset:
                shards_by_subset[subset].append(p)

        for subset in list(shards_by_subset.keys()):
            shard_list = sorted(shards_by_subset[subset])
            if self.max_shards is not None:
                shard_list = shard_list[: max(0, self.max_shards)]
            shards_by_subset[subset] = shard_list

        # Roughly split sample budget across subsets.
        per_subset_target = max(1, self.num_samples // max(1, len(self.subsets)))
        yielded_total = 0

        for subset in self.subsets:
            if yielded_total >= self.num_samples:
                break

            shard_list = list(shards_by_subset.get(subset, []))
            if not shard_list:
                continue
            rng.shuffle(shard_list)

            # 1) Collect candidate image filenames from tar headers (no payload reads).
            candidate_names: list[str] = []
            header_limit = per_subset_target * 50
            for shard in shard_list:
                local_tar = hf_hub_download(
                    repo_id=self.repo_id, filename=shard, repo_type="dataset"
                )
                with tarfile.open(local_tar, "r:*") as tf:
                    for member in tf:
                        if not member.isfile():
                            continue
                        name = member.name
                        _, ext = os.path.splitext(name)
                        if ext.lower() not in _IMG_EXTS:
                            continue
                        candidate_names.append(name)
                        if len(candidate_names) >= header_limit:
                            break
                if len(candidate_names) >= header_limit:
                    break

            if not candidate_names:
                continue

            rng.shuffle(candidate_names)
            lookup_limit = per_subset_target * 10
            candidate_set = set(candidate_names[:lookup_limit])

            # 2) Scan JSONL to map image filename -> messages.
            jsonl_path = hf_hub_download(
                repo_id=self.repo_id, filename=f"{subset}/{subset}.jsonl", repo_type="dataset"
            )
            meta_by_image: dict[str, dict[str, Any]] = {}
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    msgs = obj.get("messages")
                    img_name = extract_first_image_from_messages(msgs) if msgs is not None else None
                    if isinstance(img_name, str) and img_name in candidate_set:
                        meta_by_image[img_name] = {"id": obj.get("id"), "messages": msgs}
                        if len(meta_by_image) >= per_subset_target:
                            break

            if not meta_by_image:
                continue

            # 3) Extract matched images and yield samples.
            needed = set(meta_by_image.keys())
            for shard in shard_list:
                if yielded_total >= self.num_samples or not needed:
                    break
                local_tar = hf_hub_download(
                    repo_id=self.repo_id, filename=shard, repo_type="dataset"
                )
                with tarfile.open(local_tar, "r:*") as tf:
                    for member in tf:
                        if yielded_total >= self.num_samples or not needed:
                            break
                        if not member.isfile():
                            continue
                        name = member.name
                        if name not in needed:
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        try:
                            raw = f.read()
                            if isinstance(raw, str):
                                raw = raw.encode()
                            raw_bytes: bytes = raw
                            img = Image.open(BytesIO(raw_bytes)).convert("RGB")
                        except Exception:
                            continue
                        meta = meta_by_image.get(name)
                        if not meta:
                            continue
                        yield {
                            "id": meta.get("id", name),
                            "messages": meta.get("messages"),
                            "image": img,
                        }
                        needed.discard(name)
                        yielded_total += 1
