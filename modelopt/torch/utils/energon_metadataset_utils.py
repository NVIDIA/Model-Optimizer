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

"""Calibration samples from a Megatron-Energon ``MetadatasetV2`` blend.

Motivation
----------
Training blends for VLMs are not a single dataset. A blend is a tree of yamls whose leaves each
name **two** datasets: one holding the conversations (annotations) and a *different* one holding
the images. The image dataset is shared across many leaves -- one copy of COCO serves every task
written about COCO -- so *where the pixels are* is a property of the leaf, not of the run.

That breaks the assumption in :mod:`vlm_dataset_utils`, which resolves images against a single
``image_root`` for the whole run.

Rather than teach the read path about leaves, this module **normalises everything up front** into a
manifest: one row per sample, carrying the messages and an *absolute* image path. Every branch --
``dss://`` resolution, tar-vs-jsonl containers, image roots that live either on the URI or inside
the record -- is evaluated once at build time. Rows come out uniform, so the read path stays trivial
and the existing collate works unchanged.

Layout this understands (measured on the Nemotron 3.5 Super SFT blend: 686 leaves, 253 image sets)::

    metadataset.yaml            MetadatasetV2, splits.train.blend_epochized[]
      -> leaf yaml              (recursive; leaves may be nested several deep)
        -> leaf entry
             path:         dss://<ann>@v0[/file]        annotations
             media_source: filesystem+dss://<img>@v0[/subdir]   images
             subflavors.length                          for weighted sampling

    dss://<name>@<ver>[/suffix]
      -> <dss_root>/<name>/<ver>[/suffix]

    annotations:  shard-*.tar of NNNNNNNN.json   (467 leaves)  |  <name>.jsonl  (161)
    records:      {"conversation":[{"sender","fragments":[{"t":"image"|"text","value"}]}]}
    images:       loose files at <media_root>/<value>
"""

import json
import os
import random
import struct
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

try:  # torch is only needed to *consume* a manifest; building one is pure file I/O, so the
    # builder stays usable on a login node with no torch installed.
    import torch

    _IterableDatasetBase = torch.utils.data.IterableDataset
except ImportError:  # pragma: no cover
    torch = None
    _IterableDatasetBase = object

__all__ = [
    "DEFAULT_DSS_ROOTS",
    "build_manifest",
    "iter_manifest",
    "resolve_dss",
    "walk_metadataset",
]

# Where materialised ``dss://`` datasets live. Probed in order; the first that exists wins.
# Present on oci-hsg and aws-cmh, absent on gcp-nrt -- a manifest is therefore built per cluster.
DEFAULT_DSS_ROOTS = (
    "/lustre/fsw/portfolios/nemotron/projects/nemotron_omni_vision/dss_cache",
    "/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/dss_cache",
    "/home/svc-dss/cache/nemotron",
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _default_dss_root() -> str | None:
    for root in DEFAULT_DSS_ROOTS:
        if os.path.isdir(root):
            return root
    return None


def resolve_dss(uri: str, dss_root: str) -> str:
    """Resolve ``dss://<name>@<ver>[/suffix]`` to an absolute path.

    The suffix after the version is part of the location, not part of the record's image value:
    ``dss://coco@v0/train2017`` means the images sit under ``<root>/coco/v0/train2017``. Other
    leaves put that subpath inside the record instead (``value = "train/data/x.jpg"``), so joining
    root and value handles both without a per-leaf table.
    """
    body = uri.split("dss://", 1)[-1]
    if "@" not in body:
        return os.path.join(dss_root, body)
    name, rest = body.split("@", 1)
    parts = rest.split("/", 1)
    version, suffix = parts[0], (parts[1] if len(parts) > 1 else "")
    path = os.path.join(dss_root, name, version)
    return os.path.join(path, suffix) if suffix else path


def _iter_blend_entries(node: Any) -> list[dict]:
    """Yield the ``blend*`` entries of a MetadatasetV2 node, whatever the split/blend key is."""
    out: list[dict] = []
    if not isinstance(node, dict):
        return out
    splits = node.get("splits") or {}
    for split in splits.values():
        if not isinstance(split, dict):
            continue
        for key, entries in split.items():
            if not key.startswith("blend") or not isinstance(entries, list):
                continue
            out.extend(e for e in entries if isinstance(e, dict))
    return out


def walk_metadataset(yaml_path: str, _seen: set[str] | None = None) -> list[dict]:
    """Recursively flatten a MetadatasetV2 into leaf descriptors.

    A ``path`` that names another yaml is an inner node and is recursed into; anything else
    (in practice ``dss://...``) is a leaf. Returns dicts with ``ann_uri``, ``media_uri``,
    ``length`` and ``leaf``.
    """
    import yaml as _yaml

    _seen = _seen if _seen is not None else set()
    real = os.path.realpath(yaml_path)
    if real in _seen:  # defensive: blends have been observed to repeat sub-yamls
        return []
    _seen.add(real)

    with open(yaml_path, encoding="utf-8") as fh:
        node = _yaml.safe_load(fh)

    leaves: list[dict] = []
    base = os.path.dirname(os.path.abspath(yaml_path))
    for entry in _iter_blend_entries(node):
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        if path.endswith((".yaml", ".yml")):
            child = path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))
            if os.path.isfile(child):
                leaves.extend(walk_metadataset(child, _seen))
            continue
        sub = entry.get("subflavors") or {}
        aux = entry.get("aux") or {}
        leaves.append(
            {
                "ann_uri": path,
                "media_uri": aux.get("media_source"),
                "length": int(sub.get("length") or 0),
                "leaf": sub.get("name") or path.split("dss://")[-1].split("@")[0],
            }
        )
    return leaves


def _read_records(ann_path: str, limit: int) -> list[dict]:
    """Read up to ``limit`` records from a leaf, whichever container it uses."""
    records: list[dict] = []
    if os.path.isfile(ann_path):  # the URI named the file directly
        with open(ann_path, encoding="utf-8") as fh:
            for line in fh:
                if len(records) >= limit:
                    break
                line = line.strip()
                if line:
                    with _suppress():
                        records.append(json.loads(line))
        return records

    if not os.path.isdir(ann_path):
        return records

    entries = os.listdir(ann_path)
    tars = sorted(f for f in entries if f.endswith(".tar"))
    jsonls = sorted(f for f in entries if f.endswith(".jsonl"))

    if tars:
        for shard in tars:
            if len(records) >= limit:
                break
            with tarfile.open(os.path.join(ann_path, shard)) as tf:
                for member in tf:
                    if len(records) >= limit:
                        break
                    if not member.name.endswith(".json"):
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    with _suppress():
                        records.append(json.loads(fh.read()))
    elif jsonls:
        with open(os.path.join(ann_path, jsonls[0]), encoding="utf-8") as fh:
            for line in fh:
                if len(records) >= limit:
                    break
                line = line.strip()
                if line:
                    with _suppress():
                        records.append(json.loads(line))
    return records


class _suppress:
    """Swallow malformed records: one bad line must not fail a multi-hour PTQ job."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def _probe_dims(path: str) -> tuple[int, int]:
    """Read width/height from a PNG/JPEG header (32 bytes) without decoding the image."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    byte = fh.read(1)
                    if not byte:
                        break
                    if byte != b"\xff":
                        continue
                    marker = fh.read(1)
                    while marker == b"\xff":
                        marker = fh.read(1)
                    if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3"):
                        fh.read(3)
                        height, width = struct.unpack(">HH", fh.read(4))
                        return width, height
                    length = struct.unpack(">H", fh.read(2))[0]
                    fh.seek(length - 2, 1)
    except Exception:
        pass
    return (0, 0)


def _record_to_sample(record: dict, media_root: str) -> dict | None:
    """Flatten ``conversation[].fragments[]`` into ``messages`` + one absolute image path.

    Returns ``None`` when the record has no usable image -- no image fragment at all (~18 % of
    records), or a non-image medium (the blend contains ``.mp4``).
    """
    messages: list[dict] = []
    image_path: str | None = None
    image_wh: tuple[int, int] = (0, 0)

    for turn in record.get("conversation") or []:
        if not isinstance(turn, dict):
            continue
        role = "assistant" if turn.get("sender") == "assistant" else "user"
        texts: list[str] = []
        for frag in turn.get("fragments") or []:
            # Fragment shape is not uniform across the blend: most leaves use
            # {"t": ..., "value": ...}, some store a bare string (implicitly text).
            if isinstance(frag, str):
                texts.append(frag)
                continue
            if not isinstance(frag, dict):
                continue
            kind, value = frag.get("t"), frag.get("value")
            if kind == "text" and isinstance(value, str):
                texts.append(value)
            elif kind == "image" and image_path is None and isinstance(value, str):
                if Path(value).suffix.lower() in _IMG_EXTS:
                    image_path = os.path.join(media_root, value)
                    meta = frag.get("metadata") or {}
                    try:
                        image_wh = (int(meta.get("width") or 0), int(meta.get("height") or 0))
                    except (TypeError, ValueError):
                        image_wh = (0, 0)
        content: list[dict] = []
        if role == "user" and image_path is not None and not messages:
            content.append({"type": "image"})
        if texts:
            content.append({"type": "text", "text": "\n".join(texts)})
        if content:
            messages.append({"role": role, "content": content})

    if image_path is None or not messages:
        return None
    return {"messages": messages, "image": image_path, "wh": list(image_wh)}


def build_manifest(
    metadataset: str,
    out_path: str,
    num_samples: int = 128_000,
    dss_root: str | None = None,
    seed: int = 42,
    verify_images: bool = True,
    max_megapixels: float = 4.0,
) -> dict:
    """Materialise a calibration manifest from a MetadatasetV2 blend.

    Samples are allocated across leaves in proportion to each leaf's declared ``length``, so a
    500 k-sample leaf is not drowned out by a 5 k one. Leaves whose annotation or media dataset is
    missing from the store, or whose media uses packed shards rather than loose files, are skipped
    and counted -- a blend of this size always has some.

    Returns a stats dict, also written as the manifest's first line.
    """
    dss_root = dss_root or _default_dss_root()
    if not dss_root:
        raise FileNotFoundError(f"no DSS store found; looked in {DEFAULT_DSS_ROOTS}")

    leaves = [leaf for leaf in walk_metadataset(metadataset) if leaf.get("media_uri")]
    total_len = sum(max(1, leaf["length"]) for leaf in leaves) or 1
    rng = random.Random(seed)

    stats = Counter()
    per_leaf = Counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("")  # header rewritten at the end
        for leaf in leaves:
            if written >= num_samples:
                break
            quota = max(1, round(num_samples * max(1, leaf["length"]) / total_len))
            quota = min(quota, num_samples - written)

            ann_path = resolve_dss(leaf["ann_uri"], dss_root)
            media_root = resolve_dss(leaf["media_uri"], dss_root)
            if not os.path.exists(ann_path):
                stats["skip_ann_missing"] += 1
                continue
            if not os.path.isdir(media_root):
                stats["skip_media_missing"] += 1
                continue
            # Packed-shard media (pack_manifest.json + shard_*.tar) needs a different image
            # loader; out of scope here. ~10 % of leaves.
            if os.path.exists(os.path.join(media_root, "pack_manifest.json")):
                stats["skip_media_packed"] += 1
                continue

            records = _read_records(ann_path, quota * 3)  # over-read: many records lack an image
            if not records:
                stats["skip_no_records"] += 1
                continue

            kept = 0
            for record in records:
                if kept >= quota or written >= num_samples:
                    break
                sample = _record_to_sample(record, media_root)
                if sample is None:
                    stats["drop_no_image"] += 1
                    continue
                if verify_images and not os.path.exists(sample["image"]):
                    stats["drop_image_unresolved"] += 1
                    continue
                # Cap resolution. The blend contains images up to 45 MP (11808x3824) against a
                # 0.79 MP median; the VL processor tiles by area, so one of those inflates the
                # patch count enough to stall a rank for hours while its peers block on the
                # calibration collective. Prefer the declared size, fall back to a header read.
                if max_megapixels:
                    width, height = sample.get("wh") or (0, 0)
                    if not (width and height):
                        width, height = _probe_dims(sample["image"])
                    if width * height > max_megapixels * 1e6:
                        stats["drop_image_too_large"] += 1
                        continue
                sample["leaf"] = leaf["leaf"]
                out.write(json.dumps(sample) + "\n")
                kept += 1
                written += 1
            per_leaf[leaf["leaf"]] = kept
            stats["leaves_used"] += 1 if kept else 0

    summary = {
        "manifest": os.path.abspath(out_path),
        "metadataset": metadataset,
        "dss_root": dss_root,
        "seed": seed,
        "rows": written,
        "leaves_total": len(leaves),
        **dict(stats),
        "leaves_top10": per_leaf.most_common(10),
    }
    with open(out_path + ".stats.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def iter_manifest(manifest_path: str, num_samples: int, seed: int = 42):
    """Yield ``num_samples`` rows drawn from a manifest, shuffled deterministically.

    Subsampling a fixed manifest keeps calibration sets nested across sizes, so 256 and 1024 are
    comparable rather than independent draws.
    """
    rows: list[dict] = []
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                with _suppress():
                    rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    yield from rows[:num_samples]


class ManifestIterable(_IterableDatasetBase):
    """Stream manifest rows as ``{messages, image}``, loading each image from its absolute path."""

    def __init__(self, manifest_path: str, num_samples: int, seed: int = 42):
        super().__init__()
        self.manifest_path = manifest_path
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        from PIL import Image

        # Yield EXACTLY num_samples. Under expert parallelism every rank must run the same number
        # of forward steps, because each step carries a collective a2a dispatch -- a rank that
        # yields fewer samples exits the calibration loop early and the others block on that
        # collective forever (a silent hang, no traceback, GPUs pinned at 100 % in NCCL spin-wait).
        # Skipping an unreadable image would make the count data-dependent, so instead we draw from
        # a larger pool and refill, and only fall back to repeating an already-good sample if the
        # pool is exhausted.
        pool = list(iter_manifest(self.manifest_path, self.num_samples * 4, self.seed))
        emitted = 0
        good: list[dict] = []
        for row in pool:
            if emitted >= self.num_samples:
                break
            try:
                img = Image.open(row["image"]).convert("RGB")
            except Exception:
                continue
            sample = {"messages": row["messages"], "image": img, "id": row.get("leaf")}
            good.append(row)
            emitted += 1
            yield sample

        # Pool exhausted before the target: repeat known-good rows rather than come up short.
        idx = 0
        while emitted < self.num_samples and good:
            row = good[idx % len(good)]
            idx += 1
            try:
                img = Image.open(row["image"]).convert("RGB")
            except Exception:
                break
            emitted += 1
            yield {"messages": row["messages"], "image": img, "id": row.get("leaf")}

        # Make the count observable: a mismatch across ranks is the difference between a clean run
        # and a silent multi-hour deadlock, and it is otherwise invisible in the logs.
        print(f"[manifest] emitted {emitted}/{self.num_samples} calibration samples", flush=True)
