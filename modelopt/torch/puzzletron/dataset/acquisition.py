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

"""Bounded, reproducible acquisition of first-class Puzzletron datasets."""

from __future__ import annotations

import itertools
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

PUZZLE_KD_DATASET = "nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2"
NEMOTRON_VLM_DATASET = "nvidia/Nemotron-VLM-Dataset-v2"
VLM_HEADER_SUBSETS = ("sparsetables", "plotqa_cot", "wiki_en")
ACQUISITION_MANIFEST = "puzzletron_acquisition.json"

__all__ = [
    "ACQUISITION_MANIFEST",
    "NEMOTRON_VLM_DATASET",
    "PUZZLE_KD_DATASET",
    "VLM_HEADER_SUBSETS",
    "TextAcquisitionSpec",
    "VlmAcquisitionSpec",
    "largest_remainder_quotas",
    "materialize_nemotron_vlm_dataset",
    "materialize_puzzle_kd_dataset",
]


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def largest_remainder_quotas(
    row_counts: Mapping[str, int],
    total: int,
) -> dict[str, int]:
    """Apportion an exact sample total in mapping order by source row count."""

    total = _positive("total", total)
    entries = []
    for index, (raw_name, raw_rows) in enumerate(row_counts.items()):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("row count names must be non-empty")
        rows = _positive(f"row_counts[{name!r}]", raw_rows)
        entries.append((name, rows, index))
    if not entries:
        raise ValueError("row_counts must contain at least one subset")
    names = [name for name, _, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("row count names must be unique")
    source_total = sum(rows for _, rows, _ in entries)
    exact = [
        (name, Fraction(total * rows, source_total), index)
        for name, rows, index in entries
    ]
    quotas = {name: int(value) for name, value, _ in exact}
    remaining = total - sum(quotas.values())
    ranked = sorted(
        exact,
        key=lambda item: (-(item[1] - int(item[1])), item[2]),
    )
    for name, _, _ in ranked[:remaining]:
        quotas[name] += 1
    return quotas


@dataclass(frozen=True, kw_only=True)
class TextAcquisitionSpec:
    """A bounded local snapshot of Puzzle-KD's two canonical splits."""

    output_dir: Path
    train_samples: int = 8192
    validation_samples: int = 1024
    seed: int = 408
    revision: str | None = None
    source: str = PUZZLE_KD_DATASET

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "train_samples", _positive("train_samples", self.train_samples))
        object.__setattr__(
            self,
            "validation_samples",
            _positive("validation_samples", self.validation_samples),
        )

    def identity(self, *, revision: str) -> dict[str, Any]:
        value = asdict(self)
        value.pop("output_dir")
        value["adapter"] = "puzzle_kd_v2"
        value["revision"] = revision
        return value


@dataclass(frozen=True, kw_only=True)
class VlmAcquisitionSpec:
    """A bounded local image-conversation snapshot of Nemotron-VLM v2."""

    output_dir: Path
    subsets: tuple[str, ...] = VLM_HEADER_SUBSETS
    subset_rows: tuple[tuple[str, int], ...] = ()
    num_samples: int = 512
    seed: int = 42
    max_shards_per_subset: int = 1
    revision: str | None = None
    source: str = NEMOTRON_VLM_DATASET

    def __post_init__(self) -> None:
        subsets = tuple(str(item).strip() for item in self.subsets if str(item).strip())
        if not subsets:
            raise ValueError("subsets must contain at least one name")
        if len(subsets) != len(set(subsets)):
            raise ValueError("subsets must be unique")
        subset_rows = tuple(
            (str(name).strip(), _positive(f"subset_rows[{name!r}]", rows))
            for name, rows in self.subset_rows
        )
        if not subset_rows:
            subset_rows = tuple((name, 1) for name in subsets)
        row_names = tuple(name for name, _ in subset_rows)
        if row_names != subsets:
            raise ValueError("subset_rows names and order must exactly match subsets")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "subsets", subsets)
        object.__setattr__(self, "subset_rows", subset_rows)
        object.__setattr__(self, "num_samples", _positive("num_samples", self.num_samples))
        object.__setattr__(
            self,
            "max_shards_per_subset",
            _positive("max_shards_per_subset", self.max_shards_per_subset),
        )

    def identity(self, *, revision: str) -> dict[str, Any]:
        value = asdict(self)
        value.pop("output_dir")
        value["adapter"] = "nemotron_vlm_v2"
        value["revision"] = revision
        value["subsets"] = list(self.subsets)
        value["subset_rows"] = dict(self.subset_rows)
        return value


def _resolve_revision(source: str, requested: str | None) -> str:
    if requested and requested not in {"main", "latest"}:
        return requested
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(source, revision=requested)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return an immutable revision for {source}")
    return str(info.sha)


def _load_existing_manifest(output_dir: Path) -> dict[str, Any] | None:
    candidates = (
        output_dir / ACQUISITION_MANIFEST,
        output_dir / "manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    return None


def _reuse_or_reject(output_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    existing = _load_existing_manifest(output_dir)
    if existing is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(
                f"dataset destination exists without an acquisition manifest: {output_dir}"
            )
        return None
    if existing.get("acquisition") != dict(identity):
        raise ValueError(
            f"existing materialization at {output_dir} does not match requested acquisition"
        )
    return existing


def _resolve_or_reuse_revision(
    *,
    output_dir: Path,
    source: str,
    requested: str | None,
    resolver: Callable[[str, str | None], str],
) -> str:
    """Reuse a cached immutable revision without requiring network access."""

    existing = _load_existing_manifest(output_dir)
    acquisition = existing.get("acquisition", {}) if existing is not None else {}
    if requested is None and acquisition.get("source") == source and acquisition.get("revision"):
        return str(acquisition["revision"])
    return resolver(source, requested)


def _bounded_rows(rows: Iterable[Mapping[str, Any]], count: int, *, seed: int) -> list[dict]:
    if hasattr(rows, "shuffle"):
        try:
            rows = rows.shuffle(seed=seed)
        except TypeError:
            rows = rows.shuffle(seed)
    elif isinstance(rows, (list, tuple)):
        rows = list(rows)
        random.Random(seed).shuffle(rows)
    selected = [dict(row) for row in itertools.islice(iter(rows), count)]
    if len(selected) != count:
        raise RuntimeError(f"only found {len(selected)}/{count} rows")
    return selected


def materialize_puzzle_kd_dataset(
    spec: TextAcquisitionSpec,
    *,
    dataset_loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    revision_resolver: Callable[[str, str | None], str] = _resolve_revision,
) -> dict[str, Any]:
    """Materialize bounded Puzzle-KD train/validation splits for offline tokenization."""

    revision = _resolve_or_reuse_revision(
        output_dir=spec.output_dir,
        source=spec.source,
        requested=spec.revision,
        resolver=revision_resolver,
    )
    identity = spec.identity(revision=revision)
    reused = _reuse_or_reject(spec.output_dir, identity)
    if reused is not None:
        return reused
    if spec.output_dir.exists():
        # `_reuse_or_reject` established that this is an empty destination.
        spec.output_dir.rmdir()
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    from datasets import Dataset, DatasetDict

    selected = {}
    for offset, (split, count) in enumerate(
        (("train", spec.train_samples), ("validation", spec.validation_samples))
    ):
        rows = dataset_loader(
            spec.source,
            split=split,
            revision=revision,
            streaming=True,
        )
        selected[split] = Dataset.from_list(_bounded_rows(rows, count, seed=spec.seed + offset))

    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{spec.output_dir.name}-", dir=str(spec.output_dir.parent))
    )
    try:
        DatasetDict(selected).save_to_disk(temporary)
        manifest = {
            "version": 1,
            "acquisition": identity,
            "splits": {name: len(dataset) for name, dataset in selected.items()},
        }
        (temporary / ACQUISITION_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, spec.output_dir)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _default_vlm_sample_loader(
    *,
    subset: str,
    num_samples: int,
    seed: int,
    max_shards: int,
    revision: str,
):
    from ...utils.nemotron_vlm_dataset_utils import (
        NemotronTarPlusJsonlIterable,
        list_repo_files_cached,
    )

    prefix = f"{subset}/media/"
    shards = [
        path
        for path in list_repo_files_cached(
            NEMOTRON_VLM_DATASET,
            repo_type="dataset",
            revision=revision,
        )
        if path.startswith(prefix) and path.lower().endswith(".tar")
    ]
    if not shards:
        raise ValueError(f"Nemotron-VLM subset {subset!r} has no in-repository media tar shards")
    return NemotronTarPlusJsonlIterable(
        repo_id=NEMOTRON_VLM_DATASET,
        subsets=[subset],
        shard_paths=shards,
        num_samples=num_samples,
        seed=seed,
        shuffle_buffer_size=10_000,
        max_shards=max_shards,
        revision=revision,
    )


def materialize_nemotron_vlm_dataset(
    spec: VlmAcquisitionSpec,
    *,
    sample_loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    revision_resolver: Callable[[str, str | None], str] = _resolve_revision,
) -> dict[str, Any]:
    """Materialize a bounded, row-proportional Nemotron image-conversation subset."""

    from .multimodal import (
        materialize_normalized_conversation_samples,
        normalize_nemotron_vlm_sample,
    )

    revision = _resolve_or_reuse_revision(
        output_dir=spec.output_dir,
        source=spec.source,
        requested=spec.revision,
        resolver=revision_resolver,
    )
    identity = spec.identity(revision=revision)
    reused = _reuse_or_reject(spec.output_dir, identity)
    if reused is not None:
        return reused
    sample_loader = sample_loader or _default_vlm_sample_loader
    iterators = [
        iter(
            sample_loader(
                subset=subset,
                num_samples=spec.num_samples,
                seed=spec.seed + index,
                max_shards=spec.max_shards_per_subset,
                revision=revision,
            )
        )
        for index, subset in enumerate(spec.subsets)
    ]
    row_counts = dict(spec.subset_rows)
    requested_quotas = largest_remainder_quotas(row_counts, spec.num_samples)
    target_quotas = dict(requested_quotas)
    materialized_rows = dict.fromkeys(spec.subsets, 0)
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    active = list(range(len(iterators)))
    redistributed_rows = 0
    while active and len(samples) < spec.num_samples:
        made_progress = False
        for index in tuple(active):
            if len(samples) == spec.num_samples:
                break
            subset = spec.subsets[index]
            if materialized_rows[subset] >= target_quotas[subset]:
                continue
            try:
                row = next(iterators[index])
            except StopIteration:
                active.remove(index)
                deficit = target_quotas[subset] - materialized_rows[subset]
                target_quotas[subset] = materialized_rows[subset]
                remaining = {
                    spec.subsets[other]: max(
                        row_counts[spec.subsets[other]]
                        - materialized_rows[spec.subsets[other]],
                        1,
                    )
                    for other in active
                }
                if deficit > 0 and remaining:
                    additions = largest_remainder_quotas(remaining, deficit)
                    for name, amount in additions.items():
                        target_quotas[name] += amount
                    redistributed_rows += deficit
                continue
            made_progress = True
            try:
                samples.append(
                    normalize_nemotron_vlm_sample(
                        row,
                        subset=subset,
                        revision=revision,
                    )
                )
                materialized_rows[subset] += 1
            except (TypeError, ValueError) as error:
                failures.append(
                    {
                        "subset": subset,
                        "row_id": str(row.get("id", "")),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if not made_progress and active:
            raise RuntimeError(
                "proportional Nemotron-VLM sampling made no progress; "
                f"targets={target_quotas}, materialized={materialized_rows}"
            )
    if len(samples) != spec.num_samples:
        raise RuntimeError(
            f"only found {len(samples)}/{spec.num_samples} valid Nemotron-VLM rows; "
            f"first failures={failures[:3]}"
        )
    return materialize_normalized_conversation_samples(
        samples,
        spec.output_dir,
        acquisition=identity,
        diagnostics={
            "rejected_rows": failures,
            "requested_quotas": requested_quotas,
            "materialized_rows": materialized_rows,
            "redistributed_rows": redistributed_rows,
        },
    )
