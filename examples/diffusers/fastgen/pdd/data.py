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

"""Authenticated data and collective batch handling for the Qwen-Image PDD recipe."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _ordered_id_sha256(sample_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256(b"modelopt-pdd-ordered-sample-ids-v1\0")
    for sample_id in sample_ids:
        digest.update(sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _dataloader_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = raw.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("dataloader"), Mapping):
        raise TypeError("PDD config requires a data.dataloader mapping.")
    options = dict(data["dataloader"])
    target = options.pop("_target_", None)
    expected_target = "fastgen_data.build_text_to_image_multiresolution_dataloader"
    if target != expected_target:
        raise ValueError(f"PDD data.dataloader._target_ must be {expected_target!r}.")
    if "base_resolution" in options:
        options["base_resolution"] = tuple(options["base_resolution"])
    return options


def _build_training_dataloader(
    raw: Mapping[str, Any],
    config: Any,
    *,
    dp_rank: int,
    dp_world_size: int,
) -> tuple[Any, Any]:
    from fastgen_data import ReplayableBatchSampler
    from fastgen_data.collate_fns import build_text_to_image_multiresolution_dataloader

    options = _dataloader_options(raw)
    if options.get("drop_last", True) is not True:
        raise ValueError("PDD exact sample accounting requires data.dataloader.drop_last=true.")
    if options.get("dynamic_batch_size", False) is not False:
        raise ValueError("PDD v1 requires data.dataloader.dynamic_batch_size=false.")
    options.update(
        split="train",
        validation_count=config.validation.count,
        split_seed=config.validation.split_seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        exact_resume=True,
        sampler_seed=config.seed,
        loader_seed=config.seed,
    )
    dataloader, sampler = build_text_to_image_multiresolution_dataloader(**options)
    if not isinstance(sampler, ReplayableBatchSampler):
        raise RuntimeError("PDD training requires the committed replayable batch sampler.")
    if options.get("batch_size", 1) != config.step_scheduler.local_batch_size:
        raise RuntimeError("resolved local batch size does not match the built dataloader.")
    return dataloader, sampler


def _build_validation_dataloader(
    raw: Mapping[str, Any],
    config: Any,
    *,
    dp_rank: int,
    dp_world_size: int,
) -> tuple[Any, Any]:
    from fastgen_data.collate_fns import build_text_to_image_multiresolution_dataloader

    options = _dataloader_options(raw)
    options.update(
        split="validation",
        validation_count=config.validation.count,
        split_seed=config.validation.split_seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        drop_last=False,
        shuffle=False,
        dynamic_batch_size=False,
        exact_resume=False,
        sampler_seed=config.validation.seed,
        loader_seed=config.validation.seed,
    )
    return build_text_to_image_multiresolution_dataloader(**options)


def _validate_dataset_contract(
    train_dataset: Any,
    validation_dataset: Any,
    config: Any,
) -> tuple[Mapping[str, Any], str, str]:
    """Collectively verify deterministic splits and the authenticated dataset snapshot."""
    import torch.distributed as dist

    try:
        train_ids = tuple(str(value) for value in train_dataset.sample_ids)
        validation_ids = tuple(str(value) for value in validation_dataset.sample_ids)
        if len(validation_ids) != config.validation.count:
            raise RuntimeError(
                f"validation split has {len(validation_ids)} samples; "
                f"expected {config.validation.count}."
            )
        if set(train_ids).intersection(validation_ids):
            raise RuntimeError("training and validation splits overlap.")
        expected = {str(index) for index in range(train_dataset.total_num_samples)}
        if set(train_ids).union(validation_ids) != expected:
            raise RuntimeError("training and validation splits do not cover metadata.json.")
        if train_dataset.total_num_samples != validation_dataset.total_num_samples:
            raise RuntimeError("training and validation datasets disagree on total sample count.")
        if train_dataset.metadata_sha256 != validation_dataset.metadata_sha256:
            raise RuntimeError("training and validation datasets disagree on metadata content.")
        if (
            not train_dataset.payload_hashes_complete
            or not validation_dataset.payload_hashes_complete
        ):
            raise RuntimeError("PDD exact resume requires a cache_sha256 for every tensor payload.")
        if train_dataset.dataset_snapshot_sha256 != validation_dataset.dataset_snapshot_sha256:
            raise RuntimeError("training and validation datasets disagree on dataset content.")
        if not isinstance(train_dataset.dataset_snapshot_sha256, str):
            raise RuntimeError("PDD dataloader did not construct a dataset snapshot identity.")
        report = {
            "cache_root": str(train_dataset.cache_root),
            "metadata_sha256": train_dataset.metadata_sha256,
            "negative_prompt_embedding_sha256": (train_dataset.negative_prompt_embedding_sha256),
            "dataset_snapshot_sha256": train_dataset.dataset_snapshot_sha256,
            "total_samples": train_dataset.total_num_samples,
            "train_samples": len(train_ids),
            "validation_samples": len(validation_ids),
            "split_seed": config.validation.split_seed,
        }
        local_status: dict[str, Any] = {
            "ok": True,
            "report": report,
            "train_hash": _ordered_id_sha256(train_ids),
            "validation_hash": _ordered_id_sha256(validation_ids),
        }
    except BaseException as error:
        local_status = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    statuses: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, local_status)
    failures: list[str] = []
    successes: list[Mapping[str, Any]] = []
    for rank, status in enumerate(statuses):
        if not isinstance(status, Mapping) or type(status.get("ok")) is not bool:
            failures.append(f"rank {rank}: malformed loader authentication status")
            continue
        if not status["ok"]:
            failures.append(f"rank {rank}: {status.get('error')}")
            continue
        successes.append(status)
    if failures:
        raise RuntimeError("PDD dataset validation failed: " + "; ".join(failures))
    canonical = {json.dumps(status, sort_keys=True) for status in successes}
    if len(canonical) != 1:
        raise RuntimeError(
            "PDD ranks resolved different dataset roots, metadata, or split membership."
        )
    return report, local_status["train_hash"], local_status["validation_hash"]


def _build_validation_plan(sampler: Any, config: Any) -> tuple[Any, tuple[tuple[bool, ...], ...]]:
    import torch.distributed as dist

    from pdd.training import build_pdd_validation_assignments

    heldout_ids = tuple(str(value) for value in sampler.dataset.sample_ids)
    assignments = build_pdd_validation_assignments(
        heldout_ids,
        config.pdd,
        validation_seed=config.validation.seed,
    )
    sampler.set_epoch(0)
    sampler.load_state_dict({"epoch": 0, "batches_yielded": 0})
    local_plan = [
        tuple(str(sampler.dataset.sample_ids[index]) for index in batch) for batch in sampler
    ]
    sampler.load_state_dict({"epoch": 0, "batches_yielded": 0})
    plans: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(plans, local_plan)
    batch_counts = {len(plan) for plan in plans}
    if len(batch_counts) != 1:
        raise RuntimeError("PDD validation sampler produced different batch counts across ranks.")

    masks = [[([False] * len(batch)) for batch in plan] for plan in plans]
    seen: set[str] = set()
    for batch_index in range(len(local_plan)):
        for rank, plan in enumerate(plans):
            for position, sample_id in enumerate(plan[batch_index]):
                if sample_id not in seen:
                    masks[rank][batch_index][position] = True
                    seen.add(sample_id)
    if seen != set(heldout_ids):
        missing = sorted(set(heldout_ids) - seen)
        extra = sorted(seen - set(heldout_ids))
        raise RuntimeError(
            f"PDD validation sampler does not cover the held-out split: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    local_masks = tuple(tuple(batch) for batch in masks[dist.get_rank()])
    return assignments, local_masks


def _iter_validation_batches(
    dataloader: Any,
    masks: tuple[tuple[bool, ...], ...],
    config: Any,
    expected_latent_channels: int,
    expected_condition_features: int,
):
    from pdd.training import prepare_qwen_pdd_batch

    count = 0
    for count, (raw_batch, valid_mask) in enumerate(zip(dataloader, masks, strict=True), start=1):
        prepared = prepare_qwen_pdd_batch(
            raw_batch,
            device=config.device,
            dtype=config.dtype,
            require_negative_condition=config.pdd.guidance_scale is not None,
            expected_latent_channels=expected_latent_channels,
            expected_condition_features=expected_condition_features,
        )
        yield dataclasses.replace(prepared, valid_mask=valid_mask)
    if count != len(masks):
        raise RuntimeError(
            f"PDD validation loader produced {count} batches for a {len(masks)}-batch plan."
        )


def _coverage_axis(counts: Any, loss_sums: Any) -> dict[int, dict[str, float | int]]:
    return {
        index: {"count": int(count), "mean_loss": float(loss_sums[index] / count)}
        for index, count in enumerate(counts.tolist())
        if count
    }


def _collective_training_iterator(dataloader: Any, sampler: Any) -> Any:
    """Advance epochs and construct rank-local iterators under a collective error gate."""
    import torch.distributed as dist

    iterator = None
    error_message = None
    try:
        if sampler.remaining_batches == 0:
            sampler.set_epoch(sampler.epoch + 1)
        iterator = iter(dataloader)
        if iterator is None:
            raise RuntimeError("PDD dataloader returned no iterator.")
    except BaseException as error:
        error_message = f"{type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error_message)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError(
            "distributed PDD training iterator construction failed; " + "; ".join(failures)
        )
    if iterator is None:
        raise RuntimeError("local PDD iterator construction succeeded without an iterator.")
    return iterator


def _collective_training_batch(
    iterator: Any,
    *,
    sampler: Any,
    resume: Any,
    resume_pending: bool,
    device: Any,
    dtype: Any,
    require_negative_condition: bool,
    expected_batch_size: int,
    expected_latent_channels: int,
    expected_condition_features: int,
) -> tuple[Any, tuple[str, ...]] | None:
    """Prepare one rank-local batch, then agree on success before any model call."""
    import torch.distributed as dist

    from pdd.training import prepare_qwen_pdd_batch

    prepared = None
    sample_ids: tuple[str, ...] = ()
    status: dict[str, Any]
    try:
        raw_batch = next(iterator)
    except StopIteration:
        if resume_pending:
            status = {
                "state": "error",
                "error": "RuntimeError: resumed dataloader ended before its first batch",
                "resume_pending": True,
            }
        else:
            status = {"state": "end", "resume_pending": False}
    except BaseException as error:
        status = {
            "state": "error",
            "error": f"{type(error).__name__}: {error}",
            "resume_pending": resume_pending,
        }
    else:
        try:
            metadata = raw_batch["metadata"]
            raw_ids = metadata.get("logical_sample_ids", metadata.get("sample_ids"))
            if hasattr(raw_ids, "tolist"):
                raw_ids = raw_ids.tolist()
            sample_ids = tuple(str(value) for value in raw_ids)
            expected_ids = sampler.expected_next_sample_ids()
            if sample_ids != expected_ids:
                raise RuntimeError(
                    "prefetched PDD batch does not match committed cursor: "
                    f"expected={expected_ids}, actual={sample_ids}."
                )
            if resume_pending:
                if resume is None:
                    raise RuntimeError("resume_pending is true without a PDD resume state.")
                resume.verify_first_batch(sample_ids)
            prepared = prepare_qwen_pdd_batch(
                raw_batch,
                device=device,
                dtype=dtype,
                require_negative_condition=require_negative_condition,
                expected_latent_channels=expected_latent_channels,
                expected_condition_features=expected_condition_features,
            )
            if prepared is None:
                raise RuntimeError("PDD batch preparation returned no prepared batch.")
            if len(sample_ids) != expected_batch_size:
                raise RuntimeError(
                    f"PDD training batch has {len(sample_ids)} samples; "
                    f"expected {expected_batch_size}."
                )
            status = {
                "state": "batch",
                "batch_size": len(sample_ids),
                "resume_pending": resume_pending,
            }
        except BaseException as error:
            status = {
                "state": "error",
                "error": f"{type(error).__name__}: {error}",
                "resume_pending": resume_pending,
            }

    statuses: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, status)
    malformed = [rank for rank, item in enumerate(statuses) if not isinstance(item, Mapping)]
    if malformed:
        raise RuntimeError(f"PDD training ranks returned malformed statuses: {malformed}.")
    failures = [
        f"rank {rank}: {item.get('error')}"
        for rank, item in enumerate(statuses)
        if item.get("state") == "error"
    ]
    if failures:
        raise RuntimeError(
            "distributed PDD training batch preflight failed; " + "; ".join(failures)
        )
    states = {item.get("state") for item in statuses}
    if states == {"end"}:
        return None
    if states != {"batch"}:
        raise RuntimeError(
            "distributed PDD training ranks produced different dataloader lengths: "
            f"{[item.get('state') for item in statuses]}."
        )
    pending = {item.get("resume_pending") for item in statuses}
    if len(pending) != 1:
        raise RuntimeError("distributed PDD training ranks disagree on resume verification state.")
    batch_sizes = {item.get("batch_size") for item in statuses}
    if batch_sizes != {expected_batch_size}:
        raise RuntimeError(
            f"distributed PDD training ranks disagree on batch size: {sorted(batch_sizes)}."
        )
    if prepared is None:
        raise RuntimeError("local PDD batch preparation succeeded without a prepared batch.")
    return prepared, sample_ids
