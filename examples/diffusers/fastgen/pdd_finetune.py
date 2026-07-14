# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Qwen-Image with the ModelOpt-owned PDD lifecycle and released AutoModel APIs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
for path in (_REPO_ROOT, _THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_THIS_DIR / "configs" / "pdd_qwen_image.yaml",
    )
    return parser.parse_args()


def _metadata_sample_ids(metadata: Any, *, split: str) -> tuple[str, ...]:
    if not isinstance(metadata, list):
        raise TypeError(f"PDD {split} dataset metadata must be a list.")
    sample_ids: list[str] = []
    for index, item in enumerate(metadata):
        if not isinstance(item, Mapping) or not isinstance(item.get("sample_id"), str):
            raise ValueError(f"{split} metadata[{index}] has no string sample_id.")
        sample_ids.append(item["sample_id"])
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"PDD {split} metadata contains duplicate logical sample IDs.")
    return tuple(sample_ids)


def _ordered_id_sha256(metadata: Any, *, split: str) -> str:
    sample_ids = _metadata_sample_ids(metadata, split=split)
    digest = hashlib.sha256()
    digest.update(f"modelopt-pdd-ordered-{split}-ids-v1\0".encode())
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
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        exact_resume=True,
        sampler_seed=config.training.seed,
        loader_seed=config.training.seed,
    )
    dataloader, sampler = build_text_to_image_multiresolution_dataloader(**options)
    if not isinstance(sampler, ReplayableBatchSampler):
        raise RuntimeError("PDD training requires the committed replayable batch sampler.")
    if options.get("batch_size", 1) != config.training.local_batch_size:
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
        metadata_index=config.validation_metadata_index,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        drop_last=False,
        shuffle=False,
        dynamic_batch_size=False,
        exact_resume=False,
        sampler_seed=config.training.validation_seed,
        loader_seed=config.training.validation_seed,
    )
    return build_text_to_image_multiresolution_dataloader(**options)


def _validate_dataset_snapshot(raw: Mapping[str, Any], config: Any) -> Mapping[str, Any]:
    import torch.distributed as dist
    from portable_cache import resolve_cache_root
    from validate_cache_snapshot import validate_snapshot

    options = _dataloader_options(raw)
    root = resolve_cache_root(options["cache_dir"])
    roots: list[str] = [""] * dist.get_world_size()
    dist.all_gather_object(roots, str(root))
    if any(candidate != roots[0] for candidate in roots[1:]):
        raise RuntimeError(f"PDD ranks resolved different dataset cache roots: {roots}.")

    status = None
    if dist.get_rank() == 0:
        try:
            report = validate_snapshot(
                root,
                all_index=config.all_metadata_index,
                train_index=config.train_metadata_index,
                heldout_index=config.validation_metadata_index,
            )
            status = {"ok": True, "report": report}
        except BaseException as error:
            status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = [status]
    dist.broadcast_object_list(payload, src=0)
    status = payload[0]
    if not isinstance(status, Mapping) or type(status.get("ok")) is not bool:
        raise RuntimeError("rank 0 broadcast a malformed dataset validation status.")
    if not status["ok"]:
        raise RuntimeError(f"PDD dataset snapshot validation failed: {status.get('error')}.")
    report = status.get("report")
    if not isinstance(report, Mapping):
        raise RuntimeError("PDD dataset snapshot report is malformed.")
    return report


def _build_validation_plan(sampler: Any, config: Any) -> tuple[Any, tuple[tuple[bool, ...], ...]]:
    import torch.distributed as dist
    from pdd_training import build_pdd_validation_assignments

    heldout_ids = _metadata_sample_ids(sampler.dataset.metadata, split="heldout")
    assignments = build_pdd_validation_assignments(
        heldout_ids,
        config.pdd,
        validation_seed=config.training.validation_seed,
    )
    sampler.set_epoch(0)
    sampler.load_state_dict({"epoch": 0, "batches_yielded": 0})
    local_plan = [
        tuple(sampler.dataset.metadata[index]["sample_id"] for index in batch) for batch in sampler
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
    from pdd_training import prepare_qwen_pdd_batch

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
    from pdd_training import prepare_qwen_pdd_batch

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
            sample_ids = tuple(metadata["sample_ids"])
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


def main() -> None:
    args = _parse_args()
    import torch
    import torch.distributed as dist
    from pdd_checkpoint import PDDCheckpointManager, build_pdd_checkpoint_identity
    from pdd_recipe import (
        build_pdd_setup,
        build_pdd_training_artifacts,
        initialize_pdd_distributed,
        resolve_pdd_recipe_config,
    )
    from pdd_training import run_pdd_validation

    raw = yaml.safe_load(args.config.read_text())
    config = resolve_pdd_recipe_config(raw)
    initialize_pdd_distributed(
        backend="nccl" if config.device.type == "cuda" else "gloo",
        timeout_minutes=60,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    snapshot_report = _validate_dataset_snapshot(raw, config)
    dataloader, sampler = _build_training_dataloader(
        raw,
        config,
        dp_rank=rank,
        dp_world_size=world_size,
    )
    validation_dataloader, validation_sampler = _build_validation_dataloader(
        raw,
        config,
        dp_rank=rank,
        dp_world_size=world_size,
    )
    validation_assignments, validation_masks = _build_validation_plan(
        validation_sampler,
        config,
    )
    setup = build_pdd_setup(config)
    transformer_config = getattr(setup.student, "config", None)
    if isinstance(transformer_config, Mapping):
        in_channels = transformer_config.get("in_channels")
    else:
        in_channels = getattr(transformer_config, "in_channels", None)
    if isinstance(transformer_config, Mapping):
        condition_features = transformer_config.get("joint_attention_dim")
    else:
        condition_features = getattr(transformer_config, "joint_attention_dim", None)
    if type(in_channels) is not int or in_channels <= 0 or in_channels % 4:
        raise RuntimeError("constructed Qwen transformer has invalid packed in_channels.")
    if type(condition_features) is not int or condition_features <= 0:
        raise RuntimeError("constructed Qwen transformer has invalid joint_attention_dim.")
    expected_latent_channels = in_channels // 4
    expected_condition_features = condition_features
    training = build_pdd_training_artifacts(setup, config)
    identity = build_pdd_checkpoint_identity(
        metadata=setup.metadata,
        model_id=config.model_id,
        model_revision=config.model_revision,
        guidance_scale=config.pdd.guidance_scale,
        guidance_rescale=config.guidance.rescale,
        guidance_eps=config.guidance.eps,
        automodel_snapshot=setup.automodel_snapshot,
        ordered_train_id_sha256=_ordered_id_sha256(sampler.dataset.metadata, split="train"),
        ordered_heldout_id_sha256=_ordered_id_sha256(
            validation_sampler.dataset.metadata,
            split="heldout",
        ),
        dataset_snapshot_sha256=snapshot_report["snapshot_sha256"],
        local_batch_size=config.training.local_batch_size,
        grad_accumulation_steps=config.training.grad_accumulation_steps,
        training_seed=config.training.seed,
        validation_seed=config.training.validation_seed,
        validation_every_steps=config.training.validation_every_steps,
        max_grad_norm=config.training.max_grad_norm,
        zero_grad_warmup_steps=config.training.zero_grad_warmup_steps,
        activation_checkpointing=config.parallel.activation_checkpointing,
        dtype=str(config.dtype).removeprefix("torch."),
        optimizer=setup.optimizer,
        scheduler=training.scheduler,
    )
    checkpoint_manager = PDDCheckpointManager(
        root=config.checkpoint.checkpoint_dir,
        checkpointer=setup.checkpointer,
        model=setup.student,
        optimizer=setup.optimizer,
        scheduler=training.scheduler,
        trainer=training.trainer,
        sampler=sampler,
        rng=training.rng,
        identity=identity,
    )
    resume = checkpoint_manager.load(config.checkpoint.restore_from)
    resume_pending = resume is not None
    if resume is not None and rank == 0:
        logging.info(
            "PDD resume selected: checkpoint=%s parent=%s step=%d sample_slots=%d "
            "expected_first_sample_ids=%s",
            resume.checkpoint_path,
            resume.parent_checkpoint,
            resume.completed_steps,
            resume.sample_slots_consumed,
            resume.expected_next_sample_ids,
        )
    if rank == 0:
        logging.info(
            "PDD dataset snapshot verified: sha256=%s splits=%s declared_files=%s",
            snapshot_report["snapshot_sha256"],
            snapshot_report["splits"],
            snapshot_report["declared_files"],
        )
        logging.info(
            "PDD setup complete: lifecycle=%s student_keys=%d AutoModel=%s",
            setup.lifecycle,
            len(setup.checkpoint_keys),
            setup.automodel_snapshot["version"],
        )
    last_saved_step = 0 if resume is None else resume.completed_steps
    try:
        while training.trainer.completed_steps < config.training.max_steps:
            iterator = _collective_training_iterator(dataloader, sampler)
            data_wait_started = time.perf_counter()
            while training.trainer.completed_steps < config.training.max_steps:
                next_batch = _collective_training_batch(
                    iterator,
                    sampler=sampler,
                    resume=resume,
                    resume_pending=resume_pending,
                    device=config.device,
                    dtype=config.dtype,
                    require_negative_condition=config.pdd.guidance_scale is not None,
                    expected_batch_size=config.training.local_batch_size,
                    expected_latent_channels=expected_latent_channels,
                    expected_condition_features=expected_condition_features,
                )
                if next_batch is None:
                    break
                data_wait_seconds = time.perf_counter() - data_wait_started
                step_started = time.perf_counter()
                batch, sample_ids = next_batch
                if resume_pending:
                    if rank == 0:
                        logging.info(
                            "PDD resume first batch verified: checkpoint=%s sample_ids=%s",
                            resume.checkpoint_path,
                            sample_ids,
                        )
                    resume_pending = False
                measure_update = (
                    training.trainer.completed_steps + 1
                ) % config.training.log_every_steps == 0
                diagnostics = training.trainer.train_step(
                    batch,
                    measure_updates=measure_update,
                )
                training.scheduler.step()
                sampler.commit(sample_ids)
                if sampler.remaining_batches == 0:
                    sampler.set_epoch(sampler.epoch + 1)
                step_seconds = time.perf_counter() - step_started

                if diagnostics.completed_step % config.training.log_every_steps == 0:
                    timing = torch.tensor(
                        [data_wait_seconds, step_seconds],
                        dtype=torch.float64,
                        device=config.device,
                    )
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                    peak_memory = (
                        torch.cuda.max_memory_allocated(config.device)
                        if config.device.type == "cuda"
                        else 0
                    )
                    memory = torch.tensor(peak_memory, dtype=torch.int64, device=config.device)
                    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                    global_samples = config.training.local_batch_size * dist.get_world_size()
                    throughput = global_samples / max(float(timing[1].item()), 1e-12)
                    coverage = training.trainer.coverage
                    bin_loss = [
                        None if count == 0 else float(loss_sum / count)
                        for loss_sum, count in zip(
                            coverage.bin_loss_sums.tolist(),
                            coverage.bin_counts.tolist(),
                        )
                    ]
                    if rank == 0:
                        logging.info(
                            "PDD step=%d loss=%.6g grad_norm=%.6g nominal_update_ratio=%.6g "
                            "projection_update_ratio=%s lr=%.6g student_rms=%.6g "
                            "teacher_rms=%.6g student_teacher_rms_ratio=%.6g "
                            "reconstruction_rms=%.6g pairs=%d n_coverage=%s k_coverage=%s "
                            "bins=%s bin_loss=%s samples_per_second=%.3f "
                            "data_wait_seconds=%.4f peak_memory_bytes=%d",
                            diagnostics.completed_step,
                            diagnostics.loss,
                            diagnostics.grad_norm,
                            diagnostics.student_adamw_nominal_update_ratio,
                            diagnostics.pdd_projection_update_ratio,
                            diagnostics.learning_rate,
                            diagnostics.student_velocity_rms,
                            diagnostics.teacher_velocity_rms,
                            diagnostics.student_teacher_velocity_rms_ratio,
                            diagnostics.reconstructed_state_rms,
                            int((coverage.pair_counts > 0).sum()),
                            _coverage_axis(coverage.n_counts, coverage.n_loss_sums),
                            _coverage_axis(coverage.k_counts, coverage.k_loss_sums),
                            coverage.bin_counts.tolist(),
                            bin_loss,
                            throughput,
                            float(timing[0].item()),
                            int(memory.item()),
                        )
                    if config.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(config.device)
                if (
                    diagnostics.completed_step % config.training.validation_every_steps == 0
                    or diagnostics.completed_step >= config.training.max_steps
                ):
                    validation_sampler.set_epoch(0)
                    validation_sampler.load_state_dict({"epoch": 0, "batches_yielded": 0})
                    validation_result = run_pdd_validation(
                        training.pipeline,
                        _iter_validation_batches(
                            validation_dataloader,
                            validation_masks,
                            config,
                            expected_latent_channels,
                            expected_condition_features,
                        ),
                        validation_assignments,
                        validation_seed=config.training.validation_seed,
                    )
                    if rank == 0:
                        logging.info(
                            "PDD validation step=%d loss=%.12g pairs=%d starts=%d heads=%d "
                            "ordered_id_sha256=%s records=%d",
                            diagnostics.completed_step,
                            validation_result.mean_loss,
                            validation_result.pair_count,
                            validation_result.start_count,
                            validation_result.head_count,
                            validation_result.ordered_id_sha256,
                            len(validation_result.records),
                        )
                if (
                    config.checkpoint.enabled
                    and diagnostics.completed_step % config.training.checkpoint_every_steps == 0
                ):
                    checkpoint_manager.save()
                    last_saved_step = diagnostics.completed_step
                if diagnostics.completed_step >= config.training.max_steps:
                    break
                data_wait_started = time.perf_counter()

        if config.checkpoint.enabled and last_saved_step != training.trainer.completed_steps:
            checkpoint_manager.save()
    finally:
        setup.checkpointer.close()


if __name__ == "__main__":
    main()
