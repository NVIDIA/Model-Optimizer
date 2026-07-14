# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict evidence-bundle validation and deterministic PDD effectiveness summaries."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pdd_artifacts import (
    canonical_json_bytes,
    load_canonical_json,
    require_sha256,
    sha256_file,
    validate_artifact_reference,
)
from pdd_export import inspect_pdd_export

from modelopt.torch.fastgen import make_shifted_flow_grid

EVALUATION_CONDITIONS = (
    "teacher_guided",
    "undistilled_euler_4",
    "undistilled_2step_4eval",
    "pdd_2",
    "pdd_4",
    "pdd_8",
)


def _grid_protocol(grid_size: int) -> Mapping[str, Any]:
    nodes = [float(value) for value in make_shifted_flow_grid(grid_size, 5.0).tolist()]
    return {
        "builder": "modelopt.torch.fastgen.make_shifted_flow_grid",
        "formula": "s*u/(1+(s-1)*u), u=1-i/grid_size, i=0..grid_size",
        "grid_size": grid_size,
        "flow_shift": 5.0,
        "nodes": nodes,
        "nodes_sha256": hashlib.sha256(canonical_json_bytes(nodes)).hexdigest(),
    }


GRID_PROTOCOLS: Mapping[str, Mapping[str, Any]] = {
    "pdd_grid_128_shift5": _grid_protocol(128),
    "teacher_grid_50_shift5": _grid_protocol(50),
}

_GUIDED_CFG_PROTOCOL = {
    "execution": "sequential_conditional_unconditional",
    "guidance_scale": 4.0,
    "rescale": 1.0,
    "eps": 1e-5,
    "negative_condition": "manifest_negative_condition",
}

_DISABLED_CFG_PROTOCOL = {
    "execution": "disabled",
    "guidance_scale": None,
    "rescale": None,
    "eps": None,
    "negative_condition": None,
}

INTEGRATOR_PROTOCOLS: Mapping[str, Mapping[str, Any]] = {
    "euler_explicit": {
        "math_dtype": "float32",
        "velocity_evaluations_per_interval": 1,
        "equations": [
            "dt=t_next-t_current",
            "v_current=velocity(x_current,t_current)",
            "x_next=x_current+dt*v_current",
        ],
        "terminal_rule": "apply the same Euler update when t_next=0; no special fallback",
    },
    "heun_explicit_trapezoid": {
        "math_dtype": "float32",
        "velocity_evaluations_per_interval": 2,
        "equations": [
            "dt=t_next-t_current",
            "v_current=velocity(x_current,t_current)",
            "x_predict=x_current+dt*v_current",
            "v_predict=velocity(x_predict,t_next)",
            "x_next=x_current+0.5*dt*(v_current+v_predict)",
        ],
        "terminal_rule": (
            "always evaluate v_predict at t_next, including t_next=0; no Euler fallback"
        ),
    },
    "pdd_fused_euler": {
        "math_dtype": "float32",
        "velocity_evaluations_per_interval": 1,
        "implementation": "modelopt.torch.fastgen.methods.pdd.PDDPipeline.sample",
        "source_identity": "manifest.modelopt.commit",
        "equations": [
            "v_fused=student_fused_block(x_start,t_start,start,end,authenticated_grid)",
            "x_end=x_start+(t_end-t_start)*v_fused",
        ],
        "terminal_rule": "apply the same fused update when t_end=0; no special fallback",
    },
}

CONDITION_PROTOCOLS: Mapping[str, Mapping[str, Any]] = {
    "teacher_guided": {
        "artifact": "pdd_export",
        "model_role": "frozen_teacher",
        "integrator": "euler_explicit",
        "cfg": _GUIDED_CFG_PROTOCOL,
        "grid": {
            "protocol": "teacher_grid_50_shift5",
            "node_indices": list(range(51)),
        },
        "pdd_blocks": [],
        "scheduler_steps": 50,
        "actual_transformer_invocations": 100,
        "batch_normalized_transformer_evaluations": 100,
    },
    "undistilled_euler_4": {
        "artifact": "pdd_export",
        "model_role": "pinned_base_model",
        "integrator": "euler_explicit",
        "cfg": _GUIDED_CFG_PROTOCOL,
        "grid": {
            "protocol": "pdd_grid_128_shift5",
            "node_indices": [0, 32, 64, 96, 128],
        },
        "pdd_blocks": [],
        "scheduler_steps": 4,
        "actual_transformer_invocations": 8,
        "batch_normalized_transformer_evaluations": 8,
    },
    "undistilled_2step_4eval": {
        "artifact": "pdd_export",
        "model_role": "pinned_base_model",
        "integrator": "heun_explicit_trapezoid",
        "cfg": _GUIDED_CFG_PROTOCOL,
        "grid": {
            "protocol": "pdd_grid_128_shift5",
            "node_indices": [0, 64, 128],
        },
        "pdd_blocks": [],
        "scheduler_steps": 2,
        "actual_transformer_invocations": 8,
        "batch_normalized_transformer_evaluations": 8,
    },
    "pdd_2": {
        "artifact": "pdd_export",
        "model_role": "pdd_student",
        "integrator": "pdd_fused_euler",
        "cfg": _DISABLED_CFG_PROTOCOL,
        "grid": {
            "protocol": "pdd_grid_128_shift5",
            "node_indices": [0, 64, 128],
        },
        "pdd_blocks": [64, 64],
        "scheduler_steps": 2,
        "actual_transformer_invocations": 2,
        "batch_normalized_transformer_evaluations": 2,
    },
    "pdd_4": {
        "artifact": "pdd_export",
        "model_role": "pdd_student",
        "integrator": "pdd_fused_euler",
        "cfg": _DISABLED_CFG_PROTOCOL,
        "grid": {
            "protocol": "pdd_grid_128_shift5",
            "node_indices": [0, 32, 64, 96, 128],
        },
        "pdd_blocks": [32, 32, 32, 32],
        "scheduler_steps": 4,
        "actual_transformer_invocations": 4,
        "batch_normalized_transformer_evaluations": 4,
    },
    "pdd_8": {
        "artifact": "pdd_export",
        "model_role": "pdd_student",
        "integrator": "pdd_fused_euler",
        "cfg": _DISABLED_CFG_PROTOCOL,
        "grid": {
            "protocol": "pdd_grid_128_shift5",
            "node_indices": [0, 16, 32, 48, 64, 80, 96, 112, 128],
        },
        "pdd_blocks": [16, 16, 16, 16, 16, 16, 16, 16],
        "scheduler_steps": 8,
        "actual_transformer_invocations": 8,
        "batch_normalized_transformer_evaluations": 8,
    },
}

_PROTOCOL_FIELDS = (
    "conditions",
    "condition_protocols",
    "grid_protocols",
    "integrator_protocols",
    "image_protocol",
    "metric_protocols",
    "timing_protocol",
    "decision_rule",
    "negative_condition",
    "data_snapshot",
    "stage_run_ids",
    "prompt_set",
    "bootstrap",
)


def _exact_mapping(value: Any, keys: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} keys mismatch: expected={sorted(keys)}, actual={actual}.")
    return value


def _commit(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{name} must be a 40-character Git commit.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error
    return value.lower()


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be finite and > 0.")
    return resolved


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _load_prompt_set(path: Path) -> tuple[dict[tuple[str, str, int], str], Mapping[str, Any]]:
    prompt_set = _exact_mapping(
        load_canonical_json(path),
        {"schema_version", "prompts"},
        name="prompt set",
    )
    if prompt_set["schema_version"] != 1 or not isinstance(prompt_set["prompts"], list):
        raise ValueError("prompt set schema is unsupported.")
    expected: dict[tuple[str, str, int], str] = {}
    sort_keys: list[str] = []
    for index, raw_prompt in enumerate(prompt_set["prompts"]):
        prompt = _exact_mapping(
            raw_prompt,
            {"prompt_id", "prompt", "prompt_sha256", "seeds"},
            name=f"prompts[{index}]",
        )
        prompt_id = prompt["prompt_id"]
        text = prompt["prompt"]
        if not isinstance(prompt_id, str) or not prompt_id or not isinstance(text, str):
            raise ValueError("prompt_id must be non-empty and prompt must be a string.")
        digest = require_sha256(prompt["prompt_sha256"], name=f"prompts[{index}].prompt_sha256")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
            raise RuntimeError(f"prompt SHA-256 does not match for {prompt_id!r}.")
        seeds = prompt["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds)
            or seeds != sorted(set(seeds))
        ):
            raise ValueError(f"prompt seeds must be sorted unique integers for {prompt_id!r}.")
        sort_keys.append(prompt_id)
        for seed in seeds:
            key = (prompt_id, digest, seed)
            if key in expected:
                raise ValueError(f"duplicate prompt/seed pair: {key}.")
            expected[key] = text
    if sort_keys != sorted(set(sort_keys)):
        raise ValueError("prompts must have unique IDs sorted lexicographically.")
    if not expected:
        raise ValueError("prompt set must contain at least one prompt/seed pair.")
    return expected, prompt_set


def _protocol_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {name: manifest[name] for name in _PROTOCOL_FIELDS}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_data_snapshot(root: Path, reference: Any) -> tuple[Path, Mapping[str, Any]]:
    path = validate_artifact_reference(root, reference, name="data_snapshot")
    snapshot = _exact_mapping(
        load_canonical_json(path),
        {
            "schema_version",
            "record_type",
            "dataset_snapshot_sha256",
            "train_ids_sha256",
            "heldout_ids_sha256",
        },
        name="data snapshot",
    )
    if snapshot["schema_version"] != 1 or snapshot["record_type"] != "pdd_dataset_snapshot":
        raise ValueError("data snapshot is not a schema-v1 PDD dataset snapshot.")
    for name in ("dataset_snapshot_sha256", "train_ids_sha256", "heldout_ids_sha256"):
        require_sha256(snapshot[name], name=f"data snapshot {name}")
    return path, snapshot


def _validate_stage_evidence(
    root: Path,
    reference: Any,
    *,
    stage: str,
    expected_run_id: str,
    protocol_sha256: str,
    model: Mapping[str, Any],
    modelopt: Mapping[str, Any],
    data_snapshot_sha256: str,
    export_source_checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = validate_artifact_reference(root, reference, name=f"stage_evidence.{stage}")
    evidence = _exact_mapping(
        load_canonical_json(path),
        {
            "schema_version",
            "record_type",
            "stage",
            "status",
            "run_id",
            "model",
            "modelopt",
            "data_snapshot_sha256",
            "evaluation_protocol_sha256",
            "checkpoint",
            "results",
        },
        name=f"{stage} evidence",
    )
    if (
        evidence["schema_version"] != 1
        or evidence["record_type"] != "pdd_stage_evidence"
        or evidence["stage"] != stage
        or evidence["status"] != "passed"
    ):
        raise ValueError(f"{stage} evidence is not a passed schema-v1 record.")
    if evidence["run_id"] != expected_run_id:
        raise RuntimeError(f"{stage} evidence run_id does not match frozen stage_run_ids.")
    if evidence["model"] != model or evidence["modelopt"] != modelopt:
        raise RuntimeError(f"{stage} evidence model/code lineage does not match evaluation.")
    if (
        require_sha256(evidence["data_snapshot_sha256"], name=f"{stage} data snapshot SHA-256")
        != data_snapshot_sha256
    ):
        raise RuntimeError(f"{stage} evidence data lineage does not match evaluation.")
    if (
        require_sha256(evidence["evaluation_protocol_sha256"], name=f"{stage} protocol SHA-256")
        != protocol_sha256
    ):
        raise RuntimeError(f"{stage} evidence does not carry the frozen evaluation protocol.")
    checkpoint = _exact_mapping(
        evidence["checkpoint"],
        {"name", "manifest_sha256", "completed_steps"},
        name=f"{stage} checkpoint",
    )
    if (
        not isinstance(checkpoint["name"], str)
        or not checkpoint["name"]
        or Path(checkpoint["name"]).name != checkpoint["name"]
        or type(checkpoint["completed_steps"]) is not int
        or checkpoint["completed_steps"] < 1
    ):
        raise ValueError(f"{stage} checkpoint lineage is malformed.")
    require_sha256(checkpoint["manifest_sha256"], name=f"{stage} checkpoint manifest SHA-256")
    if stage == "training" and checkpoint != export_source_checkpoint:
        raise RuntimeError("training evidence checkpoint does not match the exported checkpoint.")
    results_path = validate_artifact_reference(
        root, evidence["results"], name=f"{stage} evidence results"
    )
    results = _exact_mapping(
        load_canonical_json(results_path),
        {
            "schema_version",
            "record_type",
            "stage",
            "status",
            "slurm_job_ids",
            "completed_updates",
            "finite_loss",
            "finite_gradients",
            "resume_verified",
        },
        name=f"{stage} results",
    )
    expected_updates = 1_500 if stage == "canary" else 10_000
    expected_job_count = 3 if stage == "canary" else 1
    job_ids = results["slurm_job_ids"]
    if (
        results["schema_version"] != 1
        or results["record_type"] != "pdd_stage_results"
        or results["stage"] != stage
        or results["status"] != "passed"
        or not isinstance(job_ids, list)
        or len(job_ids) != expected_job_count
        or any(type(job_id) is not int or job_id <= 0 for job_id in job_ids)
        or len(set(job_ids)) != len(job_ids)
        or results["completed_updates"] != expected_updates
        or results["finite_loss"] is not True
        or results["finite_gradients"] is not True
        or results["resume_verified"] is not True
    ):
        raise ValueError(f"{stage} results do not satisfy the frozen passed-stage contract.")
    return evidence


def _validate_observations(
    root: Path,
    path: Path,
    *,
    prompt_pairs: Mapping[tuple[str, str, int], str],
    metric_protocols: Mapping[str, Mapping[str, Any]],
    image_protocol: Mapping[str, Any],
    timing_protocol: Mapping[str, Any],
    export_manifest_sha256: str,
    evaluation_protocol_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    document = _exact_mapping(
        load_canonical_json(path),
        {"schema_version", "records"},
        name="observations",
    )
    if document["schema_version"] != 1 or not isinstance(document["records"], list):
        raise ValueError("observation schema is unsupported.")
    records: list[Mapping[str, Any]] = []
    observed: set[tuple[str, str, str, int]] = set()
    order = {condition: index for index, condition in enumerate(EVALUATION_CONDITIONS)}
    sort_keys: list[tuple[str, int, int]] = []
    keys = {
        "condition",
        "prompt_id",
        "prompt_sha256",
        "seed",
        "metrics",
        "output",
        "scheduler_steps",
        "actual_transformer_invocations",
        "batch_normalized_transformer_evaluations",
        "latency_seconds",
        "throughput_images_per_second",
        "peak_device_memory_bytes",
        "height",
        "width",
        "protocol_sha256",
        "evaluation_protocol_sha256",
        "model_artifact_sha256",
    }
    for index, raw_record in enumerate(document["records"]):
        record = _exact_mapping(raw_record, keys, name=f"records[{index}]")
        condition = record["condition"]
        if condition not in order:
            raise ValueError(f"unknown evaluation condition {condition!r}.")
        prompt_id = record["prompt_id"]
        prompt_sha = require_sha256(record["prompt_sha256"], name=f"records[{index}].prompt_sha256")
        seed = _nonnegative_int(record["seed"], name=f"records[{index}].seed")
        pair = (prompt_id, prompt_sha, seed)
        if pair not in prompt_pairs:
            raise RuntimeError(f"observation does not match the prompt set: {pair}.")
        key = (condition, *pair)
        if key in observed:
            raise ValueError(f"duplicate evaluation observation: {key}.")
        observed.add(key)
        protocol = CONDITION_PROTOCOLS[condition]
        if (
            require_sha256(record["protocol_sha256"], name=f"records[{index}].protocol_sha256")
            != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        ):
            raise RuntimeError(f"records[{index}] is not bound to its condition protocol.")
        if (
            require_sha256(
                record["evaluation_protocol_sha256"],
                name=f"records[{index}].evaluation_protocol_sha256",
            )
            != evaluation_protocol_sha256
        ):
            raise RuntimeError(f"records[{index}] is not bound to the evaluation protocol.")
        if (
            require_sha256(
                record["model_artifact_sha256"],
                name=f"records[{index}].model_artifact_sha256",
            )
            != export_manifest_sha256
        ):
            raise RuntimeError(f"records[{index}] model artifact does not match the PDD export.")
        if (
            record["height"] != image_protocol["height"]
            or record["width"] != image_protocol["width"]
        ):
            raise RuntimeError(f"records[{index}] resolution does not match image_protocol.")
        metrics = record["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(metric_protocols):
            raise ValueError(f"records[{index}].metrics does not match metric_protocols.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in metrics.values()
        ):
            raise ValueError(f"records[{index}].metrics contains a non-finite value.")
        validate_artifact_reference(root, record["output"], name=f"records[{index}].output")
        counts = tuple(
            _nonnegative_int(record[name], name=f"records[{index}].{name}")
            for name in (
                "scheduler_steps",
                "actual_transformer_invocations",
                "batch_normalized_transformer_evaluations",
            )
        )
        if not all(counts) or counts[1] > counts[2]:
            raise ValueError(f"records[{index}] has invalid compute counters.")
        expected_counts = tuple(
            protocol[name]
            for name in (
                "scheduler_steps",
                "actual_transformer_invocations",
                "batch_normalized_transformer_evaluations",
            )
        )
        if counts != expected_counts:
            raise RuntimeError(
                f"{condition} compute counters must be {expected_counts}, got {counts}."
            )
        latency = _positive_finite(
            record["latency_seconds"], name=f"records[{index}].latency_seconds"
        )
        throughput = _positive_finite(
            record["throughput_images_per_second"],
            name=f"records[{index}].throughput_images_per_second",
        )
        expected_throughput = timing_protocol["batch_size"] / latency
        if not math.isclose(throughput, expected_throughput, rel_tol=1e-6, abs_tol=0.0):
            raise RuntimeError(f"records[{index}] throughput does not match batch size / latency.")
        if (
            type(record["peak_device_memory_bytes"]) is not int
            or record["peak_device_memory_bytes"] <= 0
        ):
            raise ValueError(f"records[{index}].peak_device_memory_bytes must be positive.")
        sort_keys.append((prompt_id, seed, order[condition]))
        records.append(record)
    if sort_keys != sorted(sort_keys):
        raise ValueError("observations must be sorted by prompt_id, seed, and condition order.")
    expected = {(condition, *pair) for pair in prompt_pairs for condition in EVALUATION_CONDITIONS}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"effectiveness observations are incomplete: missing={missing[:5]}, extra={extra[:5]}."
        )
    for condition in EVALUATION_CONDITIONS:
        condition_counts = {
            (
                record["scheduler_steps"],
                record["actual_transformer_invocations"],
                record["batch_normalized_transformer_evaluations"],
            )
            for record in records
            if record["condition"] == condition
        }
        if len(condition_counts) != 1:
            raise RuntimeError(f"{condition} compute counters vary across paired observations.")
    return tuple(records)


def _validate_automodel_snapshot(
    snapshot: Any,
    *,
    export_automodel: Mapping[str, Any],
) -> None:
    snapshot = _exact_mapping(
        snapshot,
        {
            "distribution",
            "files",
            "import_origin",
            "package_file_count",
            "package_tree_sha256",
            "release_commit",
            "release_tag",
            "root",
            "runtime_versions",
            "version",
            "wheel",
            "wheel_sha256",
        },
        name="AutoModel environment snapshot",
    )
    for key in ("distribution", "version", "runtime_versions"):
        if snapshot[key] != export_automodel.get(key):
            raise RuntimeError(f"AutoModel environment/export identity mismatch for {key}.")
    for key in ("package_tree_sha256", "wheel_sha256"):
        digest = require_sha256(snapshot[key], name=f"AutoModel snapshot {key}")
        if digest != export_automodel.get(key):
            raise RuntimeError(f"AutoModel environment/export identity mismatch for {key}.")
    root = Path(snapshot["root"])
    import_origin = Path(snapshot["import_origin"])
    if not root.is_absolute() or not import_origin.is_absolute():
        raise ValueError("AutoModel snapshot root and import_origin must be absolute.")
    try:
        import_origin.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            "AutoModel import origin is outside its installed distribution."
        ) from error
    files = snapshot["files"]
    if (
        not isinstance(files, list)
        or type(snapshot["package_file_count"]) is not int
        or snapshot["package_file_count"] != len(files)
        or not files
    ):
        raise ValueError("AutoModel snapshot file inventory is malformed.")
    tree = hashlib.sha256()
    previous = None
    for index, raw_record in enumerate(files):
        record = _exact_mapping(
            raw_record,
            {"path", "sha256", "size"},
            name=f"AutoModel files[{index}]",
        )
        path = record["path"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or "\\" in path
            or any(part in ("", ".", "..") for part in path.split("/"))
            or (previous is not None and path <= previous)
        ):
            raise ValueError("AutoModel snapshot file paths must be sorted normalized references.")
        digest = require_sha256(record["sha256"], name=f"AutoModel files[{index}].sha256")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"AutoModel files[{index}].size is invalid.")
        tree.update(path.encode())
        tree.update(b"\0")
        tree.update(digest.encode())
        tree.update(b"\0")
        tree.update(str(record["size"]).encode())
        tree.update(b"\n")
        previous = path
    if tree.hexdigest() != snapshot["package_tree_sha256"]:
        raise RuntimeError("AutoModel snapshot file inventory does not match its tree SHA-256.")


def _validate_evaluation_protocol(manifest: Mapping[str, Any]) -> None:
    if manifest["conditions"] != list(EVALUATION_CONDITIONS):
        raise ValueError("effectiveness manifest must contain the six fixed conditions in order.")
    if manifest["condition_protocols"] != CONDITION_PROTOCOLS:
        raise ValueError("condition_protocols must match the frozen Qwen PDD protocol exactly.")
    if manifest["grid_protocols"] != GRID_PROTOCOLS:
        raise ValueError("grid_protocols must contain the exact authenticated shifted-flow nodes.")
    if manifest["integrator_protocols"] != INTEGRATOR_PROTOCOLS:
        raise ValueError("integrator_protocols must contain the exact frozen update equations.")
    stage_run_ids = _exact_mapping(
        manifest["stage_run_ids"], {"canary", "training"}, name="stage_run_ids"
    )
    if any(not isinstance(run_id, str) or not run_id for run_id in stage_run_ids.values()):
        raise ValueError("stage_run_ids must contain non-empty run IDs.")
    image = _exact_mapping(
        manifest["image_protocol"],
        {"height", "width", "batch_size", "max_sequence_length"},
        name="image_protocol",
    )
    if image != {"height": 1024, "width": 1024, "batch_size": 1, "max_sequence_length": 512}:
        raise ValueError("image_protocol must use the frozen 1024px single-image Qwen protocol.")
    metrics = manifest["metric_protocols"]
    if not isinstance(metrics, Mapping) or not metrics or list(metrics) != sorted(metrics):
        raise ValueError("metric_protocols must be a non-empty, sorted mapping.")
    for name, raw_protocol in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric protocol names must be non-empty strings.")
        protocol = _exact_mapping(
            raw_protocol,
            {"direction", "implementation", "revision"},
            name=f"metric_protocols.{name}",
        )
        if protocol["direction"] not in ("higher", "lower"):
            raise ValueError(f"metric_protocols.{name}.direction must be higher or lower.")
        if not isinstance(protocol["implementation"], str) or not protocol["implementation"]:
            raise ValueError(f"metric_protocols.{name}.implementation must be non-empty.")
        _commit(protocol["revision"], name=f"metric_protocols.{name}.revision")
    timing = _exact_mapping(
        manifest["timing_protocol"],
        {"batch_size", "warmup_runs", "measured_runs", "scope", "synchronize_device"},
        name="timing_protocol",
    )
    if (
        timing["batch_size"] != 1
        or type(timing["warmup_runs"]) is not int
        or timing["warmup_runs"] < 1
        or type(timing["measured_runs"]) is not int
        or timing["measured_runs"] < 3
        or timing["scope"] != "transformer_sampling_and_vae_decode"
        or timing["synchronize_device"] is not True
    ):
        raise ValueError("timing_protocol does not satisfy the frozen measurement contract.")
    rule = _exact_mapping(
        manifest["decision_rule"],
        {
            "primary_condition",
            "primary_metric",
            "quality_margin",
            "quality_ci_rule",
            "efficiency_measure",
            "efficiency_baseline",
            "minimum_relative_reduction",
            "minimum_paired_samples",
        },
        name="decision_rule",
    )
    if rule["primary_condition"] not in ("pdd_2", "pdd_4", "pdd_8"):
        raise ValueError("decision_rule.primary_condition must name a supported PDD schedule.")
    if rule["primary_metric"] not in metrics:
        raise ValueError("decision_rule.primary_metric is not in metric_protocols.")
    if (
        isinstance(rule["quality_margin"], bool)
        or not isinstance(rule["quality_margin"], int | float)
        or not math.isfinite(float(rule["quality_margin"]))
        or rule["quality_margin"] < 0
        or rule["quality_ci_rule"] != "paired_bootstrap_95_noninferiority"
    ):
        raise ValueError("decision_rule quality noninferiority contract is malformed.")
    if (
        rule["efficiency_measure"] != "batch_normalized_transformer_evaluations"
        or rule["efficiency_baseline"] != "teacher_guided"
        or isinstance(rule["minimum_relative_reduction"], bool)
        or not isinstance(rule["minimum_relative_reduction"], int | float)
        or not 0 < float(rule["minimum_relative_reduction"]) < 1
        or type(rule["minimum_paired_samples"]) is not int
        or rule["minimum_paired_samples"] < 16
    ):
        raise ValueError("decision_rule efficiency/sample contract is malformed.")


def validate_effectiveness_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Authenticate a complete, paired, claim-bearing effectiveness bundle."""
    unresolved_manifest = Path(manifest_path)
    if unresolved_manifest.is_symlink():
        raise RuntimeError(f"effectiveness manifest cannot be a symlink: {unresolved_manifest}.")
    manifest_path = unresolved_manifest.resolve()
    root = manifest_path.parent
    detached = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not detached.is_file() or detached.is_symlink():
        raise RuntimeError(f"detached manifest SHA-256 is missing: {detached}.")
    detached_bytes = detached.read_bytes()
    expected_bytes = (sha256_file(manifest_path) + "\n").encode()
    if detached_bytes != expected_bytes:
        raise RuntimeError("detached effectiveness manifest SHA-256 does not match.")
    manifest = _exact_mapping(
        load_canonical_json(manifest_path),
        {
            "schema_version",
            "stage",
            "run_id",
            "model",
            "modelopt",
            "pdd_export",
            "prompt_set",
            "observations",
            "environment",
            "data_snapshot",
            "stage_evidence",
            "conditions",
            "condition_protocols",
            "grid_protocols",
            "integrator_protocols",
            "image_protocol",
            "metric_protocols",
            "timing_protocol",
            "decision_rule",
            "negative_condition",
            "stage_run_ids",
            "bootstrap",
        },
        name="effectiveness manifest",
    )
    if manifest["schema_version"] != 1 or manifest["stage"] != "effectiveness_evaluation":
        raise ValueError("only schema-v1 effectiveness_evaluation manifests can support claims.")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ValueError("effectiveness run_id must be non-empty.")
    model = _exact_mapping(manifest["model"], {"id", "revision"}, name="model")
    if not isinstance(model["id"], str) or not model["id"]:
        raise ValueError("model.id must be non-empty.")
    _commit(model["revision"], name="model.revision")
    modelopt = _exact_mapping(manifest["modelopt"], {"commit", "dirty"}, name="modelopt")
    _commit(modelopt["commit"], name="modelopt.commit")
    if modelopt["dirty"] is not False:
        raise RuntimeError("claim-bearing effectiveness runs require a clean ModelOpt commit.")
    _validate_evaluation_protocol(manifest)
    bootstrap = _exact_mapping(manifest["bootstrap"], {"replicates", "seed"}, name="bootstrap")
    if type(bootstrap["replicates"]) is not int or bootstrap["replicates"] < 1_000:
        raise ValueError("bootstrap.replicates must be an integer >= 1000.")
    _nonnegative_int(bootstrap["seed"], name="bootstrap.seed")
    export_path = validate_artifact_reference(root, manifest["pdd_export"], name="pdd_export")
    if export_path.name != "manifest.json":
        raise ValueError("pdd_export must reference the export directory's manifest.json.")
    export_descriptor = inspect_pdd_export(export_path.parent)
    if export_descriptor.root / "manifest.json" != export_path:
        raise RuntimeError("pdd_export does not identify the authenticated export manifest.")
    environment_path = validate_artifact_reference(
        root, manifest["environment"], name="environment"
    )
    export_document = export_descriptor.manifest
    environment_document = load_canonical_json(environment_path)
    if not isinstance(export_document, Mapping):
        raise ValueError("pdd_export must reference a canonical JSON object.")
    if not isinstance(environment_document, Mapping):
        raise ValueError("environment must reference a canonical JSON object.")
    export_identity = export_document.get("identity")
    export_modelopt = export_document.get("modelopt_source")
    if (
        not isinstance(export_identity, Mapping)
        or export_document.get("format") != "modelopt-pdd-safetensors"
    ):
        raise ValueError("pdd_export does not reference a ModelOpt PDD export manifest.")
    export_model = export_identity.get("model")
    if not isinstance(export_model, Mapping) or {
        "id": export_model.get("id"),
        "revision": export_model.get("revision"),
    } != dict(model):
        raise RuntimeError("effectiveness model does not match the PDD export identity.")
    if not isinstance(export_modelopt, Mapping) or export_modelopt != modelopt:
        raise RuntimeError("effectiveness ModelOpt source does not match the PDD export.")
    export_automodel = export_identity.get("automodel")
    if not isinstance(export_automodel, Mapping):
        raise ValueError("PDD export has no AutoModel identity.")
    export_guidance = export_identity.get("guidance")
    if export_guidance != {"scale": 4.0, "rescale": 1.0, "eps": 1e-5}:
        raise RuntimeError("PDD export guidance does not match the frozen evaluation protocol.")
    _validate_automodel_snapshot(environment_document, export_automodel=export_automodel)
    data_snapshot_path, data_snapshot = _validate_data_snapshot(root, manifest["data_snapshot"])
    data_snapshot_sha256 = sha256_file(data_snapshot_path)
    export_data = _exact_mapping(
        export_identity.get("data"),
        {
            "ordered_train_id_sha256",
            "ordered_heldout_id_sha256",
            "dataset_snapshot_sha256",
            "local_batch_size",
            "grad_accumulation_steps",
        },
        name="PDD export data identity",
    )
    if {
        "dataset_snapshot_sha256": export_data["dataset_snapshot_sha256"],
        "train_ids_sha256": export_data["ordered_train_id_sha256"],
        "heldout_ids_sha256": export_data["ordered_heldout_id_sha256"],
    } != {
        name: data_snapshot[name]
        for name in ("dataset_snapshot_sha256", "train_ids_sha256", "heldout_ids_sha256")
    }:
        raise RuntimeError("PDD export training-data identity does not match data_snapshot.")
    negative_path = validate_artifact_reference(
        root, manifest["negative_condition"], name="negative_condition"
    )
    negative = _exact_mapping(
        load_canonical_json(negative_path),
        {"schema_version", "record_type", "prompt_sha256", "embedding"},
        name="negative condition",
    )
    if negative["schema_version"] != 1 or negative["record_type"] != "pdd_negative_condition":
        raise ValueError("negative_condition must be a schema-v1 PDD negative condition.")
    require_sha256(negative["prompt_sha256"], name="negative condition prompt SHA-256")
    validate_artifact_reference(root, negative["embedding"], name="negative condition embedding")
    prompt_path = validate_artifact_reference(root, manifest["prompt_set"], name="prompt_set")
    observation_path = validate_artifact_reference(
        root, manifest["observations"], name="observations"
    )
    evidence = _exact_mapping(
        manifest["stage_evidence"], {"canary", "training"}, name="stage_evidence"
    )
    protocol_sha256 = _protocol_sha256(manifest)
    export_source_checkpoint = export_document["source_checkpoint"]
    for stage in ("canary", "training"):
        _validate_stage_evidence(
            root,
            evidence[stage],
            stage=stage,
            expected_run_id=manifest["stage_run_ids"][stage],
            protocol_sha256=protocol_sha256,
            model=model,
            modelopt=modelopt,
            data_snapshot_sha256=data_snapshot_sha256,
            export_source_checkpoint=export_source_checkpoint,
        )
    prompt_pairs, prompt_set = _load_prompt_set(prompt_path)
    records = _validate_observations(
        root,
        observation_path,
        prompt_pairs=prompt_pairs,
        metric_protocols=manifest["metric_protocols"],
        image_protocol=manifest["image_protocol"],
        timing_protocol=manifest["timing_protocol"],
        export_manifest_sha256=sha256_file(export_path),
        evaluation_protocol_sha256=protocol_sha256,
    )
    if len(prompt_pairs) < manifest["decision_rule"]["minimum_paired_samples"]:
        raise RuntimeError("paired sample count is below decision_rule.minimum_paired_samples.")
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_set": prompt_set,
        "records": records,
    }


def deterministic_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic SHA-256-index percentile interval for a sample mean."""
    if not values:
        raise ValueError("bootstrap values must be non-empty.")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive.")
    samples: list[float] = []
    for replicate in range(replicates):
        total = 0.0
        for draw in range(len(values)):
            payload = f"{seed}:{replicate}:{draw}".encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(values)
            total += float(values[index])
        samples.append(total / len(values))
    samples.sort()
    lower = samples[math.floor(0.025 * (replicates - 1))]
    upper = samples[math.ceil(0.975 * (replicates - 1))]
    return lower, upper


def summarize_effectiveness_bundle(validated: Mapping[str, Any]) -> dict[str, Any]:
    """Compute paired aggregate metrics without promoting smoke output to evidence."""
    manifest = validated["manifest"]
    records = validated["records"]
    replicates = manifest["bootstrap"]["replicates"]
    seed = manifest["bootstrap"]["seed"]
    by_condition = {
        condition: {
            (record["prompt_id"], record["prompt_sha256"], record["seed"]): record
            for record in records
            if record["condition"] == condition
        }
        for condition in EVALUATION_CONDITIONS
    }
    pair_keys = sorted(by_condition["teacher_guided"])
    aggregates: dict[str, Any] = {}
    for condition_index, condition in enumerate(EVALUATION_CONDITIONS):
        condition_records = by_condition[condition]
        metrics: dict[str, Any] = {}
        for metric_index, (metric, protocol) in enumerate(manifest["metric_protocols"].items()):
            direction = protocol["direction"]
            values = [float(condition_records[key]["metrics"][metric]) for key in pair_keys]
            teacher = [
                float(by_condition["teacher_guided"][key]["metrics"][metric]) for key in pair_keys
            ]
            deltas = [value - baseline for value, baseline in zip(values, teacher)]
            metric_seed = seed + condition_index * 10_000 + metric_index * 2
            metrics[metric] = {
                "direction": direction,
                "mean": sum(values) / len(values),
                "mean_ci95": list(
                    deterministic_bootstrap_mean_ci(
                        values,
                        replicates=replicates,
                        seed=metric_seed,
                    )
                ),
                "paired_delta_vs_teacher": sum(deltas) / len(deltas),
                "paired_delta_ci95": list(
                    deterministic_bootstrap_mean_ci(
                        deltas,
                        replicates=replicates,
                        seed=metric_seed + 1,
                    )
                ),
            }
        aggregates[condition] = {
            "metrics": metrics,
            "latency_seconds": {},
            "throughput_images_per_second": {},
            "peak_device_memory_bytes": {},
            "mean_batch_normalized_transformer_evaluations": sum(
                condition_records[key]["batch_normalized_transformer_evaluations"]
                for key in pair_keys
            )
            / len(pair_keys),
        }
        latencies = [float(condition_records[key]["latency_seconds"]) for key in pair_keys]
        teacher_latencies = [
            float(by_condition["teacher_guided"][key]["latency_seconds"]) for key in pair_keys
        ]
        latency_deltas = [value - baseline for value, baseline in zip(latencies, teacher_latencies)]
        latency_seed = seed + condition_index * 10_000 + len(metrics) * 2
        aggregates[condition]["latency_seconds"] = {
            "mean": sum(latencies) / len(latencies),
            "mean_ci95": list(
                deterministic_bootstrap_mean_ci(
                    latencies,
                    replicates=replicates,
                    seed=latency_seed,
                )
            ),
            "paired_delta_vs_teacher": sum(latency_deltas) / len(latency_deltas),
            "paired_delta_ci95": list(
                deterministic_bootstrap_mean_ci(
                    latency_deltas,
                    replicates=replicates,
                    seed=latency_seed + 1,
                )
            ),
        }
        for telemetry_index, name in enumerate(
            ("throughput_images_per_second", "peak_device_memory_bytes"), start=1
        ):
            values = [float(condition_records[key][name]) for key in pair_keys]
            teacher = [float(by_condition["teacher_guided"][key][name]) for key in pair_keys]
            deltas = [value - baseline for value, baseline in zip(values, teacher)]
            telemetry_seed = latency_seed + telemetry_index * 2
            aggregates[condition][name] = {
                "mean": sum(values) / len(values),
                "mean_ci95": list(
                    deterministic_bootstrap_mean_ci(
                        values,
                        replicates=replicates,
                        seed=telemetry_seed,
                    )
                ),
                "paired_delta_vs_teacher": sum(deltas) / len(deltas),
                "paired_delta_ci95": list(
                    deterministic_bootstrap_mean_ci(
                        deltas,
                        replicates=replicates,
                        seed=telemetry_seed + 1,
                    )
                ),
            }

    rule = manifest["decision_rule"]
    primary = aggregates[rule["primary_condition"]]
    primary_metric = primary["metrics"][rule["primary_metric"]]
    lower, upper = primary_metric["paired_delta_ci95"]
    margin = float(rule["quality_margin"])
    direction = primary_metric["direction"]
    if direction == "higher":
        quality_state = (
            "pass" if lower >= -margin else "fail" if upper < -margin else "inconclusive"
        )
    else:
        quality_state = "pass" if upper <= margin else "fail" if lower > margin else "inconclusive"
    baseline = aggregates[rule["efficiency_baseline"]]
    candidate_value = primary["mean_batch_normalized_transformer_evaluations"]
    baseline_value = baseline["mean_batch_normalized_transformer_evaluations"]
    relative_reduction = 1.0 - candidate_value / baseline_value
    efficiency_state = (
        "pass" if relative_reduction >= float(rule["minimum_relative_reduction"]) else "fail"
    )
    if "fail" in (quality_state, efficiency_state):
        conclusion = "not_effective"
    elif (quality_state, efficiency_state) == ("pass", "pass"):
        conclusion = "effective"
    else:
        conclusion = "inconclusive"
    return {
        "schema_version": 1,
        "record_type": "effectiveness_summary",
        "source_manifest_sha256": validated["manifest_sha256"],
        "paired_sample_count": len(pair_keys),
        "bootstrap": dict(manifest["bootstrap"]),
        "aggregates": aggregates,
        "decision": {
            "label": conclusion,
            "quality_state": quality_state,
            "efficiency_state": efficiency_state,
            "relative_efficiency_reduction": relative_reduction,
            "rule": dict(rule),
        },
    }
