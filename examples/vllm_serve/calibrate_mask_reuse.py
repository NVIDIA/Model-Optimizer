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

"""Build a fail-closed schema-v3 mask-reuse candidate from compact captures.

The input is compact capture JSONL emitted by ``collect_mask_reuse.py``. Its
streaming selector never expands the full consumer-head by donor-head matrix
into repeated row objects. This command cannot promote a serving policy until
the grouped inner/outer protocol is implemented and its preregistered gates pass.

Example::

    python examples/vllm_serve/calibrate_mask_reuse.py \
        --checkpoint /path/to/checkpoint \
        --compact-captures compact-captures.jsonl \
        --capture-manifest compact-captures.jsonl.manifest.json \
        --vanilla-config sparse_attention_config.json \
        --topology topology.json \
        --calibration-plan calibration-plan.json \
        --family-registry family-registry.json \
        --grouped-fit grouped-fit.json \
        --outer-report outer-report.json \
        --max-anchor-dropped-mass 0.02 \
        --max-reuse-dropped-mass 0.02 \
        --max-reuse-selection-dropped-mass 0.005

The final output is candidate-only and must be rejected by serving.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    StableFileSnapshot,
    read_stable_file_snapshot,
    verify_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_compact import (
    calibrate_compact_mask_reuse_policy,
    load_compact_mask_reuse_captures,
)

_EVIDENCE_ARTIFACTS = {
    "calibration_plan_sha256": "calibration_plan",
    "family_registry_sha256": "family_registry",
    "grouped_fit_sha256": "grouped_fit",
    "outer_report_sha256": "outer_report",
}

_CAPTURE_MANIFEST_FIELDS = frozenset(
    {
        "capture_manifest_schema_version",
        "capture_protocol",
        "model",
        "checkpoint_manifest_sha256",
        "checkpoint_manifest_path",
        "checkpoint_file_count",
        "checkpoint_total_size_bytes",
        "plan",
        "fa4_source",
        "fa4_source_commit",
        "engine_kwargs",
        "dense_shadow_validation_requested",
        "target_sparsity_hex",
        "vanilla_threshold_scale_factor",
        "vanilla_fit_sha256",
        "vanilla_config_file_sha256",
        "prompt_plan_file_sha256",
        "compact_capture_file_sha256",
        "capture_count",
        "candidate_cell_count",
        "captures",
    }
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json_object(payload: bytes, *, path: Path, label: str) -> dict[str, object]:
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not load {label} from {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw


def _load_json_snapshot(path: Path, *, label: str) -> tuple[dict[str, object], StableFileSnapshot]:
    snapshot = read_stable_file_snapshot(path, label=label)
    return _parse_json_object(snapshot.payload, path=path, label=label), snapshot


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Load strict JSON from one stable no-follow byte snapshot."""
    return _load_json_snapshot(path, label=label)[0]


def _stable_file_sha256(path: Path, *, label: str) -> str:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"could not open {label} at {path}") from error
    before = os.fstat(descriptor)
    digest = sha256()
    try:
        with os.fdopen(descriptor, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        named = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not hash stable {label} at {path}") from error
    identities = {
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        for value in (before, after, named)
    }
    if len(identities) != 1 or not stat.S_ISREG(named.st_mode):
        raise ValueError(f"{label} changed while it was being hashed")
    return digest.hexdigest()


def _evidence_artifacts(
    args: argparse.Namespace, *, vanilla_fit_sha256: str
) -> tuple[dict[str, str], dict[str, Path]]:
    paths = {
        field: Path(getattr(args, attribute)) for field, attribute in _EVIDENCE_ARTIFACTS.items()
    }
    paths["vanilla_fit_sha256"] = args.vanilla_config
    paths["reuse_bundle_sha256"] = args.compact_captures
    evidence = {
        field: (
            vanilla_fit_sha256
            if field == "vanilla_fit_sha256"
            else _stable_file_sha256(path, label=field)
        )
        for field, path in paths.items()
    }
    return evidence, paths


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_payload(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (observed.st_dev, observed.st_ino) == identity:
        path.unlink()
        _fsync_directory(path.parent)


def _publish_no_clobber(temporary: Path, destination: Path) -> tuple[int, int]:
    observed = temporary.stat(follow_symlinks=False)
    identity = observed.st_dev, observed.st_ino
    os.link(temporary, destination, follow_symlinks=False)
    try:
        temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        _unlink_if_identity(destination, identity)
        raise
    return identity


def _publish_candidate_outputs(
    policy_path: Path, policy_payload: bytes, report_path: Path, report_payload: bytes
) -> None:
    if policy_path.exists() or report_path.exists():
        raise FileExistsError("candidate outputs already exist; refusing to overwrite them")
    policy_temporary: Path | None = None
    report_temporary: Path | None = None
    report_identity: tuple[int, int] | None = None
    try:
        policy_temporary = _temporary_payload(policy_path, policy_payload)
        report_temporary = _temporary_payload(report_path, report_payload)
        report_identity = _publish_no_clobber(report_temporary, report_path)
        report_temporary = None
        _publish_no_clobber(policy_temporary, policy_path)
        policy_temporary = None
    except BaseException:
        if report_identity is not None:
            _unlink_if_identity(report_path, report_identity)
        if policy_temporary is not None:
            policy_temporary.unlink(missing_ok=True)
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)
        raise


def _validate_capture_manifest(
    raw: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    model: str,
    compact_capture_sha256: str,
    vanilla_config_sha256: str,
) -> None:
    missing = _CAPTURE_MANIFEST_FIELDS - raw.keys()
    extra = raw.keys() - _CAPTURE_MANIFEST_FIELDS
    if missing or extra:
        raise ValueError(
            "capture manifest fields do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    expected = {
        "capture_manifest_schema_version": 3,
        "capture_protocol": "modelopt_vllm_mask_reuse_target_sparsity_v3",
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "compact_capture_file_sha256": compact_capture_sha256,
        "vanilla_config_file_sha256": vanilla_config_sha256,
    }
    for field, value in expected.items():
        if raw[field] != value:
            raise ValueError(f"capture manifest {field} does not match its verified input")
    if not isinstance(raw["engine_kwargs"], Mapping):
        raise ValueError("capture manifest engine_kwargs must be an object")
    if not isinstance(raw["dense_shadow_validation_requested"], bool):
        raise ValueError("capture manifest dense_shadow_validation_requested must be boolean")
    if isinstance(raw["capture_count"], bool) or not isinstance(raw["capture_count"], int):
        raise ValueError("capture manifest capture_count must be an integer")
    if raw["capture_count"] <= 0:
        raise ValueError("capture manifest must contain at least one capture")
    captures = raw["captures"]
    if not isinstance(captures, list) or len(captures) != raw["capture_count"]:
        raise ValueError("capture manifest captures do not match capture_count")
    candidate_cell_count = raw["candidate_cell_count"]
    if (
        isinstance(candidate_cell_count, bool)
        or not isinstance(candidate_cell_count, int)
        or candidate_cell_count <= 0
    ):
        raise ValueError("capture manifest candidate_cell_count must be positive")
    observed_cells = 0
    for index, capture in enumerate(captures):
        if not isinstance(capture, Mapping) or not isinstance(
            capture.get("candidate_cell_count"), int
        ):
            raise ValueError(
                f"capture manifest captures[{index}].candidate_cell_count must be an integer"
            )
        observed_cells += int(capture["candidate_cell_count"])
    if observed_cells != candidate_cell_count:
        raise ValueError("capture manifest candidate-cell total is inconsistent")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed schema-v3 candidate from compact mask-reuse captures",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--compact-captures",
        type=Path,
        required=True,
        help="Compact capture JSONL emitted by collect_mask_reuse.py (recommended)",
    )
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument(
        "--vanilla-config",
        type=Path,
        required=True,
        help="ModelOpt sparse_attention_config JSON or checkpoint config.json",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        required=True,
        help="JSON object containing anchors and nearest layer mappings",
    )
    parser.add_argument("--calibration-plan", type=Path, required=True)
    parser.add_argument("--family-registry", type=Path, required=True)
    parser.add_argument("--grouped-fit", type=Path, required=True)
    parser.add_argument("--outer-report", type=Path, required=True)
    parser.add_argument(
        "--max-anchor-dropped-mass",
        type=float,
        required=True,
        help="Maximum allowed anchor dropped mass",
    )
    parser.add_argument(
        "--max-reuse-dropped-mass",
        type=float,
        required=True,
        help="Maximum allowed held-out reuse dropped mass",
    )
    parser.add_argument(
        "--max-reuse-selection-dropped-mass",
        type=float,
        default=None,
        help="Optional stricter calibration-time reuse bound",
    )
    parser.add_argument(
        "--output-policy",
        type=Path,
        default=Path("mask_reuse_candidate.json"),
        help="Fail-closed schema-v3 candidate output path",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("mask_reuse_calibration_report.json"),
        help="Standalone calibration-report output path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.output_policy.resolve() == args.output_report.resolve():
        parser.error("--output-policy and --output-report must be different paths")

    try:
        checkpoint = verify_checkpoint_manifest(args.checkpoint)
        vanilla_config, vanilla_snapshot = _load_json_snapshot(
            args.vanilla_config, label="vanilla config"
        )
        topology, topology_snapshot = _load_json_snapshot(args.topology, label="topology")
        capture_manifest, capture_manifest_snapshot = _load_json_snapshot(
            args.capture_manifest, label="capture manifest"
        )
        evidence, evidence_paths = _evidence_artifacts(
            args, vanilla_fit_sha256=vanilla_snapshot.sha256
        )
        capture_manifest_sha256 = capture_manifest_snapshot.sha256
        topology_sha256 = topology_snapshot.sha256
        _validate_capture_manifest(
            capture_manifest,
            checkpoint_sha256=checkpoint.sha256,
            model=checkpoint.model,
            compact_capture_sha256=evidence["reuse_bundle_sha256"],
            vanilla_config_sha256=evidence["vanilla_fit_sha256"],
        )
        artifact = calibrate_compact_mask_reuse_policy(
            load_compact_mask_reuse_captures(args.compact_captures),
            vanilla_calibration=vanilla_config,
            topology=topology,
            checkpoint_manifest=checkpoint,
            evidence=evidence,
            max_anchor_dropped_mass=args.max_anchor_dropped_mass,
            max_reuse_dropped_mass=args.max_reuse_dropped_mass,
            max_reuse_selection_dropped_mass=args.max_reuse_selection_dropped_mass,
            source_provenance={
                "capture_manifest_sha256": capture_manifest_sha256,
                "topology_file_sha256": topology_sha256,
            },
        )
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("calibrator returned no provenance object")
        if provenance.get("input_capture_count") != capture_manifest["capture_count"]:
            raise ValueError("calibrator capture count does not match capture manifest")
        if provenance.get("candidate_cell_count") != capture_manifest["candidate_cell_count"]:
            raise ValueError("calibrator candidate-cell count does not match capture manifest")
        for field, path in evidence_paths.items():
            if _stable_file_sha256(path, label=field) != evidence[field]:
                raise ValueError(f"{field} artifact changed during calibration")
        if (
            _stable_file_sha256(args.capture_manifest, label="capture manifest")
            != capture_manifest_sha256
        ):
            raise ValueError("capture manifest changed during calibration")
        if _stable_file_sha256(args.topology, label="topology") != topology_sha256:
            raise ValueError("topology changed during calibration")
        if verify_checkpoint_manifest(args.checkpoint) != checkpoint:
            raise ValueError("checkpoint changed during calibration")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if (
        artifact.get("promotion_status") != "candidate_only"
        or artifact.get("deployment_geometry_validated") is not False
    ):
        parser.error("calibrator did not return a fail-closed candidate-only artifact")

    report = artifact.get("calibration_report")
    if not isinstance(report, Mapping):
        parser.error("calibrator returned no calibration_report object")

    try:
        policy_payload = _canonical_json_bytes(artifact)
        report_payload = _canonical_json_bytes(report)
        _publish_candidate_outputs(
            args.output_policy, policy_payload, args.output_report, report_payload
        )
    except (OSError, FileExistsError) as error:
        parser.error(f"could not write calibration outputs: {error}")

    policy_digest = sha256(policy_payload).hexdigest()
    print(f"[ModelOpt] Wrote fail-closed mask-reuse candidate to {args.output_policy.resolve()}")
    print(f"[ModelOpt] Wrote calibration report to {args.output_report.resolve()}")
    print(f"MASK_REUSE_FA4_CANDIDATE_SHA256={policy_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
