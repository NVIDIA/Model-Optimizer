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
        --anchor-dropped-mass-report-threshold 0.02 \
        --reuse-dropped-mass-report-threshold 0.02 \
        --target-bmm1-skip-ratio 0.50

The final output is candidate-only and must be rejected by serving.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from modelopt.torch.sparsity.attention_sparsity.calibration.capture_provenance import (
    capture_source_fields,
    capture_source_provenance,
    validate_capture_source,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    StableFileSnapshot,
    read_stable_file_snapshot,
    stable_file_sha256,
    verify_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_compact import (
    calibrate_compact_mask_reuse_policy,
    load_compact_mask_reuse_captures,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_contracts import (
    selection_target_met,
    validate_capture_scope,
    validate_manifest_counts,
    validate_model_geometry,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_solver import (
    MaskReuseSolverConfig,
    MaskReuseSolverError,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_stream import (
    validate_whole_prefill_captures,
    validate_whole_prefill_evidence,
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
        "fa4_source_git_tree",
        "fa4_source_git_archive_sha256",
        "fa4_source_manifest_path",
        "fa4_source_manifest_sha256",
        "fa4_source_directory_count",
        "fa4_source_file_count",
        "fa4_source_total_size_bytes",
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
    return stable_file_sha256(path, label=label)


def _evidence_artifacts(
    args: argparse.Namespace, *, vanilla_fit_sha256: str, checkpoint_sha256: str
) -> tuple[dict[str, str], dict[str, Path]]:
    paths = {
        field: Path(getattr(args, attribute))
        for field, attribute in _EVIDENCE_ARTIFACTS.items()
        if getattr(args, attribute, None) is not None
    }
    for field, path in paths.items():
        value = _load_json_object(path, label=field)
        if not value:
            raise ValueError(f"{field} must contain a nonempty JSON evidence object")
        if "checkpoint_manifest_sha256" in value:
            if value["checkpoint_manifest_sha256"] != checkpoint_sha256:
                raise ValueError(f"{field} belongs to a different checkpoint")
        if "vanilla_fit_sha256" in value and value["vanilla_fit_sha256"] != vanilla_fit_sha256:
            raise ValueError(f"{field} refers to a different vanilla fit")
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
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        # Windows lacks portable directory fsync; no-clobber linking remains
        # atomic, while crash durability of the directory entry is best effort.
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
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
    if identity[1] == 0:
        return
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if observed.st_ino != 0 and (observed.st_dev, observed.st_ino) == identity:
        path.unlink()
        _fsync_directory(path.parent)


def _publish_no_clobber(temporary: Path, destination: Path) -> tuple[int, int]:
    observed = temporary.stat(follow_symlinks=False)
    if observed.st_ino == 0:
        raise RuntimeError("candidate temporary file has no stable identity")
    identity = observed.st_dev, observed.st_ino
    os.link(temporary, destination, follow_symlinks=False)
    try:
        published = destination.stat(follow_symlinks=False)
        if published.st_ino == 0 or (published.st_dev, published.st_ino) != identity:
            raise RuntimeError("candidate destination changed during publication")
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
    version = raw.get("capture_manifest_schema_version")
    if type(version) is not int or version not in (4, 5, 6, 7, 8, 9, 10, 11):
        raise ValueError("unsupported capture manifest schema version")
    reference = version >= 6
    discovery = version % 2 == 1
    fields = (
        _CAPTURE_MANIFEST_FIELDS - capture_source_fields(reference=False)
    ) | capture_source_fields(reference=reference)
    if discovery:
        fields = (fields - {"compact_capture_file_sha256"}) | {
            "topology_discovery_capture_file_sha256",
            "capture_mode",
            "max_reuse_span",
        }
    if version >= 8:
        fields |= {"observed_kv_page_tokens"}
    if version >= 10:
        fields |= {"whole_prefill"}
    missing = fields - raw.keys()
    extra = raw.keys() - fields
    if missing or extra:
        raise ValueError(
            "capture manifest fields do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    expected = {
        "capture_manifest_schema_version": version if reference else 4,
        "capture_protocol": (
            "modelopt_vllm_mask_reuse_reference_target_sparsity_v1"
            if reference
            else "modelopt_vllm_mask_reuse_target_sparsity_v4"
        ),
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "compact_capture_file_sha256": compact_capture_sha256,
        "vanilla_config_file_sha256": vanilla_config_sha256,
    }
    if discovery:
        expected.pop("compact_capture_file_sha256")
        expected.update(
            capture_manifest_schema_version=version,
            capture_protocol="modelopt_vllm_mask_reuse_reference_topology_discovery_v1"
            if reference
            else "modelopt_vllm_mask_reuse_topology_discovery_v1",
            capture_mode="topology_discovery",
            topology_discovery_capture_file_sha256=compact_capture_sha256,
        )
        if type(raw["max_reuse_span"]) is not int or raw["max_reuse_span"] <= 0:
            raise ValueError("discovery capture requires a positive reuse span")
    for field, value in expected.items():
        if raw[field] != value:
            raise ValueError(f"capture manifest {field} does not match its verified input")
    validate_capture_source(raw, reference=reference)
    validate_manifest_counts(raw)
    if version >= 10:
        validate_whole_prefill_evidence(raw)
    if not isinstance(raw["engine_kwargs"], Mapping):
        raise ValueError("capture manifest engine_kwargs must be an object")
    if not isinstance(raw["dense_shadow_validation_requested"], bool):
        raise ValueError("capture manifest dense_shadow_validation_requested must be boolean")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed schema-v3 candidate from schema-v4 mask-reuse capture evidence"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, help="Optional external content manifest"
    )
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
    parser.add_argument("--calibration-plan", type=Path, help="Optional additional provenance JSON")
    parser.add_argument("--family-registry", type=Path, help="Optional additional provenance JSON")
    parser.add_argument("--grouped-fit", type=Path, help="Optional additional provenance JSON")
    parser.add_argument("--outer-report", type=Path, help="Optional additional provenance JSON")
    risk = parser.add_mutually_exclusive_group()
    risk.add_argument(
        "--max-anchor-dropped-mass", type=float, help="Legacy hard-anchor-cap formulation"
    )
    risk.add_argument(
        "--anchor-dropped-mass-report-threshold",
        type=float,
        help="Report-only diagnostic for anchor-inclusive risk; default 0.02",
    )
    parser.add_argument("--solver", choices=("cbc", "highs"), default="highs")
    parser.add_argument(
        "--diagnostic-final-chunk",
        action="store_true",
        help="Allow final-chunk evidence for diagnostics only",
    )
    parser.add_argument(
        "--allow-unmet-target",
        action="store_true",
        help="Export a diagnostic maximum-skip candidate when the target is infeasible",
    )
    parser.add_argument(
        "--benchmark-candidate",
        action="store_true",
        help="Export the best replayed work-feasible fixed-topology assignment even if optimality is unproved",
    )
    parser.add_argument(
        "--previous-candidate",
        type=Path,
        help="Replay this prior policy alongside new benchmark assignments",
    )
    parser.add_argument("--topology-sha256", help="Expected SHA-256 of the topology file")
    parser.add_argument("--solver-time-limit-seconds", type=float, default=600.0)
    parser.add_argument(
        "--reuse-dropped-mass-report-threshold",
        type=float,
        required=True,
        help="Diagnostic reuse threshold; does not affect selection",
    )
    parser.add_argument(
        "--target-bmm1-skip-ratio",
        type=float,
        required=True,
        help="Minimum model-wide BMM1 tile skip ratio required in every context bucket",
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
    if args.previous_candidate and not args.benchmark_candidate:
        parser.error("--previous-candidate requires --benchmark-candidate")
    if args.benchmark_candidate and (args.allow_unmet_target or args.diagnostic_final_chunk):
        parser.error("--benchmark-candidate requires whole-prefill captures and a met work target")
    if args.output_policy.resolve() == args.output_report.resolve():
        parser.error("--output-policy and --output-report must be different paths")
    for path in (args.output_policy, args.output_report):
        if path.exists() or path.is_symlink():
            parser.error(f"output already exists: {path}")

    try:
        checkpoint = verify_checkpoint_manifest(
            args.checkpoint, manifest_path=args.checkpoint_manifest
        )
        vanilla_config, vanilla_snapshot = _load_json_snapshot(
            args.vanilla_config, label="vanilla config"
        )
        topology, topology_snapshot = _load_json_snapshot(args.topology, label="topology")
        capture_manifest, capture_manifest_snapshot = _load_json_snapshot(
            args.capture_manifest, label="capture manifest"
        )
        evidence, evidence_paths = _evidence_artifacts(
            args, vanilla_fit_sha256=vanilla_snapshot.sha256, checkpoint_sha256=checkpoint.sha256
        )
        capture_manifest_sha256 = capture_manifest_snapshot.sha256
        topology_sha256 = topology_snapshot.sha256
        if args.topology_sha256 is not None and args.topology_sha256 != topology_sha256:
            raise ValueError("topology file SHA-256 does not match --topology-sha256")
        previous = None
        prior_provenance = {}
        donor_mode = "calibrated"
        if "topology_selection_schema_version" in topology:
            if (
                type(topology["topology_selection_schema_version"]) is not int
                or topology["topology_selection_schema_version"] != 1
                or topology.get("checkpoint_manifest_sha256") != checkpoint.sha256
                or topology.get("model") != checkpoint.model
                or topology.get("promotion_status") != "topology_candidate_only"
            ):
                raise ValueError("discovered topology does not match its schema or checkpoint")
            if not selection_target_met(topology) and not args.allow_unmet_target:
                raise ValueError("discovered topology did not meet the BMM1 skip target")
            validate_model_geometry(
                checkpoint, topology["attention_layers"], topology["global_num_heads"]
            )
            if sorted(int(layer) for layer in topology["nearest"]) != topology["attention_layers"]:
                raise ValueError("topology nearest mapping does not cover attention_layers")
            donor_mode = (
                topology["provenance"]
                .get("candidate_restrictions", {})
                .get("donor_head_mode", "calibrated")
                if args.benchmark_candidate
                else "calibrated"
            )
            discovery_evidence = topology.get("evidence")
            if (
                args.benchmark_candidate
                and not args.previous_candidate
                and isinstance(discovery_evidence, Mapping)
                and discovery_evidence.get("topology_discovery_capture_sha256")
                == evidence["reuse_bundle_sha256"]
            ):
                previous = {
                    **topology,
                    "context_policies": topology["discovery_context_policies"],
                    "evidence": {"reuse_bundle_sha256": evidence["reuse_bundle_sha256"]},
                }
                prior_provenance["previous_candidate_sha256"] = topology_sha256
            topology = {key: topology[key] for key in ("anchors", "nearest")}
        validate_capture_scope(capture_manifest, diagnostic=args.diagnostic_final_chunk)
        _validate_capture_manifest(
            capture_manifest,
            checkpoint_sha256=checkpoint.sha256,
            model=checkpoint.model,
            compact_capture_sha256=evidence["reuse_bundle_sha256"],
            vanilla_config_sha256=evidence["vanilla_fit_sha256"],
        )
        source = (
            load_compact_mask_reuse_captures(
                args.compact_captures,
                topology=topology,
                discovery_span=capture_manifest["max_reuse_span"],
            )
            if capture_manifest.get("capture_mode") == "topology_discovery"
            else load_compact_mask_reuse_captures(args.compact_captures)
        )
        whole_prefill = "whole_prefill" in capture_manifest
        if whole_prefill:
            validate_whole_prefill_captures(source, capture_manifest)
        first = next(iter(source), None)
        if first is None:
            raise ValueError("capture file is empty")
        validate_model_geometry(
            checkpoint, [int(layer) for layer in topology["nearest"]], first.global_num_heads
        )
        if args.previous_candidate:
            previous, snapshot = _load_json_snapshot(
                args.previous_candidate, label="prior candidate"
            )
            prior_provenance["previous_candidate_sha256"] = snapshot.sha256
        artifact = calibrate_compact_mask_reuse_policy(
            source,
            vanilla_calibration=vanilla_config,
            topology=topology,
            checkpoint_manifest=checkpoint,
            evidence=evidence,
            max_anchor_dropped_mass=args.max_anchor_dropped_mass,
            anchor_dropped_mass_report_threshold=getattr(
                args, "anchor_dropped_mass_report_threshold", None
            ),
            solver_config=MaskReuseSolverConfig(
                backend=getattr(args, "solver", "highs"),
                time_limit_seconds=getattr(args, "solver_time_limit_seconds", 600.0),
            ),
            reuse_dropped_mass_report_threshold=(args.reuse_dropped_mass_report_threshold),
            target_bmm1_skip_ratio=args.target_bmm1_skip_ratio,
            allow_same_group_prefixes=whole_prefill,
            benchmark_candidate=args.benchmark_candidate,
            previous_candidate=previous,
            donor_head_mode=donor_mode,
            source_provenance={
                **prior_provenance,
                "capture_manifest_sha256": capture_manifest_sha256,
                "topology_file_sha256": topology_sha256,
                "capture_provider": capture_source_provenance(capture_manifest),
                "capture_scope": "final_chunk_diagnostic"
                if args.diagnostic_final_chunk
                else "whole_prefill",
                **(
                    {
                        "whole_prefill_plan_sha256": sha256(
                            _canonical_json_bytes(capture_manifest["whole_prefill"])
                        ).hexdigest()
                    }
                    if "whole_prefill" in capture_manifest
                    else {}
                ),
            },
        )
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("calibrator returned no provenance object")
        if provenance.get("input_capture_count") != capture_manifest["capture_count"]:
            raise ValueError("calibrator capture count does not match capture manifest")
        source_cells = provenance.get("candidate_cell_count")
        if capture_manifest.get("capture_mode") == "topology_discovery":
            nearest = topology["nearest"]
            assert isinstance(nearest, Mapping)
            layers = len(nearest)
            span = capture_manifest["max_reuse_span"]
            assert isinstance(span, int)
            source_cells = (
                provenance["input_capture_count"]
                * first.global_num_heads**2
                * sum(min(span, i) for i in range(layers))
            )
            provenance["discovery_projection"] = {
                "source_candidate_cell_count": source_cells,
                "topology_sha256": topology_sha256,
                "source_capture_sha256": evidence["reuse_bundle_sha256"],
            }
        if source_cells != capture_manifest["candidate_cell_count"]:
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
        if (
            verify_checkpoint_manifest(args.checkpoint, manifest_path=checkpoint.manifest_path)
            != checkpoint
        ):
            raise ValueError("checkpoint changed during calibration")
    except MaskReuseSolverError as error:
        payload = _canonical_json_bytes(
            {"selection_status": "unresolved", "error": str(error), "solver": error.diagnostics}
        )
        try:
            _publish_no_clobber(_temporary_payload(args.output_report, payload), args.output_report)
        except OSError as publication_error:
            print(f"[ModelOpt] Could not publish unresolved report: {publication_error}")
        print(f"[ModelOpt] Selection unresolved; report: {args.output_report.resolve()}")
        return 3
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

    if not selection_target_met(artifact) and not args.allow_unmet_target:
        payload = _canonical_json_bytes(report)
        try:
            _publish_no_clobber(_temporary_payload(args.output_report, payload), args.output_report)
        except OSError as error:
            parser.error(f"could not publish infeasibility report: {error}")
        print(
            f"[ModelOpt] BMM1 target infeasible; no policy exported. Report: {args.output_report.resolve()}"
        )
        return 2

    try:
        policy_payload = _canonical_json_bytes(artifact)
        report_payload = _canonical_json_bytes(
            {**report, "candidate_policy_sha256": sha256(policy_payload).hexdigest()}
            if args.benchmark_candidate
            else report
        )
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
