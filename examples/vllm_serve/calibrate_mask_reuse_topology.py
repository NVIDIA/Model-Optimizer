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

"""Select a global mask-reuse topology from bounded discovery captures.

The output is a development-selected, held-out-evaluated topology candidate.
Freeze its ``anchors`` and ``nearest`` fields, then rerun the ordinary compact
collector/calibrator to obtain the final context-bucket policy.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    read_stable_file_snapshot,
    verify_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_topology import (
    calibrate_mask_reuse_topology,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_capture import (
    canonical_json_sha256,
    parse_vanilla_prefill_fit,
)

_MANIFEST_FIELDS = frozenset(
    {
        "capture_manifest_schema_version",
        "capture_protocol",
        "capture_mode",
        "model",
        "checkpoint_manifest_sha256",
        "checkpoint_manifest_path",
        "checkpoint_file_count",
        "checkpoint_total_size_bytes",
        "plan",
        "max_reuse_span",
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
        "topology_discovery_capture_file_sha256",
        "capture_count",
        "candidate_cell_count",
        "captures",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a development-only global mask-reuse topology"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--captures", required=True)
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--vanilla-config", required=True)
    parser.add_argument("--prompt-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-anchor-dropped-mass", type=float, required=True)
    parser.add_argument(
        "--reuse-dropped-mass-report-threshold",
        type=float,
        required=True,
        help="Diagnostic threshold for reported reuse violations; does not affect selection",
    )
    parser.add_argument(
        "--target-bmm1-skip-ratio",
        type=float,
        required=True,
        help=(
            "Minimum calibration-split BMM1 tiles skipped by reuse, divided by eligible "
            "BMM1 tiles across all attention layers"
        ),
    )
    return parser


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw


def _validate_manifest(
    raw: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    model: str,
    capture_sha256: str,
    vanilla_sha256: str,
    prompt_sha256: str,
    fit: Mapping[str, object],
) -> None:
    missing = _MANIFEST_FIELDS - raw.keys()
    extra = raw.keys() - _MANIFEST_FIELDS
    if missing or extra:
        raise ValueError(
            "topology capture manifest fields differ; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    expected = {
        "capture_manifest_schema_version": 5,
        "capture_protocol": "modelopt_vllm_mask_reuse_topology_discovery_v1",
        "capture_mode": "topology_discovery",
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "topology_discovery_capture_file_sha256": capture_sha256,
        "vanilla_config_file_sha256": vanilla_sha256,
        "prompt_plan_file_sha256": prompt_sha256,
        "vanilla_fit_sha256": canonical_json_sha256(fit),
    }
    for field, value in expected.items():
        if raw[field] != value:
            raise ValueError(f"topology capture manifest {field} does not match its verified input")
    if not isinstance(raw["plan"], str) or not raw["plan"].endswith("_topology_discovery"):
        raise ValueError("topology capture manifest plan is not a discovery preset")
    max_reuse_span = raw["max_reuse_span"]
    if (
        isinstance(max_reuse_span, bool)
        or not isinstance(max_reuse_span, int)
        or max_reuse_span <= 0
    ):
        raise ValueError("topology capture manifest max_reuse_span must be positive")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _publish_no_clobber(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> Path:
    captures_path = Path(args.captures)
    manifest_path = Path(args.capture_manifest)
    vanilla_path = Path(args.vanilla_config)
    prompt_path = Path(args.prompt_plan)
    output_path = Path(args.output)
    capture_snapshot = read_stable_file_snapshot(captures_path, label="topology discovery captures")
    manifest_snapshot = read_stable_file_snapshot(manifest_path, label="topology capture manifest")
    vanilla_snapshot = read_stable_file_snapshot(vanilla_path, label="vanilla calibration")
    prompt_snapshot = read_stable_file_snapshot(prompt_path, label="prompt plan")
    checkpoint = verify_checkpoint_manifest(args.checkpoint)
    fit = parse_vanilla_prefill_fit(vanilla_snapshot.payload)
    manifest = _json_object(manifest_snapshot.payload, label="topology capture manifest")
    _validate_manifest(
        manifest,
        checkpoint_sha256=checkpoint.sha256,
        model=checkpoint.model,
        capture_sha256=capture_snapshot.sha256,
        vanilla_sha256=vanilla_snapshot.sha256,
        prompt_sha256=prompt_snapshot.sha256,
        fit=fit,
    )
    result = calibrate_mask_reuse_topology(
        captures_path,
        vanilla_calibration=fit,
        checkpoint_manifest=checkpoint,
        evidence={
            "topology_discovery_capture_sha256": capture_snapshot.sha256,
            "vanilla_fit_sha256": str(manifest["vanilla_fit_sha256"]),
            "prompt_plan_sha256": prompt_snapshot.sha256,
        },
        max_anchor_dropped_mass=args.max_anchor_dropped_mass,
        reuse_dropped_mass_report_threshold=args.reuse_dropped_mass_report_threshold,
        target_bmm1_skip_ratio=args.target_bmm1_skip_ratio,
    )
    final_inputs = (
        read_stable_file_snapshot(captures_path, label="topology discovery captures"),
        read_stable_file_snapshot(manifest_path, label="topology capture manifest"),
        read_stable_file_snapshot(vanilla_path, label="vanilla calibration"),
        read_stable_file_snapshot(prompt_path, label="prompt plan"),
    )
    if tuple(item.sha256 for item in final_inputs) != (
        capture_snapshot.sha256,
        manifest_snapshot.sha256,
        vanilla_snapshot.sha256,
        prompt_snapshot.sha256,
    ):
        raise RuntimeError("a topology calibration input changed; result was discarded")
    _publish_no_clobber(output_path, _canonical_json_bytes(result))
    print(f"[ModelOpt] Wrote topology candidate to {output_path.resolve()}")
    print(f"[ModelOpt] anchors={result['anchors']}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
