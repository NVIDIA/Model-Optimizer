# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare vendor-runtime and vLLM DeepSeek-V4 native-MTP feature dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

FEATURE_KEYS = ("target_mtp_hidden_states", "target_lm_head_hidden_states")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Maximum matching samples to compare; 0 compares every match.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_elements: int = 1_000_000,
) -> dict[str, float]:
    """Compute stable global error metrics without materializing full FP32 copies."""
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    if reference.dtype != candidate.dtype:
        raise TypeError(f"dtype mismatch: {reference.dtype} != {candidate.dtype}")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    if not reference.is_floating_point():
        raise TypeError(f"feature tensors must be floating point, got {reference.dtype}")

    reference = reference.reshape(-1)
    candidate = candidate.reshape(-1)
    dot = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    difference_squared = 0.0
    max_absolute = 0.0
    for start in range(0, reference.numel(), chunk_elements):
        stop = min(start + chunk_elements, reference.numel())
        ref_chunk = reference[start:stop].float()
        candidate_chunk = candidate[start:stop].float()
        difference = candidate_chunk - ref_chunk
        dot += float(torch.dot(ref_chunk, candidate_chunk))
        reference_squared += float(torch.dot(ref_chunk, ref_chunk))
        candidate_squared += float(torch.dot(candidate_chunk, candidate_chunk))
        difference_squared += float(torch.dot(difference, difference))
        max_absolute = max(max_absolute, float(difference.abs().max()))

    denominator = math.sqrt(reference_squared * candidate_squared)
    cosine = dot / denominator if denominator else float(reference_squared == candidate_squared)
    relative_l2 = (
        math.sqrt(difference_squared / reference_squared)
        if reference_squared
        else math.sqrt(difference_squared)
    )
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_absolute": max_absolute,
    }


def _files_by_name(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in directory.rglob("*.pt"):
        if path.name in files:
            raise ValueError(f"Duplicate feature basename under {directory}: {path.name}")
        files[path.name] = path
    if not files:
        raise ValueError(f"No .pt feature dumps found under {directory}")
    return files


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {path}")
    return payload


def main(args: argparse.Namespace) -> None:
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if not -1 <= args.min_cosine <= 1:
        raise ValueError("--min-cosine must be in [-1, 1]")
    if args.max_relative_l2 < 0:
        raise ValueError("--max-relative-l2 must be non-negative")

    reference_files = _files_by_name(args.reference_dir)
    candidate_files = _files_by_name(args.candidate_dir)
    matching_names = sorted(reference_files.keys() & candidate_files.keys())
    if not matching_names:
        raise ValueError("Reference and candidate directories have no matching .pt basenames")
    if args.max_samples:
        matching_names = matching_names[: args.max_samples]

    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in matching_names:
        reference = _load(reference_files[name])
        candidate = _load(candidate_files[name])
        sample_report: dict[str, Any] = {"file": name, "features": {}}

        for key in ("input_ids", "loss_mask"):
            if key not in reference or key not in candidate:
                failures.append(f"{name}: missing {key}")
            elif not torch.equal(reference[key], candidate[key]):
                failures.append(f"{name}: {key} mismatch")

        for key in FEATURE_KEYS:
            if key not in reference or key not in candidate:
                failures.append(f"{name}: missing {key}")
                continue
            try:
                metrics = tensor_error_metrics(
                    reference[key],
                    candidate[key],
                    chunk_elements=args.chunk_elements,
                )
            except (TypeError, ValueError) as error:
                failures.append(f"{name}: {key}: {error}")
                continue
            sample_report["features"][key] = metrics
            if not math.isfinite(metrics["cosine"]) or not math.isfinite(metrics["relative_l2"]):
                failures.append(f"{name}: {key}: non-finite metrics")
            elif metrics["cosine"] < args.min_cosine:
                failures.append(
                    f"{name}: {key}: cosine {metrics['cosine']:.8f} < {args.min_cosine}"
                )
            elif metrics["relative_l2"] > args.max_relative_l2:
                failures.append(
                    f"{name}: {key}: relative_l2 {metrics['relative_l2']:.8f} "
                    f"> {args.max_relative_l2}"
                )
        samples.append(sample_report)

    report = {
        "reference_dir": str(args.reference_dir),
        "candidate_dir": str(args.candidate_dir),
        "reference_files": len(reference_files),
        "candidate_files": len(candidate_files),
        "compared_samples": len(samples),
        "min_cosine": args.min_cosine,
        "max_relative_l2": args.max_relative_l2,
        "failures": failures,
        "samples": samples,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if failures:
        raise RuntimeError(f"DSV4 MTP parity validation failed with {len(failures)} errors")


if __name__ == "__main__":
    main(_parse_args())
