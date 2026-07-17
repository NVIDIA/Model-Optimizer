#!/usr/bin/env python3
"""ModelOpt-backed synthetic check of FP8 scale granularity under outliers.

This is deliberately a toy numerical sanity check, not a Qwen3.6 measurement. It uses
ModelOpt's FP8QTensor and MXFP8QTensor implementations so the E4M3/E8M0 mappings match the
study code rather than a hand-written FP8 approximation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from modelopt.torch.quantization.qtensor.fp8_tensor import FP8QTensor
from modelopt.torch.quantization.qtensor.mxfp8_tensor import MXFP8QTensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "theory" / "toy_scale_sweep.json"


def quantize_fp8(tensor: torch.Tensor, block_sizes: dict[int, int] | None = None) -> torch.Tensor:
    quantized, scale = FP8QTensor.quantize(tensor, block_sizes=block_sizes)
    return quantized.dequantize(scale=scale, block_sizes=block_sizes, dtype=torch.float32)


def quantize_mxfp8(tensor: torch.Tensor) -> torch.Tensor:
    quantized, scale = MXFP8QTensor.quantize(tensor)
    return quantized.dequantize(scale=scale, dtype=torch.float32)


def distribution_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_logp = torch.log_softmax(reference, dim=-1)
    candidate_logp = torch.log_softmax(candidate, dim=-1)
    reference_p = reference_logp.exp()
    candidate_p = candidate_logp.exp()
    midpoint = 0.5 * (reference_p + candidate_p)
    midpoint_log = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    forward_kl = (reference_p * (reference_logp - candidate_logp)).sum(dim=-1).mean()
    reverse_kl = (candidate_p * (candidate_logp - reference_logp)).sum(dim=-1).mean()
    js = 0.5 * (
        (reference_p * (reference_logp - midpoint_log)).sum(dim=-1).mean()
        + (candidate_p * (candidate_logp - midpoint_log)).sum(dim=-1).mean()
    )
    centered_reference = reference - reference.mean(dim=-1, keepdim=True)
    centered_candidate = candidate - candidate.mean(dim=-1, keepdim=True)
    normalized_mse = torch.mean((centered_candidate - centered_reference).square()) / (
        torch.mean(centered_reference.square()) + torch.finfo(torch.float32).eps
    )
    return {
        "mse": torch.mean((candidate - reference).square()).item(),
        "normalized_mse": normalized_mse.item(),
        "kl_forward": forward_kl.item(),
        "kl_reverse": reverse_kl.item(),
        "js": js.item(),
        "top1_agreement": (reference.argmax(dim=-1) == candidate.argmax(dim=-1))
        .float()
        .mean()
        .item(),
    }


def make_tensor(rows: int, columns: int, outlier_multiplier: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.randn(rows, columns, generator=generator, dtype=torch.float32)
    # Place one positive or negative outlier in a random block of each row. Every method sees
    # the same tensor; smaller scale domains isolate the damage to fewer ordinary values.
    positions = torch.randint(columns, (rows,), generator=generator)
    signs = torch.where(torch.rand(rows, generator=generator) < 0.5, -1.0, 1.0)
    tensor[torch.arange(rows), positions] = signs * outlier_multiplier
    return tensor


def run(rows: int, columns: int, multipliers: list[float], seed: int) -> dict:
    if columns % 128:
        raise ValueError("--columns must be divisible by 128 (and therefore by MXFP8 block 32)")
    sweeps = []
    for multiplier in multipliers:
        reference = make_tensor(rows, columns, multiplier, seed)
        candidates = {
            "per_tensor_fp8": quantize_fp8(reference),
            "block128_full_precision_scale": quantize_fp8(reference, {-1: 128}),
            "mxfp8_block32_e8m0_scale": quantize_mxfp8(reference),
        }
        sweeps.append(
            {
                "outlier_multiplier": multiplier,
                "metrics": {
                    name: distribution_metrics(reference, candidate)
                    for name, candidate in candidates.items()
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_modelopt_scale_granularity_sanity_check",
        "is_qwen_result": False,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation": {
            "per_tensor_and_block": "modelopt.torch.quantization.qtensor.FP8QTensor",
            "mxfp8": "modelopt.torch.quantization.qtensor.MXFP8QTensor",
            "torch_version": torch.__version__,
            "shape": [rows, columns],
            "seed": seed,
            "construction": "standard-normal values with one signed outlier per row",
        },
        "interpretation": (
            "For one instantaneous tensor, static versus dynamic full-precision block scaling "
            "has the same numerical map; calibration across tensors is the additional distinction."
        ),
        "sweeps": sweeps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[1, 16, 256, 4096, 65536, 1048576],
        help="Signed outlier magnitudes in standard-deviation units (default is a stress sweep)",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.rows <= 0
        or args.columns <= 0
        or not all(math.isfinite(x) and x > 0 for x in args.multipliers)
    ):
        raise ValueError("rows, columns, and all outlier multipliers must be positive")
    payload = run(args.rows, args.columns, args.multipliers, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
