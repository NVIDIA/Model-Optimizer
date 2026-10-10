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

"""Calibrate skip-softmax thresholds *through vLLM* and write the serving config.

Runs calibration prompts through a vLLM ``LLM`` whose attention layers carry
the ModelOpt calibration adapters (installed by
``sparse_attn_worker.SkipSoftmaxCalibWorker`` via
``install_vllm_skip_softmax_calibration``). The paged Triton calibration
kernel measures, per candidate threshold, how many KV tiles would be skipped —
over the paged KV cache, for both prefill and decode — then this driver
aggregates the raw counts from every tensor-parallel rank and fits the
exponential model ``scale_factor = a * exp(b * sparsity)`` once per phase.

The fit initializes a bounded search using actual serving-kernel tile counters.
Export requires calibration and held-out sparsity to meet the requested
tolerance for every requested phase. Refinement adjusts ``a`` at the selected
target; other targets and serving schedules require separate validation.
The canonical export schema and existing N:M groups are preserved.

Usage:
    python calibrate_sparse_attn.py <ckpt> \
        --calib_data_dir <ruler-data-dir> \
        --validation_prompts_file held_out.txt \
        --target_sparse_ratio 0.5 \
        --decode_tokens 32 \
        --update_checkpoint_config

Calibration prompts default to the RULER dataset — the same
``RulerDatasetBuilder`` the PyTorch (HF) calibration path uses — so both paths
calibrate on identical data. NIAH tasks need the essay haystack downloaded by
``examples/llm_sparsity/attention_sparsity/download_ruler_data.sh`` (point
``--calib_data_dir`` at its ``data`` directory). ``--prompts_file`` (one prompt
per line) overrides the RULER set with custom calibration data.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

from modelopt.torch.sparsity.attention_sparsity.calibration.ruler_dataset import RulerDatasetBuilder
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_calibration import (
    DEFAULT_THRESHOLD_TRIALS,
    build_sparse_attention_config,
    fit_from_counts,
    merge_phase_counts,
)

_LOCKED_ENGINE_KWARGS = frozenset(
    {
        "model",
        "worker_cls",
        "enforce_eager",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "max_num_seqs",
        "seed",
    }
)


def _sparse_ratio(value: str) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite value between 0.0 and 1.0")
    return ratio


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _engine_kwargs(value: str) -> dict:
    try:
        kwargs = json.loads(value)
    except json.JSONDecodeError as err:
        raise argparse.ArgumentTypeError(f"must be a JSON object: {err.msg}") from err
    if not isinstance(kwargs, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    if locked := sorted(_LOCKED_ENGINE_KWARGS & kwargs.keys()):
        raise argparse.ArgumentTypeError(
            "cannot override calibration-controlled option(s): " + ", ".join(locked)
        )
    if kwargs.get("pipeline_parallel_size", 1) != 1:
        raise argparse.ArgumentTypeError("pipeline_parallel_size must be 1 for calibration")
    if kwargs.get("data_parallel_size", 1) != 1:
        raise argparse.ArgumentTypeError("data_parallel_size must be 1 for calibration")
    return kwargs


def _load_prompts(llm, args) -> list[str]:
    """Load override prompts from a file, or build the default RULER set."""
    if args.prompts_file is not None:
        lines = [
            ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()
        ]
        if not lines:
            raise ValueError(f"No prompts found in {args.prompts_file}")
        print(f"[ModelOpt] Loaded {len(lines)} calibration prompts from {args.prompts_file}")
        return lines

    # Same dataset as the HF calibration path (calibration/calibrate.py), so the
    # vLLM- and PyTorch-calibrated thresholds are fit on identical data.
    builder = RulerDatasetBuilder(
        samples=args.calib_samples,
        max_seqlen=args.calib_max_seqlen,
        tokenizer_name_or_path=llm.get_tokenizer(),
        max_length_filter=int(args.calib_max_seqlen * 1.5),
        data_dir=args.calib_data_dir,
    )
    samples = builder.build_calibration_dataset()
    if not samples:
        raise ValueError(
            "RULER produced no calibration samples (all candidates exceeded "
            f"max_length_filter={int(args.calib_max_seqlen * 1.5)} tokens). "
            "Adjust --calib_max_seqlen / --calib_samples, or pass --prompts_file."
        )
    prompts = [sample["input"] for sample in samples]
    lengths = sorted(sample["length"] for sample in samples)
    print(
        f"[ModelOpt] Built {len(prompts)} RULER calibration prompts "
        f"(token lengths {lengths[0]}..{lengths[-1]})"
    )
    return prompts


def _preflight_prompt_inputs(args, parser: argparse.ArgumentParser) -> list[str] | None:
    """Validate prompt sources before the vLLM engine is initialized."""
    if args.prompts_file is not None:
        try:
            return _load_prompts(None, args)
        except (OSError, ValueError) as err:
            parser.error(str(err))
    if args.calib_data_dir is None:
        parser.error(
            "the default RULER tasks require --calib_data_dir; pass --prompts_file "
            "to supply custom prompts instead"
        )
    data_dir = Path(args.calib_data_dir)
    if not data_dir.is_dir():
        parser.error(f"--calib_data_dir {args.calib_data_dir!r} is not a directory")
    essays_dir = data_dir / "essays"
    if not essays_dir.is_dir() or next(essays_dir.glob("*.txt"), None) is None:
        parser.error(
            f"--calib_data_dir {args.calib_data_dir!r} must contain essays/*.txt; "
            "run examples/llm_sparsity/attention_sparsity/download_ruler_data.sh first"
        )
    return None


def _existing_sparse_config(ckpt: str) -> dict | None:
    """Read the checkpoint's sparse_attention_config so non-skip groups survive."""
    config_json = Path(ckpt) / "config.json"
    if not config_json.is_file():
        return None
    existing = json.loads(config_json.read_text()).get("sparse_attention_config")
    return existing if isinstance(existing, dict) else None


def _write_config(
    ckpt: str, sparse_config: dict, update_checkpoint: bool, output_dir: Path = Path(".")
) -> None:
    """Dump the sparse_attention_config and optionally merge into config.json."""
    out_path = output_dir / "sparse_attention_config.json"
    out_path.write_text(json.dumps(sparse_config, indent=2))
    print(f"[ModelOpt] Wrote calibrated config to {out_path.resolve()}")

    if not update_checkpoint:
        print(
            "[ModelOpt] Checkpoint not modified. Merge the generated configuration as "
            f"'sparse_attention_config' in {ckpt}/config.json before serving. On future "
            "calibration runs, pass --update_checkpoint_config to do this automatically."
        )
        return

    config_json = Path(ckpt) / "config.json"
    config = json.loads(config_json.read_text())
    config["sparse_attention_config"] = sparse_config
    # Atomic replace: a crash mid-write must not truncate the checkpoint's
    # config.json (write_text would rewrite it in place).
    tmp_path = config_json.with_name(config_json.name + ".tmp")
    tmp_path.write_text(json.dumps(config, indent=2))
    os.replace(tmp_path, config_json)
    print(f"[ModelOpt] Merged sparse_attention_config into {config_json}")


def _summarize_validation(
    worker_stats: list[dict], target: float, tolerance: float, phases=("prefill", "decode")
) -> dict:
    """Combine integer counts, failing closed on missing ranks, phases, or counters."""
    result = {}
    for phase in phases:
        records = [worker.get(phase, {}) for worker in worker_stats]
        total = sum(r.get("total", 0) for r in records)
        skipped = sum(r.get("skipped", 0) for r in records)
        complete = bool(records) and all(
            r.get("launches", 0) > 0
            and r.get("total", 0) > 0
            and r.get("unmeasured_launches", 0) == 0
            and 0 <= r.get("skipped", -1) <= r["total"]
            for r in records
        )
        achieved = skipped / total if complete else None
        result[phase] = {
            "target": target,
            "total": total,
            "skipped": skipped,
            "achieved": achieved,
            "error_pp": 100 * (achieved - target) if complete else None,
            "min_seq_len": min((r.get("min_seq_len", 0) for r in records), default=0),
            "complete": complete,
            "passed": complete and target - tolerance <= achieved <= target + tolerance,
        }
    return result


def _refine_scales(config: dict, measure, max_steps: int) -> tuple[dict, dict]:
    """Refine prefill then decode, retaining the best measured calibration candidate."""
    config = deepcopy(config)
    group = config["config_groups"]["group_0"]
    measured = measure(config)
    for phase in ("prefill", "decode"):
        if phase not in measured:
            continue
        params = group["threshold_scale_factor"][phase]
        target = group["target_sparsity"][phase]
        multiplier = math.exp(params["b"] * target)
        best_a = params["a"]
        best_error = float("inf")
        lower = 0.0
        # lambda >= 1 disables skipping in the serving resolver. Stay inside
        # that domain for every request, including the shortest one.
        upper = measured[phase]["min_seq_len"] * (1.0 - 1e-6)
        bracketed = False
        for step in range(max_steps + 1):
            stats = measured[phase]
            if not stats["complete"]:
                if step < max_steps and upper > 0 and params["a"] * multiplier >= upper:
                    params["a"] = upper / multiplier
                    measured = measure(config)
                    continue
                break
            error = abs(stats["error_pp"])
            if error < best_error:
                best_error, best_a = error, params["a"]
            if stats["passed"] or step == max_steps:
                break
            scale = params["a"] * multiplier
            if stats["achieved"] < target:
                lower = scale
            else:
                upper, bracketed = scale, True
            candidate = (lower + upper) / 2 if bracketed else min(scale * 4, upper)
            if candidate <= 0 or math.isclose(candidate, scale, rel_tol=1e-6):
                break
            params["a"] = candidate / multiplier
            measured = measure(config)
        if params["a"] != best_a:
            params["a"] = best_a
            measured = measure(config)
    return config, measured


def _export_validated_config(
    model: str, config: dict, report: dict, output_dir: Path, update_checkpoint: bool
) -> None:
    """Always retain the receipt; never export a candidate that failed the gate."""
    passed = all(
        report.get(split, {}).get(phase, {}).get("passed", False)
        for split in ("calibration", "held_out")
        for phase in report.get("phases", ("prefill", "decode"))
    )
    report.update({"passed": passed, "candidate_config": config})
    report_path = output_dir / "sparse_attention_validation.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError(
            f"Sparsity validation failed; no config exported or checkpoint updated. "
            f"See {report_path} for measured counts and every candidate."
        )
    _write_config(model, config, update_checkpoint, output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate skip-softmax thresholds via vLLM")
    parser.add_argument("model", type=str, help="Path to the HF checkpoint to calibrate")
    parser.add_argument(
        "--validation_prompts_file", required=True, help="Disjoint held-out prompts, one per line"
    )
    parser.add_argument(
        "--sparsity_tolerance",
        type=_sparse_ratio,
        default=0.02,
        help="Absolute ratio tolerance for each phase (0.02 = two percentage points)",
    )
    parser.add_argument(
        "--max_refinement_steps",
        type=_nonnegative_int,
        default=10,
        help="Maximum search steps per phase on calibration prompts",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Optional custom calibration prompts (one per line), overriding the "
        "default RULER dataset",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=24,
        help="Total RULER samples, distributed across length bins (HF-path default: 24)",
    )
    parser.add_argument(
        "--calib_max_seqlen",
        type=int,
        default=32768,
        help="Maximum RULER sequence length; length bins descend in powers of 2. "
        "Must fit within --max_model_len together with --decode_tokens.",
    )
    parser.add_argument(
        "--calib_data_dir",
        type=str,
        default=None,
        help="RULER data directory containing the 'essays' haystack (populated by "
        "examples/llm_sparsity/attention_sparsity/download_ruler_data.sh)",
    )
    parser.add_argument(
        "--target_sparse_ratio",
        type=_sparse_ratio,
        default=0.5,
        help="Target sparsity baked into the exported config (applied to both phases)",
    )
    parser.add_argument(
        "--decode_tokens",
        type=_nonnegative_int,
        default=32,
        help="Decode attention steps per prompt (drives decode-phase calibration). "
        "Generation runs decode_tokens + 1 output tokens: the first output token "
        "comes from the prefill forward and performs no decode attention.",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None, help="vLLM max_model_len override"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="vLLM tensor-parallel size"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization fraction",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for custom model classes (e.g. NemotronH)",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype, e.g. bfloat16")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        help="Force the vLLM attention backend, e.g. FLASH_ATTN or FLASHINFER. "
        "Default: let vLLM choose (the installer supports whichever of FlashAttention "
        "/ FlashInfer is selected).",
    )
    parser.add_argument(
        "--engine_kwargs",
        type=_engine_kwargs,
        default=None,
        help="JSON dict of extra vLLM engine kwargs, e.g. "
        '\'{"enable_expert_parallel": true, "mamba_cache_mode": "align"}\' '
        "for hybrid MoE/Mamba models",
    )
    parser.add_argument(
        "--fit_logspace",
        action="store_true",
        help="Fit the exponential model in log space (wide scale_factor ranges)",
    )
    parser.add_argument(
        "--update_checkpoint_config",
        action="store_true",
        help="Merge the calibrated config into <ckpt>/config.json in place",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.update_checkpoint_config and not (Path(args.model) / "config.json").is_file():
        # Fail before the (expensive, multi-GPU) calibration run, not after:
        # merging requires a local checkpoint directory, not a HF hub ID.
        parser.error(
            f"--update_checkpoint_config requires a local checkpoint directory "
            f"containing config.json; {args.model!r} has none"
        )

    # Custom prompts do not need a tokenizer, so read them eagerly as well.
    prompts = _preflight_prompt_inputs(args, parser)
    try:
        held_out = _load_prompts(
            None, argparse.Namespace(prompts_file=args.validation_prompts_file)
        )
    except (OSError, ValueError) as err:
        parser.error(str(err))
    if not 0 < args.target_sparse_ratio < 1:
        parser.error("--target_sparse_ratio must be strictly between 0 and 1 for refinement")
    if not 0 < args.sparsity_tolerance < 1:
        parser.error("--sparsity_tolerance must be strictly between 0 and 1")
    if prompts is not None and set(prompts) & set(held_out):
        parser.error("Calibration and held-out prompt files must not overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (args.output_dir / name).exists()
        for name in ("sparse_attention_config.json", "sparse_attention_validation.json")
    ):
        parser.error("--output_dir already contains calibration artifacts; use a fresh directory")

    # Workers run in separate processes and must import the calibration worker.
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    current = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([current, repo_root]) if current else repo_root

    # Deferred heavy import: keep argparse/--help (and arg errors) fast, and
    # only import vLLM after the PYTHONPATH setup above.
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "worker_cls": "sparse_attn_worker.SkipSoftmaxCalibWorker",
        # The calibration installer requires eager execution: the per-request
        # calibration loop cannot be CUDA-graph captured.
        "enforce_eager": True,
        # Shared-prefix reuse would make prefill measurements cover only the
        # non-cached suffix of each prompt; the installer rejects it.
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": 1,
        "seed": 0,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.tensor_parallel_size and args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.dtype is not None:
        llm_kwargs["dtype"] = args.dtype
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.engine_kwargs:
        llm_kwargs.update(args.engine_kwargs)
    llm = LLM(**llm_kwargs)

    # Built after engine init so the RULER builder reuses the engine's tokenizer.
    if prompts is None:
        prompts = _load_prompts(llm, args)
    if set(prompts) & set(held_out):
        parser.error("Calibration and held-out prompts must not overlap")

    trials = list(DEFAULT_THRESHOLD_TRIALS)
    n_layers = llm.collective_rpc("sparse_calib_enable", args=(trials,))[0]
    status = llm.collective_rpc("sparse_calib_status")[0]
    print(f"[ModelOpt] Calibration enabled on {n_layers} attention layers")
    print(f"[ModelOpt] Active sparse impls: {status['impl_types']}")

    # generate() drives prefill (prefill-phase stats) then decode steps
    # (decode-phase stats). No sparsification is applied during calibration —
    # the kernel computes full dense attention while recording tile-skip
    # counts. ignore_eos forces the full decode length so early EOS cannot
    # thin the decode-phase statistics. max_tokens is decode_tokens + 1: the
    # first output token comes from the prefill forward, so decode_tokens
    # decode-attention steps need one extra output token.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens + 1, ignore_eos=True)
    llm.generate(prompts, sampling)

    # Aggregate RAW counts from every TP rank (each rank only measures its
    # attention-head shard), then fit once per phase on the global counts.
    rank_counts = llm.collective_rpc("sparse_calib_counts")
    merged = merge_phase_counts(rank_counts)
    calibration_params = fit_from_counts(merged, trials, fit_logspace=args.fit_logspace)

    requested_phases = ["prefill"] + (["decode"] if args.decode_tokens > 0 else [])
    missing = [phase for phase in requested_phases if phase not in calibration_params]
    if missing:
        print(
            f"[ModelOpt] Calibration FAILED: no valid fit for phase(s) {', '.join(missing)}. "
            "No config was written — a partially calibrated export would silently serve "
            "the missing phase dense. Try more/longer prompts (and more decode tokens) "
            "so observed sparsity spans the (10%, 90%) fitting window."
        )
        sys.exit(1)
    # Export only requested phases: a stray record (e.g. a scheduling corner
    # case classified into an unrequested phase) must not bake an
    # uncalibrated-by-intent phase into the config.
    calibration_params = {
        phase: params for phase, params in calibration_params.items() if phase in requested_phases
    }

    sparse_config = build_sparse_attention_config(
        calibration_params,
        {"prefill": args.target_sparse_ratio, "decode": args.target_sparse_ratio},
        existing_config=_existing_sparse_config(args.model),
    )
    print("[ModelOpt] Calibrated threshold_scale_factor:")
    print(json.dumps(sparse_config["config_groups"]["group_0"]["threshold_scale_factor"], indent=2))
    report = {
        "model": args.model,
        "engine_args": llm_kwargs,
        "decode_tokens": args.decode_tokens,
        "phases": requested_phases,
        "target": args.target_sparse_ratio,
        "tolerance": args.sparsity_tolerance,
        "refinement": {"max_steps_per_phase": args.max_refinement_steps, "expansion_factor": 4},
        "metric": "sum(skipped_tiles) / sum(eligible_tiles), across selected layers and TP ranks",
        "scope": "Single-sequence serving; refined coefficients validated only at this target",
        "prompts": {
            split: {
                "count": len(values),
                "sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest(),
            }
            for split, values in (("calibration", prompts), ("held_out", held_out))
        },
        "initial_config": deepcopy(sparse_config),
        "history": [],
    }

    def measure(config, *, split="calibration"):
        layers = llm.collective_rpc("sparse_validation_enable", args=(config,))
        if not layers or any(n <= 0 for n in layers):
            raise RuntimeError("No selected sparse attention layers on a worker; cannot validate")
        llm.generate(prompts if split == "calibration" else held_out, sampling)
        counters = llm.collective_rpc("sparse_validation_collect")
        stats = _summarize_validation(
            counters, args.target_sparse_ratio, args.sparsity_tolerance, requested_phases
        )
        report["history"].append(
            {"split": split, "config": deepcopy(config), "workers": counters, "measured": stats}
        )
        for phase, row in stats.items():
            actual = f"{row['achieved']:.2%}" if row["complete"] else "UNMEASURED"
            print(
                f"[ModelOpt] {split} {phase}: target={args.target_sparse_ratio:.2%}, "
                f"achieved={actual}, skipped={row['skipped']}/{row['total']}, "
                f"passed={row['passed']}",
                flush=True,
            )
        (args.output_dir / "sparse_attention_validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        return stats

    sparse_config, report["calibration"] = _refine_scales(
        sparse_config, measure, args.max_refinement_steps
    )
    # Do not consume held-out data until calibration passes, or feed its results
    # back into the threshold search.
    if all(row["passed"] for row in report["calibration"].values()):
        report["held_out"] = measure(sparse_config, split="held_out")
    else:
        report["held_out_status"] = "not_run_calibration_failed"
    _export_validated_config(
        args.model, sparse_config, report, args.output_dir, args.update_checkpoint_config
    )


if __name__ == "__main__":
    main()
