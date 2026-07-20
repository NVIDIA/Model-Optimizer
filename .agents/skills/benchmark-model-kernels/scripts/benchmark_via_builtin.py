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

"""Run FlashInfer's built-in GEMM and fused-MoE microbenchmarks.

Plain rows contain kernel time. Most ``*_with_quant`` rows add a separately
measured activation-quantization time; NVFP4 MoE with quantization is measured
directly. Logical shapes label each case while backend-specific physical
padding follows vLLM. A local FlashInfer source checkout is required for its
benchmark driver and utilities.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    from vllm import _custom_ops as vllm_ops
except ImportError:
    vllm_ops = None

_ERROR_CASE_PREFIX = "[ERROR] Error running test:"
_ERROR_MESSAGE_PREFIX = "[ERROR] Error:"
_FP8_QUANT_UNAVAILABLE = "ERROR: vLLM is unavailable for FP8 activation quantization"
_MOE_ACTIVATIONS = (
    "Gelu",
    "Relu",
    "Silu",
    "Swiglu",
    "Geglu",
    "SwigluBias",
    "Relu2",
    "SwigluStep",
    "Identity",
)
_MOE_ROUTINGS = ("renormalize", "deepseek_v3", "llama4", "renormalize_naive", "topk")

_ResultValue = float | str


@dataclass
class _Case:
    section: str
    tag: str
    key: str
    argv: list[str]
    quant: tuple[str, int, int] | None = None


def _index_cases(cases: list[_Case]) -> dict[str, _Case]:
    indexed = {}
    for case in cases:
        if case.tag in indexed:
            raise RuntimeError(f"duplicate benchmark case tag: {case.tag}")
        indexed[case.tag] = case
    return indexed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")
    return parsed


def _nk_pair(value: str) -> tuple[int, int]:
    try:
        n, k = value.split(",")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive N,K pair, got {value!r}") from exc
    try:
        return _positive_int(n), _positive_int(k)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive N,K pair, got {value!r}") from exc


def _named_nks(
    nks: list[tuple[int, int]], names: list[str] | None
) -> tuple[list[tuple[int, int]], list[str] | None]:
    if names is not None and len(names) != len(nks):
        raise ValueError("--nk_names must contain exactly one name for each --nks pair")

    unique_nks = list(dict.fromkeys(nks))
    if names is None:
        return unique_nks, None

    names_by_nk: dict[tuple[int, int], list[str]] = {}
    for nk, name in zip(nks, names, strict=True):
        labels = names_by_nk.setdefault(nk, [])
        if name not in labels:
            labels.append(name)
    return unique_nks, ["/".join(names_by_nk[nk]) for nk in unique_nks]


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _parse_driver_errors(lines: list[str]) -> dict[str, str]:
    errors = {}
    pending_tag = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(_ERROR_CASE_PREFIX):
            command = stripped.removeprefix(_ERROR_CASE_PREFIX).strip()
            try:
                argv = shlex.split(command)
                tag_index = argv.index("--case_tag")
                pending_tag = argv[tag_index + 1]
            except (ValueError, IndexError):
                pending_tag = None
        elif stripped.startswith(_ERROR_MESSAGE_PREFIX) and pending_tag is not None:
            message = stripped.removeprefix(_ERROR_MESSAGE_PREFIX).strip().replace(",", ";")
            errors[pending_tag] = message
            pending_tag = None
    return errors


def _run_driver(
    benchmarks_dir: Path, testlist: Path, output: Path, driver_log: Path
) -> tuple[int, list[str]]:
    # This invokes the explicitly selected FlashInfer checkout without a shell.
    process = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "flashinfer_benchmark.py",
            "--testlist",
            str(testlist.resolve()),
            "--output_path",
            str(output.resolve()),
        ],
        cwd=benchmarks_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines = []
    with driver_log.open("w") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
    return process.wait(), lines


def _gemm_cases(
    ms: list[int],
    nks: list[tuple[int, int]],
    common: list[str],
) -> list[_Case]:
    cases = []

    for m in ms:
        for n, k in nks:
            variants: list[tuple[str, str, str, int, int, list[str], str | None]] = [
                ("bf16", "mm_bf16", "cudnn", n, k, [], None)
            ]
            for backend in ("cudnn", "cutlass", "trtllm"):
                layout = "128x4" if backend != "trtllm" or m > 32 else "8x4"
                extra = ["--use_nvfp4"]
                if layout == "128x4":
                    extra.append("--use_128x4_sf_layout")
                run_n, run_k = n, k
                if backend != "trtllm":
                    run_n, run_k = _round_up(n, 32), _round_up(k, 32)
                variants.append(
                    (
                        f"nvfp4_{backend}",
                        "mm_fp4",
                        backend,
                        run_n,
                        run_k,
                        extra,
                        f"nvfp4_{layout}",
                    )
                )
            variants += [
                (
                    f"fp8_{backend}",
                    "bmm_fp8",
                    backend,
                    n,
                    k,
                    ["--batch_size", "1"],
                    "fp8_static",
                )
                for backend in ("cudnn", "cutlass")
            ]
            variants.append(("fp8_trtllm", "mm_fp8", "trtllm_low_latency", n, k, [], "fp8_static"))

            for name, routine, backend, run_n, run_k, extra, quant_kind in variants:
                key = f"{name}_MxNxK={m}x{n}x{k}"
                quant = (quant_kind, m, run_k) if quant_kind else None
                cases.append(
                    _Case(
                        section="gemm",
                        tag=f"gemm_{key}",
                        key=key,
                        argv=[
                            "--routine",
                            routine,
                            "--backends",
                            backend,
                            *extra,
                            "--m",
                            str(m),
                            "--n",
                            str(run_n),
                            "--k",
                            str(run_k),
                            *common,
                        ],
                        quant=quant,
                    )
                )
    return cases


def _moe_cases(
    ms: list[int],
    shape: tuple[int, int, int, int] | None,
    common: list[str],
    activation: str | None,
    routing: str | None,
    routing_args: list[str] | None = None,
) -> list[_Case]:
    if shape is None:
        return []
    routine = "cutlass_fused_moe"

    hidden, intermediate, experts, top_k = shape
    shape_args = [
        "--hidden_size",
        str(hidden),
        "--num_experts",
        str(experts),
        "--top_k",
        str(top_k),
    ]
    if activation:
        shape_args += ["--activation-type", activation]
    if routing:
        shape_args += ["--routing_method", routing]
    shape_args += routing_args or []

    cases = []
    gated = activation is None or activation in {
        "Swiglu",
        "Geglu",
        "SwigluBias",
        "SwigluStep",
    }
    fp8_intermediate = _round_up(intermediate, 16 if gated else 128)
    # Pad a dimension only when vLLM pads it: vLLM pads non-gated NVFP4 expert
    # weights up to the 128-aligned swizzled scale rows, but raises instead of
    # padding gated NVFP4, so the gated case stays exact and may fail like vLLM.
    nvfp4_intermediate = intermediate if gated else _round_up(intermediate, 128)
    variants = (
        ("bf16_cutlass_moe", intermediate, []),
        ("fp8_cutlass_moe", fp8_intermediate, ["--cutlass_variant", "fp8"]),
        (
            "nvfp4_cutlass_moe",
            nvfp4_intermediate,
            ["--cutlass_variant", "nvfp4", "--quantized_input"],
        ),
        ("nvfp4_cutlass_moe_with_quant", nvfp4_intermediate, ["--cutlass_variant", "nvfp4"]),
    )
    for row, run_intermediate, extra in variants:
        for m in ms:
            key = f"{row}_M={m}"
            quant = ("fp8_static", m, hidden) if row == "fp8_cutlass_moe" else None
            cases.append(
                _Case(
                    section="moe",
                    tag=f"moe_{key}",
                    key=key,
                    argv=[
                        "--routine",
                        routine,
                        "--num_tokens",
                        str(m),
                        *shape_args,
                        "--intermediate_size",
                        str(run_intermediate),
                        *extra,
                        *common,
                    ],
                    quant=quant,
                )
            )
    return cases


def _nvfp4_runner(tensor: torch.Tensor, layout: str):
    import flashinfer

    global_scale = (448 * 6) / tensor.float().abs().nan_to_num().max()
    sf_layout = (
        flashinfer.SfLayout.layout_128x4 if layout == "128x4" else flashinfer.SfLayout.layout_8x4
    )

    def kernel(value, scale):
        return flashinfer.nvfp4_quantize(value, scale, sfLayout=sf_layout, do_shuffle=False)

    return kernel, (tensor, global_scale)


def _fp8_runner(tensor: torch.Tensor):
    import torch

    scale = tensor.abs().max().float() / torch.finfo(torch.float8_e4m3fn).max

    def kernel(value, value_scale):
        quantized, _ = vllm_ops.scaled_fp8_quant(value.contiguous(), value_scale)
        return quantized

    return kernel, (tensor, scale)


def _quant_times(
    cases: list[_Case], dry_runs: int, iterations: int, cuda_graph: bool
) -> dict[tuple[str, int, int], _ResultValue]:
    results: dict[tuple[str, int, int], _ResultValue] = {}
    specs = {case.quant for case in cases if case.quant is not None}
    warned_fp8 = False
    for kind, m, k in sorted(specs):
        if not kind.startswith("nvfp4_") and vllm_ops is None:
            results[(kind, m, k)] = _FP8_QUANT_UNAVAILABLE
            if not warned_fp8:
                print(f"[WARN] {_FP8_QUANT_UNAVAILABLE.removeprefix('ERROR: ')}")
                warned_fp8 = True
            continue
        # The GPU stack is imported lazily so shape planning, result parsing,
        # and their tests work without FlashInfer or torch installed.
        import numpy as np
        import torch
        from flashinfer.testing import bench_gpu_time

        tensor = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        runner = (
            _nvfp4_runner(tensor, kind.removeprefix("nvfp4_"))
            if kind.startswith("nvfp4_")
            else _fp8_runner(tensor)
        )
        kernel, inputs = runner
        times = bench_gpu_time(
            fn=kernel,
            input_args=inputs,
            dry_run_iters=dry_runs,
            repeat_iters=iterations,
            enable_cupti=True,
            use_cuda_graph=cuda_graph,
            cold_l2_cache=True,
            sleep_after_run=True,
        )
        results[(kind, m, k)] = float(np.median(times)) * 1000
    return results


def _combine(
    cases_by_tag: dict[str, _Case],
    rows: list[dict[str, str]],
    quant: dict[tuple[str, int, int], _ResultValue],
    errors: dict[str, str] | None = None,
) -> dict[str, dict[str, _ResultValue]]:
    results: dict[str, dict[str, _ResultValue]] = {"gemm": {}, "moe": {}}
    for row in rows:
        case = cases_by_tag.get(row.get("case_tag", ""))
        if case is None:
            continue
        value = float(row["median_time"]) * 1000
        results[case.section][case.key] = value
        if case.quant is not None and case.quant in quant:
            separator = "_MxNxK=" if case.section == "gemm" else "_M="
            name, shape = case.key.split(separator, 1)
            quant_value = quant[case.quant]
            results[case.section][f"{name}_with_quant{separator}{shape}"] = (
                value + quant_value if isinstance(quant_value, float) else quant_value
            )
    for tag, reason in (errors or {}).items():
        case = cases_by_tag.get(tag)
        if case is None:
            continue
        error = f"ERROR: {reason}"
        results[case.section].setdefault(case.key, error)
        if case.quant is not None:
            separator = "_MxNxK=" if case.section == "gemm" else "_M="
            name, shape = case.key.split(separator, 1)
            results[case.section].setdefault(f"{name}_with_quant{separator}{shape}", error)
    return results


def _format_result(value: _ResultValue | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def _write_results(
    path: Path,
    results: dict[str, dict[str, _ResultValue]],
    ms: list[int],
    nks: list[tuple[int, int]],
    nk_names: list[str] | None = None,
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        gemm = results["gemm"]
        if gemm:
            writer.writerow(["GEMM"])
            writer.writerow(["M", *ms])
            names = sorted({key.split("_MxNxK=", 1)[0] for key in gemm})
            for index, (n, k) in enumerate(nks):
                shape = f"{n}x{k}"
                label = f"{nk_names[index]}: {shape}" if nk_names is not None else shape
                writer.writerow([label])
                for name in names:
                    cells = [gemm.get(f"{name}_MxNxK={m}x{n}x{k}") for m in ms]
                    writer.writerow(
                        [
                            name,
                            *(_format_result(value) for value in cells),
                        ]
                    )

        moe = results["moe"]
        if moe:
            if gemm:
                writer.writerow([])
            writer.writerow(["MoE"])
            writer.writerow(["M", *ms])
            names = sorted({key.split("_M=", 1)[0] for key in moe})
            for name in names:
                cells = [moe.get(f"{name}_M={m}") for m in ms]
                writer.writerow([name, *(_format_result(value) for value in cells)])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flashinfer_repo",
        type=Path,
        required=True,
        help="checkout containing benchmarks/flashinfer_benchmark.py",
    )
    parser.add_argument(
        "--ms",
        type=_positive_int,
        nargs="+",
        default=[1, 8, 64, 512],
        help="token counts, for example: 1 8 64 512",
    )
    parser.add_argument("--nks", type=_nk_pair, nargs="+", help="GEMM N,K pairs, e.g. 4096,4096")
    parser.add_argument(
        "--nk_names",
        nargs="+",
        help="optional names parallel to --nks, e.g. qkv_proj o_proj",
    )
    parser.add_argument("--dry_run_iters", type=_positive_int, help="warmup iterations, e.g. 5")
    parser.add_argument("--num_iters", type=_positive_int, help="timed iterations, e.g. 30")
    parser.add_argument("--no_cuda_graph", action="store_true")
    parser.add_argument("--no_autotune", action="store_true")
    parser.add_argument(
        "--moe_hidden_size", type=_positive_int, help="model hidden size, e.g. 4096"
    )
    parser.add_argument(
        "--moe_intermediate_size", type=_positive_int, help="expert width, e.g. 14336"
    )
    parser.add_argument("--moe_num_experts", type=_positive_int, help="local expert count, e.g. 8")
    parser.add_argument("--moe_top_k", type=_positive_int, help="experts per token, e.g. 2")
    parser.add_argument(
        "--moe_activation_type",
        choices=_MOE_ACTIVATIONS,
        help="FlashInfer activation, e.g. Swiglu",
    )
    parser.add_argument(
        "--moe_routing_method",
        choices=_MOE_ROUTINGS,
        default="topk",
        help="FlashInfer routing, e.g. renormalize",
    )
    parser.add_argument("--moe_num_expert_group", type=_positive_int)
    parser.add_argument("--moe_topk_group", type=_positive_int)
    parser.add_argument("--moe_routed_scaling_factor", type=float)
    parser.add_argument("--moe_use_routing_bias", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path("benchmark_via_builtin_out"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Validate inputs, run the FlashInfer driver, and combine its results."""
    parser = _parser()
    args = parser.parse_args(argv)
    ms = list(dict.fromkeys(args.ms))
    try:
        nks, nk_names = _named_nks(args.nks or [], args.nk_names)
    except ValueError as exc:
        parser.error(str(exc))
    moe_values = (
        args.moe_hidden_size,
        args.moe_intermediate_size,
        args.moe_num_experts,
        args.moe_top_k,
    )
    if any(moe_values) and not all(moe_values):
        parser.error("all four --moe_* shape arguments are required together")
    if not nks and not any(moe_values):
        parser.error("pass --nks and/or all four --moe_* shape arguments")
    if all(moe_values) and args.moe_top_k > args.moe_num_experts:
        parser.error("--moe_top_k cannot exceed --moe_num_experts")
    deepseek_values = (
        args.moe_num_expert_group,
        args.moe_topk_group,
        args.moe_routed_scaling_factor,
    )
    if args.moe_routing_method == "deepseek_v3" and not all(
        value is not None for value in deepseek_values
    ):
        parser.error(
            "DeepSeekV3 routing requires --moe_num_expert_group, --moe_topk_group, "
            "and --moe_routed_scaling_factor"
        )
    if args.moe_routed_scaling_factor is not None and args.moe_routed_scaling_factor <= 0:
        parser.error("--moe_routed_scaling_factor must be positive")
    if (
        args.moe_num_expert_group is not None
        and args.moe_topk_group is not None
        and args.moe_topk_group > args.moe_num_expert_group
    ):
        parser.error("--moe_topk_group cannot exceed --moe_num_expert_group")

    benchmarks_dir = args.flashinfer_repo / "benchmarks"
    driver = benchmarks_dir / "flashinfer_benchmark.py"
    if not driver.is_file():
        parser.error(f"{driver} does not exist")

    common = []
    if args.dry_run_iters is not None:
        common += ["--dry_run_iters", str(args.dry_run_iters)]
    if args.num_iters is not None:
        common += ["--num_iters", str(args.num_iters)]
    if not args.no_autotune:
        common.append("--autotune")
    if args.no_cuda_graph:
        common.append("--no_cuda_graph")

    gemm_cases = _gemm_cases(ms, nks, common)
    moe_shape = moe_values if all(moe_values) else None
    moe_routing_args = []
    if args.moe_num_expert_group is not None:
        moe_routing_args += ["--n_group", str(args.moe_num_expert_group)]
    if args.moe_topk_group is not None:
        moe_routing_args += ["--topk_group", str(args.moe_topk_group)]
    if args.moe_routed_scaling_factor is not None:
        moe_routing_args += [
            "--routed_scaling_factor",
            str(args.moe_routed_scaling_factor),
        ]
    if args.moe_use_routing_bias:
        moe_routing_args.append("--use_routing_bias")
    moe_cases = _moe_cases(
        ms,
        moe_shape,
        common,
        args.moe_activation_type,
        args.moe_routing_method,
        moe_routing_args,
    )
    cases = gemm_cases + moe_cases

    args.workdir.mkdir(parents=True, exist_ok=True)
    testlist = args.workdir / "testlist.txt"
    builtin_csv = args.workdir / "builtin_results.csv"
    combined_csv = args.workdir / "combined_results.csv"
    driver_log = args.workdir / "driver.log"
    if builtin_csv.exists() or combined_csv.exists():
        parser.error(f"{args.workdir} already contains results; choose a fresh --workdir")
    cases_by_tag = _index_cases(cases)
    testlist.write_text(
        "\n".join(shlex.join([*case.argv, "--case_tag", case.tag]) for case in cases) + "\n"
    )

    returncode, driver_output = _run_driver(benchmarks_dir, testlist, builtin_csv, driver_log)

    rows = []
    if builtin_csv.is_file():
        with builtin_csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    completed_tags = {row.get("case_tag") for row in rows}
    parsed_errors = _parse_driver_errors(driver_output)
    missing_reason = (
        f"FlashInfer driver exited with status {returncode} before this case completed"
        f" (an earlier CUDA fault can poison later cases); see {driver_log}"
        if returncode
        else f"FlashInfer produced no result row and no error message; see {driver_log}"
    )
    empty_error_reason = (
        f"FlashInfer reported an error without a message (empty exception); see {driver_log}"
    )
    errors = {}
    for case in cases:
        if case.tag in completed_tags:
            continue
        if case.tag in parsed_errors:
            errors[case.tag] = parsed_errors[case.tag] or empty_error_reason
        else:
            errors[case.tag] = missing_reason

    completed_cases = [case for case in cases if case.tag in completed_tags]
    quant = _quant_times(
        completed_cases,
        args.dry_run_iters if args.dry_run_iters is not None else 5,
        args.num_iters if args.num_iters is not None else 30,
        not args.no_cuda_graph,
    )
    results = _combine(cases_by_tag, rows, quant, errors)
    _write_results(combined_csv, results, ms, nks, nk_names)
    print(f"Wrote {combined_csv}")
    if errors:
        failed = [cases_by_tag[tag].key for tag in errors if tag in cases_by_tag]
        raise RuntimeError(
            "FlashInfer failed benchmark cases: "
            + ", ".join(failed)
            + f"; wrote failure details to {combined_csv}"
        )
    if returncode:
        raise RuntimeError(f"FlashInfer driver exited with status {returncode}")


if __name__ == "__main__":
    main()
