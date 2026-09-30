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
measured activation-quantization time in the scale-factor layout the backend
consumes; the NVFP4 CUTLASS MoE row is instead a single fused measurement.
Logical shapes label each case while backend-specific physical padding follows
vLLM. A local FlashInfer source checkout is required for its benchmark driver
and utilities.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TextIO

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
_ResultValue = float | str


@dataclass(frozen=True, order=True)
class _QuantSpec:
    """One shared activation-quantization measurement of an m-by-k BF16 tile.

    Attributes:
        dtype: Quantized element type, ``"nvfp4"`` or ``"fp8"``.
        layout: Scale-factor layout the consuming kernel expects: ``"128x4"``,
            ``"8x4"``, or ``"linear"`` for NVFP4; ``"static"`` for FP8.
        m: Token count of the activation tile.
        k: Inner dimension of the activation tile.
    """

    dtype: str
    layout: str
    m: int
    k: int


@dataclass
class _Case:
    """One driver invocation and, once it has run, its outcome.

    Attributes:
        section: ``"gemm"`` or ``"moe"``.
        tag: Case label passed to the driver as ``--case_tag``; validates
            returned rows and labels the artifacts. Never an internal key.
        backend: Value of the output ``backend`` column.
        m: Token count.
        n: Logical output size; the driver may run a padded physical shape
            from ``argv``. ``None`` for MoE cases.
        k: Logical reduction size, as ``n``.
        argv: Driver arguments, before ``--case_tag``/``--output_path``.
        with_quant: ``with_quant`` column of the measured row itself; ``True``
            only for the fused NVFP4 CUTLASS MoE measurement.
        quant: Spec of the activation-quantization time a derived
            ``with_quant`` row adds to ``result``.
        result: Median kernel time in microseconds, an ``ERROR: ...`` message,
            or ``None`` until the case has run.
        quant_result: Measured time (or error) of ``quant``, recorded by
            ``_attach_quant_times``.
    """

    section: str
    tag: str
    backend: str
    m: int
    n: int | None
    k: int | None
    argv: list[str]
    with_quant: bool = False
    quant: _QuantSpec | None = None
    result: _ResultValue | None = None
    quant_result: _ResultValue | None = None


@dataclass(frozen=True)
class _MoeShape:
    """The per-rank fused-MoE problem all MoE cases share.

    Attributes:
        hidden: Model hidden size.
        intermediate: Per-rank expert width.
        experts: Per-rank expert count.
        top_k: Experts activated per token.
        activation: FlashInfer activation name; ``None`` means the driver
            default (gated SwiGLU).
        name: Module path of the expert container, used as the MoE rows'
            ``module_name``.
    """

    hidden: int
    intermediate: int
    experts: int
    top_k: int
    activation: str | None = None
    name: str = "experts"

    def label(self) -> str:
        label = f"H={self.hidden} F={self.intermediate} E={self.experts} top_k={self.top_k}"
        if self.activation:
            label += f" activation={self.activation}"
        return label


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")
    return parsed


def _nk_arg(value: str) -> tuple[int, int, str | None]:
    """Parse one GEMM shape argument: ``N,K`` or ``N,K,NAME``.

    Module names never contain commas, so a fourth field is a typo (two
    shapes missing their separating space) and is rejected loudly.
    """
    fields = value.split(",")
    if len(fields) not in (2, 3) or (len(fields) == 3 and not fields[2]):
        raise argparse.ArgumentTypeError(f"expected N,K or N,K,NAME, got {value!r}")
    try:
        n, k = _positive_int(fields[0]), _positive_int(fields[1])
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(f"expected positive N and K in {value!r}: {exc}") from exc
    return n, k, fields[2] if len(fields) == 3 else None


def _labels_by_nk(nk_args: list[tuple[int, int, str | None]]) -> dict[tuple[int, int], list[str]]:
    """Map each unique N,K, in first-seen order, to its module labels.

    Unnamed shapes label themselves ``"NxK"``.
    """
    labels_by_nk: dict[tuple[int, int], list[str]] = {}
    for n, k, name in nk_args:
        labels = labels_by_nk.setdefault((n, k), [])
        label = name if name is not None else f"{n}x{k}"
        if label not in labels:
            labels.append(label)
    return labels_by_nk


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _parse_driver_error(lines: list[str]) -> str | None:
    """Extract the driver's error message from one case's output.

    Returns the message, ``""`` when the driver reported an error without a
    message, or ``None`` when no error was reported. Each case runs in its own
    driver process, so any reported error belongs to that case.
    """
    error = None
    pending = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(_ERROR_CASE_PREFIX):
            pending = True
        elif stripped.startswith(_ERROR_MESSAGE_PREFIX) and pending:
            error = stripped.removeprefix(_ERROR_MESSAGE_PREFIX).strip().replace(",", ";")
            pending = False
    return error


def _failure_message(case_output: list[str], returncode: int, driver_log: Path) -> str:
    """Classify a case that produced no result row into an ``ERROR: ...`` cell."""
    message = _parse_driver_error(case_output)
    if message:
        return f"ERROR: {message}"
    if message is not None:
        return (
            "ERROR: FlashInfer reported an error without a message (empty exception); "
            f"see {driver_log}"
        )
    if returncode:
        return (
            f"ERROR: FlashInfer driver exited with status {returncode} for this case; "
            f"see {driver_log}"
        )
    return f"ERROR: FlashInfer produced no result row and no error message; see {driver_log}"


def _run_case(benchmarks_dir: Path, argv: list[str], log: TextIO) -> tuple[int, list[str]]:
    # Each case gets its own driver process: a fatal CUDA fault (for example a
    # misaligned address) permanently poisons the CUDA context, so sharing one
    # process would fail every later case (verified empirically). This invokes
    # the explicitly selected FlashInfer checkout without a shell.
    process = subprocess.Popen(  # nosec B603
        [sys.executable, "flashinfer_benchmark.py", *argv],
        cwd=benchmarks_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        lines.append(line)
    return process.wait(), lines


def _gpu_description() -> str:
    try:
        import torch

        # The driver name can be a placeholder on pre-release GPUs, so record
        # compute capability, SM count, and memory to pin down the exact part.
        properties = torch.cuda.get_device_properties(0)
        name = (
            f"{properties.name} (sm_{properties.major}{properties.minor} / "
            f"{properties.multi_processor_count} SMs / "
            f"{properties.total_memory / (1 << 30):.0f} GiB)"
        )
    except Exception:
        return "unknown GPU"
    watts = "unknown power limit"
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            # NVML does not honor CUDA_VISIBLE_DEVICES, so map the first
            # visible device back to its physical NVML handle.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
            if visible.startswith(("GPU-", "MIG-")):
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible) if visible else 0)
            limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            watts = f"{limit / 1000:.0f} W power limit"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return f"{name}; {watts}"


def _environment_header(flashinfer_repo: Path) -> str:
    try:
        import flashinfer

        version = flashinfer.__version__
    except Exception:
        version = "unknown"
    try:
        # Reads the revision of the explicitly selected checkout, no shell.
        result = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(flashinfer_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        revision = result.stdout.strip() or "unknown"
    except OSError:
        revision = "unknown"
    return (
        f"flashinfer {version}; checkout {flashinfer_repo.resolve()} @ {revision}; "
        f"{_gpu_description()}"
    )


def _write_builtin(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: dict[str, None] = {}
    for row in rows:
        for key in row:
            fieldnames.setdefault(key, None)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), restval="", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _gemm_cases(
    ms: list[int],
    nks: list[tuple[int, int]],
    common: list[str],
) -> list[_Case]:
    cases: list[_Case] = []

    def add(
        m: int,
        n: int,
        k: int,
        backend: str,
        routine: str,
        driver_backend: str,
        run_n: int | None = None,
        run_k: int | None = None,
        extra: list[str] | None = None,
        quant: _QuantSpec | None = None,
    ) -> None:
        cases.append(
            _Case(
                section="gemm",
                tag=f"gemm_{backend}_MxNxK={m}x{n}x{k}",
                backend=backend,
                m=m,
                n=n,
                k=k,
                argv=[
                    "--routine",
                    routine,
                    "--backends",
                    driver_backend,
                    *(extra or []),
                    "--m",
                    str(m),
                    "--n",
                    str(run_n if run_n is not None else n),
                    "--k",
                    str(run_k if run_k is not None else k),
                    *common,
                ],
                quant=quant,
            )
        )

    for m in ms:
        for n, k in nks:
            # Physical padding follows vLLM: dense NVFP4 on cuDNN, CUTLASS,
            # and CuteDSL pads N and K to multiples of 32; trtllm keeps the
            # exact shape with the shuffled layout; BF16 and FP8 stay exact.
            add(m, n, k, "bf16", "mm_bf16", "cudnn")
            for row_suffix, driver_backend in (
                ("cudnn", "cudnn"),
                ("cutlass", "cutlass"),
                ("cutedsl", "cute-dsl"),
                ("trtllm", "trtllm"),
            ):
                layout = "128x4" if driver_backend != "trtllm" or m > 32 else "8x4"
                extra = ["--use_nvfp4"]
                if layout == "128x4":
                    extra.append("--use_128x4_sf_layout")
                run_n, run_k = n, k
                if driver_backend != "trtllm":
                    run_n, run_k = _round_up(n, 32), _round_up(k, 32)
                add(
                    m,
                    n,
                    k,
                    f"nvfp4_{row_suffix}",
                    "mm_fp4",
                    driver_backend,
                    run_n,
                    run_k,
                    extra,
                    _QuantSpec("nvfp4", layout, m, run_k),
                )
            for driver_backend in ("cudnn", "cutlass"):
                add(
                    m,
                    n,
                    k,
                    f"fp8_{driver_backend}",
                    "bmm_fp8",
                    driver_backend,
                    extra=["--batch_size", "1"],
                    quant=_QuantSpec("fp8", "static", m, k),
                )
            add(
                m,
                n,
                k,
                "fp8_trtllm",
                "mm_fp8",
                "trtllm_low_latency",
                quant=_QuantSpec("fp8", "static", m, k),
            )
    return cases


def _moe_cases(ms: list[int], shape: _MoeShape, common: list[str]) -> list[_Case]:
    cases: list[_Case] = []

    def add(
        backend: str,
        routine: str,
        intermediate: int,
        hidden: int = shape.hidden,
        extra: list[str] | None = None,
        quant: tuple[str, str] | None = None,
        with_quant: bool = False,
    ) -> None:
        suffix = "_with_quant" if with_quant else ""
        cases.extend(
            _Case(
                section="moe",
                tag=f"moe_{backend}_moe{suffix}_M={m}",
                backend=backend,
                m=m,
                n=None,
                k=None,
                with_quant=with_quant,
                quant=_QuantSpec(*quant, m, hidden) if quant else None,
                argv=[
                    "--routine",
                    routine,
                    "--num_tokens",
                    str(m),
                    "--hidden_size",
                    str(hidden),
                    "--num_experts",
                    str(shape.experts),
                    "--top_k",
                    str(shape.top_k),
                    *(["--activation-type", shape.activation] if shape.activation else []),
                    "--intermediate_size",
                    str(intermediate),
                    *(extra or []),
                    *common,
                ],
            )
            for m in ms
        )

    gated = shape.activation is None or shape.activation in {
        "Swiglu",
        "Geglu",
        "SwigluBias",
        "SwigluStep",
    }
    # Pad a dimension only when vLLM pads it. FP8 per-tensor (CUTLASS and
    # trtllm-gen) pads the intermediate to 16 gated / 128 non-gated. NVFP4
    # CUTLASS pads non-gated intermediate up to the 128-aligned swizzled scale
    # rows but raises instead of padding gated, so gated stays exact and may
    # fail like vLLM. NVFP4 trtllm-gen additionally pads hidden to 256.
    fp8_intermediate = _round_up(shape.intermediate, 16 if gated else 128)
    nvfp4_intermediate = shape.intermediate if gated else _round_up(shape.intermediate, 128)

    add("bf16_cutlass", "cutlass_fused_moe", shape.intermediate)
    add(
        "fp8_cutlass",
        "cutlass_fused_moe",
        fp8_intermediate,
        extra=["--cutlass_variant", "fp8"],
        quant=("fp8", "static"),
    )
    add(
        "nvfp4_cutlass",
        "cutlass_fused_moe",
        nvfp4_intermediate,
        extra=["--cutlass_variant", "nvfp4", "--quantized_input"],
    )
    # The NVFP4 CUTLASS with_quant row is its own fused driver measurement
    # (unquantized input), not a derived base-plus-quant-time row.
    add(
        "nvfp4_cutlass",
        "cutlass_fused_moe",
        nvfp4_intermediate,
        extra=["--cutlass_variant", "nvfp4"],
        with_quant=True,
    )
    # Routing is synthetic in this benchmark (uniform random logits), so the
    # trtllm-gen rows, which route in-kernel, use a fixed renormalize method
    # to stay comparable across models; the model's real routing scheme is
    # not derivable from its config alone. CUTLASS and CuteDSL rows receive
    # precomputed indices and have no routing stage to time.
    add(
        "fp8_trtllm",
        "trtllm_fp8_per_tensor_scale_moe",
        fp8_intermediate,
        extra=["--routing_method", "renormalize"],
        quant=("fp8", "static"),
    )
    add(
        "nvfp4_trtllm",
        "trtllm_fp4_block_scale_moe",
        fp8_intermediate,
        hidden=_round_up(shape.hidden, 256),
        extra=["--routing_method", "renormalize"],
        quant=("nvfp4", "linear"),
    )
    if shape.activation in (None, "Swiglu"):
        # FlashInfer's CuteDSL fused MoE supports only gated Swiglu.
        add(
            "nvfp4_cutedsl",
            "cute_dsl_fp4_block_scale_moe",
            shape.intermediate,
            quant=("nvfp4", "linear"),
        )
    return cases


def _nvfp4_runner(tensor: torch.Tensor, layout: str):
    import flashinfer

    global_scale = (448 * 6) / tensor.float().abs().nan_to_num().max()
    if layout == "linear":
        # The trtllm-gen and CuteDSL fused-MoE kernels consume activation
        # scale factors in linear (unswizzled) layout.
        def linear_kernel(value, scale):
            return flashinfer.fp4_quantize(value, scale, is_sf_swizzled_layout=False)

        return linear_kernel, (tensor, global_scale)
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


def _attach_quant_times(
    cases: list[_Case], dry_runs: int, iterations: int, cuda_graph: bool
) -> None:
    """Measure each distinct quantization spec once and attach shared times."""
    specs = sorted(
        {case.quant for case in cases if case.quant is not None and isinstance(case.result, float)}
    )
    results: dict[_QuantSpec, _ResultValue] = {}
    if vllm_ops is None and any(spec.dtype == "fp8" for spec in specs):
        print(f"[WARN] {_FP8_QUANT_UNAVAILABLE.removeprefix('ERROR: ')}")
        results = {spec: _FP8_QUANT_UNAVAILABLE for spec in specs if spec.dtype == "fp8"}
        specs = [spec for spec in specs if spec.dtype != "fp8"]
    if specs:
        # The GPU stack is imported lazily so shape planning, result parsing,
        # and their tests work without FlashInfer or torch installed.
        import numpy as np
        import torch
        from flashinfer.testing import bench_gpu_time

        for spec in specs:
            tensor = torch.randn(spec.m, spec.k, device="cuda", dtype=torch.bfloat16)
            kernel, inputs = (
                _nvfp4_runner(tensor, spec.layout) if spec.dtype == "nvfp4" else _fp8_runner(tensor)
            )
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
            results[spec] = float(np.median(times)) * 1000
    for case in cases:
        if case.quant is not None and isinstance(case.result, float):
            case.quant_result = results[case.quant]


def _format_result(value: _ResultValue) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def _output_rows(case: _Case) -> list[tuple[bool, _ResultValue]]:
    """Expand a case into its (with_quant, runtime) output rows.

    A case with a quant spec gets a derived ``with_quant`` row adding the
    shared activation-quantization time; an error on either measurement
    propagates into the derived row.
    """
    assert case.result is not None
    rows = [(case.with_quant, case.result)]
    if case.quant is None:
        return rows
    if isinstance(case.result, str):
        rows.append((True, case.result))
        return rows
    quant_result = case.quant_result
    assert quant_result is not None
    rows.append(
        (True, case.result + quant_result if isinstance(quant_result, float) else quant_result)
    )
    return rows


def _write_results(
    path: Path,
    cases: list[_Case],
    labels_by_nk: dict[tuple[int, int], list[str]],
    header: str | None = None,
    moe_shape: _MoeShape | None = None,
) -> None:
    columns = ["module_name", "M", "N", "K", "backend", "with_quant", "runtime"]
    gemm = [case for case in cases if case.section == "gemm" and case.result is not None]
    moe = [case for case in cases if case.section == "moe" and case.result is not None]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        if header:
            writer.writerow([header])
        if gemm:
            writer.writerow(["GEMM"])
            writer.writerow(columns)
            for (n, k), labels in labels_by_nk.items():
                group = sorted(
                    (case for case in gemm if (case.n, case.k) == (n, k)),
                    key=lambda case: (case.backend, case.m),
                )
                # Modules fused into one GEMM are joined with "|" inside one
                # name; distinct same-shape modules each get their own rows,
                # duplicating the shared measurement.
                for label in labels:
                    for case in group:
                        for with_quant, value in _output_rows(case):
                            writer.writerow(
                                [
                                    label,
                                    case.m,
                                    n,
                                    k,
                                    case.backend,
                                    with_quant,
                                    _format_result(value),
                                ]
                            )
        if moe:
            if gemm:
                writer.writerow([])
            writer.writerow(["MoE"])
            if moe_shape is not None:
                writer.writerow([moe_shape.label()])
            writer.writerow(columns)
            for case in sorted(moe, key=lambda case: (case.backend, case.m, case.with_quant)):
                for with_quant, value in _output_rows(case):
                    writer.writerow(
                        [
                            moe_shape.name if moe_shape is not None else "experts",
                            case.m,
                            "",
                            "",
                            case.backend,
                            with_quant,
                            _format_result(value),
                        ]
                    )


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
    parser.add_argument(
        "--nks",
        type=_nk_arg,
        nargs="+",
        help="GEMM shapes as N,K or N,K,NAME, e.g. 4096,4096 2688,4096,mixer.o_proj",
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
        "--moe_name",
        default="experts",
        help="expert container module path, e.g. model.layers.*.mlp.experts",
    )
    parser.add_argument(
        "--moe_activation_type",
        choices=_MOE_ACTIVATIONS,
        help="FlashInfer activation, e.g. Swiglu",
    )
    parser.add_argument("--workdir", type=Path, default=Path("benchmark_via_builtin_out"))
    return parser


def _execute_cases(
    cases: list[_Case], benchmarks_dir: Path, workdir: Path, driver_log: Path, header: str
) -> list[dict[str, str]]:
    """Run each case in its own driver process and record outcomes on it.

    Returns the raw driver CSV rows for ``builtin_results.csv``.
    """
    case_csv = workdir / "case_result.csv"
    rows: list[dict[str, str]] = []
    with driver_log.open("w") as log:
        print(header, flush=True)
        log.write(header + "\n")
        for case in cases:
            marker = f"[CASE] {case.tag}\n"
            print(marker, end="", flush=True)
            log.write(marker)
            case_csv.unlink(missing_ok=True)
            returncode, case_output = _run_case(
                benchmarks_dir,
                [*case.argv, "--case_tag", case.tag, "--output_path", str(case_csv.resolve())],
                log,
            )
            case_rows = []
            if case_csv.is_file():
                with case_csv.open(newline="") as stream:
                    # A row that does not carry this case's tag cannot be
                    # trusted as this case's measurement; fail the case.
                    case_rows = [
                        row for row in csv.DictReader(stream) if row.get("case_tag") == case.tag
                    ]
            if case_rows:
                rows.extend(case_rows)
                case.result = float(case_rows[-1]["median_time"]) * 1000
            else:
                case.result = _failure_message(case_output, returncode, driver_log)
    case_csv.unlink(missing_ok=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Validate inputs, run the FlashInfer driver, and combine its results."""
    parser = _parser()
    args = parser.parse_args(argv)
    ms = list(dict.fromkeys(args.ms))
    labels_by_nk = _labels_by_nk(args.nks or [])
    moe_values = (
        args.moe_hidden_size,
        args.moe_intermediate_size,
        args.moe_num_experts,
        args.moe_top_k,
    )
    if any(moe_values) and not all(moe_values):
        parser.error("all four --moe_* shape arguments are required together")
    moe_shape = None
    if all(moe_values):
        moe_shape = _MoeShape(
            args.moe_hidden_size,
            args.moe_intermediate_size,
            args.moe_num_experts,
            args.moe_top_k,
            args.moe_activation_type,
            args.moe_name,
        )
    if not labels_by_nk and moe_shape is None:
        parser.error("pass --nks and/or all four --moe_* shape arguments")
    if moe_shape is not None and moe_shape.top_k > moe_shape.experts:
        parser.error("--moe_top_k cannot exceed --moe_num_experts")

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

    cases = _gemm_cases(ms, list(labels_by_nk), common)
    if moe_shape is not None:
        cases += _moe_cases(ms, moe_shape, common)

    args.workdir.mkdir(parents=True, exist_ok=True)
    testlist = args.workdir / "testlist.txt"
    builtin_csv = args.workdir / "builtin_results.csv"
    combined_csv = args.workdir / "combined_results.csv"
    driver_log = args.workdir / "driver.log"
    if builtin_csv.exists() or combined_csv.exists():
        parser.error(f"{args.workdir} already contains results; choose a fresh --workdir")
    testlist.write_text(
        "\n".join(shlex.join([*case.argv, "--case_tag", case.tag]) for case in cases) + "\n"
    )

    header = _environment_header(args.flashinfer_repo)
    rows = _execute_cases(cases, benchmarks_dir, args.workdir, driver_log, header)
    if rows:
        _write_builtin(builtin_csv, rows)

    _attach_quant_times(
        cases,
        args.dry_run_iters if args.dry_run_iters is not None else 5,
        args.num_iters if args.num_iters is not None else 30,
        not args.no_cuda_graph,
    )
    _write_results(combined_csv, cases, labels_by_nk, header, moe_shape)
    print(f"Wrote {combined_csv}")
    failed = [case.tag for case in cases if isinstance(case.result, str)]
    if failed:
        raise RuntimeError(
            "FlashInfer failed benchmark cases: "
            + ", ".join(failed)
            + f"; wrote failure details to {combined_csv}"
        )


if __name__ == "__main__":
    main()
