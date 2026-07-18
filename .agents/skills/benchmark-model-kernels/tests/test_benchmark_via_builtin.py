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

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_via_builtin.py"
SPEC = importlib.util.spec_from_file_location("benchmark_via_builtin", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("value", ["1", "1,2,3", "a,2", "0,2", "2,-1"])
def test_nk_pair_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._nk_pair(value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        benchmark._positive_int(value)


@pytest.mark.parametrize(
    "option",
    [
        "--ms",
        "--dry_run_iters",
        "--num_iters",
        "--moe_hidden_size",
        "--moe_intermediate_size",
        "--moe_num_experts",
        "--moe_top_k",
    ],
)
def test_parser_rejects_zero_for_numeric_options(option, capsys):
    with pytest.raises(SystemExit, match="2"):
        benchmark._parser().parse_args(["--flashinfer_repo", "/unused", option, "0"])
    assert "not a positive integer" in capsys.readouterr().err


def test_gemm_cases_are_data_driven_and_preserve_requested_shapes():
    cases = benchmark._gemm_cases([1], [(65, 129)], [])

    assert {case.key.split("_MxNxK=", 1)[0] for case in cases} == {
        "bf16",
        "nvfp4_cudnn",
        "nvfp4_cutlass",
        "nvfp4_trtllm",
        "fp8_cudnn",
        "fp8_cutlass",
        "fp8_trtllm",
    }
    assert len({case.tag for case in cases}) == len(cases)
    assert {case.key.split("_MxNxK=", 1)[1] for case in cases} == {"1x65x129"}
    physical_shapes = {
        case.key.split("_MxNxK=", 1)[0]: (
            case.argv[case.argv.index("--n") + 1],
            case.argv[case.argv.index("--k") + 1],
        )
        for case in cases
    }
    assert physical_shapes["nvfp4_cudnn"] == ("96", "160")
    assert physical_shapes["nvfp4_cutlass"] == ("96", "160")
    assert physical_shapes["nvfp4_trtllm"] == ("65", "129")
    assert all(
        shape == ("65", "129")
        for name, shape in physical_shapes.items()
        if not name.startswith("nvfp4_")
    )
    assert {case.quant for case in cases if case.quant is not None} == {
        ("nvfp4_128x4", 1, 160),
        ("nvfp4_8x4", 1, 129),
        ("fp8_static", 1, 129),
    }


def test_nk_names_are_parallel_and_merge_duplicate_shapes():
    nks, names = benchmark._named_nks(
        [(32, 64), (32, 64), (128, 256)],
        ["o_proj", "out_proj", "qkv_proj"],
    )

    assert nks == [(32, 64), (128, 256)]
    assert names == ["o_proj/out_proj", "qkv_proj"]
    with pytest.raises(ValueError, match="exactly one name"):
        benchmark._named_nks([(32, 64)], ["o_proj", "out_proj"])


def test_case_tags_are_unique_join_keys():
    case = benchmark._Case("gemm", "duplicate", "bf16_MxNxK=1x2x3", [])

    assert benchmark._index_cases([case]) == {case.tag: case}
    with pytest.raises(RuntimeError, match="duplicate benchmark case tag: duplicate"):
        benchmark._index_cases([case, case])


def test_moe_cases_and_result_combination():
    cases = benchmark._moe_cases([1], (32, 50, 4, 2), [], None, None)
    assert len(cases) == 4
    assert [case.key.split("_M=", 1)[0] for case in cases] == [
        "bf16_cutlass_moe",
        "fp8_cutlass_moe",
        "nvfp4_cutlass_moe",
        "nvfp4_cutlass_moe_with_quant",
    ]
    intermediate_sizes = {
        case.key.split("_M=", 1)[0]: case.argv[case.argv.index("--intermediate_size") + 1]
        for case in cases
    }
    assert intermediate_sizes == {
        "bf16_cutlass_moe": "50",
        "nvfp4_cutlass_moe": "50",
        "nvfp4_cutlass_moe_with_quant": "50",
        "fp8_cutlass_moe": "64",
    }

    fp8 = next(case for case in cases if case.key == "fp8_cutlass_moe_M=1")
    assert fp8.quant is not None
    rows = [{"case_tag": fp8.tag, "median_time": "0.001"}]
    quant = {fp8.quant: 2.0}

    results = benchmark._combine({fp8.tag: fp8}, rows, quant)

    assert results["moe"]["fp8_cutlass_moe_M=1"] == 1.0
    assert results["moe"]["fp8_cutlass_moe_with_quant_M=1"] == 3.0


def test_swiglustep_uses_gated_fp8_alignment():
    cases = benchmark._moe_cases([1], (32, 50, 4, 2), [], "SwigluStep", "topk")

    fp8 = next(case for case in cases if case.key == "fp8_cutlass_moe_M=1")
    assert fp8.argv[fp8.argv.index("--intermediate_size") + 1] == "64"


def test_non_gated_moe_pads_fp8_and_nvfp4_intermediate_to_128():
    cases = benchmark._moe_cases([1], (32, 50, 4, 2), [], "Relu2", "topk")

    intermediate_sizes = {
        case.key.split("_M=", 1)[0]: case.argv[case.argv.index("--intermediate_size") + 1]
        for case in cases
    }
    assert intermediate_sizes == {
        "bf16_cutlass_moe": "50",
        "fp8_cutlass_moe": "128",
        "nvfp4_cutlass_moe": "128",
        "nvfp4_cutlass_moe_with_quant": "128",
    }


def test_moe_cases_forward_deepseek_routing_metadata():
    routing_args = [
        "--n_group",
        "1",
        "--topk_group",
        "1",
        "--routed_scaling_factor",
        "2.5",
        "--use_routing_bias",
    ]
    cases = benchmark._moe_cases(
        [1],
        (32, 48, 4, 2),
        [],
        "Relu2",
        "deepseek_v3",
        routing_args,
    )

    for case in cases:
        assert case.argv[case.argv.index("--routing_method") + 1] == "deepseek_v3"
        assert case.argv[case.argv.index("--n_group") + 1] == "1"
        assert case.argv[case.argv.index("--topk_group") + 1] == "1"
        assert case.argv[case.argv.index("--routed_scaling_factor") + 1] == "2.5"
        assert "--use_routing_bias" in case.argv


def test_unavailable_fp8_quantization_is_written_as_an_error(monkeypatch, capsys, tmp_path):
    case = benchmark._Case(
        section="gemm",
        tag="gemm_fp8_cutlass_MxNxK=1x32x64",
        key="fp8_cutlass_MxNxK=1x32x64",
        argv=[],
        quant=("fp8_static", 1, 64),
    )
    monkeypatch.setattr(benchmark, "vllm_ops", None)

    quant = benchmark._quant_times([case], 1, 1, False)
    results = benchmark._combine(
        {case.tag: case}, [{"case_tag": case.tag, "median_time": "0.001"}], quant
    )
    output = tmp_path / "combined_results.csv"
    benchmark._write_results(output, results, [1], [(32, 64)])

    assert quant == {case.quant: benchmark._FP8_QUANT_UNAVAILABLE}
    assert "[WARN] vLLM is unavailable for FP8 activation quantization" in capsys.readouterr().out
    assert results["gemm"][case.key] == 1.0
    assert (
        results["gemm"]["fp8_cutlass_with_quant_MxNxK=1x32x64"] == benchmark._FP8_QUANT_UNAVAILABLE
    )
    assert f"fp8_cutlass_with_quant,{benchmark._FP8_QUANT_UNAVAILABLE}\n" in output.read_text()


def test_driver_errors_are_added_to_kernel_and_with_quant_rows(tmp_path):
    case = benchmark._Case(
        section="gemm",
        tag="gemm_fp8_trtllm_MxNxK=8x1280x2880",
        key="fp8_trtllm_MxNxK=8x1280x2880",
        argv=[],
        quant=("fp8_static", 8, 2880),
    )
    output = [
        f"[ERROR] Error running test: --routine mm_fp8 --case_tag {case.tag}\n",
        "[ERROR] Error: K must be divisible by 128, got 2880\n",
    ]

    errors = benchmark._parse_driver_errors(output)
    results = benchmark._combine({case.tag: case}, [], {}, errors)
    csv_path = tmp_path / "combined_results.csv"
    benchmark._write_results(csv_path, results, [8], [(1280, 2880)])

    expected = "ERROR: K must be divisible by 128; got 2880"
    assert errors == {case.tag: "K must be divisible by 128; got 2880"}
    assert results["gemm"][case.key] == expected
    assert results["gemm"]["fp8_trtllm_with_quant_MxNxK=8x1280x2880"] == expected
    assert f"fp8_trtllm,{expected}\n" in csv_path.read_text()
    assert f"fp8_trtllm_with_quant,{expected}\n" in csv_path.read_text()


def test_empty_driver_error_has_no_synthetic_reason():
    tag = "gemm_nvfp4_cutlass_MxNxK=8x2880x1024"
    output = [
        f"[ERROR] Error running test: --routine mm_fp4 --case_tag {tag}\n",
        "[ERROR] Error:\n",
    ]

    assert benchmark._parse_driver_errors(output) == {tag: ""}


def test_write_results_preserves_the_original_sectioned_tables(tmp_path):
    output = tmp_path / "combined_results.csv"
    benchmark._write_results(
        output,
        {
            "gemm": {
                "bf16_MxNxK=1x2x3": 1.25,
                "fp8_cutlass_MxNxK=8x4x5": 3.5,
            },
            "moe": {"fp8_cutlass_moe_with_quant_M=8": 2.5},
        },
        [1, 8],
        [(4, 5), (2, 3)],
        ["qkv_proj", "in_proj"],
    )

    assert output.read_text() == (
        "GEMM\n"
        "M,1,8\n"
        "qkv_proj: 4x5\n"
        "bf16,,\n"
        "fp8_cutlass,,3.500\n"
        "in_proj: 2x3\n"
        "bf16,1.250,\n"
        "fp8_cutlass,,\n"
        "\n"
        "MoE\n"
        "M,1,8\n"
        "fp8_cutlass_moe_with_quant,,2.500\n"
    )


def test_top_k_cannot_exceed_expert_count(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--flashinfer_repo",
            "/unused",
            "--moe_hidden_size",
            "4",
            "--moe_intermediate_size",
            "8",
            "--moe_num_experts",
            "1",
            "--moe_top_k",
            "2",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert "--moe_top_k cannot exceed --moe_num_experts" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("returncode", "expected_reason"),
    [
        (0, "FlashInfer produced no result row"),
        (1, "FlashInfer driver exited with status 1"),
    ],
)
def test_missing_builtin_results_still_writes_combined_errors(
    monkeypatch, tmp_path, returncode, expected_reason
):
    benchmarks_dir = tmp_path / "flashinfer" / "benchmarks"
    benchmarks_dir.mkdir(parents=True)
    (benchmarks_dir / "flashinfer_benchmark.py").write_text("")
    workdir = tmp_path / "results"
    monkeypatch.setattr(benchmark, "_run_driver", lambda *_: (returncode, []))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--flashinfer_repo",
            str(benchmarks_dir.parent),
            "--ms",
            "1",
            "--nks",
            "2,3",
            "--workdir",
            str(workdir),
        ],
    )

    with pytest.raises(RuntimeError, match="FlashInfer failed benchmark cases"):
        benchmark.main()

    assert not (workdir / "builtin_results.csv").exists()
    combined = (workdir / "combined_results.csv").read_text()
    assert f"bf16,ERROR: {expected_reason}" in combined
    assert "driver.log" in combined


def test_run_driver_streams_and_persists_the_driver_output(tmp_path, capsys):
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    (benchmarks_dir / "flashinfer_benchmark.py").write_text(
        "print('line one')\nprint('line two')\n"
    )
    driver_log = tmp_path / "driver.log"

    returncode, lines = benchmark._run_driver(
        benchmarks_dir, tmp_path / "testlist.txt", tmp_path / "out.csv", driver_log
    )

    assert returncode == 0
    assert lines == ["line one\n", "line two\n"]
    assert driver_log.read_text() == "line one\nline two\n"
    assert "line one" in capsys.readouterr().out
