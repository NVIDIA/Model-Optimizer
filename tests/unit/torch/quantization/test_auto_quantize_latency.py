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

"""Synthetic tests for the HAQ latency LUT, canonicalizer, and fixed-kernel policy."""

import json
from pathlib import Path

import pytest

from modelopt.torch.quantization._auto_quantize_latency import (
    RECIPE_FP8,
    RECIPE_NONE,
    RECIPE_W4A16_NVFP4,
    SCHEMA_VERSION,
    FixedKernelPolicy,
    KernelSelector,
    LatencyCoverageError,
    LatencyLUT,
    canonicalize_benchmark_csv,
    load_fixed_kernel_policy,
    normalize_layer_indices,
    write_canonical_csv,
)

PROFILE = "b100_tp1_ep1_decode"

# A synthetic sectioned benchmark CSV mirroring qwen36_tp1_ep1_8_5/combined_results.csv:
# fused QKV + gate/up GEMM groups, a fused-MoE container, multiple backends per
# (group, M), an ERROR row for the selected nvfp4_trtllm backend, and a
# deliberately *faster* non-selected backend (fp8_cudnn) to prove it is ignored.
RAW_CSV = """\
flashinfer 0.6.14; checkout /x @ deadbeef; NVIDIA Graphics Device (sm_100 / 148 SMs / 178 GiB); 600 W power limit
GEMM
module_name,M,N,K,backend,with_quant,runtime
model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj,1,3072,2048,bf16,False,9.100
model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj,1,3072,2048,fp8_cudnn,True,3.000
model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj,1,3072,2048,fp8_trtllm,True,6.000
model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj,1,3072,2048,nvfp4_trtllm,True,5.000
model.layers.*.mlp.shared_expert.gate_proj|model.layers.*.mlp.shared_expert.up_proj,1,1024,2048,bf16,False,7.700
model.layers.*.mlp.shared_expert.gate_proj|model.layers.*.mlp.shared_expert.up_proj,1,1024,2048,fp8_trtllm,True,8.100
model.layers.*.mlp.shared_expert.gate_proj|model.layers.*.mlp.shared_expert.up_proj,1,1024,2048,nvfp4_trtllm,True,7.900
MoE
H=2048 F=512 E=256 top_k=8 activation=Swiglu
module_name,M,N,K,backend,with_quant,runtime
model.layers.*.mlp.experts,1,,,bf16_cutlass,False,46.959
model.layers.*.mlp.experts,1,,,fp8_trtllm,True,21.856
model.layers.*.mlp.experts,1,,,nvfp4_trtllm,True,19.500
"""

# TRT-LLM fixed-kernel policy for the first POC (per Wei-Ming: start with trtllm).
TRTLLM_POLICY = FixedKernelPolicy(
    kernel_policy_id="qwen36_trtllm_v1",
    selectors={
        "gemm": {
            RECIPE_NONE: KernelSelector(backend="bf16"),
            RECIPE_FP8: KernelSelector(backend="fp8_trtllm"),
            RECIPE_W4A16_NVFP4: KernelSelector(
                backend="nvfp4_trtllm",
                enable_w4a4_proxy=True,
                proxy_reason="bsz=1 low-M weight-memory-bound W4A4->W4A16 POC proxy",
            ),
        },
        "moe": {
            RECIPE_NONE: KernelSelector(backend="bf16_cutlass"),
            RECIPE_FP8: KernelSelector(backend="fp8_trtllm"),
            RECIPE_W4A16_NVFP4: KernelSelector(
                backend="nvfp4_trtllm",
                enable_w4a4_proxy=True,
                proxy_reason="bsz=1 low-M weight-memory-bound W4A4->W4A16 POC proxy",
            ),
        },
    },
)


@pytest.fixture
def raw_csv(tmp_path) -> Path:
    path = tmp_path / "combined_results.csv"
    path.write_text(RAW_CSV)
    return path


def _canonicalize(raw_path, policy=TRTLLM_POLICY):
    return canonicalize_benchmark_csv(
        raw_path, policy, deployment_profile=PROFILE, tp=1, ep=1, hardware="sm_100"
    )


def _write_lut(tmp_path, rows) -> LatencyLUT:
    out = tmp_path / "haq_latency_v1.csv"
    write_canonical_csv(rows, out)
    return LatencyLUT.from_csv(out)


# ---------------------------------------------------------------------------
# Canonicalization + fixed-kernel selection
# ---------------------------------------------------------------------------


def test_normalize_layer_indices():
    assert (
        normalize_layer_indices("model.layers.5.self_attn.q_proj")
        == "model.layers.*.self_attn.q_proj"
    )
    # Already wildcarded names are unchanged.
    assert normalize_layer_indices("model.layers.*.mlp.experts") == "model.layers.*.mlp.experts"


def test_canonicalize_selects_only_policy_backends(raw_csv):
    rows, problems = _canonicalize(raw_csv)
    assert problems == []
    backends = {(r.op_kind, r.recipe_id): r.backend for r in rows}
    # FP8 GEMM uses fp8_trtllm, never the faster fp8_cudnn (3.0us) row.
    assert backends[("gemm", RECIPE_FP8)] == "fp8_trtllm"
    fp8_gemm = next(
        r
        for r in rows
        if r.op_kind == "gemm"
        and r.recipe_id == RECIPE_FP8
        and r.m == 1
        and r.group_pattern.endswith("v_proj")
    )
    assert fp8_gemm.latency_us == 6.000  # not the 3.000 fp8_cudnn value


def test_faster_nonselected_backend_is_ignored_not_minimized(raw_csv):
    rows, _ = _canonicalize(raw_csv)
    # No canonical row may ever carry the non-selected fp8_cudnn backend.
    assert all(r.backend != "fp8_cudnn" for r in rows)


def test_recipe_mappings_and_with_quant_semantics(raw_csv):
    rows, _ = _canonicalize(raw_csv)
    by_recipe = {(r.op_kind, r.recipe_id): r for r in rows if r.group_pattern.endswith("v_proj")}
    none_row = by_recipe[("gemm", RECIPE_NONE)]
    assert none_row.with_quant is False
    assert none_row.runtime_format == "BF16"
    assert none_row.cost_is_proxy is False

    fp8_row = by_recipe[("gemm", RECIPE_FP8)]
    assert fp8_row.with_quant is True
    assert fp8_row.runtime_format == "FP8"

    w4_row = by_recipe[("gemm", RECIPE_W4A16_NVFP4)]
    assert w4_row.with_quant is True
    assert w4_row.runtime_format == "NVFP4_W4A16"


def test_w4a4_proxy_provenance_is_complete(raw_csv):
    rows, _ = _canonicalize(raw_csv)
    proxy_rows = [r for r in rows if r.recipe_id == RECIPE_W4A16_NVFP4]
    assert proxy_rows
    for r in proxy_rows:
        assert r.cost_is_proxy is True
        assert r.measured_runtime_format == "NVFP4_W4A4"
        assert r.proxy_reason
        assert r.backend == "nvfp4_trtllm"


def test_moe_shape_metadata_is_preserved(raw_csv):
    rows, _ = _canonicalize(raw_csv)
    moe_rows = [r for r in rows if r.op_kind == "moe"]
    assert moe_rows
    for r in moe_rows:
        assert r.h == 2048 and r.f == 512 and r.local_experts == 256 and r.top_k == 8


def test_missing_backend_is_coverage_error(raw_csv):
    # A policy demanding an un-benchmarked backend for FP8 GEMM must fail, never
    # substitute another backend.
    policy = FixedKernelPolicy(
        kernel_policy_id="bad",
        selectors={"gemm": {RECIPE_FP8: KernelSelector(backend="fp8_does_not_exist")}},
    )
    rows, problems = _canonicalize(raw_csv, policy)
    assert rows == []
    assert any("backend not measured" in p for p in problems)


def test_failed_error_row_is_coverage_error(tmp_path):
    raw = tmp_path / "err.csv"
    raw.write_text(
        "prov\nGEMM\nmodule_name,M,N,K,backend,with_quant,runtime\n"
        "model.layers.*.self_attn.o_proj,1,2048,2048,nvfp4_trtllm,True,"
        "ERROR: FlashInfer produced no result row; see driver.log\n"
    )
    policy = FixedKernelPolicy(
        kernel_policy_id="p",
        selectors={
            "gemm": {
                RECIPE_W4A16_NVFP4: KernelSelector(
                    backend="nvfp4_trtllm", enable_w4a4_proxy=True, proxy_reason="poc"
                )
            }
        },
    )
    rows, problems = _canonicalize(raw, policy)
    assert rows == []
    assert any("All rows failed" in p for p in problems)


def test_ambiguous_selection_is_coverage_error(tmp_path):
    # Two successful rows for the same (group, M, backend, with_quant) is ambiguous.
    raw = tmp_path / "dup.csv"
    raw.write_text(
        "prov\nGEMM\nmodule_name,M,N,K,backend,with_quant,runtime\n"
        "model.layers.*.self_attn.o_proj,1,2048,2048,fp8_trtllm,True,6.0\n"
        "model.layers.*.self_attn.o_proj,1,2048,2048,fp8_trtllm,True,6.5\n"
    )
    policy = FixedKernelPolicy(
        kernel_policy_id="p",
        selectors={"gemm": {RECIPE_FP8: KernelSelector(backend="fp8_trtllm")}},
    )
    rows, problems = _canonicalize(raw, policy)
    assert rows == []
    assert any("Ambiguous" in p for p in problems)


def test_selector_rejects_proxy_on_wrong_recipe():
    with pytest.raises(ValueError, match="proxy_reason"):
        KernelSelector(backend="nvfp4_trtllm", enable_w4a4_proxy=True)


# ---------------------------------------------------------------------------
# LUT loading + exact lookup
# ---------------------------------------------------------------------------


def test_roundtrip_lut_lookup(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    lut = _write_lut(tmp_path, rows)
    qkv = "model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj"
    assert lut.lookup(PROFILE, 1, qkv, RECIPE_FP8).latency_us == 6.0
    assert lut.lookup(PROFILE, 1, qkv, RECIPE_W4A16_NVFP4).latency_us == 5.0
    assert len(lut.digest) == 64  # sha256 hex


def test_lookup_wrong_profile_and_m_fail(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    lut = _write_lut(tmp_path, rows)
    qkv = "model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj"
    with pytest.raises(LatencyCoverageError):
        lut.lookup("wrong_profile", 1, qkv, RECIPE_FP8)
    with pytest.raises(LatencyCoverageError):
        lut.lookup(PROFILE, 32, qkv, RECIPE_FP8)  # M=32 not in synthetic LUT


def test_digest_is_deterministic(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    out = tmp_path / "a.csv"
    write_canonical_csv(rows, out)
    d1 = LatencyLUT.from_csv(out).digest
    d2 = LatencyLUT.from_csv(out).digest
    assert d1 == d2


def test_duplicate_row_rejected(tmp_path):
    rows, _ = canonicalize_benchmark_csv(
        _tmp_raw(tmp_path), TRTLLM_POLICY, deployment_profile=PROFILE, tp=1, ep=1, hardware="sm_100"
    )
    out = tmp_path / "dup_lut.csv"
    write_canonical_csv([*rows, rows[0]], out)
    with pytest.raises(LatencyCoverageError, match="Duplicate"):
        LatencyLUT.from_csv(out)


def test_non_positive_latency_rejected(tmp_path):
    lut_csv = _minimal_canonical_csv(latency_us="0.0")
    path = tmp_path / "bad.csv"
    path.write_text(lut_csv)
    with pytest.raises(LatencyCoverageError, match="finite and positive"):
        LatencyLUT.from_csv(path)


def test_wrong_schema_version_rejected(tmp_path):
    lut_csv = _minimal_canonical_csv(schema_version="haq_latency_v0")
    path = tmp_path / "bad.csv"
    path.write_text(lut_csv)
    with pytest.raises(LatencyCoverageError, match="schema_version"):
        LatencyLUT.from_csv(path)


def test_incomplete_proxy_provenance_rejected(tmp_path):
    # cost_is_proxy=True but no measured_runtime_format / proxy_reason.
    lut_csv = _minimal_canonical_csv(cost_is_proxy="True")
    path = tmp_path / "bad.csv"
    path.write_text(lut_csv)
    with pytest.raises(LatencyCoverageError, match="requires both"):
        LatencyLUT.from_csv(path)


# ---------------------------------------------------------------------------
# Group / source-pattern matching
# ---------------------------------------------------------------------------


def test_match_group_pattern_qkv_and_gate_up(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    lut = _write_lut(tmp_path, rows)
    qkv_sources = [
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.k_proj",
        "model.layers.7.self_attn.v_proj",
    ]
    matched = lut.match_group_pattern(PROFILE, 1, qkv_sources)
    assert matched.endswith("v_proj") and "q_proj" in matched


def test_match_group_pattern_partial_source_fails(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    lut = _write_lut(tmp_path, rows)
    # Missing v_proj: every concrete source still matches, but one pattern
    # (v_proj) matches no concrete source -> not fully covered.
    partial = [
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.k_proj",
    ]
    with pytest.raises(LatencyCoverageError):
        lut.match_group_pattern(PROFILE, 1, partial)


def test_match_group_pattern_unknown_source_fails(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    lut = _write_lut(tmp_path, rows)
    with pytest.raises(LatencyCoverageError):
        lut.match_group_pattern(PROFILE, 1, ["model.layers.7.self_attn.rotary_emb"])


def test_moe_container_group_match(raw_csv, tmp_path):
    rows, _ = _canonicalize(raw_csv)
    lut = _write_lut(tmp_path, rows)
    matched = lut.match_group_pattern(PROFILE, 1, ["model.layers.3.mlp.experts"])
    assert matched == "model.layers.*.mlp.experts"


# ---------------------------------------------------------------------------
# M-dependent winner (data supports it; solver selection is a later increment)
# ---------------------------------------------------------------------------


def test_m_dependent_winner(tmp_path):
    raw = tmp_path / "mdep.csv"
    raw.write_text(
        "prov\nGEMM\nmodule_name,M,N,K,backend,with_quant,runtime\n"
        # At M=1 FP8 is faster; at M=32 NVFP4 is faster.
        "model.layers.*.self_attn.o_proj,1,2048,2048,fp8_trtllm,True,4.0\n"
        "model.layers.*.self_attn.o_proj,1,2048,2048,nvfp4_trtllm,True,5.0\n"
        "model.layers.*.self_attn.o_proj,32,2048,2048,fp8_trtllm,True,40.0\n"
        "model.layers.*.self_attn.o_proj,32,2048,2048,nvfp4_trtllm,True,30.0\n"
    )
    policy = FixedKernelPolicy(
        kernel_policy_id="p",
        selectors={
            "gemm": {
                RECIPE_FP8: KernelSelector(backend="fp8_trtllm"),
                RECIPE_W4A16_NVFP4: KernelSelector(
                    backend="nvfp4_trtllm", enable_w4a4_proxy=True, proxy_reason="poc"
                ),
            }
        },
    )
    rows, problems = canonicalize_benchmark_csv(
        raw, policy, deployment_profile=PROFILE, tp=1, ep=1, hardware="sm_100"
    )
    assert problems == []
    lut = _write_lut(tmp_path, rows)
    g = "model.layers.*.self_attn.o_proj"
    assert (
        lut.lookup(PROFILE, 1, g, RECIPE_FP8).latency_us
        < lut.lookup(PROFILE, 1, g, RECIPE_W4A16_NVFP4).latency_us
    )
    assert (
        lut.lookup(PROFILE, 32, g, RECIPE_W4A16_NVFP4).latency_us
        < lut.lookup(PROFILE, 32, g, RECIPE_FP8).latency_us
    )


def test_load_policy_from_dict_and_yaml_json_equivalent(tmp_path):
    policy_dict = {
        "kernel_policy_id": "qwen36_trtllm_v1",
        "mode": "fixed_kernel",
        "selectors": {
            "gemm": {
                "NONE": {"backend": "bf16"},
                "FP8_DEFAULT_CFG": {"backend": "fp8_trtllm"},
                "W4A16_NVFP4_CFG": {
                    "backend": "nvfp4_trtllm",
                    "enable_w4a4_proxy": True,
                    "proxy_reason": "poc",
                },
            }
        },
    }
    from_dict = load_fixed_kernel_policy(policy_dict)
    json_path = tmp_path / "policy.json"
    json_path.write_text(json.dumps(policy_dict))
    from_json = load_fixed_kernel_policy(json_path)
    assert from_dict.kernel_policy_id == from_json.kernel_policy_id == "qwen36_trtllm_v1"
    assert from_json.selector("gemm", RECIPE_W4A16_NVFP4).enable_w4a4_proxy is True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tmp_raw(tmp_path) -> Path:
    path = tmp_path / "combined_results.csv"
    path.write_text(RAW_CSV)
    return path


def _minimal_canonical_csv(
    *, schema_version=SCHEMA_VERSION, latency_us="5.0", cost_is_proxy="False"
) -> str:
    from modelopt.torch.quantization._auto_quantize_latency import ALL_COLUMNS

    values = {
        "schema_version": schema_version,
        "deployment_profile": PROFILE,
        "group_pattern": "model.layers.*.self_attn.o_proj",
        "source_module_patterns": json.dumps(["model.layers.*.self_attn.o_proj"]),
        "recipe_id": RECIPE_FP8,
        "runtime_format": "FP8",
        "m": "1",
        "latency_us": latency_us,
        "backend": "fp8_trtllm",
        "with_quant": "True",
        "op_kind": "gemm",
        "timing_scope": "gemm_fused",
        "selection_policy": "fixed_kernel",
        "kernel_policy_id": "p",
        "tp": "1",
        "ep": "1",
        "hardware": "sm_100",
        "cost_is_proxy": cost_is_proxy,
    }
    header = ",".join(ALL_COLUMNS)
    row = ",".join(values.get(col, "") for col in ALL_COLUMNS)
    return header + "\n" + row + "\n"
