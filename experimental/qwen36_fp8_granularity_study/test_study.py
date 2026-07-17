# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103, TC003
"""Focused CPU tests for the Qwen3.6 FP8 granularity study."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import study
import torch


def _rule(config, quantizer_name):
    return next(
        rule for rule in config["quant_cfg"] if rule.get("quantizer_name") == quantizer_name
    )


def test_output_metrics_identity() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 4, 11, generator=generator)
    input_ids = torch.randint(0, 11, (2, 4), generator=generator)
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    accumulator = study.OutputMetricAccumulator()

    accumulator.add_batch(logits, logits.clone(), input_ids, attention_mask)
    result = accumulator.finalize()

    metrics = result["aggregate_per_token"]
    for name in (
        "logit_mse",
        "logit_rmse",
        "logit_mae",
        "centered_logit_mse",
        "variance_normalized_logit_mse",
        "forward_kl_ref_to_quant",
        "reverse_kl_quant_to_ref",
        "jensen_shannon",
        "target_logprob_error",
        "target_logprob_absolute_error",
        "target_logprob_squared_error",
        "target_logprob_rmse",
        "nll_delta_quant_minus_ref",
    ):
        assert metrics[name] == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["top5_set_overlap"] == 1.0
    assert result["token_count"] == 5
    assert result["sample_count"] == 2


def test_left_padding_does_not_score_first_real_token_from_pad_position() -> None:
    """Both sides of a next-token pair must be non-padding positions."""
    logits = torch.zeros(1, 4, 3)
    quantized = logits.clone()
    # Only the logit at the final real source position differs. The difference at
    # source position 1 is padding -> first-real-token and must not be counted.
    quantized[0, 1, 0] = 100.0
    quantized[0, 2, 0] = 2.0
    accumulator = study.OutputMetricAccumulator()
    accumulator.add_batch(
        logits,
        quantized,
        torch.tensor([[0, 0, 1, 2]]),
        torch.tensor([[0, 0, 1, 1]]),
    )

    result = accumulator.finalize()

    assert result["token_count"] == 1
    assert result["aggregate_per_token"]["logit_mse"] == pytest.approx(4.0 / 3.0)


def test_kl_orientation_matches_manual_definition() -> None:
    reference_probability = torch.tensor([0.8, 0.2])
    quantized_probability = torch.tensor([0.55, 0.45])
    reference = reference_probability.log().reshape(1, 1, 2).repeat(1, 2, 1)
    quantized = quantized_probability.log().reshape(1, 1, 2).repeat(1, 2, 1)
    input_ids = torch.tensor([[0, 1]])
    attention_mask = torch.ones_like(input_ids)
    accumulator = study.OutputMetricAccumulator()

    accumulator.add_batch(reference, quantized, input_ids, attention_mask)
    metrics = accumulator.finalize()["aggregate_per_token"]

    expected_forward = torch.sum(
        reference_probability * (reference_probability.log() - quantized_probability.log())
    ).item()
    expected_reverse = torch.sum(
        quantized_probability * (quantized_probability.log() - reference_probability.log())
    ).item()
    assert metrics["forward_kl_ref_to_quant"] == pytest.approx(expected_forward)
    assert metrics["reverse_kl_quant_to_ref"] == pytest.approx(expected_reverse)
    assert metrics["forward_kl_ref_to_quant"] != pytest.approx(metrics["reverse_kl_quant_to_ref"])
    assert metrics["nll_delta_quant_minus_ref"] == pytest.approx(
        -quantized_probability[1].log().item() + reference_probability[1].log().item()
    )


def test_variance_normalized_mse_preserves_gain_error() -> None:
    reference = torch.tensor([[[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]]])
    quantized = 2 * reference
    accumulator = study.OutputMetricAccumulator(epsilon=1.0e-12)

    accumulator.add_batch(
        reference,
        quantized,
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.long),
    )
    metric = accumulator.finalize()["aggregate_per_token"]["variance_normalized_logit_mse"]

    # q_center - r_center == r_center, so numerator and reference variance match.
    # Independent standardization would incorrectly erase this gain error and return zero.
    assert metric == pytest.approx(1.0, rel=1.0e-6)


def test_aggregation_is_token_weighted_and_keeps_sample_distribution() -> None:
    accumulator = study.OutputMetricAccumulator()
    reference_a = torch.zeros(1, 3, 2)
    quantized_a = reference_a.clone()
    quantized_a[:, :-1, 0] = 1.0
    accumulator.add_batch(
        reference_a,
        quantized_a,
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, 3, dtype=torch.long),
        [10],
    )
    reference_b = torch.zeros(1, 2, 2)
    quantized_b = reference_b.clone()
    quantized_b[:, :-1, 0] = 3.0
    accumulator.add_batch(
        reference_b,
        quantized_b,
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.long),
        [11],
    )

    result = accumulator.finalize()

    assert result["token_count"] == 3
    assert result["sample_count"] == 2
    assert result["aggregate_per_token"]["logit_mse"] == pytest.approx((2 * 0.5 + 4.5) / 3)
    assert [row["sample_index"] for row in result["per_sample"]["values"]] == [10, 11]
    distribution = result["per_sample"]["distributions"]["logit_mse"]
    assert distribution["count"] == 2
    assert distribution["quantiles"]["p50"] == pytest.approx(2.5)


def test_recipe_structures_distinguish_both_dynamic_switches() -> None:
    dynamic = study.resolve_recipe("block128_dynamic_w8a8_research")
    dynamic_weight = _rule(dynamic.config, "*weight_quantizer")["cfg"]
    dynamic_input = _rule(dynamic.config, "*input_quantizer")["cfg"]
    for attribute_config, expected_blocks in (
        (dynamic_weight, {-2: 128, -1: 128}),
        (dynamic_input, {-1: 128}),
    ):
        assert attribute_config["type"] == "dynamic"
        assert attribute_config["block_sizes"] == expected_blocks
        assert "type" not in attribute_config["block_sizes"]
        assert (
            study.classify_block_semantics(attribute_config)
            == "static_block_reshape_with_dynamic_full_precision_amax"
        )

    mxfp8 = study.resolve_recipe("mxfp8")
    mx_weight = _rule(mxfp8.config, "*weight_quantizer")["cfg"]
    assert "type" not in mx_weight
    assert mx_weight["block_sizes"] == {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
    assert study.classify_block_semantics(mx_weight) == "nested_dynamic_block_kernel"

    bad_yaml_equivalent = copy.deepcopy(dynamic_weight)
    bad_yaml_equivalent.pop("type")
    assert study.classify_block_semantics(bad_yaml_equivalent) == "static_block_calibrated_amax"
    with pytest.raises(ValueError, match="without calibrated amax"):
        study.validate_research_dynamic_attribute(bad_yaml_equivalent)


@pytest.mark.parametrize(
    "recipe_id",
    [
        "per_tensor_fp8_weight_only_control",
        "block128_static_weight_only",
        "block128_dynamic_weight_only_control",
        "mxfp8_weight_only_control",
    ],
)
def test_weight_only_candidates_disable_input_and_share_exclusions(recipe_id: str) -> None:
    recipe = study.resolve_recipe(recipe_id)
    input_rules = [
        rule
        for rule in recipe.config["quant_cfg"]
        if rule.get("quantizer_name") == "*input_quantizer"
    ]
    assert input_rules[-1].get("enable") is False
    assert recipe.activation_quantized is False
    for pattern in study.EXTRA_EXCLUSIONS:
        assert {"quantizer_name": pattern, "enable": False} in recipe.config["quant_cfg"]


def test_static_block_candidate_retains_builtin_calibrated_structure() -> None:
    recipe = study.resolve_recipe("block128_static_weight_only")
    weight = _rule(recipe.config, "*weight_quantizer")["cfg"]
    assert recipe.config["algorithm"] == "max"
    assert weight["block_sizes"] == {-2: 128, -1: 128}
    assert "type" not in weight
    assert study.classify_block_semantics(weight) == "static_block_calibrated_amax"


def test_cost_arithmetic() -> None:
    per_tensor = study.estimate_tensor_cost((128, 128), "per_tensor_fp8")
    block = study.estimate_tensor_cost((128, 128), "block128_static_weight_only")
    mx = study.estimate_tensor_cost((128, 128), "mxfp8")

    assert per_tensor["scale_count"] == 1
    assert per_tensor["effective_bits_per_weight"] == pytest.approx(8 + 32 / 128**2)
    assert block["scale_count"] == 1
    assert block["effective_bits_per_weight"] == pytest.approx(8 + 32 / 128**2)
    assert mx["scale_count"] == 128 * 4
    assert mx["effective_bits_per_weight"] == pytest.approx(8.25)
    assert (
        study.estimate_tensor_cost((129, 129), "block128_dynamic_w8a8_research")["scale_count"] == 4
    )


def test_dry_run_writes_resolved_schema_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_loader(_args):
        raise AssertionError("dry-run must not load a model")

    monkeypatch.setattr(study, "load_model_and_tokenizer", forbidden_loader)
    output_dir = tmp_path / "plan"
    exit_code = study.main(
        [
            "--model",
            "Qwen/Qwen3.6-27B",
            "--recipe",
            "block128_dynamic_w8a8_research",
            "--output-dir",
            str(output_dir),
            "--reference-cache",
            str(tmp_path / "references"),
            "--dry-run-plan",
        ]
    )

    assert exit_code == 0
    plan = json.loads((output_dir / "plan.json").read_text())
    printed = json.loads(capsys.readouterr().out)
    assert printed == plan
    assert plan["schema_version"] == study.SCHEMA_VERSION
    assert plan["status"] == "plan_only"
    assert plan["one_candidate_per_process"] is True
    assert plan["model"]["id"] == "Qwen/Qwen3.6-27B"
    assert plan["recipe"]["recipe_id"] == "block128_dynamic_w8a8_research"
    assert plan["recipe"]["deployable"] is False
    assert plan["data"]["fixed_shape_enforced"] is True
    assert plan["data"]["evaluation"]["row_offset"] == 16 * 8
    assert (
        plan["data"]["evaluation"]["row_offset_derivation"]
        == "same_dataset_packed_calibration_8x_raw_sample_multiplier"
    )
    assert plan["reference_cache"]["mismatch_policy"] == "fail"
    assert not (output_dir / "results.json").exists()


def test_reference_hash_changes_with_sample_tokens() -> None:
    base = {"model": "m", "sample_ids": ["0:abc"], "dtype": "bfloat16"}
    changed = {**base, "sample_ids": ["0:def"]}
    assert study.canonical_hash(base) != study.canonical_hash(changed)
    assert study.canonical_hash(base) == study.canonical_hash(dict(reversed(list(base.items()))))


def test_reference_signature_hash_changes_with_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        model="Qwen/Qwen3.6-27B",
        revision="abc123",
        tokenizer=None,
        tokenizer_revision=None,
        eval_dataset="snapshot.jsonl",
        eval_offset=8,
        eval_offset_derivation="explicit_cli_value",
        eval_size=1,
        eval_seq_len=2,
        eval_batch_size=1,
        seed=1234,
        trust_remote_code=False,
    )
    # Special methods are resolved on the type, not an instance SimpleNamespace.
    tokenizer = type("Tokenizer", (), {"__len__": lambda self: 100})()
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 2
    batches = [{"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}]
    runtime = {"torch": "v1"}
    monkeypatch.setattr(study, "reference_runtime_provenance", lambda *_args: dict(runtime))

    first = study.build_reference_signature(
        args,
        tokenizer,
        batches,
        torch.bfloat16,
        torch.nn.Linear(1, 1),
        {},
        {"sha256": "dataset-a"},
    )
    runtime["torch"] = "v2"
    second = study.build_reference_signature(
        args,
        tokenizer,
        batches,
        torch.bfloat16,
        torch.nn.Linear(1, 1),
        {},
        {"sha256": "dataset-a"},
    )

    assert study.canonical_hash(first) != study.canonical_hash(second)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("layers.0.self_attn.q_proj.weight_quantizer", "weight"),
        ("layers.0.mlp.experts.gate_up_proj_weight_quantizers.17", "weight"),
        ("layers.0.mlp.experts.gate_up_proj_weight_quantizer.0", "weight"),
        ("layers.0.mlp.experts.down_proj_input_quantizer", "input"),
        ("layers.0.output_quantizer", "other"),
    ],
)
def test_quantizer_role_includes_indexed_fused_experts(name: str, expected: str) -> None:
    assert study.quantizer_role(name) == expected


def test_fused_expert_weight_cost_maps_each_3d_parameter_slice() -> None:
    from modelopt.torch.quantization.nn import TensorQuantizer

    class FusedExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.randn(3, 8, 4))
            self.gate_up_proj_weight_quantizers = torch.nn.ModuleList(
                [TensorQuantizer() for _ in range(3)]
            )

    model = FusedExperts()
    result = study.estimate_model_weight_cost(model, "per_tensor_fp8")

    assert result["unmapped_weight_quantizers"] == []
    assert len(result["logical_quantized_modules"]) == 3
    assert result["logical_totals"]["element_count"] == model.gate_up_proj.numel()
    assert result["logical_totals"]["scale_count"] == 3
    assert {record["expert_index"] for record in result["logical_quantized_modules"]} == {
        0,
        1,
        2,
    }


def test_dynamic_shape_refresh_accepts_varying_routed_token_counts() -> None:
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    class SharedExpertInput(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert_input_quantizer = TensorQuantizer(
                QuantizerAttributeConfig(
                    num_bits=(4, 3),
                    type="dynamic",
                    block_sizes={-1: 2},
                    fake_quant=True,
                )
            )

    unprotected = SharedExpertInput()
    unprotected.expert_input_quantizer(torch.randn(3, 4))
    with pytest.raises(ValueError, match="Input shape has changed"):
        unprotected.expert_input_quantizer(torch.randn(5, 4))

    protected = SharedExpertInput()
    handles, names = study.install_dynamic_shape_refresh_hooks(protected)
    assert names == ["expert_input_quantizer"]
    assert len(handles) == 1
    assert protected.expert_input_quantizer(torch.randn(3, 4)).shape == (3, 4)
    assert protected.expert_input_quantizer(torch.randn(5, 4)).shape == (5, 4)


def test_paired_document_bootstrap_is_deterministic_and_document_weighted() -> None:
    first = study.paired_document_bootstrap([0.0, 2.0, 4.0], resamples=1000, seed=9)
    second = study.paired_document_bootstrap([0.0, 2.0, 4.0], resamples=1000, seed=9)

    assert first == second
    assert first["point_estimate_equal_document_mean"] == pytest.approx(2.0)
    assert first["percentile_interval"]["lower"] <= 2.0
    assert first["percentile_interval"]["upper"] >= 2.0


def test_eval_row_offset_produces_disjoint_sample_ids() -> None:
    rows = torch.arange(8, dtype=torch.long).reshape(8, 1)
    source = [{"input_ids": rows, "attention_mask": torch.ones_like(rows)}]
    calibration = study.select_row_range(source, sample_offset=0, sample_count=4, batch_size=2)
    evaluation = study.select_row_range(source, sample_offset=4, sample_count=3, batch_size=1)

    calibration_ids = set(study.sample_ids_from_batches(calibration, ordinal_offset=0))
    evaluation_ids = set(study.sample_ids_from_batches(evaluation, ordinal_offset=4))
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert torch.equal(
        torch.cat([batch["input_ids"] for batch in evaluation]).flatten(), rows[4:7].flatten()
    )


def test_dynamic_shape_validation() -> None:
    parser = study.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "Qwen/Qwen3.6-27B",
            "--recipe",
            "block128_dynamic_w8a8_research",
            "--output-dir",
            "/tmp/not-used",
            "--calib-seq-len",
            "128",
            "--eval-seq-len",
            "256",
        ]
    )
    args.eval_offset = args.calib_size
    with pytest.raises(ValueError, match="comparison contract"):
        study.validate_args(args)


@pytest.mark.parametrize(
    ("extra_args", "expected_offset", "expected_derivation"),
    [
        ([], 16 * 8, "same_dataset_packed_calibration_8x_raw_sample_multiplier"),
        (["--no-pack-calibration"], 16, "same_dataset_unpacked_calibration_prefix"),
        (
            ["--eval-dataset", "wikitext"],
            0,
            "different_dataset_default_zero",
        ),
        (["--eval-offset", "77"], 77, "explicit_cli_value"),
    ],
)
def test_eval_offset_derivation(extra_args, expected_offset, expected_derivation) -> None:
    parser = study.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "Qwen/Qwen3.6-27B",
            "--recipe",
            "per_tensor_fp8",
            "--output-dir",
            "/tmp/not-used",
            *extra_args,
        ]
    )
    offset, derivation = study.resolve_eval_offset(args)
    assert offset == expected_offset
    assert derivation == expected_derivation
