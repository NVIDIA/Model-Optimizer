#!/usr/bin/env python3
"""Regression tests for the self-contained Qwen3.6 report renderer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("render_report.py")
SPEC = importlib.util.spec_from_file_location("qwen36_render_report", SCRIPT)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def complete_payload(model: str, recipe: str, reference_hash: str, value: float) -> dict:
    is_w8a8 = recipe in {
        "per_tensor_fp8",
        "block128_dynamic_w8a8_research",
        "mxfp8",
    }
    weight_names = [
        "model.layers.0.self_attn.q_proj.weight_quantizer",
        "model.layers.1.mlp.down_proj.weight_quantizer",
    ]
    input_names = ["model.layers.0.self_attn.q_proj.input_quantizer"] if is_w8a8 else []
    aggregate = {
        "logit_mse": value,
        "variance_normalized_logit_mse": value * 0.5,
        "forward_kl_ref_to_quant": value * 0.25,
        "reverse_kl_quant_to_ref": value * 0.3,
        "jensen_shannon": value * 0.1,
        "target_logprob_squared_error": value * 0.75,
        "nll_delta_quant_minus_ref": value * 0.05,
        "top1_agreement": max(0.0, 1.0 - value),
    }
    mse_summary = {
        "by_quantizer": {
            "model.layers.0.self_attn.q_proj.weight_quantizer": value,
            "model.layers.1.mlp.down_proj.weight_quantizer": value * 2,
        },
        "families": {
            "attention": {
                "count": 1,
                "mean": value,
                "std": 0.0,
                "min": value,
                "max": value,
                "quantiles": {"p95": value},
            },
            "mlp": {
                "count": 1,
                "mean": value * 2,
                "std": 0.0,
                "min": value * 2,
                "max": value * 2,
                "quantiles": {"p95": value * 2},
            },
        },
        "coverage": {
            "eligible_count": 2,
            "executed_count": 2,
            "missing_quantizers": [],
        },
    }
    input_summary = {
        "by_quantizer": (
            {"model.layers.0.self_attn.q_proj.input_quantizer": value * 0.1} if is_w8a8 else {}
        ),
        "families": (
            {
                "attention": {
                    "count": 1,
                    "mean": value * 0.1,
                    "std": 0.0,
                    "min": value * 0.1,
                    "max": value * 0.1,
                    "quantiles": {"p95": value * 0.1},
                }
            }
            if is_w8a8
            else {}
        ),
        "coverage": {
            "eligible_count": 1 if is_w8a8 else 0,
            "executed_count": 1 if is_w8a8 else 0,
            "missing_quantizers": [],
        },
    }
    return {
        "schema_version": renderer.STUDY_RESULT_SCHEMA,
        "status": "complete",
        "plan": {"status": "resolved"},
        "model": model,
        "recipe": recipe,
        "reference": {"signature_hash": reference_hash},
        "output_similarity": {
            "orientation": {
                "forward_kl": "KL(reference || quantized)",
                "reverse_kl": "KL(quantized || reference)",
            },
            "token_count": 123,
            "sample_count": 32,
            "aggregate_per_token": aggregate,
            "paired_document_bootstrap": {
                "metrics": {
                    "variance_normalized_logit_mse": {
                        "document_count": 32,
                        "resamples": 10_000,
                        "point_estimate_equal_document_mean": value * 0.55,
                        "percentile_interval": {
                            "lower": value * 0.45,
                            "upper": value * 0.65,
                        },
                    }
                }
            },
        },
        "quantization": {
            "coverage_contract": {
                "status": "passed",
                "weight_quantizer_names": weight_names,
                "input_quantizer_names": input_names,
            },
            "quantizer_inventory": [
                {
                    "name": "model.layers.0.self_attn.q_proj.weight_quantizer",
                    "family": "attention",
                },
                {
                    "name": "model.layers.1.mlp.down_proj.weight_quantizer",
                    "family": "mlp",
                },
                {
                    "name": "model.layers.0.self_attn.q_proj.input_quantizer",
                    "family": "attention",
                },
            ],
            "weight_cost_estimate": {
                "logical_totals": {
                    "element_count": 1_000_000,
                    "scale_overhead_bits": 250_000,
                    "total_bits": 8_250_000,
                    "effective_bits_per_weight": 8.25,
                },
                "unique_parameter_slice_count": 99,
                "unmapped_weight_quantizers": [],
            },
        },
        "quantization_mse": {"weight": mse_summary, "input": input_summary},
        "phase_walltime_seconds": {
            "initialization": 61.0,
            "dataset_materialization": 4.0,
            "reference_logits": 120.0,
            "quantization": 180.0,
            "quantizer_mse": 80.0,
            "output_similarity": 60.0,
        },
        "total_walltime_seconds": 505.0,
    }


class RendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = renderer.load_json(renderer.DEFAULT_MANIFEST)

    @staticmethod
    def write_result(root: Path, name: str, payload: dict) -> None:
        path = root / name / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def test_wrong_schema_is_rejected_without_path_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_result(root, "wrong", {"schema_version": "other", "status": "complete"})
            records, errors = renderer.load_results(root, self.manifest)
        self.assertEqual(records, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("unsupported or missing study result schema", errors[0])
        self.assertIn("results.json", errors[0])

    def test_partial_valid_and_failed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = self.manifest["models"][0]["handle"]
            first, second = self.manifest["candidates"][:2]
            self.write_result(root, "valid", complete_payload(model, first["id"], "ref-a", 0.01))
            self.write_result(
                root,
                "failed",
                {
                    "schema_version": renderer.STUDY_RESULT_SCHEMA,
                    "status": "failed",
                    "model": model,
                    "recipe": second["id"],
                    "failed_phase": "quantization",
                },
            )
            records, errors = renderer.load_results(root, self.manifest)
            report = renderer.render(self.manifest, records, errors, None)
        self.assertEqual(len(records), 2)
        self.assertEqual(sum(record["comparable"] for record in records), 1)
        self.assertIn("Partial measurements are available", report)
        self.assertIn("Quantizer-level diagnostics", report)
        self.assertNotIn("#1 ·", report)

    def test_top_level_status_wins_over_nested_plan_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = self.manifest["models"][0]["handle"]
            candidate = self.manifest["candidates"][0]
            self.write_result(
                root,
                "nested-status",
                complete_payload(model, candidate["id"], "ref-a", 0.01),
            )
            records, errors = renderer.load_results(root, self.manifest)

        self.assertEqual(errors, [])
        self.assertEqual(records[0]["status"], "complete")
        self.assertTrue(records[0]["valid_complete"])

    def test_complete_matrix_renders_rankings_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = 0
            for model_index, model in enumerate(self.manifest["models"]):
                for candidate in self.manifest["candidates"]:
                    index += 1
                    payload = complete_payload(
                        model["handle"],
                        candidate["id"],
                        f"reference-{model_index}",
                        0.001 * index,
                    )
                    self.write_result(root, f"{model_index}-{candidate['id']}", payload)
            records, errors = renderer.load_results(root, self.manifest)
            report = renderer.render(self.manifest, records, errors, None)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 14)
        self.assertTrue(all(record["comparable"] for record in records))
        self.assertIn("All 14 expected, comparable candidate artifacts were parsed", report)
        self.assertIn("Within-scope screen rankings", report)
        self.assertIn("Measured findings", report)
        self.assertIn("Matched activation penalty", report)
        self.assertIn("#1 ·", report)
        self.assertIn("Weight MSE coverage", report)
        self.assertIn("Highest named quantizer MSE", report)
        self.assertIn("Phase wall times", report)
        self.assertIn("8.25", report)

    def test_measured_findings_handles_zero_control_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = self.manifest["models"][0]
            for candidate in self.manifest["candidates"]:
                value = 0.0 if candidate["id"] == "per_tensor_fp8_weight_only_control" else 0.01
                payload = complete_payload(model["handle"], candidate["id"], "reference-0", value)
                self.write_result(root, candidate["id"], payload)
            records, errors = renderer.load_results(root, self.manifest)
            report = renderer.render(self.manifest, records, errors, None)

        self.assertEqual(errors, [])
        self.assertIn("Matched activation penalty", report)
        self.assertIn("<td>—</td>", report)


if __name__ == "__main__":
    unittest.main()
