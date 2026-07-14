# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic tests for paired PDD effectiveness evidence and conclusions."""

from __future__ import annotations

import copy
import hashlib
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pdd_artifacts import (
    canonical_json_bytes,
    load_canonical_json,
    sha256_file,
    write_canonical_json,
)
from pdd_evaluation import (
    CONDITION_PROTOCOLS,
    EVALUATION_CONDITIONS,
    GRID_PROTOCOLS,
    INTEGRATOR_PROTOCOLS,
    summarize_effectiveness_bundle,
    validate_effectiveness_bundle,
)
from pdd_export import write_pdd_export

from modelopt.torch.fastgen import PDDConfig, PDDMetadata, PDDOutputProjection
from modelopt.torch.fastgen.plugins.qwen_image_pdd import QWEN_IMAGE_PDD_LAYER_SPEC


def _reference(root: pathlib.Path, path: pathlib.Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _rewrite_manifest(manifest: pathlib.Path, data: dict) -> None:
    manifest.unlink()
    write_canonical_json(manifest, data)
    detached = manifest.with_suffix(".json.sha256")
    detached.unlink(missing_ok=True)
    detached.write_bytes((sha256_file(manifest) + "\n").encode())


def _export(root: pathlib.Path, automodel: dict) -> pathlib.Path:
    config = PDDConfig(
        grid_size=128,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=4,
        block_size_max=64,
        inference_blocks=[32, 32, 32, 32],
        student_sample_steps=4,
        guidance_scale=4.0,
        num_train_timesteps=None,
    )
    projection = PDDOutputProjection(1, 4, 128, QWEN_IMAGE_PDD_LAYER_SPEC)
    metadata = PDDMetadata.from_config(config, projection)
    identity = {
        "model": {"id": "Qwen/Qwen-Image", "revision": "1" * 40, "dtype": "float32"},
        "pdd_metadata": metadata.to_dict(),
        "guidance": {"scale": 4.0, "rescale": 1.0, "eps": 1e-5},
        "automodel": automodel,
        "data": {
            "ordered_train_id_sha256": "8" * 64,
            "ordered_heldout_id_sha256": "9" * 64,
            "dataset_snapshot_sha256": "7" * 64,
            "local_batch_size": 1,
            "grad_accumulation_steps": 1,
        },
        "topology": {"world_size": 1, "pure_data_parallel": True},
    }
    return write_pdd_export(
        root / "export",
        projection.state_dict(),
        metadata=metadata,
        transformer_config={"_class_name": "QwenImageTransformer2DModel", "in_channels": 4},
        identity=identity,
        source_checkpoint={
            "name": "step_00010000",
            "manifest_sha256": "6" * 64,
            "completed_steps": 10_000,
        },
        modelopt_source={"commit": "2" * 40, "dirty": False},
        max_shard_bytes=1 << 20,
    )


def _write_bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "run"
    root.mkdir()
    model = {"id": "Qwen/Qwen-Image", "revision": "1" * 40}
    modelopt = {"commit": "2" * 40, "dirty": False}
    file_path = "nemo_automodel/__init__.py"
    file_sha = "3" * 64
    file_size = 7
    tree = hashlib.sha256()
    tree.update(file_path.encode())
    tree.update(b"\0")
    tree.update(file_sha.encode())
    tree.update(b"\0")
    tree.update(str(file_size).encode())
    tree.update(b"\n")
    tree_sha = tree.hexdigest()
    automodel = {
        "distribution": "nemo_automodel",
        "version": "0.5.0",
        "package_tree_sha256": tree_sha,
        "wheel_sha256": "4" * 64,
        "runtime_versions": {"diffusers": "0.38.0"},
    }
    export_dir = _export(root, automodel)
    export_manifest = export_dir / "manifest.json"

    environment = root / "environment.json"
    write_canonical_json(
        environment,
        {
            "distribution": "nemo_automodel",
            "files": [{"path": file_path, "sha256": file_sha, "size": file_size}],
            "import_origin": "/opt/pdd/site-packages/nemo_automodel/__init__.py",
            "package_file_count": 1,
            "package_tree_sha256": tree_sha,
            "release_commit": "5" * 40,
            "release_tag": "v0.5.0",
            "root": "/opt/pdd/site-packages",
            "runtime_versions": {"diffusers": "0.38.0"},
            "version": "0.5.0",
            "wheel": "nemo_automodel-0.5.0-py3-none-any.whl",
            "wheel_sha256": "4" * 64,
        },
    )
    data_snapshot = root / "data_snapshot.json"
    write_canonical_json(
        data_snapshot,
        {
            "schema_version": 1,
            "record_type": "pdd_dataset_snapshot",
            "dataset_snapshot_sha256": "7" * 64,
            "train_ids_sha256": "8" * 64,
            "heldout_ids_sha256": "9" * 64,
        },
    )

    prompts = []
    prompt_pairs = []
    for index in range(16):
        prompt = f"a small red cube on a white table, view {index:02d}"
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_id = f"prompt-{index:04d}"
        seed = 100 + index
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "seeds": [seed],
            }
        )
        prompt_pairs.append((prompt_id, prompt_sha, seed))
    prompt_set = root / "prompts.json"
    write_canonical_json(prompt_set, {"schema_version": 1, "prompts": prompts})
    negative_embedding = root / "negative_prompt_embedding.bin"
    negative_embedding.write_bytes(b"authenticated fixed negative condition")
    negative_condition = root / "negative_condition.json"
    write_canonical_json(
        negative_condition,
        {
            "schema_version": 1,
            "record_type": "pdd_negative_condition",
            "prompt_sha256": "c" * 64,
            "embedding": _reference(root, negative_embedding),
        },
    )

    protocol_fields = {
        "conditions": list(EVALUATION_CONDITIONS),
        "condition_protocols": CONDITION_PROTOCOLS,
        "grid_protocols": GRID_PROTOCOLS,
        "integrator_protocols": INTEGRATOR_PROTOCOLS,
        "image_protocol": {
            "height": 1024,
            "width": 1024,
            "batch_size": 1,
            "max_sequence_length": 512,
        },
        "metric_protocols": {
            "clip_score": {
                "direction": "higher",
                "implementation": "open_clip.ViT-H-14",
                "revision": "a" * 40,
            }
        },
        "timing_protocol": {
            "batch_size": 1,
            "warmup_runs": 3,
            "measured_runs": 5,
            "scope": "transformer_sampling_and_vae_decode",
            "synchronize_device": True,
        },
        "decision_rule": {
            "primary_condition": "pdd_4",
            "primary_metric": "clip_score",
            "quality_margin": 0.02,
            "quality_ci_rule": "paired_bootstrap_95_noninferiority",
            "efficiency_measure": "batch_normalized_transformer_evaluations",
            "efficiency_baseline": "teacher_guided",
            "minimum_relative_reduction": 0.5,
            "minimum_paired_samples": 16,
        },
        "negative_condition": _reference(root, negative_condition),
        "data_snapshot": _reference(root, data_snapshot),
        "stage_run_ids": {"canary": "canary-run", "training": "training-run"},
        "prompt_set": _reference(root, prompt_set),
        "bootstrap": {"replicates": 1_000, "seed": 91},
    }
    protocol_sha = hashlib.sha256(canonical_json_bytes(protocol_fields)).hexdigest()
    evidence_references = {}
    for stage in ("canary", "training"):
        checkpoint = (
            {
                "name": "step_00001500",
                "manifest_sha256": "b" * 64,
                "completed_steps": 1_500,
            }
            if stage == "canary"
            else {
                "name": "step_00010000",
                "manifest_sha256": "6" * 64,
                "completed_steps": 10_000,
            }
        )
        results = root / f"{stage}_results.json"
        write_canonical_json(
            results,
            {
                "schema_version": 1,
                "record_type": "pdd_stage_results",
                "stage": stage,
                "status": "passed",
                "slurm_job_ids": [101, 102, 103] if stage == "canary" else [201],
                "completed_updates": 1_500 if stage == "canary" else 10_000,
                "finite_loss": True,
                "finite_gradients": True,
                "resume_verified": True,
            },
        )
        evidence = root / f"{stage}_evidence.json"
        write_canonical_json(
            evidence,
            {
                "schema_version": 1,
                "record_type": "pdd_stage_evidence",
                "stage": stage,
                "status": "passed",
                "run_id": f"{stage}-run",
                "model": model,
                "modelopt": modelopt,
                "data_snapshot_sha256": sha256_file(data_snapshot),
                "evaluation_protocol_sha256": protocol_sha,
                "checkpoint": checkpoint,
                "results": _reference(root, results),
            },
        )
        evidence_references[stage] = _reference(root, evidence)

    records = []
    export_sha = sha256_file(export_manifest)
    metric_values = {
        "teacher_guided": 0.80,
        "undistilled_euler_4": 0.73,
        "undistilled_2step_4eval": 0.75,
        "pdd_2": 0.77,
        "pdd_4": 0.79,
        "pdd_8": 0.795,
    }
    latencies = {
        "teacher_guided": 10.0,
        "undistilled_euler_4": 1.4,
        "undistilled_2step_4eval": 1.2,
        "pdd_2": 0.8,
        "pdd_4": 1.0,
        "pdd_8": 1.8,
    }
    for prompt_id, prompt_sha, seed in prompt_pairs:
        for condition in EVALUATION_CONDITIONS:
            output = root / f"{prompt_id}-{condition}.png"
            output.write_bytes(b"png" + prompt_id.encode() + condition.encode())
            protocol = CONDITION_PROTOCOLS[condition]
            latency = latencies[condition]
            records.append(
                {
                    "condition": condition,
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_sha,
                    "seed": seed,
                    "metrics": {"clip_score": metric_values[condition]},
                    "output": _reference(root, output),
                    "scheduler_steps": protocol["scheduler_steps"],
                    "actual_transformer_invocations": protocol["actual_transformer_invocations"],
                    "batch_normalized_transformer_evaluations": protocol[
                        "batch_normalized_transformer_evaluations"
                    ],
                    "latency_seconds": latency,
                    "throughput_images_per_second": 1.0 / latency,
                    "peak_device_memory_bytes": 24_000_000_000,
                    "height": 1024,
                    "width": 1024,
                    "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
                    "evaluation_protocol_sha256": protocol_sha,
                    "model_artifact_sha256": export_sha,
                }
            )
    observations = root / "observations.json"
    write_canonical_json(observations, {"schema_version": 1, "records": records})
    manifest = root / "manifest.json"
    write_canonical_json(
        manifest,
        {
            "schema_version": 1,
            "stage": "effectiveness_evaluation",
            "run_id": "test-run",
            "model": model,
            "modelopt": modelopt,
            "pdd_export": _reference(root, export_manifest),
            "observations": _reference(root, observations),
            "environment": _reference(root, environment),
            "stage_evidence": evidence_references,
            **protocol_fields,
        },
    )
    manifest.with_suffix(".json.sha256").write_bytes((sha256_file(manifest) + "\n").encode())
    return manifest


def test_effectiveness_bundle_is_authenticated_and_emits_effective_conclusion(tmp_path) -> None:
    manifest = _write_bundle(tmp_path)
    validated = validate_effectiveness_bundle(manifest)
    first = summarize_effectiveness_bundle(validated)
    second = summarize_effectiveness_bundle(validated)

    assert first == second
    assert first["paired_sample_count"] == 16
    assert first["decision"]["label"] == "effective"
    assert set(first["aggregates"]) == set(EVALUATION_CONDITIONS)
    assert first["aggregates"]["pdd_2"]["mean_batch_normalized_transformer_evaluations"] == 2
    assert first["aggregates"]["pdd_4"]["metrics"]["clip_score"][
        "paired_delta_vs_teacher"
    ] == pytest.approx(-0.01)
    assert first["aggregates"]["pdd_4"]["peak_device_memory_bytes"]["mean"] > 0


def test_grid_protocol_pins_fastgen_precision_and_raw_noise_initialization() -> None:
    pdd_grid = GRID_PROTOCOLS["pdd_grid_128_shift5"]
    teacher_grid = GRID_PROTOCOLS["teacher_grid_50_shift5"]

    assert pdd_grid["grid_max_t"] == teacher_grid["grid_max_t"] == 0.999
    assert pdd_grid["construction_dtype"] == "float64"
    assert pdd_grid["runtime_dtype"] == "float32"
    assert pdd_grid["initial_state"] == "float32(float64(noise)*float64(grid_max_t))"
    assert pdd_grid["nodes"][0] == 0.9990000128746033
    assert pdd_grid["nodes"][-1] == 0.0
    assert (
        pdd_grid["nodes_sha256"]
        == hashlib.sha256(canonical_json_bytes(pdd_grid["nodes"])).hexdigest()
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "detached",
        "stage",
        "shadow",
        "count",
        "incomplete",
        "missing_shard",
        "tampered_shard",
        "unrelated_training",
        "failed_stage_results",
        "unrelated_run",
        "mismatched_export_data",
        "bootstrap",
        "prompt_reference",
        "guided_count",
        "grid_construction_dtype",
        "grid_initial_state",
        "grid_max_t",
        "grid_nodes",
        "grid_nodes_hash",
        "latency_decision",
        "integrator_formula",
        "protocol",
        "resolution",
    ],
)
def test_effectiveness_bundle_rejects_unclaimable_evidence(tmp_path, corruption) -> None:
    manifest = _write_bundle(tmp_path)
    root = manifest.parent
    data = copy.deepcopy(load_canonical_json(manifest))
    if corruption == "detached":
        manifest.with_suffix(".json.sha256").write_bytes(("0" * 64 + "\n").encode())
    elif corruption == "stage":
        data["stage"] = "smoke"
    elif corruption == "shadow":
        environment_path = root / data["environment"]["path"]
        environment = copy.deepcopy(load_canonical_json(environment_path))
        environment["import_origin"] = "/project/automodel/nemo_automodel/__init__.py"
        environment_path.unlink()
        write_canonical_json(environment_path, environment)
        data["environment"] = _reference(root, environment_path)
    elif corruption in ("missing_shard", "tampered_shard"):
        shard = next((root / "export").glob("*.safetensors"))
        if corruption == "missing_shard":
            shard.unlink()
        else:
            with shard.open("ab") as stream:
                stream.write(b"tampered")
    elif corruption == "unrelated_training":
        evidence_path = root / data["stage_evidence"]["training"]["path"]
        evidence = copy.deepcopy(load_canonical_json(evidence_path))
        evidence["checkpoint"]["manifest_sha256"] = "d" * 64
        evidence_path.unlink()
        write_canonical_json(evidence_path, evidence)
        data["stage_evidence"]["training"] = _reference(root, evidence_path)
    elif corruption == "failed_stage_results":
        evidence_path = root / data["stage_evidence"]["training"]["path"]
        evidence = copy.deepcopy(load_canonical_json(evidence_path))
        results_path = root / evidence["results"]["path"]
        results = copy.deepcopy(load_canonical_json(results_path))
        results["finite_gradients"] = False
        results_path.unlink()
        write_canonical_json(results_path, results)
        evidence["results"] = _reference(root, results_path)
        evidence_path.unlink()
        write_canonical_json(evidence_path, evidence)
        data["stage_evidence"]["training"] = _reference(root, evidence_path)
    elif corruption == "unrelated_run":
        evidence_path = root / data["stage_evidence"]["training"]["path"]
        evidence = copy.deepcopy(load_canonical_json(evidence_path))
        evidence["run_id"] = "unrelated-run"
        evidence_path.unlink()
        write_canonical_json(evidence_path, evidence)
        data["stage_evidence"]["training"] = _reference(root, evidence_path)
    elif corruption == "mismatched_export_data":
        snapshot_path = root / data["data_snapshot"]["path"]
        snapshot = copy.deepcopy(load_canonical_json(snapshot_path))
        snapshot["dataset_snapshot_sha256"] = "e" * 64
        snapshot_path.unlink()
        write_canonical_json(snapshot_path, snapshot)
        data["data_snapshot"] = _reference(root, snapshot_path)
    elif corruption == "bootstrap":
        data["bootstrap"]["seed"] += 1
    elif corruption == "prompt_reference":
        source = root / data["prompt_set"]["path"]
        alternate = root / "alternate_prompts.json"
        write_canonical_json(alternate, load_canonical_json(source))
        data["prompt_set"] = _reference(root, alternate)
    elif corruption == "grid_nodes":
        data["grid_protocols"]["pdd_grid_128_shift5"]["nodes"][32] += 1e-4
    elif corruption == "grid_nodes_hash":
        data["grid_protocols"]["pdd_grid_128_shift5"]["nodes_sha256"] = "0" * 64
    elif corruption == "grid_max_t":
        data["grid_protocols"]["pdd_grid_128_shift5"]["grid_max_t"] = 1.0
    elif corruption == "grid_construction_dtype":
        data["grid_protocols"]["pdd_grid_128_shift5"]["construction_dtype"] = "float32"
    elif corruption == "grid_initial_state":
        data["grid_protocols"]["pdd_grid_128_shift5"]["initial_state"] = "noise"
    elif corruption == "latency_decision":
        data["decision_rule"]["efficiency_measure"] = "latency_seconds"
    elif corruption == "integrator_formula":
        data["integrator_protocols"]["heun_explicit_trapezoid"]["terminal_rule"] = (
            "fall back to Euler at t_next=0"
        )
    elif corruption == "protocol":
        data["condition_protocols"]["pdd_4"]["pdd_blocks"] = [64, 64]
    else:
        observations_path = root / data["observations"]["path"]
        observations = copy.deepcopy(load_canonical_json(observations_path))
        if corruption == "count":
            observations["records"][3]["actual_transformer_invocations"] = 3
        elif corruption == "guided_count":
            observations["records"][0]["actual_transformer_invocations"] = 50
            observations["records"][0]["batch_normalized_transformer_evaluations"] = 50
        elif corruption == "resolution":
            observations["records"][3]["height"] = 512
        else:
            observations["records"].pop()
        observations_path.unlink()
        write_canonical_json(observations_path, observations)
        data["observations"] = _reference(root, observations_path)
    if corruption not in ("detached", "missing_shard", "tampered_shard"):
        _rewrite_manifest(manifest, data)

    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        validate_effectiveness_bundle(manifest)


@pytest.mark.parametrize(
    ("pdd_values", "label"),
    [
        ([0.75] * 16, "not_effective"),
        ([0.80] * 8 + [0.76] * 8, "inconclusive"),
    ],
)
def test_predeclared_decision_rule_emits_boundary_conclusions(tmp_path, pdd_values, label) -> None:
    manifest = _write_bundle(tmp_path)
    root = manifest.parent
    data = copy.deepcopy(load_canonical_json(manifest))
    observations_path = root / data["observations"]["path"]
    observations = copy.deepcopy(load_canonical_json(observations_path))
    primary = [record for record in observations["records"] if record["condition"] == "pdd_4"]
    for record, value in zip(primary, pdd_values):
        record["metrics"]["clip_score"] = value
    observations_path.unlink()
    write_canonical_json(observations_path, observations)
    data["observations"] = _reference(root, observations_path)
    _rewrite_manifest(manifest, data)

    summary = summarize_effectiveness_bundle(validate_effectiveness_bundle(manifest))
    assert summary["decision"]["label"] == label
