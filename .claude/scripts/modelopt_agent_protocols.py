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

"""Strawman shared schemas for ModelOpt agent workflows (v0.0.1).

Shared schemas across the three ModelOpt-workflow projects:
- bigpareto (strategy / sweep)
- quant_flow (standard CI / pipeline execution backend)
- ModelOpt agentic skills (day-0 onboarding / reliability / debugging)

STRAWMAN — treat the field set as a starting point for discussion. Each
project keeps its own internal types and translates at the boundary
where it talks to other layers.

Requires: Python >= 3.10, pydantic >= 2.0
"""

from __future__ import annotations

from datetime import (
    datetime,  # noqa: TC003  (pydantic v2 needs runtime symbol to resolve annotations)
)
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# ──────────────────────────────────────────────────────────────────────
# Recipe reference (discriminated union)
# ──────────────────────────────────────────────────────────────────────


class RecipePath(BaseModel):
    """Path to a recipe YAML file. bigpareto's primary mode."""

    type: Literal["path"] = "path"
    path: str  # e.g., "examples/recipes/nvfp4/max.yaml"


class RecipePreset(BaseModel):
    """Named ModelOpt recipe identifier.

    e.g., 'general/ptq/fp8_default-fp8_kv' — resolved from ModelOpt's
    built-in recipe library.
    """

    type: Literal["preset"] = "preset"
    name: str


class RecipeInline(BaseModel):
    """Inline ptq_cfg dict — for one-off experiments without a file."""

    type: Literal["inline"] = "inline"
    ptq_cfg: dict


RecipeRef = Annotated[
    RecipePath | RecipePreset | RecipeInline,
    Field(discriminator="type"),
]


# ──────────────────────────────────────────────────────────────────────
# Deployment + Evaluation
# ──────────────────────────────────────────────────────────────────────


class DeploymentSpec(BaseModel):
    """How to serve the model for evaluation."""

    framework: Literal["vllm", "sglang", "trtllm", "external"] = "vllm"
    image: str | None = None  # e.g., "nvcr.io/nvidia/vllm:..."
    command: str | None = None  # full override; if None, framework default
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    expert_parallel_size: int | None = None
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    """What benchmarks to run."""

    nel_config_path: str | None = None  # path to NEL config YAML
    nel_inline_config: dict | None = None  # alternative to nel_config_path
    tasks: list[str] | None = None  # subset; None = all in config
    limit_samples: int | None = None  # per-task sample cap
    time_limit: str | None = None  # HH:MM:SS for the eval job

    @model_validator(mode="after")
    def _check_mutual_exclusion(self) -> EvaluationSpec:
        if self.nel_config_path is not None and self.nel_inline_config is not None:
            raise ValueError("nel_config_path and nel_inline_config are mutually exclusive")
        return self


# ──────────────────────────────────────────────────────────────────────
# Core: OptimizationConfig — portable unit of execution
# ──────────────────────────────────────────────────────────────────────


class OptimizationConfig(BaseModel):
    """One optimization point.

    Produced by bigpareto or by agentic day-0 onboarding. Consumed by any
    executor that can emit a standard RunResult: quant_flow, agentic direct
    execution, local scripts, or another backend.
    """

    # Identity
    config_id: str  # stable hash or sweep-assigned ID
    model: str  # HF handle ('org/name') or JETArt key
    recipe: RecipeRef

    # Where to run
    launcher: str  # 'cw', 'oci-hsg', 'prenyx', ...
    slurm_account: str | None = None

    # How to deploy + eval
    deployment: DeploymentSpec
    evaluation: EvaluationSpec

    # Free-form metadata.
    # Suggested keys: sweep_id, pareto_iteration, owner_team, user, parent_run_id.
    metadata: dict = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────


class FailureClass(str, Enum):
    """Routable failure categories.

    Executors emit these when they can classify a failed RunResult; consumers
    (bigpareto, reliability agent) dispatch on them. Kept small on purpose -
    easier to agree on in one pass.
    """

    INFRA_TRANSIENT = "infra_transient"
    # SLURM evicted, network flake, registry timeout, JET runner died.
    # Verdict: usually retry without change.

    CONTAINER_BUILD_FAILED = "container_build_failed"
    # Docker build stage failed (dep conflict, missing package).
    # Verdict: inspect Dockerfile, possibly PATCH.

    MODEL_UNSUPPORTED = "model_unsupported"
    # Recipe couldn't be applied to model architecture; no quantizer matched.
    # Verdict: PATCH (recipe wildcard fix) or POINT_INFEASIBLE.

    QUANT_COVERAGE_FAILURE = "quant_coverage_failure"
    # Post-quantization validation failed (e.g. NVFP4 layers got BF16).
    # Verdict: PATCH (recipe pattern fix).

    DEPLOYMENT_HEALTH_FAILED = "deployment_health_failed"
    # Server didn't come up, health check timed out.
    # Verdict: PATCH (image / command / env / TP) or POINT_INFEASIBLE.

    EVAL_JUDGE_FAILED = "eval_judge_failed"
    # Judge auth, parse, or rate limit. Often transient.
    # Verdict: usually retry or wait.

    SAMPLE_ACCOUNTING_FAILED = "sample_accounting_failed"
    # Incomplete run, dropped samples, unknown_agent_error.
    # Verdict: investigation needed.

    USER_CONFIG_ERROR = "user_config_error"
    # Invalid config (typo, missing required field).
    # Verdict: NEEDS_HUMAN.

    UNKNOWN = "unknown"
    # Couldn't classify; default to NEEDS_HUMAN.


class PerfStats(BaseModel):
    """Server-side perf stats during evaluation."""

    throughput_tokens_per_sec: float | None = None
    kv_cache_pct: float | None = None
    request_queue_max: int | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None


class ArtifactRefs(BaseModel):
    """Pointers to result blobs — never payloads through the protocol."""

    mlflow_run_id: str | None = None  # MLflow run ID (canonical key)
    jet_pipeline_id: int | None = None  # JET CI pipeline ID
    nel_invocation_id: str | None = None  # NEL invocation ID
    checkpoint_uri: str | None = None  # quantized checkpoint location
    # Log URIs keyed by stage. Suggested keys: client, server, slurm, judge, build.
    log_uris: dict[str, str] = Field(default_factory=dict)


class RunExecutor(str, Enum):
    """Execution backend that produced a RunResult."""

    QUANT_FLOW = "quant_flow"
    AGENTIC_DIRECT = "agentic_direct"
    LOCAL_SCRIPT = "local_script"
    MANUAL = "manual"
    OTHER = "other"


class RunResult(BaseModel):
    """One config's outcome.

    Produced by any executor that runs quantization/evaluation. quant_flow is
    the standard CI/pipeline backend, but agentic skills may also execute
    directly during day-0 onboarding or repair. This object is the shared
    source of truth for metrics and artifact pointers.
    """

    config: OptimizationConfig
    run_id: str | None = None

    # Provenance
    executor: RunExecutor = RunExecutor.QUANT_FLOW
    executor_details: dict = Field(default_factory=dict)

    # Outcome
    status: Literal["success", "failed", "partial", "anomalous", "in_progress"]
    failure_class: FailureClass | None = None
    failure_message: str | None = None  # human-readable summary

    # Scores (task_name -> metric value)
    scores: dict[str, float] = Field(default_factory=dict)
    score_stderrs: dict[str, float] = Field(default_factory=dict)

    perf: PerfStats | None = None

    artifacts: ArtifactRefs = Field(default_factory=ArtifactRefs)

    # Timing
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None


# ──────────────────────────────────────────────────────────────────────
# Reliability hand-off
# ──────────────────────────────────────────────────────────────────────


class ReliabilityVerdict(BaseModel):
    """Agent decision based on config, RunResult evidence, and failures.

    Used both for day-0 success cases and failure triage.
    """

    decision: Literal[
        "READY_FOR_SWEEP",
        "WORKS_OOTB",
        "PATCH",
        "POINT_INFEASIBLE",
        "SYSTEMIC",
        "NEEDS_HUMAN",
    ]
    # READY_FOR_SWEEP — config has enough evidence to enter bigpareto search
    # WORKS_OOTB — default config ran successfully without patching
    # PATCH — try this modified config (set in `patched_config`)
    # POINT_INFEASIBLE — this (model, recipe, launcher) point can't work; skip
    # SYSTEMIC — infra-wide issue (cluster down, dataset unavailable); pause sweep
    # NEEDS_HUMAN — agent can't decide; escalate

    reason: str  # human-readable

    # Config to continue with. For PATCH, this is usually the patched config.
    # For WORKS_OOTB / READY_FOR_SWEEP, this can be the default config.
    recommended_config: OptimizationConfig | None = None
    patched_config: OptimizationConfig | None = None  # set iff decision == PATCH

    # RunResult IDs used as evidence. IDs can be protocol run_id, MLflow run IDs,
    # JET pipeline IDs rendered as strings, or another stable external handle.
    run_result_ids: list[str] = Field(default_factory=list)

    # Evidence the agent used: log excerpts, traceback snippets, playbook refs
    evidence: list[str] = Field(default_factory=list)

    confidence: Literal["high", "medium", "low"] = "medium"


class AgenticSkillOutput(BaseModel):
    """Output envelope for ModelOpt agentic skills.

    Agentic skills may call quant_flow or run quant/eval directly. If they
    execute directly, they should return the resulting RunResult here. Either
    way, the verdict summarizes what to do next.
    """

    config: OptimizationConfig | None = None
    run_results: list[RunResult] = Field(default_factory=list)
    verdict: ReliabilityVerdict


# ──────────────────────────────────────────────────────────────────────
# Smoke test: round-trip via JSON
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = OptimizationConfig(
        config_id="bp_sweep001_nano30b_nvfp4_oci-hsg",
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        recipe=RecipePath(path="examples/recipes/nvfp4/max.yaml"),
        launcher="oci-hsg",
        slurm_account="coreai_dlalgo_modelopt",
        deployment=DeploymentSpec(framework="vllm", tensor_parallel_size=4),
        evaluation=EvaluationSpec(
            nel_config_path="configs/AA/nano30b.yaml",
            tasks=["simple_evals.gpqa_diamond_aa_v3"],
            limit_samples=10,
        ),
        metadata={"sweep_id": "exp001", "pareto_iteration": 3},
    )

    result = RunResult(
        config=cfg,
        run_id="qf_pipeline_44881906",
        executor=RunExecutor.QUANT_FLOW,
        executor_details={"backend": "jet", "pipeline_id": "44881906"},
        status="failed",
        failure_class=FailureClass.QUANT_COVERAGE_FAILURE,
        failure_message=(
            "0 of 224 attention projections matched recipe wildcard `*self_attn*proj*`"
        ),
        artifacts=ArtifactRefs(
            jet_pipeline_id=44881906,
            checkpoint_uri="model/nano30b/modelopt_nvfp4:pipeline_44881906",
            log_uris={
                "client": "/lustre/.../client.log",
                "server": "/lustre/.../server.log",
            },
        ),
    )

    verdict = ReliabilityVerdict(
        decision="PATCH",
        reason=(
            "Recipe wildcard `*self_attn*proj*` doesn't match Nemotron-3's "
            "module names `self_attention.proj_*`. Patched recipe to use "
            "`*self_attention*proj*`."
        ),
        recommended_config=OptimizationConfig(
            **{**cfg.model_dump(), "config_id": cfg.config_id + "_patch1"}
        ),
        patched_config=OptimizationConfig(
            **{**cfg.model_dump(), "config_id": cfg.config_id + "_patch1"}
        ),
        run_result_ids=["qf_pipeline_44881906"],
        evidence=[
            "debugging-playbooks: nemotron-3-attn-naming",
            "Module list grep: 0 matches for original pattern",
        ],
        confidence="high",
    )

    ootb_result = RunResult(
        config=cfg,
        run_id="agentic_direct_day0_001",
        executor=RunExecutor.AGENTIC_DIRECT,
        executor_details={"runner": "modelopt-agentic-skill", "mode": "day0"},
        status="success",
        scores={"simple_evals.gpqa_diamond_aa_v3": 0.42},
        artifacts=ArtifactRefs(
            mlflow_run_id="mlflow-agentic-day0-001",
            checkpoint_uri="model/nano30b/modelopt_default:agentic_day0_001",
            log_uris={"agent": "/lustre/.../agentic_day0.log"},
        ),
    )

    ootb_verdict = ReliabilityVerdict(
        decision="WORKS_OOTB",
        reason="Default config ran successfully in the day-0 agentic path.",
        recommended_config=cfg,
        run_result_ids=["agentic_direct_day0_001"],
        evidence=["RunResult agentic_direct_day0_001 completed with status=success"],
        confidence="high",
    )

    agentic_output = AgenticSkillOutput(
        config=cfg,
        run_results=[ootb_result],
        verdict=ootb_verdict,
    )

    # Round-trip via JSON to confirm schemas serialize cleanly
    samples: list[tuple[BaseModel, str]] = [
        (cfg, "OptimizationConfig"),
        (result, "RunResult"),
        (verdict, "ReliabilityVerdict"),
        (agentic_output, "AgenticSkillOutput"),
    ]
    for obj, label in samples:
        s = obj.model_dump_json(indent=2)
        roundtrip = type(obj).model_validate_json(s)
        assert roundtrip == obj, f"{label} did not round-trip"
        print(f"=== {label} ===")
        print(s)
        print()
    print("All schemas round-trip cleanly via JSON.")
