# Qwen 3.5 0.8B text pruning smoke

The `full_smoke` recipe runs a small end-to-end test of text-only
pruning for Qwen 3.5 0.8B. It searches the FFN intermediate sizes
`[3072, 2048]`, evaluates the candidates, saves the two strongest candidates as
physical checkpoints, and reloads each saved directory through vLLM for two
IFEval samples. It then measures both checkpoints with AIPerf, distills the
candidate with higher measured output-token throughput for two steps, evaluates
the resulting checkpoint with another two IFEval samples and the internal
two-sample LM-loss check, and selects the final checkpoint. The recipe pins the
public checkpoint revision so repeated runs use the same starting model. See
[evaluate saved checkpoints](post_mip_pipeline.md#evaluate-saved-checkpoints)
for how both Hugging Face directories are loaded without an AnyModel-to-AutoModel
conversion.

These small budgets check that the complete workflow runs and resumes
correctly. They do not establish model quality or production throughput.

## Before you start

Prepare the controller venv and worker environment described in
[environment setup](environment_setup.md). The worker environment must provide
the [pinned evaluator installation](checkpoint_evaluation.md#quick-start).
IFEval task data must be fetchable from each worker or already present in its
Hugging Face cache.

## Generate a complete bundle with the setup wizard

For a new run, start with the [setup wizard](setup_wizard.md) and select Qwen
3.5 0.8B. Its model-specific defaults generate both smoke and production
bundles covering conversion, pruning, search, MIP selection, materialization,
serving measurement, short distillation, final selection, and a pinned
student-versus-teacher quality comparison. Inspect the generated
`dry-run-plan.txt` and materialize the site-specific runner settings before
launching. The comparison is measurement-only and does not accept the current
scores as a quality baseline.

The tracked recipes below remain useful as reviewable reference configurations
and for reproducing the bounded GPU comparison.

## Run and resume the text workflow

Use the `full_smoke.yaml` experiment with the shared `execution.single_gpu.yaml`
profile. The `runner.slurm.yaml` file is a portable template, not a
runnable site configuration. Copy it to a site-specific location and replace its
`REPLACE_WITH_` values before launching. Dry-run accepts the portable template
for plan inspection, but the orchestrator rejects unresolved placeholders
before submitting work.

This is the per-attempt Slurm route. Each scheduled GPU attempt requests one
GPU and may run in a separate allocation, so it is not the reusable single-node
VLM smoke.

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_full_smoke
```

Inspect the complete plan, which uses one GPU per scheduled GPU attempt,
without submitting work:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

After reviewing the dry-run, launch with the same three inputs and omit only
`--dry-run`:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

Resume by rerunning that exact launch command with the same experiment,
materialized runner, execution config, and output root.

The flow deliberately uses two candidate-evaluation samples, two
IFEval samples, four AIPerf requests per serving candidate, and two
distillation steps. These budgets validate workflow correctness, comparative
serving selection, and resumability; they are not quality or throughput claims.

After completion, inspect the `checkpoint_eval` and `post_kd_checkpoint_eval`
nodes under `artifacts/post_mip/nodes`. Their summaries must name the corresponding
pre-KD and post-KD checkpoints, report two effective IFEval samples, and contain
finite metrics. Also verify the cumulative report and confirm that resuming
submits no work for completed stages.
