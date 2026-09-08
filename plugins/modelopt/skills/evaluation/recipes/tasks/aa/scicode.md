# SciCode

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/index.html>

## Params

SciCode is a NeMo Skills code/reasoning benchmark with multi-step prompts and a
code-execution sandbox. Check this reference before creating or modifying NEL
configs for SciCode; the benchmark has deployment, parallelism, and score
harvesting requirements beyond the task YAML fragment.

## Config Requirements

- **Deployment context length:** at least `--max-model-len 65536` (SciCode
  multi-step prompts can exceed 32K). The example template's default of
  `--max-model-len 131072` satisfies this and is preferred — do not lower
  it unless you have a memory reason to.
- **Parallelism:** set task-level `parallelism: 8` exactly. Use the same value
  for baseline and candidate.
- **Repeats: separate runs, not `num_repeats: 8`.** Keep `num_repeats: 1` —
  in-run repeats multiply code-execution sandbox exposure — and submit the task
  8+ times instead. See [At least 8 runs](#at-least-8-runs-mandatory).

## At least 8 runs (mandatory)

One run is never a reportable SciCode score: at `temperature 1.0` its noise
rivals a 1 % gate — a paired comparison moved **3.92 pp and changed sign** when
repeated, and a DeepSeek-V4-Pro drop read 2.96 pp (`REGRESSION`) at 1 run but
**-0.96 pp** (`PASS`) at 8. **8 is a floor, not a judgment call** — pool more
when you have them, never fewer.

- **Submit until you hold at least 8 *valid* runs** — normally 8 submissions,
  more if any fail validation; at least 8 on each side of a comparison (8-vs-1
  is not apples-to-apples). In a multi-task AA config, run the suite once, then
  SciCode alone for the rest:
  `for _ in $(seq 7); do nel run --config <cfg> -t ns_scicode; done`.
- **Each must be a fresh `nel run`.** Re-submitting a run's `run.sub` resumes
  from its response cache and replays generations. Equal scores are a **signal
  to check, not proof** — SciCode's score is discrete, so independent runs tie
  often; confirm replay from provenance (invocation id, output dir, response
  artifacts) before discarding, then resubmit to restore the pool.
- **Report the mean** of the n pooled runs, with the **sample** stdev (n-1
  denominator) over `sqrt(n)` as the standard error, and n itself.
  Validate each run (`references/run-validation.md`) before averaging it in;
  resubmit to replace invalid runs rather than pooling a sandbox-crashed one.
- **Expect an export to time out.** The exports land on the CPU partition
  together and each reinstalls the launcher against a fixed 30 min sbatch
  limit. The score survives in the run artifacts — re-submit that run's
  `export.sbatch`; it is not a lost run.
- **Fewer than 8 valid runs is `INDETERMINATE`**, never a pass/fail verdict.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_scicode
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        parallelism: 8
        extra:
          args: ++prompt_config=eval/scicode/default ++with_background=true
          num_repeats: 1 # keep at 1; submit this config 8x and average (see above)
```

## Score Extraction from mlflow

Per-run score (0-100): `scicode_pass_at_1_subtask_accuracy`

**No `avg-of-N` segment at `num_repeats: 1`** — the harness adds one only when
N > 1 (cf. `gpqa_pass_at_1_avg-of-16_symbolic_correct`); the `avg-of-1` name
silently harvests nothing.

The reported SciCode result is the **mean of that field across all pooled
runs**, not any single run's value.
