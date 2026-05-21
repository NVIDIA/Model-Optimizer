---
name: compare-results
description: Establish baseline-vs-candidate evaluation plans, delegate missing evaluations, compare validated results, and decide quantization feasibility. Use when the user asks to compare baseline vs quantized runs, explain an accuracy drop/regression, verify whether a quantized checkpoint is acceptable, or compare NEL/MLflow evaluation outputs. Do NOT use for generic single-model evaluation without comparison intent (use evaluation), live NEL status/debugging (use launching-evals), or generic MLflow browsing without a comparison goal (use accessing-mlflow).
license: Apache-2.0
---

# Compare Results

Use this to plan and complete a baseline-vs-candidate comparison. The baseline
is the reference checkpoint, and the candidate is the checkpoint whose accuracy
change is being measured, typically a further quantized version of the baseline.

## Workflow

1. Establish the candidate checkpoint/run and the matching baseline. Infer the
   baseline from the PTQ source model/checkpoint in the workspace or config used
   to create the candidate. If it cannot be inferred, ask the user for the
   baseline checkpoint or an existing baseline invocation/run path.
2. If a required baseline or candidate evaluation is missing, delegate to the
   evaluation skill to create, run, and verify it. The companion evaluation
   config should match benchmark versions, task configs, serving args, token
   limits, dataset setup, credentials, cluster, and container as closely as
   possible; change only the model/checkpoint and checkpoint-specific serving or
   quantization flags.
3. Fetch the baseline and candidate task list, configs, score artifacts, and
   logs. If the user provides MLflow runs or invocation IDs, use the
   accessing-mlflow skill to fetch configs and artifacts.
4. Confirm each run passed evaluation Step 9, "Verify completed evaluation run",
   before comparing scores. If not, validate logs, server health,
   judge/code-execution status, sample accounting, and reasoning parsing before
   computing deltas.
5. For each task, use the canonical score field from the matching
   `.claude/skills/evaluation/recipes/tasks/<task>.md` Score Extraction
   section.
6. Compute exact deltas outside the chat context when there are multiple tasks
   or repeated runs.
7. Report comparability and quantized-feasibility verdicts before interpreting
   the delta as model quality. If the user did not provide an acceptance
   threshold, report feasibility as inconclusive instead of inventing one.

## Comparability Checklist

Before treating a baseline-vs-quantized delta as a model quality result, verify
the validated runs are comparable:

1. Prompt text, system prompt, chat template, and rendered messages match.
2. Task name, benchmark version, dataset split, container, harness, and task
   fragment match.
3. Generation settings match, including temperature, top_p, top_k, max tokens,
   stop strings, chat-template kwargs, reasoning mode/budget, and task-specific
   overrides.
4. Reasoning traces are enabled, disabled, parsed, stripped, or ignored
   consistently between runs.
5. The number of evaluated and scored samples/repeats matches for each task and
   split.
6. Judge-backed or simulator-backed tasks use the same judge/user model,
   endpoint class, prompt, and scoring config.
7. The same accuracy metric and score field is used for both runs.

If any item differs, either rerun with matched settings or label the result as
not an apples-to-apples quantization comparison.

## Report Format

Include:

- Baseline and candidate identifiers.
- Per-task metric path, baseline score, candidate score, delta, and stderr if
  available.
- Comparability status for prompt/template, generation settings, sample counts,
  reasoning handling, judge/simulator setup, and score field.
- Comparability verdict: comparable, not comparable, or inconclusive.
- Quantization feasibility verdict: acceptable, not acceptable, or inconclusive.
