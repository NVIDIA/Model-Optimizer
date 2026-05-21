---
name: compare-results
description: Compare completed baseline and quantized evaluation results, verify apples-to-apples comparability, and report accuracy deltas. Use when the user asks to compare baseline vs quantized runs, explain an accuracy drop/regression, verify a quantization result, or compare completed NEL/MLflow evaluation outputs. Do NOT use for creating or launching new evaluation configs (use evaluation), live NEL status/debugging (use launching-evals), or generic MLflow browsing without a comparison goal (use accessing-mlflow).
license: Apache-2.0
---

# Compare Results

Use this after baseline and candidate evaluation runs have completed. The
baseline is the reference checkpoint, and the candidate is the checkpoint whose
accuracy change is being measured, typically a further quantized version of the
baseline.

## Workflow

1. Identify the baseline run, candidate run, task list, and result artifacts.
   If the user provides MLflow runs or invocation IDs, use the accessing-mlflow
   skill to fetch configs and artifacts. If a required run does not exist, use
   the evaluation skill to create it.
2. Confirm each run passed evaluation Step 9, "Verify completed evaluation run",
   before comparing scores. If not, validate logs, server health,
   judge/code-execution status, sample accounting, and reasoning parsing before
   computing deltas.
3. For each task, use the canonical score field from the matching
   `.claude/skills/evaluation/recipes/tasks/<task>.md` Score Extraction
   section.
4. Compute exact deltas outside the chat context when there are multiple tasks
   or repeated runs.
5. Report a comparability verdict before interpreting the delta as model
   quality.

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
- Final verdict: comparable, not comparable, or inconclusive.
