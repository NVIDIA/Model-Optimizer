---
name: modelopt-eval-guidance
description: Tool-neutral guidance for judging a quantized checkpoint's accuracy, whatever evaluation tool runs it - pick quantization-sensitive benchmarks, check the baseline against the published score, take the tool's run verdict and report per-side limit counts for the comparison, and decide on doom-loop detection. Use it next to the evaluation tool's own skills whenever a baseline or candidate is evaluated; the tool fills the config and runs the evaluation.
license: Apache-2.0
---

# Evaluation Guidance

The evaluation tool owns what values go in the config and how the run goes (protocol, judge, sampling, budget,
parsers, quantized-variant settings, submit, babysit, recover, MLflow tags, and whether the run is healthy and
complete). This skill holds what the Day 0 workflow does around the tool: which benchmarks to run, asking for the baseline
and the quantized evaluation (both go through the tool), the serving record, whether the baseline is believable, and
what the comparison needs from a finished run. It applies to every tool.

Tool-specific details live in [references/tools/](references/tools/index.md), one file per tool: the skills that
run it, where its health verdict and settings are, and its MLflow names. Use the skills and the reference of the
tool the task names (the default is named in the index), and do not load another tool's skills.

## What to read, in order

0. **First, the reference file of the tool you are using** (`references/tools/<tool>.md`). Do not guess its
   name: open [references/tools/index.md](references/tools/index.md) and follow the link for your tool. It holds the rules that apply only to that tool, such as what must not be cancelled, where its
   sampling and health rules are, and its MLflow names; the files below defer to it.
1. **When choosing benchmarks for a quantized checkpoint:**
   [quantization-benchmarks.md](references/quantization-benchmarks.md).
2. **When a model repeats itself until the token limit:** [doom-loops.md](references/doom-loops.md).
3. **After each run finishes, before any score is compared:** [run-validity.md](references/run-validity.md) (take the tool's verdict; report per-side limit counts).
4. **Before accepting a baseline for a baseline-versus-candidate comparison:**
   [external-baseline-check.md](references/external-baseline-check.md), then `modelopt-compare-results`.

## Rules for every evaluation, whatever the tool

- **Serving deployment of a quantized checkpoint is ours.** The tool's model-card reading knows the baseline, not the
  quantized format. Before any run of a quantized checkpoint, load `modelopt-deployment`, write the serving record
  (step 5c: image that supports the format, parallelism, quantization and KV-cache flags from the checkpoint's
  `hf_quant_config.json`) and hand it to the tool, which then applies it to the baseline's config. This holds even
  when the tool can derive a deployment on its own. For a BF16 baseline the tool's own deployment stands.
- **Tool-managed deployments.** Evaluate through the tool's own managed deployments. Evaluate an external or
  retained model-under-test endpoint only when the user explicitly asks; a retained PTQ smoke server is not
  authorization to evaluate it.
- **Never report scores from an incomplete or invalid run.** The tool's verdict decides; label the result
  incomplete or invalid and return the findings (see [run-validity.md](references/run-validity.md)).
- **MLflow.** The tool exports the results; confirm the tool's export of the results to MLflow arrived and put the run id
  in the handoff.
- **Report timeout and output-limit counts** for both sides of a comparison, as `modelopt-compare-results` asks.

## Rules that belong to the tool, not to this skill

- **The output-token budget, the repeat count and the score definition** are set by the tool's benchmark
  configuration. Record the model card's value next to the tool's value; do not replace the tool's value.
- **The health verdict** (what counts as an evaluation fault, the fault limit, what counts as the model's
  own result such as a truncated answer) is the tool's. Its reference says where that rule is; apply a tool's
  policy only to that tool's runs.
- **Never change the evaluation to make a run fit:** no extra row limits beyond a smoke run, fewer repeats,
  smaller token budgets, or shorter timeouts.

## Reading results

Read results from MLflow when the run was exported there, with the run id from the handoff. Both tools
export to the same experiment; the metric names differ by tool, so confirm the name in the run before using
it. Take scores and token counts from a run only when its status is `FINISHED` and the metric exists, and take
the task status from the tool's own verdict (a benchmark's run can be `FINISHED` while its submission failed
overall). Details are in [run-validity.md](references/run-validity.md#reading-results-from-mlflow) and the tool's
reference.
