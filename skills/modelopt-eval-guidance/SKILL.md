---
name: modelopt-eval-guidance
description: Tool-neutral guidance for the accuracy evaluation of a quantized checkpoint, whatever evaluation tool runs it - read the vendor model card, pick quantization-sensitive benchmarks, validate a finished run, check the baseline against the published score, and handle doom loops. Use it next to the evaluation tool's own skills whenever a baseline or candidate is evaluated; it holds the duties that do not depend on the tool.
license: Apache-2.0
---

# Evaluation Guidance

The evaluation tool runs the evaluation: it composes the config, submits, resumes, and reports its own
health verdict. This skill holds what a quantization workflow needs on top of that, and it applies to
every tool.

Tool-specific details live in [references/tools/](references/tools/index.md), one file per tool: the skills that
run it, where its health verdict and settings are, and its MLflow names. Use the skills and the reference of the
tool the task names (the default is named in the index), and do not load another tool's skills.

## What to read, in order

0. **First, the reference file of the tool you are using** (`references/tools/<tool>.md`). Do not guess its
   name: open [references/tools/index.md](references/tools/index.md) and follow the link for your tool. It holds the rules that apply only to that tool, such as what must not be cancelled, where its
   sampling and health rules are, and its MLflow names; the files below defer to it.
1. **Before choosing settings:** [vendor-reference-and-sampling.md](references/vendor-reference-and-sampling.md)
   - the quantization-specific part of reading the model card (the tool's own skill does the extraction), the
   output-budget rule, and the published NVFP4 sampling table
   [nvfp4-modelcard-sampling.md](references/nvfp4-modelcard-sampling.md).
2. **When choosing benchmarks for a quantized checkpoint:**
   [quantization-benchmarks.md](references/quantization-benchmarks.md).
3. **When a model repeats itself until the token limit:** [doom-loops.md](references/doom-loops.md).
4. **After each run finishes, before any score is reported:** [run-validity.md](references/run-validity.md).
5. **Before accepting a baseline for a baseline-versus-candidate comparison:**
   [external-baseline-check.md](references/external-baseline-check.md), then `modelopt-compare-results`.

## Rules for every evaluation, whatever the tool

- **Serving deployment.** Before evaluating a checkpoint, check whether the tool's skills need a serving
  deployment from the caller (image, parallelism, serve flags). If they do, load `modelopt-deployment`, write the
  serving record (step 5c) and hand it to the tool. A tool that composes its own deployment keeps doing so.
- **Tool-managed deployments.** Evaluate through the tool's own managed deployments. Evaluate an external or
  retained model-under-test endpoint only when the user explicitly asks; a retained PTQ smoke server is not
  authorization to evaluate it.
- **Complete the tool's own steps** (dry run, canary, full run, validation) before reporting anything.
- **Never report scores from an incomplete or invalid run.** Label the result incomplete or invalid and return
  the findings (see [run-validity.md](references/run-validity.md)).
- **MLflow.** Configure the tool's export of the results to MLflow, verify that the run arrived, and say in the
  handoff whether it did.
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
