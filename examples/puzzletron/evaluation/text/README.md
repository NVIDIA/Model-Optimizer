# Text evaluation

Puzzletron uses `lmms-eval` by default. Use NeMo Evaluator when the pinned
`lmms-eval` revision does not provide the required benchmark, or when an exact
NeMo/Artificial Analysis task contract is required for an external comparison.

Both routes share one Puzzletron command:

- `python -m examples.puzzletron.evaluation.text ...` runs `lmms-eval` by default.
- `python -m examples.puzzletron.evaluation.text --backend nemo ...` prepares a NeMo
  Evaluator configuration, which is then launched with NeMo Evaluator.

The command forwards backend-specific options unchanged. It does not translate
task contracts or results between evaluators. The backends can evaluate the
same checkpoint, but overlapping task names do not imply identical prompts,
sampling, or scoring. Do not combine or directly compare scores from different
contracts.

## Default routes

| Benchmark | Default | Other available route | Reason |
| --- | --- | --- | --- |
| IFEval | `lmms-eval` | None maintained here | Included in the default checkpoint smoke. |
| GSM8K | `lmms-eval` | None maintained here | Included in the default checkpoint smoke. |
| GPQA Diamond | `lmms-eval` | NeMo Evaluator `gpqa_diamond_aa_v3` | Prefer the pinned `lmms-eval` task unless the existing simple-evals AA v3 contract is required. |
| MMLU-Pro | `lmms-eval` | NeMo Evaluator `mmlu_pro_aa_v3` | Prefer the pinned `lmms-eval` task unless the existing simple-evals AA v3 contract is required. |
| AIME 2025 | `lmms-eval` | NeMo Evaluator `AIME_2025_aa_v2` | The NeMo route uses a different 64-sample, judge-scored contract. |
| LiveCodeBench v6 | NeMo Evaluator | Existing standalone LiveCodeBench script | The pinned `lmms-eval` revision does not include LiveCodeBench. |
| SciCode | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision. |
| IFBench | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision; IFBench is not IFEval. |
| HLE | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision and requires a judge. |
| AA-LCR | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision and requires a judge and long context. |
| Tau2 Telecom | NeMo Evaluator | `lmms-eval` infrastructure smoke only | The pinned `lmms-eval` task is a small agent-loop demonstration, not the complete Tau2 contract. |

The NeMo compiler defaults only to LiveCodeBench v6, SciCode, and IFBench, which
fill gaps in the pinned `lmms-eval` catalog. GPQA Diamond and MMLU-Pro are
available only when explicitly selected for their alternate NeMo contracts.

## Run the default route

The smallest `lmms-eval` smoke uses IFEval and GSM8K:

```bash
python -m examples.puzzletron.evaluation.text \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/checkpoint-smoke
```

Select another task from the pinned `lmms-eval` catalog with `--tasks`. For
example:

```bash
python -m examples.puzzletron.evaluation.text \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/gpqa-smoke \
  --tasks gpqa_diamond_openai \
  --limit 8
```

## Use the NeMo route

Prepare the NeMo task configuration without launching it:

```bash
python -m examples.puzzletron.evaluation.text --backend nemo \
  --base-config path/to/base_nel_config.yaml \
  --output path/to/text_benchmarks.yaml
```

Use repeated `--task` options to select an explicit NeMo task contract. Keep
this generated config fixed, then use NeMo Evaluator's native
`limit_samples=2` or `limit_samples=10` launcher override for smoke or short
runs. See
[`NEMO_EVALUATOR.md`](../../../llm_eval/NEMO_EVALUATOR.md) for preparation,
launch, and validation.

## Result identity

Every reported result must state:

- evaluator: `lmms-eval` or `nemo-evaluator`;
- exact task name and task contract;
- evaluator revision or evaluation-container tag;
- profile or sample limit; and
- judge or user-simulator identity when the task uses one.

For `lmms-eval`, retain `command.json`, `summary.json`, and the raw result in the
attempt directory together with the pinned revision from
[`ci_environment.json`](../../ci_environment.json). For NeMo Evaluator, retain
the generated YAML, whose task entries include the task name and container, with
the launcher result artifacts. If both routes are run, publish separate rows
labeled with their evaluator and contract; do not average them.
