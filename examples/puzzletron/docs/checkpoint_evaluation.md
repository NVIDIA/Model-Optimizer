# Text checkpoint evaluation

Use the unified text-evaluation command to evaluate a compatible local Hugging
Face checkpoint without creating or running a Puzzletron campaign. Puzzletron
uses `lmms-eval` by default. Use NeMo Evaluator when the pinned `lmms-eval`
revision does not provide the required benchmark or when an exact
NeMo/Artificial Analysis task contract is required for an external comparison.

Both routes share one Puzzletron command:

- `python -m examples.puzzletron.evaluation.text ...` runs `lmms-eval`.
- `python -m examples.puzzletron.evaluation.text --backend nemo ...` prepares
  a NeMo Evaluator configuration for a later NeMo Evaluator launch.

Backend-specific options are forwarded unchanged. The command does not
translate task contracts or results between evaluators. Overlapping task names
do not imply identical prompts, sampling, or scoring, so keep results from
different contracts separate.

For pinned image and video benchmarks on a Qwen 3.5 0.8B checkpoint, use the
separate [VLM checkpoint evaluator](vlm_checkpoint_evaluation.md).

## Quick start

Run the command in the Puzzletron worker image described in the
[environment setup guide](environment_setup.md#worker-environment). Mount the
checkpoint and output paths into the container, then run the default smoke:

```bash
python -m examples.puzzletron.evaluation.text \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/checkpoint-smoke
```

This evaluates eight samples each from IFEval and GSM8K on one GPU. Results and
logs are written under the output directory.

## Choose a benchmark route

| Benchmark | Default | Other available route | Reason |
| --- | --- | --- | --- |
| IFEval | `lmms-eval` | None maintained here | Included in the default checkpoint smoke. |
| GSM8K | `lmms-eval` | None maintained here | Included in the default checkpoint smoke. |
| GPQA Diamond | `lmms-eval` | NeMo Evaluator `gpqa_diamond_aa_v3` | Use the NeMo route only when the simple-evals AA v3 contract is required. |
| MMLU-Pro | `lmms-eval` | NeMo Evaluator `mmlu_pro_aa_v3` | Use the NeMo route only when the simple-evals AA v3 contract is required. |
| AIME 2025 | `lmms-eval` | NeMo Evaluator `AIME_2025_aa_v2` | The NeMo route uses a separate 64-sample, judge-scored contract. |
| LiveCodeBench v6 | NeMo Evaluator | Existing standalone LiveCodeBench script | The pinned `lmms-eval` revision does not include LiveCodeBench. |
| SciCode | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision. |
| IFBench | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision; IFBench is not IFEval. |
| HLE | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision and requires a judge. |
| AA-LCR | NeMo Evaluator | None maintained here | Not included in the pinned `lmms-eval` revision and requires a judge and long context. |
| Tau2 Telecom | NeMo Evaluator | `lmms-eval` infrastructure smoke only | The pinned `lmms-eval` task is a small agent-loop demonstration, not the complete Tau2 contract. |

The NeMo compiler defaults to LiveCodeBench v6, SciCode, and IFBench. GPQA
Diamond and MMLU-Pro are available only when explicitly selected for their
alternate NeMo contracts.

## Run with lmms-eval

Choose tasks and common runtime settings with command-line options:

```bash
python -m examples.puzzletron.evaluation.text \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/custom-smoke \
  --tasks ifeval,gsm8k \
  --limit 32 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192
```

Qwen 3.5 checkpoints are detected from their local `config.json` and configured
automatically. Use `--reasoning-parser` to override the detected parser or
`--model-profile none` to disable model detection.

Use `--trust-remote-code` only after reviewing the checkpoint-provided Python
code. After the smoke succeeds, use `--full` with a separate output directory
to evaluate the complete task datasets. Use `--timeout-seconds` if the full run
needs a different limit.

Pass additional native options after `--lmms-eval-args`, which must be the last
wrapper option. See `python -m lmms_eval --help` for the available options.

## Prepare a NeMo Evaluator run

Prepare the NeMo task configuration without launching it:

```bash
python -m examples.puzzletron.evaluation.text --backend nemo \
  --base-config path/to/base_nel_config.yaml \
  --output path/to/text_benchmarks.yaml
```

Use repeated `--task` options to select explicit NeMo task contracts. Keep the
generated configuration fixed, then use NeMo Evaluator's native
`limit_samples=2` or `limit_samples=10` launcher override for smoke or short
runs. See the [NeMo Evaluator guide](../../llm_eval/NEMO_EVALUATOR.md) for
preparation, launch, and validation.

## Record and troubleshoot results

Each run creates a new `attempt_<id>/` directory. Start with `summary.json` for
metrics. If a run fails, inspect `stderr.txt`; the command and raw evaluator
output are retained in the same directory. Rerunning creates another attempt
without overwriting the earlier one.

Every reported result must state the evaluator, exact task contract, evaluator
revision or evaluation-container tag, sample limit, and judge or user-simulator
identity when applicable. For `lmms-eval`, retain `command.json`,
`summary.json`, the raw result, and the pinned revision from the
[CI environment](../ci_environment.json). For NeMo Evaluator, retain the
generated YAML and launcher result artifacts.

To evaluate candidates as part of a pruning campaign, use
[saved-checkpoint evaluation](post_mip_pipeline.md#evaluate-saved-checkpoints) instead.
