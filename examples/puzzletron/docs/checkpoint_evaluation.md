# Checkpoint evaluation

Use the standalone command to evaluate a compatible local Hugging Face
checkpoint without creating or running a Puzzletron campaign.

## Quick start

Install the Puzzletron worker requirements:

```bash
python -m pip install -r examples/puzzletron/requirements.txt
```

Then run the default smoke:

```bash
python examples/puzzletron/evaluate_lmms_checkpoint.py \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/checkpoint-smoke
```

This evaluates eight samples each from IFEval and GSM8K on one GPU. Results and
logs are written under the output directory.

## Customize the evaluation

Choose tasks and common runtime settings with command-line options:

```bash
python examples/puzzletron/evaluate_lmms_checkpoint.py \
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

## Results and troubleshooting

Each run creates a new `attempt_<id>/` directory. Start with `summary.json` for
metrics. If a run fails, inspect `stderr.txt`; the command and raw evaluator
output are retained in the same directory. Rerunning creates another attempt
without overwriting the earlier one.

To evaluate candidates as part of a pruning campaign, use
[downstream evaluation](post_mip_pipeline.md#downstream-evaluation) instead.
