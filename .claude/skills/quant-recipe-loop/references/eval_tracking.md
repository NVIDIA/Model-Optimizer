# Evaluation Tracking Reference

## Metrics

Choose one primary accuracy metric per benchmark and keep it stable in the table.

Common examples:

- Exact/symbolic QA: symbolic correctness.
- Coding: pass@1 accuracy or pass@1 subtask accuracy, depending on benchmark definition.
- Instruction following: loose prompt accuracy when that is the accepted metric.
- Judge-based tasks: judge correctness/score, with judge model/config recorded.
- Agent/tool tasks: domain pass@1 plus failed-sample count.

Also collect verbosity:

```text
response_stats.avg_completion_tokens
```

When available, also inspect `count`, `successful_count`, `max_completion_tokens`, failure count, timeout count, and average latency.

## Result Sources

Preferred source order:

1. Final evaluator summary such as `results.yml`.
2. Detailed benchmark metrics such as `eval-results/**/metrics.json`.
3. Response stats such as `eval_factory_metrics.json`.
4. Client/server logs for backend confirmation, failures, and timing.

Every table row should be traceable to:

- Checkpoint and converted checkpoint path.
- Evaluator output directory.
- Evaluator config.
- Client and server logs.
- Job ID or invocation ID.
- Git branch/commit and container image.

## Comparison Hygiene

Keep these settings fixed within a table unless the row explicitly studies them:

- Sampling: temperature, top_p, top_k, min_p, penalties.
- Max new tokens / token cap.
- Reasoning parser and answer parser.
- Tool-call parser and auto-tool-choice flags.
- KV-cache dtype.
- Quantization backend/kernel.
- CUDA graph vs eager mode.
- Number of repeats/samples and benchmark split/version.

## Resume Policy

- Resume interrupted evaluator jobs from their run directory when the checkpoint/config is still correct.
- Prefer resume over manual relaunch for timeouts, transient judge/API errors, and rate limits.
- Relaunch only when the config, checkpoint, image, or serving flags were wrong.
- Preserve old output folders when changing configs so table provenance stays clear.
