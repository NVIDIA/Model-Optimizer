# Analyze the results

Copy this checklist and track your progress:

```
Analysis progress:
- [ ] Step 1: Gather information
- [ ] Step 2: Scan logs for runtime problems (per run)
- [ ] Step 3: Validate config and methodology (per run)
- [ ] Step 4: Report findings
```

Steps 2-3 are executed for EACH run separately.

## Step 1: Gather information

**IMPORTANT**: Copy what you need (and only what you need) locally BEFORE analysis — each SSH command requires user approval, so remote one-by-one reads are disruptive, and copying too much is slow.

- Get one or more successful invocation IDs to analyze from the user. You might already have the invocation ID in your memory from the previous step.
- Get paths: `uv run nemo-evaluator-launcher info <invocation_id>`
- If artifacts are local, read them directly from the paths shown by `nel info`.
- If artifacts are remote:
  - Copy logs: `uv run nemo-evaluator-launcher info <invocation_id> --copy-logs ./evaluation-results/`
  - Rsync analysis-relevant artifacts: `rsync -avzP <user>@<host>:<artifacts_path>/{results.yml,eval_factory_metrics.json,config.yml} ./evaluation-results/<invocation_id>.<job_index>/artifacts/`
- For MLflow access, see the `accessing-mlflow` skill.
- Read benchmark-specific analysis notes from `references/benchmarks/` if available for the evaluated benchmarks.
  - For Terminal Bench agent trace analysis, follow the procedure in `references/benchmarks/terminal-bench-trace-analysis.md`.

## Step 2: Scan logs for runtime problems

Access logs from locally copied files (`./evaluation-results/<invocation_id>.<job_index>/logs/`). Do NOT read logs via SSH — use the local copies from Step 1.

Check logs for silent errors that may invalidate results:

1. **Tool calling failures**: Search `client-*.log` for "failed" tests, `server-*.log` for "invalid tool call"
2. **Unfinished reasoning**: Check `server-*.log` for `finish_reason: length`, or truncation warnings in `client-*.log`
3. **API errors**: HTTP status != 200 in `client-*.log`, trace to `server-*.log` or `proxy-*.log`
4. **Config mismatches**: Compare `config.yml` params with actual values in `server-*.log` startup and `client-*.log` command
5. **Performance anomalies**: Low throughput, 0% prefix cache hit rate in `server-*.log`
6. **Cached responses**: Count "Returning cached response" in `client-*.log`
7. **KV cache preemptions**: Search `server-*.log` for `PreemptionMode.RECOMPUTE`. If found, consider increasing `tensor_parallel_size` (even at the cost of `data_parallel_size`) to relieve KV cache memory pressure.

## Step 3: Validate config and methodology

1. **Methodology consistency**: Verify same benchmark versions, prompt templates, sampling params, and infrastructure across all models. Flag discrepancies.
2. **Generation settings**: Follow the evaluation skill's provenance policy: explicit user/task requirements first, then applicable model-card settings explicitly used for evaluation/benchmarking; otherwise preserve checkpoint/vLLM defaults. Flag unsupported overrides, not deviations from general inference recommendations.
3. **Reasoning model validation**: Verify the intended reasoning mode and effective generation settings; do not impose generic sampling or token budgets.
   NOTE: `use_reasoning: False` in adapter_config does NOT mean reasoning is disabled — it only controls the reasoning interceptor. Whether reasoning is active depends on the model's own controls (deployment args, system prompt, API payload fields, etc.).
4. **Effective requests**: Check evaluator/adapter defaults and actual requests. Omitted fields and explicit `null` are not interchangeable; confirm required settings reach the server.
5. **Max model length**: Check prompt + output capacity, including multi-turn history, against the supported context window. Do not infer generation limits from `max_position_embeddings` or impose a generic context override.
6. **RULER tasks**: Check thinking disabled, walltime=4h, rope-scaling for Qwen models
7. **AA baseline comparison**: Compare results against Artificial Analysis published scores. Exact match not expected — flag significant deviations.
8. **Model baseline comparison**: For quantized runs, compare results against the matching baseline model run when available. The baseline may be unquantized or simply less quantized (for example, FP8 as the baseline for NVFP4). Use the same benchmark version, task config, serving args, token limits, dataset setup, and infrastructure before treating the delta as a quantization effect.

## Step 4: Report findings

Present key metrics from `results.yml` in a table and summarize the metrics from `eval_factory_metrics.json` in a concise manner (include only the most important metrics or anomalies). If multiple runs, include side-by-side comparison of metrics (e.g. accuracy, latency, tokens count, memory). Summarize any issues found. Recommend improvements if applicable.
