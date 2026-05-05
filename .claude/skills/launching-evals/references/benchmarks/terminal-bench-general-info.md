# Terminal Bench

Terminal Bench is an agentic benchmark where models interact with a terminal environment to solve tasks.

## Key files

- `terminal_bench/agents/terminus_2/terminus_2.py` — main agent implementation
- `terminal_bench/agents/failure_mode.py` — failure mode definitions
- `terminal_bench/harness/harness.py` — harness and result aggregation
- `core_evals/nvidia_terminal_bench/framework.yml` — default config values

### Key Facts

- **Task-first ordering**: `task1.1-of-N, task1.2-of-N, ..., task2.1-of-N, ...` — mid-run results are biased toward early tasks.

## Failure Modes

All failure modes (see `failure_mode.py`):
- `UNSET` — no failure mode triggered (task ran to completion)
- `NONE` — explicitly set: no failure (task solved)
- `UNSOLVED` — task not completed within constraints
- `TOKEN_LIMIT_EXCEEDED` — agent hit `max_input_tokens_per_task` (cumulative input tokens across all turns). Shows as `outcome: token_limit_exceeded` in `task_status.json`.
- `PARSE_ERROR` — harness couldn't parse the **test output** (`post-test.txt`), e.g. pytest output missing `short test summary info`
- `FATAL_LLM_PARSE_ERROR` — unrecoverable LLM/agent response parse error
- `CONTEXT_LENGTH_EXCEEDED` — input exceeded model's context window (see [Context Recovery](#context-recovery))
- `OUTPUT_LENGTH_EXCEEDED` — response truncated by `max_completion_tokens`; agent retries; recorded when all retries exhausted. Shows as `finish_reason: length` in `eval_factory_metrics.json`.
- `TEST_TIMEOUT` — test verification timed out
- `AGENT_TIMEOUT` — agent execution timed out (see [Mitigating Agent Timeouts](#mitigating-agent-timeouts))
- `UNKNOWN_AGENT_ERROR` — unexpected agent error (stops eval on default policy)
- `AGENT_INSTALLATION_FAILED` — agent setup failed (stops eval on default policy)
- `UNKNOWN` — unknown harness error (stops eval on default policy)

`failed_samples_policy` (default: `default`) — only stops on "no fair chance" failures: `UNKNOWN`, `UNKNOWN_AGENT_ERROR`, `AGENT_INSTALLATION_FAILED`. All other failures continue with score 0.

## Artifacts

All paths relative to `<output_dir>/<invocation>/terminal-bench-hard/`.

### Client logs

`logs/client-*.log` — contains rich/ANSI formatting (binary), always use `grep -a`. Shows live progress (`Running tasks (X/Y, Accuracy: Z%)`) and crash diagnostics.

### Run-level artifacts

Path: `artifacts/terminal-bench/`

| File | Written | Updated | Content |
|------|---------|---------|---------|
| `tb.lock` | Run start | Never | Full resolved config: invocation args, agent kwargs (`max_episodes`, `temperature`, `max_input_tokens_per_task`), run config (`n_concurrent_trials`, `global_agent_timeout_sec`, `failed_samples_policy`), ECS/sandbox settings. Best for reproducing runs. |
| `run_metadata.json` | Run start | Once at end | `model_name`, `dataset_name`/`dataset_version`, `n_concurrent_trials`, `task_ids`, `start_time`/`end_time`, `accuracy`, `pass_at_k` |
| `task_status.json` | After 1st task | After each task | One entry per task (not per trial). `status` (success/failed), `outcome`, `trial_name`. "Success is sticky" — once a task succeeds, later failures don't overwrite. 48 entries total. |
| `tb_results.json` | After 1st task | After each task | See below |

**Mid-run**: `task_status.json` and `tb_results.json` grow incrementally. `run_metadata.json` exists but lacks final metrics.

#### `tb_results.json` details

The richest single artifact.

**Per-trial fields:**
- `is_resolved` (bool) — ground truth for whether the task was solved. Use this, not `passed` or `score`.
- `failure_mode`, `parser_results` (dict of test name → "passed"/"failed")
- `instruction` — full task description given to the agent
- Token usage: `total_input_tokens`, `total_output_tokens`
- `trajectory_length` — number of agent episodes (turns)
- Timestamps: `trial_started_at`, `agent_started_at/ended_at`, `test_started_at/ended_at`
- `recording_path` — asciinema `.cast` file for replaying terminal sessions
- `error_type`, `error_message` — populated on crashes

**Aggregate fields:**
- `pass_at_k`, `accuracy`, `n_resolved`, `n_unresolved`
- `resolved_ids`, `unresolved_ids`
- `failure_mode_counts`, `error_type_counts`, `token_limit_exceeded_count`
- `total_input_tokens`, `total_output_tokens` — run-wide totals

Per-trial `artifacts/terminal-bench/<task>/<trial>/results.json` files are the source — `tb_results.json` aggregates them (same schema).

### Per-trial artifacts

Path: `artifacts/terminal-bench/<task>/<trial>/`

**Agent logs** (`agent-logs/episode-N/`, N = 0, 1, 2, ...):
- `prompt.txt` — full prompt sent to the model (system instructions + task + terminal state)
- `response.txt` — model's raw response (JSON with `analysis`, `plan`, `commands`, `task_complete`)
- `debug.json` — LiteLLM trace: model, messages, optional_params, `reasoning`/`reasoning_content` (chain-of-thought), token usage, `llm_api_duration_ms`, response headers

**Panes** (`panes/`) — terminal screen snapshots:
- `pre-agent.txt` — before agent starts (initial prompt)
- `post-agent.txt` — after agent finishes (all commands and outputs)
- `post-test.txt` — after test verification. If `failure_mode: parse_error`, check this first; for pytest tasks the summary block may be missing.

Panes are useful for quick triage without reading episode logs.

## Troubleshooting

### Mitigating Agent Timeouts

High `AGENT_TIMEOUT` rates (e.g. 85%+) are caused by inference contention: too many concurrent agent sessions competing for the same vLLM instance.

Two levers reduce contention: **lower parallelism** (fewer concurrent tasks) and **scale inference** (more deployment nodes / data-parallel replicas). Scaling inference has diminishing returns — requesting 32–64 nodes means long queue times and harder Slurm scheduling. The recommended approach combines both:

**Split into independent single-sample runs with lower parallelism (8x1 pattern):**

Instead of one run with `n_samples: 8, parallelism: 100`, submit 8 independent runs each with `n_samples: 1` and reduced `parallelism: 24`. This scales horizontally with multiple smaller jobs.

### Context Recovery

When the agent's input exceeds the model's context window, terminus_2 has two recovery paths. Both rely on `litellm.get_max_tokens(model_name)` to determine the context limit.

**Proactive path** (`_check_proactive_summarization`): Fires when `free_tokens < 8000` *before* the API call. Summarizes while the **full** conversation history is still available. This is the healthier path.

**Reactive path** (on `ContextLengthExceededError`): Fires after the API *rejects* a request:
1. **Unwind** (`_unwind_messages_to_free_tokens`): Drops the most recent user+assistant pairs until `free_tokens >= 4000`. Destructive — removed messages are permanently lost.
2. **Summarize** (`_summarize`): Asks the model (using truncated history) to summarize, generates questions from summary + `capture_pane()`, answers from truncated history, resets `chat._messages` to just 3 messages (original instruction + Q&A).

**Reactive path flaw**: Unwind drops recent messages *before* summarize runs. The terminal reflects those actions but the summary doesn't contain them. Only `capture_pane()` partially compensates.

**LiteLLM context limit is often wrong**: `litellm.get_max_tokens()` returns the *advertised* context window, not the deployment limit. For unknown models it falls back to 1M tokens; for `--max-model-len` smaller than default, it reports the full spec. When the limit is too high, unwind removes nothing, summarize hits the same error, and recovery is a no-op — propagates as `CONTEXT_LENGTH_EXCEEDED`.

## Agent Trace Analysis

See `references/benchmarks/terminal-bench-trace-analysis.md` for analyzing per-task agent traces, extracting behavior patterns, and categorizing failures.
