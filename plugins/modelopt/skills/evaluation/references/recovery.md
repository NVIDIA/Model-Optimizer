# Recovery After Failed Run Validation

## Choose recovery from evidence

If acting as a subagent, return recovery guidance under `Blockers` to the parent
instead of rerunning; the parent coordinates matched baseline/candidate settings.
For analysis-only requests, recommend recovery without submitting jobs.

| Failure | Recovery action |
|---|---|
| Excessive truncation from output-budget exhaustion | Inspect raw responses to distinguish unfinished reasoning from repetition or context exhaustion. For repetition, apply [doom_loops.md](doom_loops.md) instead of raising the cap. For unfinished reasoning, **increase `max_new_tokens` and rerun** (e.g. `65536 → 131072`) when task policy permits; check context fit, KV headroom, and timeouts first. If truncation persists after trying the maximum supported context, reduce reasoning effort where supported; label the protocol change and match baseline/candidate settings. |
| Prompt/history exhausts context | Increase supported deployed context; a larger output cap alone cannot help. |
| GPU/host OOM, KV pressure, disk or quota exhaustion | Reduce concurrency, adjust supported resources, or clean trial-owned temporary data after preserving evidence; verify capacity before retrying. |
| Preemption or interrupted infrastructure work | Prefer supported same-config resume when cache identity and complete accounting can be proved. |
| Endpoint overload, rate limits, transient transport errors | Check health, back off, and reduce offending concurrency. Retry after recovery, not repeatedly into the same outage. |
| Sandbox provisioning/executor failure | Repair image/dependencies/capacity. Distinguish infrastructure faults from protocol-valid failed trials. |
| Authentication, permission, unsupported protocol | Verify credential validity, propagation, and endpoint protocol. Use anonymous access only when both the endpoint and task recipe permit it; otherwise report the access/protocol blocker. |
