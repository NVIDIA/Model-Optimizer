# Recovery After Failed Run Validation

Diagnose, repair, canary, rerun, and revalidate autonomously without prompting the
user. Do not retry solely for accepted bounded failures or ordinary wrong answers.

Record the cause, evidence, config diff, resume/rerun strategy, submission budget,
and stop conditions. Count canaries and failed submissions against existing
limits. A recurring failure needs a revised diagnosis, not identical retries;
report a blocker when recovery cannot proceed.

## Choose recovery from evidence

| Failure | Recovery action |
|---|---|
| Output budget exhausted with unfinished reasoning | Increase `max_new_tokens` and rerun; see below. |
| Prompt/history exhausts context | Increase supported deployed context; a larger output cap alone cannot help. |
| Repeated reasoning or malformed actions | Inspect raw outputs and parser/tool protocol; more tokens may only prolong a loop. |
| GPU/host OOM, KV pressure, disk or quota exhaustion | Reduce concurrency, adjust supported resources, or clean trial-owned temporary data after preserving evidence; verify capacity before retrying. |
| Preemption or interrupted infrastructure work | Prefer supported same-config resume when cache identity and complete accounting can be proved. |
| Endpoint overload, rate limits, transient transport errors | Check health, back off, and reduce offending concurrency. Retry after recovery, not repeatedly into the same outage. |
| Sandbox provisioning/executor failure | Repair image/dependencies/capacity; canary code execution before full. Distinguish infrastructure faults from protocol-valid failed trials. |
| Missing scores/coverage, parser errors, wrong identity/configuration | Repair accounting or configuration; never invent missing outcomes or retry only wrong answers. |
| Authentication, permission, secret exposure, unsupported protocol | Stop affected work and report the blocker; retries cannot bypass access controls or waive benchmark rules. |

## Excessive truncation

Inspect raw provider responses: `finish_reason: length` alone does not distinguish
output caps from context exhaustion. For genuine unfinished reasoning, **increase
`max_new_tokens` and rerun** rather than declaring the model infeasible; for example,
`65536 → 131072` when task policy permits. Check prompt-plus-output context fit,
model/runtime support, KV headroom, and timeouts. This is a recovery diagnostic,
not a universal default; do not override fixed benchmark budgets.

Canary the changed configuration, then rerun the entire affected benchmark, not
only truncated samples. Apply protocol changes identically to baseline and
candidate; reuse an existing validated run only if it matches the new protocol.
Use fresh invocation/output/cache namespaces for changed settings; preserve
original results and never mix old- and new-budget responses. Reapply all
validation checks before reporting replacement scores or comparisons.
