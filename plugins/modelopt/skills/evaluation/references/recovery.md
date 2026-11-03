# Recovery After Failed Run Validation

Use after [run-validation.md](run-validation.md) reports an invalid or incomplete
run. Validation owns acceptance; recovery does not relax its gates. Do not retry
solely for accepted bounded failures or ordinary wrong answers. Expected NEL
walltime/resume chains are not failed evaluations.

## Parent and evaluator contract

The **evaluator sub-agent** diagnoses and proposes recovery; the **parent** owns
the decision, comparison protocol, shared resources, and submission budget. In a
standalone evaluation, the agent acts as parent within the user's authorization.

1. **Evaluator → parent:** return run/task IDs, validation verdict, fault counts
   and denominators, effective limits, and evidence paths. Identify the first
   causative failure, uncertainty, affected baseline/candidate tasks, and a
   proposed config diff with expected benefit and cost. Do not resubmit merely
   because validation failed.
2. **Parent → evaluator:** approve, revise, or reject the proposal. Approval must
   name the tasks/models, exact changes, resume versus fresh-run strategy,
   canary/full stages, maximum submissions and resource/time limits, and stop
   conditions. Count canaries and failed submissions under existing user limits;
   recovery does not reset them. A bounded recovery plan may be approved before
   delegation; the evaluator must return for changes outside it.
3. **Evaluator executes only that plan:** register invocation/job IDs before
   further delegation; monitor existing jobs rather than duplicate them. Return
   each stage's evidence and revalidation verdict. Proceed canary → full only if
   explicitly authorized and the canary passes the applicable operational checks.
4. **Parent closes the loop:** accept a validated replacement, authorize another
   evidence-backed plan within the remaining budget, or report the blocker.
   Repeated failure is not permission for an unbounded retry loop. Ask the user
   before exceeding authority, resource limits, or fixed benchmark requirements.

## Choose recovery from evidence

| Failure | Proposed next action |
|---|---|
| Output budget exhausted with unfinished, varied reasoning | Increase `max_new_tokens` (or the harness-equivalent field) and rerun; check context fit and timeout/KV headroom first. |
| Prompt/history plus generation exhausts deployed context | Increase deployed context within supported model/runtime limits and rerun; raising the output cap alone cannot create context capacity. |
| Repeated reasoning or malformed actions | Inspect raw outputs and parser/tool protocol first. More tokens may only prolong a loop; use [doom-loop guidance](doom_loops.md) for an approved matched intervention. |
| GPU/host OOM, KV pressure, preemption, disk or quota exhaustion | Identify the exhausted resource; propose lower concurrency, more supported resources, or cleanup of trial-owned temporary data after preserving evidence. Verify capacity before retrying. |
| Overloaded model/judge endpoint, rate limits, transient transport errors | Respect backoff; check health/capacity and reduce offending concurrency. Propose a bounded retry after recovery, not repeated submissions into the same outage. |
| Sandbox provisioning/executor failure | Separate infrastructure failure from generated-code errors or protocol-defined timeouts. Repair image/dependencies/capacity or retry confirmed transient provisioning failures; canary code execution before full. |
| Missing scores/coverage, parser errors, wrong identity/configuration | Repair accounting or configuration first. Never fill missing outcomes with invented zeros or retry only wrong answers. |
| Authentication, permission, secret exposure, unsupported protocol | Stop affected work and escalate. Retries cannot bypass access controls, hide exposure, or waive benchmark rules. |

Transient failures may justify one unchanged retry in the approved plan. A
recurrence requires a new diagnosis/parent decision, not another identical retry.
Delegate serving failures to **deployment**, evaluation/harness failures to
**evaluation**, and comparison decisions to **compare-results**.

### Length-capped responses

Inspect original provider responses, not only Harbor's recovered/fallback action
or final transcript. Record termination reason, actual input/output usage,
configured cap/context, final-answer presence, and representative identities.
Distinguish output caps, total-context exhaustion, agent step/total-token limits,
parser loss, and repetition; `finish_reason: length` alone does not identify which
limit fired. A normal stop can still contain an empty final.

For a failed validation caused by output/context limits, the parent should
approve a supported increase and rerun unless task policy, resources, or remaining
budget prevents it; otherwise return the specific blocker. For example, try
`65536 → 131072` output tokens when task policy permits and evidence shows genuine
unfinished reasoning. This is a recovery diagnostic, not a new universal default.

- Check [task policy and model-card guidance](model-card-research.md#max_new_tokens--mandatory-model-card-lookup).
  Do not silently override a fixed evaluation budget or add a cap to a task that
  forbids one. An approved departure is a separate diagnostic protocol, not an
  official benchmark-equivalent result.
- Verify tokenized prompt/history plus the proposed output fits deployed context,
  including subsequent agent turns. If context is the bottleneck, increase
  `--max-model-len` (or equivalent) only to a supported capacity. Context extension
  beyond the trained window requires explicit approval and documented scaling;
  never silently clip input or invent support.
- Recheck KV capacity, concurrency, and effective request/task timeouts for longer
  generations. Do not change thinking effort, sampling, or repetition penalties
  alongside a cap increase without separately stating and approving those changes.
- Canary the changed configuration, including representative previously capped
  cases as diagnostics, then rerun the entire affected benchmark with its full
  intended samples/repeats. A canary or repaired subset is not a replacement score.

## Resume, rerun, and revalidate

- Prefer supported same-config resume for interrupted infrastructure work when
  cache identity and complete accounting can be proved. Preserve failed attempts
  and distinguish recovery from new independent evaluation repeats.
- Changed generation/context settings require fresh invocation/output/cache
  namespaces. Never mix old-budget responses with new ones or overwrite the
  original invalid results. Preserve evidence before cancelling only owned jobs;
  clean up only trial-owned resources.
- Apply protocol changes identically to baseline and candidate. Rerun affected
  tasks on both sides, unless an existing validated run exactly matches the new
  protocol. Unaffected validated tasks may be reused with explicit provenance.
- Reapply all of [run-validation.md](run-validation.md), not only the previously
  failing check. Carry original/replacement IDs, config diffs, fault rates,
  approval/budget ledger, and unresolved limitations into the handoff. Keep
  scheduler completion, MLflow delivery, and evaluation validity separate.
- After both sides validate, repeat external baseline sanity and use
  **compare-results**. Otherwise withhold the failed task's score/comparison and
  return the remaining blocker; never select the best retry or silently regrade.
