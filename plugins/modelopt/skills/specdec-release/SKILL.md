---
name: specdec-release
description: Deterministic readiness gates for a speculative-decoding drafter — training/export (gate_train), serving canary (gate_smoke), matched A|B benchmark validity (gate_bench), and a non-blocking greedy-match report (gate_correctness). Use when the user asks whether a trained drafter (EAGLE3, DFlash, DFlash2, DSpark, Domino, LiLiCorr) is ready for release review, or to gate the artifacts of a draft-training pipeline run. Do NOT use to configure, debug or triage a draft-training run (use speculative-decoding), to quantize (use ptq), or to serve a model (use deployment).
license: Apache-2.0
---

# Spec-Dec Release

Gates that decide whether a drafter is **ready for release review**: trained cleanly,
exported to fit its target, servable with speculation active, and measured validly. They
set no per-model performance bar. AL, speedup and TPS are returned as evidence. The
checkpoint release criteria (Addendum A: AL and throughput bars) are a later stage and are
not implemented here.

## Gate contract

Same as `day0-release`. Each script prints JSON `{pass, failure_class, detail, notes,
inputs, ...}` and exits `0` pass, `1` gate failed, `2` unreadable or invalid input
(`USER_CONFIG_ERROR`). Run the gates on local copies of the artifacts or on the cluster
login node; they are stdlib-only. **Never read a verdict from a log or a Slurm state.**

| Stage | Command | Reads |
| --- | --- | --- |
| Train + export | `python "$SKILL_DIR/scripts/gate_train.py" --regression <out>/regression_check.json --export <out>/exported-checkpoint-final --target-config <target>/config.json --algorithm <algo>` | `check_regression.py --json` output, export dir, target config |
| Smoke | `python "$SKILL_DIR/scripts/gate_smoke.py" --smoke <SMOKE_JSON> --draft <drafter> --target <target>` | `vllm_smoke_test.sh` `SMOKE_JSON` |
| A\|B bench | `python "$SKILL_DIR/scripts/gate_bench.py" --summary <SAVE_ROOT>/ab_summary.json --concurrencies 1,8,32 --repeats N --algorithm DFLASH --num-spec-tokens 7 --draft <drafter> --target <target>` | `compare_ab.py` summary |
| Correctness | `python "$SKILL_DIR/scripts/gate_correctness.py" --summary <SAVE_ROOT>/ab_summary.json` | same summary; **report-only**, always passes |

A gate never passes a run its producing script failed, classifies the failure itself only
when the script cannot (so `gate_smoke` reads raw values; `gate_train` keeps
`check_regression.py`'s verdict), and never re-implements a script's comparison.

Pass `gate_bench` what the run was **submitted** with, not targets. Artifact files are read
from host paths (`/cicd/<exp_id>/...`), but `--draft`/`--target` are compared to the paths the
job recorded, so pass them as the job saw them (e.g. `/scratchspace/...`). An empty
summary is `USER_CONFIG_ERROR`: check `--summary`; if the path is right, the bench job died
before writing any arm, so resubmit as for `BENCH_INCOMPLETE`. `gate_smoke`
classifies failures from the raw values and its `stage`; a SMOKE_JSON without `stage` comes
from an older script, so re-run the smoke test.

## Failure classes

| `failure_class` | Gate | Action |
| --- | --- | --- |
| `USER_CONFIG_ERROR` | any | Fix the invocation (paths, `--algorithm`, unknown family → `--expect-architecture`); re-run the gate only |
| `METRICS_MISSING` | train | No regression verdict, or the recipe set no thresholds. Check the training log; set `MAX_FINAL_LOSS`/`MIN_FINAL_ACC` in the launcher YAML |
| `TRAIN_REGRESSION` | train | The recipe's own floors failed. Triage with the **speculative-decoding** skill; do not serve |
| `EXPORT_MISSING` | train | Re-run the training task to re-export (an export counts only once `config.json` and weights exist) |
| `DRAFT_CONFIG_MISMATCH` | train | Export does not fit the target (architecture, hidden/vocab size, mask token, layer ids). Fix the recipe overrides per the algorithm sheet; retrain |
| `CHECKPOINT_NOT_SERVABLE` | smoke | Same as `day0-release`: fix the export or serving flags; do not benchmark |
| `SPEC_NOT_ACTIVE` | smoke | AL ≤ 1.05: drafter not wired up or useless. Check `SPEC_METHOD`, token count, drafter path |
| `ACCEPTANCE_REGRESSION` | smoke | Serves and speculates, but AL < the recipe's `MIN_ACCEPTANCE_LENGTH`. Triage training with **speculative-decoding**; do not benchmark |
| `SPEC_CONFIG_MISMATCH` | smoke, bench | Wrong or unrecorded (pass `--block_size`/`--draft_length`) drafter/target/algorithm/token count, or speculation on in arm A / off in arm B. Fix the submission; re-run |
| `BENCH_INCOMPLETE` | bench | Missing rows or arms, appended results, arms with different response counts. Resubmit into a **fresh** `SAVE_ROOT` (≤ 2 automatic retries) |
| `ARMS_UNMATCHED` | bench | Arms or rows differ in target, engine, GPU, TP, dataset, length or temperature, or an arm ran at another concurrency than its directory says. The comparison is invalid; re-run in one job |
| `ANOMALOUS` | bench | Implausible numbers (AL or speedup above tokens + 1, non-positive TPS, unpaired turns). Do not read the numbers; investigate |

Verdict: every blocking gate passed → `DRAFTER_READY` (not a publish decision);
training, export or serving failed for good → `INFEASIBLE`; the measurement is invalid →
`ANOMALOUS`.
