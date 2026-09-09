# GDPVal (NeMo Gym "Stirrup" agent)

## Task Details

- Reference: `references/gym-gdpval.md` (SIF build, gym machinery, deploy sizing,
  scoring modes, failure modes) — **read it before editing a GDPVal config.**
- Upstream README:
  <https://github.com/NVIDIA-NeMo/Evaluator/blob/main/examples/nemotron/nemotron-3-ultra/v0.2/README.md>

GDPVal is an **agentic** benchmark: the Stirrup agent produces office/PDF
deliverables inside a per-task Apptainer code-exec sandbox, then a pairwise/rubric
judge (**Gemini 3.1 Pro**) scores them. It is the most resource-intensive benchmark
in the suite — **220 tasks**, `num_repeats=1`, 4 judge trials per rollout.

It is currently validated with the **0.2.6 `nel` launcher** as a `nemo_gym` task
(NOT nel-next), so Steps 1–9 apply — but with the branch differences below.

Run it through `"$SKILL_DIR/scripts/nel-gdpval.sh"`, which pins the currently
validated launcher. Do not use an unversioned `nel` from PATH: an incompatible
launcher can fail before client startup when the config forwards
`NEL_INVOCATION_ID`. The shared reference explains the failure signature, dry-run
check, and procedure for validating and adopting a newer launcher.

**Part of the AA suite** — generate it for every AA request, as its own config
alongside the `aa/` multi-task one. It shares `recipes/tasks/gym/` with MRCR,
which is *not* AA: the directory groups tasks by **harness** (NeMo Gym), not by
suite membership, so read that per-task, not from the path.

## What makes GDPVal different (not a normal `aa/` task)

- **Standalone** — one gym eval per config. Never add GDPVal to a multi-task
  `evaluation.tasks` list, and never add other tasks to a GDPVal config.
- **Apptainer SIF sandbox** — prefer a site-provided SIF; otherwise
  `$SKILL_DIR/scripts/gdpval-sif.sh` builds one into `$GDPVAL_SIF_DIR` (build-if-absent,
  never copied between clusters). Missing/misnamed → **silent** unsandboxed exec.
- **Thinking mode is mandatory** — non-thinking loses ~86% of pairwise judgements.
  Serve with the model's `--reasoning-parser` and force it on via the adapter's
  `chat_template_kwargs`.
- **Scoring:** `rubric` (template default, no references, no ELO) vs `comparison`
  (the AA-comparable `normalized_elo`; a conversion, not a flag flip).
- Needs `INFERENCE_API_KEY`, `TAVILY_API_KEY`, `INFERENCE_JUDGE_URL`,
  `GDPVAL_SIF_DIR` in `.env`, plus `NEMO_EVALUATOR_TRUST_PRE_CMD=1` (the config has a
  `pre_cmd`).

All of the above — SIF handling, the SIF↔Gym-commit coupling, scoring modes, judge
panel, preflight and failure modes — is detailed in **`references/gym-gdpval.md`**.
Read it before editing a GDPVal config.

## Config

Start from the self-contained example and edit it — **do not** copy a fragment into
another config:

```text
recipes/examples/gym/example_gdpval.yaml   # SLURM + single-node vLLM,
                                                    # rubric mode, self-contained
```

`num_repeats=1` — already set by the template via `++num_repeats=1`; both
current goldens use it. A full 220-task run of a large MoE typically needs multi-node.

## Canary — `limit_samples` does NOT work here

**`++…params.limit_samples=N` is inert on the gym path.** The gym does its own data
prep and rollout collection, so the launcher-level limiter is ignored: you get the
full 220-task run. Do not use it believing you launched a two-task smoke test — this
is the heaviest benchmark in the suite.

There is no cheap sample-limited canary. Instead, **launch the real run and treat its
first ~20–30 minutes as the canary**, cancelling if any of these is wrong:

```bash
RD=<output_dir>/<run>/nemo_gym.0
grep -c "Using Apptainer container"  $RD/logs/client-*.log          # sandbox actually used
grep -c "falling back\|not a git repo" $RD/logs/client-*.log        # unsandboxed / inert pin
grep -ciE " 401 | 403 |Internal Server Error" $RD/artifacts/nemo_gym_logs/gdpval_judge_model.log
wc -l $RD/artifacts/evaluator_rollouts.jsonl                        # rollouts flowing
```

In comparison mode stage 1 (45 tasks) is a natural early checkpoint — an ELO estimate
appears before the full 220-task stage 2 starts.

## Score Extraction

> **The GDPVal score is NOT in `artifacts/eval_factory_metrics.json`.** That file
> holds only `response_stats` / `reasoning` / `evaluation` (request-level telemetry).
> Looking there and finding no ELO does not mean the run failed to score.

**The reported GDPVal score is `normalized_elo`** — the AA 0–1 scale, comparable
across models and to the published AA index. `eval_elo` is the same fit on the raw
Elo axis (`normalized_elo = (eval_elo - 500) / 2000`); quote it as supporting
detail, not as the score.

The final numbers live in **`artifacts/results.yml`** (authoritative, local) and are
mirrored to MLflow. Read them by metric name:

| Mode | Metric (results.yml → `groups.nemo_gym.metrics.<name>.scores.<name>.value`) |
| --- | --- |
| comparison | `gdpval_stirrup_agent/comparison/normalized_elo` ← **REPORT THIS** (AA 0–1 scale) |
| comparison | `gdpval_stirrup_agent/comparison/eval_elo` (raw Elo; supporting detail) |
| comparison | `gdpval_stirrup_agent/comparison/win_rate`, `/judged`, `/wins`, `/losses`, `/ties` |
| comparison | per-reference: `gdpval_stirrup_agent/comparison/ref/<ref_key>/{win_rate,wins,losses,ties,judged}` |
| comparison | per-stage estimate: `gdpval_stirrup_agent/comparison/stage_0/eval_elo` (stage 1, all refs) — the **final** value is the top-level one, from the last stage |
| rubric | mean of `reward` across `artifacts/evaluator_rollouts.jsonl` (per-rollout 0–1) |

```bash
# COMPARISON mode — final score from the local results file (no MLflow needed)
python3 -c "
import yaml
m=yaml.safe_load(open('<output_dir>/<run>/nemo_gym.0/artifacts/results.yml'))['groups']['nemo_gym']['metrics']
for k in ('normalized_elo','eval_elo','win_rate'):
    n=f'gdpval_stirrup_agent/comparison/{k}'
    print(k, '=', m[n]['scores'][n]['value'])"

# RUBRIC mode (the template default) — there is no ELO; average the per-rollout reward
python3 -c "
import json
r=[json.loads(l).get('reward') for l in open('<output_dir>/<run>/nemo_gym.0/artifacts/evaluator_rollouts.jsonl')]
r=[x for x in r if isinstance(x,(int,float))]
print('mean reward =', sum(r)/len(r), 'over', len(r), 'rollouts')"
```

In **MLflow** the same values are prefixed `nemo_gym_` and duplicated under a
`key_metrics/` path — query these exact keys rather than browsing the UI, because a
comparison run logs **~200 metrics and most of them are per-reference**, so the
headline is easy to miss:

```text
nemo_gym_gdpval_stirrup_agent/key_metrics/comparison/normalized_elo   <- report this
nemo_gym_gdpval_stirrup_agent/key_metrics/comparison/eval_elo
nemo_gym_gdpval_stirrup_agent/key_metrics/comparison/win_rate
```

Sanity checks before quoting a score: `…/comparison/judged` should be large (a few
hundred+), `num_stages`/`num_references` should match your multistage config, and the
unique `task_id` count in `evaluator_rollouts.jsonl` should be close to 220 — a short
count means tasks were lost (e.g. across a walltime resume) and the ELO is computed on
fewer tasks than the references were. Per-task detail is in
`evaluator_rollouts.jsonl` + `nemo_gym_logs/`; raw judge responses are under
`PERSIST_DELIVERABLES_DIR`.

## Feasibility pre-check

The **per-task ceiling** binds, not the total budget. On a large reasoning model a
12600 s (3.5 h) ceiling timed out ~88% of rollouts, leaving n=7 paired tasks —
a full benchmark's GPU-h for no usable signal.

Run a handful of tasks first and measure the timeout rate. If a material fraction
hits the ceiling, either raise it or drop GDPval and record the infeasibility.
