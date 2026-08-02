# GDPVal (NeMo Gym "Stirrup" agent)

## Task Details

- Reference: `references/gym-gdpval.md` (SIF build, gym machinery, deploy sizing,
  scoring modes, failure modes) — **read it before editing a GDPVal config.**
- Upstream README:
  <https://github.com/NVIDIA-NeMo/Evaluator/blob/main/examples/nemotron/nemotron-3-ultra/v0.2/README.md>

GDPVal is an **agentic** benchmark: the Stirrup agent produces office/PDF
deliverables inside a per-task Apptainer code-exec sandbox, then a pairwise/rubric
judge (**Gemini 3.1 Pro**) scores them. It is the most resource-intensive benchmark
in the suite — **220 tasks**, `num_repeats=2` in the reviewed golden (= 440
rollouts), each rollout using 4 judge trials.

It runs on the **0.2.6 `nel` launcher** as a `nemo_gym` task (NOT nel-next), so
Steps 1–9 apply — but with the branch differences below.

## What makes GDPVal different (do NOT treat it as a normal `aa/` task)

- **Standalone — one gym eval per config.** Never add GDPVal to a multi-task
  `evaluation.tasks` list, and never add other tasks to a GDPVal config.
- **Apptainer SIF sandbox (self-contained).** Set `GDPVAL_SIF_DIR` in `.env`, then
  run `.agents/scripts/gdpval-sif.sh` (uses `$GDPVAL_SIF_DIR`) — it **builds if
  absent, reuses if present**, and never copies from another cluster. The config
  bind-mounts `$GDPVAL_SIF_DIR` at **exactly** `/gdpval/sif/python-3.12.gdpval.sif`
  (matches `GDPVAL_CONTAINER_PATH`). Missing/mispathed → the agent **silently** runs
  code-exec unsandboxed and results are not comparable. Details in `references/gym-gdpval.md`.
- **Thinking mode is mandatory.** Non-thinking loses ~86% of pairwise judgements.
  Serve the policy with its `--reasoning-parser` and force thinking on via the
  adapter `chat_template_kwargs` (see the example).
- **Judge + web search + gym plumbing.** Needs `INFERENCE_API_KEY` (judge auth),
  `TAVILY_API_KEY` (agent web search), `INFERENCE_JUDGE_URL` (judge host, from
  `.env`), a `pre_cmd` that installs apptainer/squashfuse, and the co-located
  `_gym_prepare.yaml` include.

## Scoring modes

Set `gdpval.reward_mode` (override: `-o gdpval.reward_mode=comparison`):

- **`rubric`** (default) — standalone LLM-judge scoring; **no reference
  deliverables** needed. Use this unless you specifically need pairwise-vs-baseline.
- **`comparison`** — pairwise scoring vs a reference model's deliverables. Also
  mount the ref dir at `/gdpval/refs/test_ref` and set `gdpval.reference_elo`
  (golden uses Kimi-K2.5-Thinking refs, elo=1290). Two-step baseline→comparison
  flow in `references/gym-gdpval.md`.

## Config

**Do not copy a fragment into another config.** GDPVal is standalone — start from
the self-contained example and edit it:

```text
recipes/examples/gym_gdpval/
  example_gym_gdpval.yaml   # SLURM + single-node vLLM self-deploy template
  _gym_prepare.yaml         # co-located Hydra include (${gym_prepare.*}); travels with the yaml
```

Copy the **whole `gym_gdpval/` directory** to your workspace (the `- _gym_prepare`
default resolves relative to the config dir — copying the yaml alone breaks it).

- **num_repeats — the right value depends on the flow, so check which one you're in:**
  - **Multistage comparison** (the current golden for AA-comparable ELO): **1**, set
    with a top-level `++num_repeats=1`, which *does* work. Recent Gym pins already
    ship `num_repeats: 1` in `benchmarks/gdpval/config.yaml`, so the `sed` below is a
    no-op there.
  - **Rubric / older single-reference comparison:** the pre-multistage golden used
    **2** (220 tasks × 2 = 440 rollouts). On old pins the per-dataset key could not be
    set via `++` (OmegaConf `ListConfig` merge error), hence the `sed` in the task
    `command:`; delete that line to keep 2.

  Do not carry a `=2` from an old single-reference config into a multistage run.
- **SIF ↔ Gym version (rebuild on bump):** the SIF is built from `gdpval.def` at
  `install_on_the_fly.commit`. **If you change that commit, rebuild the SIF** with a
  matching `gdpval-sif.sh --commit <sha>` (to a new version-tagged filename, then
  repoint `GDPVAL_CONTAINER_PATH`) — the def's base image + package stack change
  across commits, and running a new gym with an old SIF makes the agent's generated
  code fail imports in the sandbox → silently degraded deliverables. See
  `references/gym-gdpval.md` → "Rebuild the SIF when the Gym version changes".
- **Deployment:** single-node vLLM in the template; the full 220×2 run of a large
  MoE typically needs multi-node — see `references/gym-gdpval.md`.
- Required `.env` keys: `HF_TOKEN`, `INFERENCE_API_KEY`, `TAVILY_API_KEY`,
  `INFERENCE_JUDGE_URL`, `GDPVAL_SIF_DIR` (see `recipes/env.example`).
  `NEMO_EVALUATOR_TRUST_PRE_CMD=1` is needed because the config has a `pre_cmd`.

## Canary

Validate the SIF sandbox + judge + gym plumbing on a couple of tasks before the
full run:

```bash
nel run --config example_gym_gdpval.yaml --env-file .env \
  -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=2
```

Inspect logs for the SIF fallback warning, judge auth/429s, and Ray/gym shutdown
hangs (see `references/gym-gdpval.md` → failure modes).

## Score Extraction

> **The GDPVal score is NOT in `artifacts/eval_factory_metrics.json`.** That file
> holds only `response_stats` / `reasoning` / `evaluation` (request-level telemetry).
> Looking there and finding no ELO does not mean the run failed to score.

The final numbers live in **`artifacts/results.yml`** (authoritative, local) and are
mirrored to MLflow. Read them by metric name:

| Mode | Metric (results.yml → `groups.nemo_gym.metrics.<name>.scores.<name>.value`) |
| --- | --- |
| comparison | `gdpval_stirrup_agent/comparison/eval_elo` ← **the headline** |
| comparison | `gdpval_stirrup_agent/comparison/normalized_elo` (AA 0–1 scale) |
| comparison | `gdpval_stirrup_agent/comparison/win_rate`, `/judged`, `/wins`, `/losses`, `/ties` |
| comparison | per-reference: `gdpval_stirrup_agent/comparison/ref/<ref_key>/{win_rate,wins,losses,ties,judged}` |
| comparison | per-stage estimate: `gdpval_stirrup_agent/comparison/stage_0/eval_elo` (stage 1, all refs) — the **final** value is the top-level one, from the last stage |
| rubric | mean of `reward` across `artifacts/evaluator_rollouts.jsonl` (per-rollout 0–1) |

```bash
# final ELO from the local results file (no MLflow needed)
python3 -c "
import yaml,sys
m=yaml.safe_load(open('results.yml'))['groups']['nemo_gym']['metrics']
for k in ('eval_elo','normalized_elo','win_rate'):
    n=f'gdpval_stirrup_agent/comparison/{k}'
    print(k, '=', m[n]['scores'][n]['value'])"
```

In **MLflow** the same values are prefixed `nemo_gym_` and duplicated under a
`key_metrics/` path — query these exact keys rather than browsing the UI, because a
comparison run logs **~200 metrics and most of them are per-reference**, so the
headline is easy to miss:

```text
nemo_gym_gdpval_stirrup_agent/key_metrics/comparison/eval_elo
nemo_gym_gdpval_stirrup_agent/key_metrics/comparison/normalized_elo
nemo_gym_gdpval_stirrup_agent/key_metrics/comparison/win_rate
```

Sanity checks before quoting a score: `…/comparison/judged` should be large (a few
hundred+), `num_stages`/`num_references` should match your multistage config, and the
unique `task_id` count in `evaluator_rollouts.jsonl` should be close to 220 — a short
count means tasks were lost (e.g. across a walltime resume) and the ELO is computed on
fewer tasks than the references were. Per-task detail is in
`evaluator_rollouts.jsonl` + `nemo_gym_logs/`; raw judge responses are under
`PERSIST_DELIVERABLES_DIR`.
