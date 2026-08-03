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

It runs on the **0.2.6 `nel` launcher** as a `nemo_gym` task (NOT nel-next), so
Steps 1–9 apply — but with the branch differences below.

## What makes GDPVal different (not a normal `aa/` task)

- **Standalone** — one gym eval per config. Never add GDPVal to a multi-task
  `evaluation.tasks` list, and never add other tasks to a GDPVal config.
- **Apptainer SIF sandbox** — prefer a site-provided SIF; otherwise
  `.agents/scripts/gdpval-sif.sh` builds one into `$GDPVAL_SIF_DIR` (build-if-absent,
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
recipes/examples/gym_gdpval/
  example_gym_gdpval.yaml   # SLURM + single-node vLLM self-deploy template (rubric)
  _gym_prepare.yaml         # co-located Hydra include; MUST travel with the yaml
```

Copy the **whole dir** (the `- _gym_prepare` default resolves relative to the config
dir). `num_repeats=1` — already set by the template via `++num_repeats=1`; both
current goldens use it. A full 220-task run of a large MoE typically needs multi-node.

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
# final score from the local results file (no MLflow needed)
python3 -c "
import yaml,sys
m=yaml.safe_load(open('<output_dir>/<run>/nemo_gym.0/artifacts/results.yml'))['groups']['nemo_gym']['metrics']
for k in ('normalized_elo','eval_elo','win_rate'):
    n=f'gdpval_stirrup_agent/comparison/{k}'
    print(k, '=', m[n]['scores'][n]['value'])"
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
