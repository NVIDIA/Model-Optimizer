---
name: evaluation
description: Evaluates accuracy of quantized or unquantized LLMs using NeMo Evaluator Launcher (NEL). Triggers on "evaluate model", "benchmark accuracy", "run MMLU", "evaluate quantized model", "run nel". Handles deployment, config generation, and evaluation execution. Not for quantizing models (use ptq), deploying/serving models (use deployment), or comparing completed baseline-vs-quantized results (use compare-results).
license: Apache-2.0
# Based on nel-assistant skill from NeMo Evaluator Launcher (commit f1fa073).
# https://github.com/NVIDIA-NeMo/Evaluator/tree/f1fa073/packages/nemo-evaluator-launcher/.claude/skills/nel-assistant
---

## NeMo Evaluator Launcher Assistant

Guide the user through creating NEL YAML configs, running evaluations, and monitoring progress.

### Workspace integration

If `MODELOPT_WORKSPACE_ROOT` is set, use the common skill's `workspace-management.md` and reuse existing workspaces (this skill is usually the final stage of PTQ → Deploy → Eval; carry any deployment-time patches into `deployment.command`).

### Workflow

```text
- [ ] Step 0: Check workspace (if MODELOPT_WORKSPACE_ROOT set)
- [ ] Step 1: Check `nel` install + existing config; set up `.env` (+ `modelopttools:eval-config` for judge-scored runs)
- [ ] Step 2: Build base config (5-question flow OR shortcut)
- [ ] Step 3: Configure deployment (model path, params, cross-check)
- [ ] Step 4: Fill remaining ??? values
- [ ] Step 5: Confirm tasks (iterative)
- [ ] Step 6: Multi-node (if needed)
- [ ] Step 7: Interceptors (if needed)
- [ ] Step 7.5: Container auth (SLURM private images)
- [ ] Step 8: Dry-run → canary → full run
- [ ] Step 9: Verify completed run
```

---

### nel-next path (Terminal-Bench 2.x, SWE-bench, …) — branch here FIRST

A few **agentic** AA benchmarks do **not** run on the currently validated
`nemo-evaluator-launcher` 0.2.6 path (Steps 1–9 don't apply). They run on **nel-next**
(`nemo-evaluator[harbor]` 0.4.x) — a separate package, CLI (`nel eval run`), `-O`
overrides, and `services`/`benchmarks`/`cluster`/`output` schema. If the user asks
for one, do **not** add it to a 0.2.6 `evaluation.tasks` list — instead:

1. Read **`references/nel-next.md`** (shared: venv, schema, AWS creds, architecture, timeout strategy, MLflow, run flow) + the per-benchmark recipe `recipes/tasks/aa_next/{terminal_bench_2_1,swebench_verified}.md`; start from `recipes/examples/example_eval_next.yaml`.
2. Isolated nel-next venv: `"$SKILL_DIR/scripts/nel-next.sh" --setup-only` (keeps 0.2.6 `nel` untouched).
3. Run **`modelopttools:eval-config`** (Step 3b) to write the AWS-sandbox creds + harbor infra rows (`${NEL_NEXT_EVAL_IMAGE}`, `${HARBOR_*_ECR_REPOSITORY}`) into `.env`; always include the `output.export_config.mlflow` block.
4. Dry-run → canary → full (`nel-next.sh eval run`), then **push to MLflow** — SLURM doesn't auto-export, so run `nel-next.sh mlflow-push -r <run_id> -c <cfg>` after (config-driven; see `references/nel-next.md`).

Steps 1–9 below are currently validated with 0.2.6 — use them for everything else.

---

### MRCR (NeMo Gym `simple_agent`) path — branch here too

MRCR **does** run on the currently validated 0.2.6 `nel` launcher (as a `nemo_gym`
task, not nel-next), so Steps 1–9 apply — but it is mechanically special and
**standalone** (one gym eval per config; never mix it with `aa/` tasks). It is
simple as gym tasks go: `simple_agent`, **no judge** — deterministic prefix-gated
grading, `HF_TOKEN` the only secret. **Not an AA benchmark** — never generate it
for an "AA" request. If the user asks for MRCR:

1. Read **`references/gym.md`** (pinned launcher, gym prepare/reap machinery,
   pin↔container coupling, preflight gaps, failure modes) + **`recipes/tasks/gym/mrcr.md`**;
   start from **`recipes/examples/gym/example_mrcr.yaml`** (1M variant, like the golden).
2. **Pick the variant first** (`config_n3_1m` / `config_n3_128k` / `config`) — it
   sets the context cap, dataset *and* metric prefix; the three are not
   comparable; set it in **both** `data_prep_params` and `collect_rollout_params`.
3. `.env`: `HF_TOKEN` (dataset + n3 tokenizer are gated) plus
   `NEMO_EVALUATOR_TRUST_PRE_CMD=1` (the `pre_cmd` installs `tiktoken` +
   `transformers`; prepare fails without it) and
   `NEMO_EVALUATOR_TRUST_UNLISTED_TASKS=1` (`nemo_gym` is not in the FDF map).
4. **MRCR needs a git-backed Gym image.** The pin is newer than any image's baked
   Gym and must apply, so the template's `container:` is `???` and the bootstrap
   exits 1 on a non-git `/opt/Gym` (the public `eval-factory/nemo-gym:*` images).
   NVIDIA-internal: `modelopttools:eval-config` Step 3d names a working image.
5. Long-context deploy (`--max-model-len 1100000` +
   `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `gpu_memory_utilization: 0.95`,
   multi-instance fan-out); **never cap output tokens**; report the needle-count
   strata alongside `pass@1/accuracy`.
6. Run both dry-run and launch through `"$SKILL_DIR/scripts/nel-gym.sh"`; it
   enforces the currently validated 0.2.6 launcher even if `nel` on PATH is stale
   and avoids an unset `NEL_INVOCATION_ID` failure before client startup.
   **`limit_samples` is inert on the gym path** — canary with the gym's own
   `++limit=N` (see the recipe's Canary section), remembering the prepare pass
   still runs in full.

---

Detailed launcher instructions live in [launcher-workflow.md](references/launcher-workflow.md).
Read only the sections needed for the current stage; retain the dry-run → canary →
full-run gates. Existing configs can start at Step 8. Paths in that reference are
relative to this skill directory.

### Step 1 — Prerequisites

Read [Step 1](references/launcher-workflow.md#step-1--prerequisites).

### Step 2 — Build base config (when not using shortcut)

Read [Step 2](references/launcher-workflow.md#step-2--build-base-config-when-not-using-shortcut).

### Step 3 — Configure deployment

Read [Step 3](references/launcher-workflow.md#step-3--configure-deployment).

### Step 4 — Fill remaining ??? values

Read [Step 4](references/launcher-workflow.md#step-4--fill-remaining--values).

### Step 5 — Confirm tasks (iterative)

Read [Step 5](references/launcher-workflow.md#step-5--confirm-tasks-iterative).

### Step 6 — Multi-node

Read [Step 6](references/launcher-workflow.md#step-6--multi-node).

### Step 7 — Interceptors

Read [Step 7](references/launcher-workflow.md#step-7--interceptors).

### Step 7.5 — Container registry auth (SLURM private images only)

Read [Step 7.5](references/launcher-workflow.md#step-75--container-registry-auth-slurm-private-images-only).

### Step 8 — Run evaluation (gated dry-run → canary → full)

Read [Step 8](references/launcher-workflow.md#step-8--run-evaluation-gated-dry-run--canary--full).

### Step 9 — Verify completed run

Read [run-validation.md](references/run-validation.md) before reporting scores:
validate logs and sample coverage, complete **Timeout and Output-Limit Accounting**
for every task, and report missing telemetry as unknown. For comparisons, also
apply its **External Baseline Sanity Check**, then use `compare-results`.

---

Issues: <https://github.com/NVIDIA-NeMo/Evaluator/issues> · <https://github.com/NVIDIA-NeMo/Evaluator/discussions>
