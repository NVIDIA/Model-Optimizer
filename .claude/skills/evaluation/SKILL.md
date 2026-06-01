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

If `MODELOPT_WORKSPACE_ROOT` is set, read `skills/common/workspace-management.md` and reuse existing workspaces (this skill is usually the final stage of PTQ → Deploy → Eval; carry any deployment-time patches into `deployment.command`).

### Workflow

```text
- [ ] Step 0: Check workspace (if MODELOPT_WORKSPACE_ROOT set)
- [ ] Step 1: Check `nel` install + existing config
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

### Step 1 — Prerequisites

Run `nel --version`; if missing, instruct `pip install nemo-evaluator-launcher`. If user has an existing config, skip to Step 8 (optionally review for `???` and quantization flags first).

**Task recipes** (always read before editing the relevant task in the config):

- AA Index v2 suite (default for quantized-checkpoint validation, see `references/quantization-benchmarks.md`): `recipes/tasks/aa/{gpqa_diamond,hle,lcr,scicode,ifbench,mmmu_pro,tau2_bench_telecom}.md`
- Optional: `recipes/tasks/mmlu_pro.md`, `recipes/tasks/aime_2025.md`, `recipes/tasks/livecodebench.md`

**AA rule:** If the user mentions "AA" / "Artificial Analysis", generate **only** tasks under `recipes/tasks/aa/`. Do not add MMLU-Pro, AIME 2025, or LiveCodeBench unless explicitly asked.

**Shortcut path** (when task list is known up front, e.g. "run AA"):

1. Read the task reference file(s).
2. Use `recipes/examples/example_eval.yaml` as the base.
3. Copy the YAML fragment(s) into `evaluation.tasks`, applying any per-task notes.
4. **MLflow auto-export is on by default** — it needs **two** pieces, both in `example_eval.yaml`: (a) the **trigger** `execution.auto_export.destinations: [mlflow]` (without it the run is *not* uploaded), and (b) the `export.mlflow` block that configures it. In the `export.mlflow` block use **literal** values for `experiment_name` / `description` / `tags` — substitute the actual `served_model_name` and sampling params. Do **not** use `${deployment.*}` / `${evaluation.*}` cross-references: with auto-export on, NEL resolves the export block at submit time in a scope without those nodes and fails with `Interpolation key '...' not found` (`${oc.env:USER}` is fine — it's an env var). Fill `tracking_uri` in Step 4.
5. Proceed to Step 3, then Step 4, then Step 7.5/8. Skip Step 2's 5-question flow.

---

### Step 2 — Build base config (when not using shortcut)

Ask the 5 questions via AskUserQuestion (categories must match `nel skills build-config --help` — **run that first** to confirm the current option names; CLI options override this list).

1. **Execution:** Local / SLURM
2. **Deployment:** None (External) / vLLM / SGLang / NIM / TRT-LLM. Prefer vLLM unless the user/card says otherwise.
3. **Auto-export:** None / MLflow / wandb
4. **Model type:** Base / Chat / Reasoning
5. **Benchmarks** (multi-select): standard / code / math_reasoning / safety / multilingual

Build the base:

```bash
nel skills build-config --execution <...> --deployment <...> --model_type <...> --benchmarks <...> [--export <...>] [--output <...>]
```

(`--output` omitted = cwd auto-named; directory = dir + auto-name; `*.yaml` = exact path. Never overwrites.)

---

### Step 3 — Configure deployment

**Model path.** Checkpoint path (`/`, `./`, `../`, `~`, or exists on disk) → set `deployment.checkpoint_path`, leave `hf_model_handle: null`. Else HF handle (one `/`, not on disk) → set `deployment.hf_model_handle`, leave `checkpoint_path: null`.

**Auto-detect ModelOpt quantization** (checkpoint paths). Check `config.json` for `quantization_config` (or legacy `hf_quant_config.json`):

- **vLLM:** no `--quantization` flag by default — vLLM auto-detects from `quantization_config` / `hf_quant_config.json`. Add only when the card, vLLM version, or dry-run error requires it.
- **SGLang:** may need `--quantization modelopt_fp8` / `modelopt_fp4` / `modelopt` — verify against installed version.

Some models need extra env vars (e.g. `VLLM_NVFP4_GEMM_BACKEND=marlin` for Nemotron Super) — discovered via model card research.

**Auto-detect from `config.json`:**

| Field | Flag |
| --- | --- |
| `max_position_embeddings` | `--max-model-len <value>` |
| `auto_map` exists | `--trust-remote-code` |

#### Cross-check both sources for vLLM (mandatory, neither replaces the other)

**Source 1 — `recipes.vllm.ai/<org>/<model>`** (curated vLLM recipes; authoritative for parallelism, family-specific flags like `--reasoning-parser` / `--tool-call-parser` / `--mm-encoder-tp-mode`, vLLM version, spec-decoding, GPU count). Pin variants via query params (e.g. `?variant=fp8&strategy=single_node_tep`).

> **WebFetch caveat — triage the summary:**
>
> 1. **"No `vllm serve` commands found" / "page is a usage guide":** JS-rendering miss. recipes.vllm.ai pages always have ≥1 command. Ask the user to paste it or share the variant URL.
> 2. **Single recipe returned** for a model with known multiple variants → retry with variant-pinned URL. Axis names differ per model (Qwen: `?variant=&strategy=`; Kimi: `?advanced=`; others vary — no fixed pattern).
> 3. **Variant label contradicts the command** (e.g. label "TEP" but command shows DP+EP) → summarizer conflated variants; ask user.
>
> For non-trivial deployments (≥120B, multi-node, novel arch), ask the user which variant *before* fetching.

**Source 2 — HF model card + `config.json`** (authoritative for):

| Signal | Flag |
| --- | --- |
| `max_position_embeddings` | `--max-model-len <value>` |
| `auto_map` | `--trust-remote-code` |
| Reasoning/CoT documented | `--reasoning-parser` (and `--reasoning-parser-plugin` if custom) |
| Tool-calling documented | `--enable-auto-tool-choice --tool-call-parser <parser>` |
| Custom flags in card | Add as specified (e.g. `--mamba_ssm_cache_dtype float32`) |

**Cross-check rules:**

1. Read both sources before composing the command.
2. Agree → use with confidence.
3. Disagree → **do not silently pick one.** Surface both values to the user. Common conflicts: stale cards, parser rename between generations (Qwen2.5 `hermes` → Qwen3 `qwen3_coder`), recipe-only flags like `--language-model-only`, ARM64-specific card notes.
4. Resolve in Step 3 — don't defer to dry-run.

#### vLLM deployment command structure — single `command:` field

Rewrite the build-config output into one `command:` field. Move all parallelism (`--tensor-parallel-size`, `--data-parallel-size`, `--pipeline-parallel-size`) into the command; do not keep separate `tensor_parallel_size` / `data_parallel_size` / `extra_args` YAML fields.

```yaml
deployment:
  command: >-
    vllm serve /checkpoint
    --host 0.0.0.0
    --port ${deployment.port}
    --tensor-parallel-size <N>
    --data-parallel-size <M>
    --max-model-len <value>
    <... rest of cross-checked flags ...>
```

Conventions: always start `vllm serve /checkpoint` (NEL mounts here); always `--host 0.0.0.0 --port ${deployment.port}`; use folded scalar (`>-`) for one flag per line. Example fallback `--max-model-len 131072` covers AA-LCR (~120K + 16K gen) and SciCode (≥ 65536) — prefer `config.json` / recipe value.

**Image / vLLM version.** Default `image: vllm/vllm-openai:v0.19.1` (pinned for reproducibility). If `recipes.vllm.ai` states a higher minimum version for the chosen variant (e.g. "vLLM >= 0.20.0"), bump the image tag accordingly (e.g. `v0.20.0`) — do **not** stay on `0.19.1` when the recipe explicitly requires newer. Do **not** use `:latest` (drifts across re-runs, breaks reproducibility). The version is part of the cross-check: surface to the user when bumping.

#### vLLM-backend defaults — always include unless the recipe *contradicts*

Silence is not contradiction. Drop/override only when the recipe sets a different value for the same setting (e.g. recipe pins `--max-num-batched-tokens 16384` → use 16384).

- `--max-num-batched-tokens 8192` — caps per-step batched tokens; prevents long-prefill stalls.
- `--enable-chunked-prefill` — interleaves long prefills with decode steps (required for AA-LCR's ~120K input). Modern vLLM defaults this on for many models; set explicitly to avoid drift.
- `--enable-expert-parallel` — **MoE-only default.** Detect MoE from handle suffix (`-A10B`, `-A3B`, etc.), `num_experts` / `num_local_experts` / `n_routed_experts` in `config.json`, or card. No-op when TP=DP=1, safe to always include for MoE. Do not add for dense models.
- `--max-num-seqs N` — **omit at generation time** (top-level `parallelism` is `???`). Add this comment above `command:`:

  ```text
  # After filling in `parallelism` values (top-level + per-task overrides),
  # append `--max-num-seqs N` where N = ceil(max_parallelism / data_parallel_size).
  ```

  In Step 4 compute and append. Example: top-level=16, Tau2=128, DP=8 → `ceil(128/8)=16`. Too small → request queuing; too large → wasted KV reservation.

#### Evaluation params template (top-level params)

The top-level `nemo_evaluator_config.config.params` must contain **exactly these six fields** — no `top_k` / `presence_penalty` / `repetition_penalty` / `min_p`:

```yaml
nemo_evaluator_config:
  config:
    params:
      parallelism: ???    # Required — ask user in Step 4 (depends on cluster + judge rate limits)
      request_timeout: 3600
      max_retries: 10
      max_new_tokens: 65536  # see rule below
      temperature: 1.0    # from model card (reasoning); adjust
      top_p: 0.95         # from model card (reasoning); adjust
```

Per-task `max_new_tokens` overrides are forbidden — set one top-level ceiling everywhere.

#### `max_new_tokens` — mandatory model-card lookup

1. **Fetch the HF model card before writing the value.** Not optional.
2. Scan for any `max_tokens` / `max_new_tokens` / "output length" recommendation. Pick the **highest** value the card mentions (Qwen3.6: 32768 general + 81920 math-coding → use **81920**). Annotate with a citing comment.
3. If the card is genuinely silent after a thorough read, fall back to: **65536** (reasoning), **16384** (non-reasoning); surface the silence to the user.
4. **Forbidden:** writing `max_new_tokens: <generic_default>` with a "card not yet checked" comment. Either fetch and apply, or fetch and confirm silence.

#### Quantization-aware benchmark defaults

For quantized checkpoints, read `references/quantization-benchmarks.md` for sensitivity rankings and recommended sets; present and ask which to include. Read `references/model-card-research.md` for the full extraction checklist (sampling, reasoning config, ARM64, `pre_cmd`, output length — see the dedicated bullet there).

Reasoning models: prefer reasoning mode (highest scores). For lower variance / cost / apples-to-apples vs non-reasoning baselines, also consider a non-reasoning companion run.

---

### Step 4 — Fill remaining ??? values

- Find every `???` left. Ask the user only for what can't be inferred (SLURM hostname/account/output_dir, MLflow tracking URI, etc.). Don't propose defaults; let them give plain text.
- Ask about other defaults they may want to change (partition, walltime, MLflow tags).

**Walltime cap: 4 hours.** Always `execution.walltime: "04:00:00"`. The cluster does not schedule jobs longer than 4h — this is a hard limit, not a preference.

Evals that exceed 4h of wall-clock time are handled by **NEL's built-in dependency-chain resume**, not by shrinking the eval. NEL submits the first SLURM job; if it hits walltime, a dependent follow-on job resumes from the response/result caches the first job wrote, then queues another follow-on. Long evals continue across walltime windows automatically. See `references/run-validation.md#nel-timeout-and-resume-behavior` for the full mechanism.

Implications for the agent:

- Do **not** lower `num_repeats`, split heavy tasks (AA-LCR, SciCode) into separate configs, or otherwise carve up the eval to fit inside 4h. Let NEL chain.
- Do **not** treat a walltime timeout as a failed run. Check `nel status` / `nel info` and the dependent job's logs before declaring failure. `references/run-validation.md` covers what a real failure looks like vs an expected resume event.
- Bumping `data_parallel_size` / `parallelism` to finish faster is fine when the goal is wall-clock latency, not a walltime workaround — but it's optional, not required, for runs longer than 4h.

---

### Step 5 — Confirm tasks (iterative)

1. Tell user: "Run `nel ls tasks` for the full task list."
2. For any task with a `recipes/tasks/` reference, read it and prefer its YAML fragment + repeat counts.
3. Ask about add/remove/modify. Per-task overrides under task's `nemo_evaluator_config.config.params`:

   ```yaml
   tasks:
     - name: <task>
       nemo_evaluator_config:
         config:
           params:
             temperature: <value>
             ...
   ```

4. Apply, show updated list, ask "Final, or more changes?" Loop until confirmed.

**Known issue — nemo-skills self-deployment:** If using `nemo_skills.*` tasks with self-deployment (vLLM/SGLang/NIM), add at top level:

```yaml
target:
  api_endpoint:
    api_key_name: DUMMY_API_KEY
```

External-deployment configs already define `api_key_name`. Export of `DUMMY_API_KEY` is handled in Step 8.

---

### Step 6 — Multi-node

For models > ~120B or higher throughput needs, read `references/multi-node.md` for HAProxy multi-instance / Ray TP/PP / combined patterns.

### Step 7 — Interceptors

Direct user to <https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/interceptors/index.html>. Do not provide generic interceptor info — read the specific interceptor's page if asked, then configure via `evaluation.nemo_evaluator_config.config.target.api_endpoint.adapter_config` (NOT under `target.api_endpoint.adapter_config`). Use the per-field syntax from the CLI Configuration section, not a full `interceptors:` list (that overrides the default chain).

**Errata:** Logging field names are `max_logged_requests` / `max_logged_responses` (NOT `max_saved_*` / `max_*` as some docs show).

### Step 7.5 — Container registry auth (SLURM private images only)

Default images:

| Framework | Image | Registry |
| --- | --- | --- |
| vLLM | `vllm/vllm-openai:v0.19.1` (bump per recipe; never `:latest`) | DockerHub |
| SGLang | `lmsysorg/sglang:latest` | DockerHub |
| TRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:...` | NGC |
| Eval tasks | `nvcr.io/nvidia/eval-factory/*:26.03` | NGC |

Public images → submit without preflight. Private/restricted → check credentials:

```bash
ssh <host> "grep -E '^\s*machine\s+' ~/.config/enroot/.credentials 2>/dev/null"
```

Add credentials per `skills/common/slurm-setup.md` §6 if missing. If you can't add, switch to a compatible public image (e.g. `nvcr.io/nvidia/vllm:<YY.MM>-py3` — check catalog.ngc.nvidia.com). **Do not retry more than once** after an auth failure.

---

### Step 8 — Run evaluation (gated dry-run → canary → full)

Run directly when the user asked to launch; otherwise ask before submitting.

**Env setup:** Copy `recipes/env.example` → `.env`, fill, source:

```bash
cp recipes/env.example .env
set -a && source .env && set +a

# If pre_cmd/post_cmd in config (review pre_cmd first — runs arbitrary commands):
export NEMO_EVALUATOR_TRUST_PRE_CMD=1
# If nemo_skills.* + self-deployment:
export DUMMY_API_KEY=dummy
```

**Step 8.1 — Dry-run** (config validation):

```bash
nel run --config <path> --dry-run
```

Fix unresolved `???`, bad Hydra overrides, missing env vars, invalid mounts, image issues, sbatch errors, obvious deployment errors before proceeding.

**Step 8.2 — Canary** (limited-samples, validates everything dry-run can't):

```bash
nel run --config <path> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

Catches judge auth/rate-limits, container failures, sandbox issues, OOM, bad request formatting, low evaluated counts. Always inspect logs:

```bash
nel status <id>
nel info <id> --logs
ssh <user>@<host> "grep -i 'traceback\|exception\|error\|failed\|oom\|killed\|timeout\|unauthorized\|rate limit\|sandbox\|container\|judge\|parse\|scoring' <log_path>/*.log"
```

Canary each risky task class separately (judge-scored, code-execution, model-only). Start `parallelism` conservatively; raise only after judge/sandbox logs are clean — they bottleneck before the model.

Single-task rerun: `nel run --config <path> -t <task_name>` (combine with `-o ++...limit_samples=10` for canary).

**Step 8.3 — Full run** (after canary passes):

```bash
nel run --config <path>
```

Remove `limit_samples` overrides; keep canary-validated parallelism. If the canary fails, fix and rerun the canary — don't skip to full.

**Monitoring:** Register the job per the **monitor** skill for cross-session tracking. One-off live status / debugging → **launching-evals** skill. Past-run MLflow queries → **accessing-mlflow** skill. NEL timeout/resume → read `references/run-validation.md` before treating the run as failed.

---

### Step 9 — Verify completed run

Before pulling/reporting scores, validate the run. Read `references/run-validation.md` for NEL timeout/resume behavior, completed-run validation, diagnostics, score harvesting, and the handoff to `compare-results` for baseline-vs-candidate deltas.

---

Issues: <https://github.com/NVIDIA-NeMo/Evaluator/issues> · <https://github.com/NVIDIA-NeMo/Evaluator/discussions>
