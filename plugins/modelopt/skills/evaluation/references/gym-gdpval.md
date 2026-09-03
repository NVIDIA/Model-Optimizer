# GDPVal (NeMo Gym "Stirrup" agent) — reference for the gym / agentic path

GDPVal is currently validated with the **0.2.6 `nel` launcher** as a `nemo_gym`
task, but it is mechanically unlike the `aa/` nemo-skills tasks: the Stirrup agent
produces office/PDF **deliverables** in a per-task **Apptainer** code-exec sandbox,
a pairwise/rubric **judge** (Gemini 3.1 Pro) scores them, and NeMo Gym is pulled
and run **inline in the eval container** (`install_on_the_fly`) via
`ng_prepare_benchmark` + `ng_e2e_collect_rollouts`. This file is the shared
machinery; the config template is `recipes/examples/gym/example_gdpval.yaml` and
the per-task pointer is `recipes/tasks/gym/gdpval.md`. The gym bootstrap machinery
described here is shared with MRCR (`recipes/tasks/gym/mrcr.md`) — fixes here apply
to both examples.

Always invoke GDPVal through the pinned wrapper, even if `nel` is already on PATH:

```bash
"$SKILL_DIR/scripts/nel-gdpval.sh" --version  # must report nemo_evaluator_launcher: 0.2.6
"$SKILL_DIR/scripts/nel-gdpval.sh" run --config <gdpval-config.yaml> --dry-run
"$SKILL_DIR/scripts/nel-gdpval.sh" run --config <gdpval-config.yaml>
```

Using the wrapper with a validated pin is a correctness and reproducibility
requirement. The currently pinned 0.2.6 launcher writes the generated
`NEL_INVOCATION_ID` into `run.sub` before environment-variable re-exports. The
failure this pin fixes emits
`export NEL_INVOCATION_ID="${NEL_INVOCATION_ID}"` without first assigning it, then
exits with `NEL_INVOCATION_ID: unbound variable` under `set -u` before the evaluation
client starts. In a dry-run, verify that a literal assignment appears before the
re-export; do not substitute `SLURM_JOB_ID`, because it changes across the
benchmark's walltime-resume chain.

The exact version is a reproducibility baseline, not a claim that future launchers
are incompatible. Keep baseline and candidate evaluations on the same validated
launcher so a harness change does not become part of the measured model delta.

## Updating the launcher pin

When a newer `nemo-evaluator-launcher` release is available:

1. Review its release notes for launcher schema, generated Slurm, resume, and export
   changes.
2. Update `NEL_GDPVAL_VERSION` in `scripts/nel-gdpval.sh` and the expected spec in
   `tests/test_nel_gdpval.py`.
3. Run the focused test and pre-commit checks. Verify `nel-gdpval.sh --version`
   reports the candidate version.
4. Dry-run a known GDPVal config and confirm the literal `NEL_INVOCATION_ID`
   assignment still precedes its runtime re-export in every generated `run.sub`.
5. Launch with the candidate version and monitor the first 20–30 minutes for SIF
   sandbox startup and judge authentication. GDPVal ignores `limit_samples`, so
   there is no cheap reduced-sample canary.

Only then update the validated version used for scored runs. Do not mix launcher
versions within a baseline-versus-candidate comparison.

## Where each piece runs

| Component | Where |
|---|---|
| Policy model (under test) | your self-deployed vLLM endpoint (SLURM GPU node) — or an external endpoint |
| NeMo Gym + Stirrup agent orchestration | inside the **eval** container (`nemo_gym` task), pulled via `install_on_the_fly` |
| Per-task code-exec | **Apptainer SIF** launched by the agent inside the eval container |
| Judge (pairwise/rubric) | external OpenAI-compatible endpoint (`gdpval_judge`, e.g. Gemini 3.1 Pro) |
| Agent web search | Tavily (`TAVILY_API_KEY`) |

## Apptainer SIF sandbox

The Stirrup agent runs each task's generated code in an Apptainer SIF, bind-mounted
at **exactly** the path `GDPVAL_CONTAINER_PATH` names (template:
`/gdpval/sif/python-3.13.gdpval.sif`). Missing or misnamed → the agent **silently**
runs code-exec unsandboxed; the run "succeeds" but the numbers aren't comparable.

**If your site provides a prebuilt SIF, use it** — a self-built one resolves its pip
stack at *your* build time and can drift from the sandbox a published reference set
was generated in. Mount the provided dir at `/gdpval/sif` and point
`GDPVAL_CONTAINER_PATH` at its filename. (NVIDIA-internal: `modelopttools:eval-config`
Step 3c has the path.) Otherwise build it on the target cluster — never copy a SIF
between clusters:

```bash
srun -p cpu -t 01:00:00 --pty "$SKILL_DIR/scripts/gdpval-sif.sh"   # uses $GDPVAL_SIF_DIR
```

`gdpval-sif.sh` is idempotent (flock-guarded, atomic): it builds from `gdpval.def` at
the pinned commit if absent and is a no-op once present. It needs
apptainer/singularity with unprivileged-build support and network egress — run it on a
login or CPU node, **not** inside the eval job. The eval image doesn't ship apptainer,
so the config's `pre_cmd` installs the **runtime** (arch-aware: use the Ubuntu PPA, not
an amd64 `.deb` — most Blackwell/Grace clusters are aarch64), which needs
`NEMO_EVALUATOR_TRUST_PRE_CMD=1`.

**The SIF is versioned with the Gym repo.** `gdpval.def` changes across commits (e.g.
`2502893977` → `049b1fd0` moved python 3.12 → 3.13 and added TeX Live, playwright,
polars, geospatial), and the newer agent's prompt advertises that richer runtime. So
when you bump `install_on_the_fly.commit`, **diff the def at the two commits**
(`raw.githubusercontent.com/NVIDIA-NeMo/Gym/<sha>/responses_api_agents/stirrup_agent/containers/gdpval.def`);
if it changed, rebuild to a **new version-tagged filename** (`GDPVAL_SIF_NAME=… gdpval-sif.sh --commit <sha>`)
and repoint `GDPVAL_CONTAINER_PATH`. Running a new gym on an old SIF makes the
generated code fail its imports *inside the sandbox* — deliverables silently degrade
with no error in the eval.

**Exception — a site-provided SIF paired with a site-provided gym image.** Those
images typically bake Gym as a non-git dir, so `install_on_the_fly.commit` is inert
and the two provided artifacts are already matched to each other; that pairing is the
coherent one even when the SIF filename encodes a different SHA than your pin. You
cannot check the image's baked Gym from the config — confirm from the client log
(`=== NeMo Gym commit ===` + a SHA, or the "not a git repo" line).

## Gym prepare / reap (why the task `command:` is long)

The task `command:` carries two workaround blocks, inlined in the template:

1. **prepare** — activate the baked Gym venv, checkout the `install_on_the_fly` pin
   (only if `/opt/Gym` is a git repo), repair the image's incomplete per-server venvs
   (drop the editable `-e nemo-gym[dev]` line, which forces a ray re-resolve that
   breaks venv-less servers; pin `ray==2.49.2` + `tqdm`), and front the main venv on
   `PYTHONPATH`.
2. **run** — data prep, then `ng_e2e_collect_rollouts` executed from a script written
   via a **quoted heredoc** and launched under `setsid`, so the whole server/Ray
   process tree can be reaped by process group. Without that reap, orphaned Ray
   workers hold the launcher's stdout open and the run **hangs in post-eval**; the
   quoted heredoc keeps `$$` and `$*_API_KEY` unexpanded until run time and survives
   params that contain single quotes (comparison mode's `stages='[{...}]'`).

Both compensate for the eval image's deployment-oriented packaging and Gym's
incomplete shutdown — remove them once the image ships complete ray-consistent venvs
and Gym reaps its own process groups. Avoid bash `${VAR}` inside these blocks:
OmegaConf parses `${...}`. `$(...)`, `$$` and `$VAR` are fine.

## Deployment sizing

GDPVal is heavy: 220 tasks × `num_repeats` rollouts, each a long multi-turn agent
episode with code-exec + judge calls (`request_timeout: 36000`). The example
self-deploys single-node vLLM, which is fine for a canary or a small policy. For the
**full run of a large MoE** (e.g. MiniMax-M2.7), the reviewed golden uses **multi-node
`vllm_ray`** (16 × 4-GPU HSG = 64 GPUs, `walltime 04:00:00`). To scale up:

+ Switch `defaults: - deployment: vllm_ray` and add nodes (`execution.num_nodes`);
  see `references/multi-node.md` for the Ray TP/PP layout.
+ `parallelism` (`16384`) is **gym-internal concurrency**, not a server cap. The
  real throttles are the agent's `stirrup_agent.concurrency` and the judge's
  `max_concurrent_requests` — raise those only after the judge logs are clean of 429s.
+ **`--max-num-seqs`: derive it from `stirrup_agent.concurrency`, NOT `parallelism`.**
  SKILL Step 3/4's `ceil(parallelism / DP)` rule assumes `parallelism` is the in-flight
  request count; on the gym path it is not, and applying it literally gives an absurd
  cap. Use `ceil(stirrup_agent.concurrency / DP)` — e.g. 220 / DP 4 → 55, round to 64.
+ **`max_new_tokens`:** the reviewed golden **does** set it alongside the adapter's
  `params_to_remove: [max_tokens, max_completion_tokens]`, so do the Step 3 model-card
  lookup as normal. The template omits it (five params) because the adapter strips the
  per-request cap anyway; adding it back matches the golden and is harmless.
+ **Match `temperature` / `top_p` to whatever the reference deliverables were generated
  with.** A pairwise ELO compares your deliverables against theirs, so a sampling
  difference lands in the score as if it were a quality difference.
+ Long runs exceed 4h; rely on NEL's walltime dependency-chain resume
  (`resume_from_cache=true` is already set). See SKILL Step 4 + `run-validation.md`.

## Scoring modes — rubric vs comparison

+ **`rubric`** (template default) — judge scores each deliverable against its rubric.
  0–1 reward, **no ELO** (undefined without an opponent). Runs on the public gym image.
+ **`comparison`** — pairwise vs anchored reference deliverables; the **only** mode
  yielding the AA-comparable `normalized_elo`. It is a conversion, not a flag flip:
  it needs a reference set, a gym image whose Gym has the `reference_models` map, ref
  mounts on **both** deployment and evaluation, and multistage overrides. Setting
  `reward_mode=comparison` alone exits at startup with
  `reward_mode=comparison requires reference_deliverables_dir to be set`, surfaced
  only as `Process gdpval_resources_server finished unexpectedly!`.
  NVIDIA-internal: `modelopttools:eval-config` Step 3c is the conversion checklist.

## Judge

Rubric mode uses a single judge. **Comparison mode uses a 3-member panel** —
`openai/gpt-5.5`, `gcp/google/gemini-3.1-pro-preview`,
`aws/anthropic/bedrock-claude-opus-4-8` — one **sampled per trial**, all routed
through the single `gdpval_judge_model` proxy (`<INFERENCE_JUDGE_URL>` from `.env`).
`++...judge_sampling_seed=42` makes that sampling reproducible.

+ **Inject the key's VALUE, not its name:** `openai_api_key=$INFERENCE_API_KEY`.
  Passing an env-var *name* (e.g. via a `${...api_key}` interpolation that resolves to
  the literal string `INFERENCE_API_KEY`) makes the proxy reply `LiteLLM Virtual Key
  expected`, which the gym wraps as an opaque **500** — it looks like a judge outage,
  not a config error. One key covers all three panel members.
+ **Do not set `judge_responses_create_params_overrides.model`.** Pinning a model
  collapses the panel to a single judge, silently changing the scoring methodology.
+ **Throttles:** judge `max_concurrent_requests=10` and Stirrup `concurrency=220` are
  the golden values — the judge rate-limits long before the served model does, so raise
  these only after the judge logs are clean of 429s.

## Preflight — what NEL validates, and what it does NOT

NEL validates mount paths at **submit** time (`_collect_mount_paths` +
`_validate_remote_paths_exist`): it ssh's to the cluster, runs `test -d` on every
mount source, and `raise ValueError` listing the missing ones **before** any
`sbatch` — so a missing reference dir or cache costs you nothing. Three gaps to know:

| Artifact | Missing → | Loud? |
|---|---|---|
| mounted dirs (refs, caches, SIF **dir**, checkpoint) | `ValueError` at submit, no job queued | ✅ pre-allocation |
| **the SIF file inside that dir** | **agent silently runs code-exec unsandboxed** | ❌ **silent** |
| task `container:` (image / `.sqsh`) | not collected for validation → pyxis import failure | ⚠️ only after allocation |

1. **`test -d` proves the directory, not the SIF.** A `$GDPVAL_SIF_DIR` that exists but
   holds the *wrong* filename (e.g. `python-3.12…` after bumping to a 3.13 def) passes
   validation, and the run then silently degrades. Guard with the verify-only mode:

   ```bash
   "$SKILL_DIR/scripts/gdpval-sif.sh" --check     # uses $GDPVAL_SIF_DIR; exit 1 + lists what IS there
   ```

   Keep `GDPVAL_SIF_NAME` / the helper's default in sync with the config's
   `GDPVAL_CONTAINER_PATH`; they are the same string in two places.
2. **`--dry-run` skips remote validation entirely** (it never opens the ssh
   connection). A clean dry-run says nothing about whether your mounts exist — run
   the preflight separately.
3. **The container is never checked.** A wrong/rotated image path fails at pyxis
   import, i.e. after the allocation is granted. Verify it with `ls -l` first
   (comparison mode's internal image especially — see `modelopttools:eval-config`).

## Env vars

| Var | Prefix | Purpose |
|---|---|---|
| `HF_TOKEN` | host | model/dataset downloads |
| `INFERENCE_API_KEY` | host | **judge** auth (and policy if external) |
| `TAVILY_API_KEY` | host | Stirrup agent web search |
| `DUMMY_API_KEY` | lit:dummy | self-deployed vLLM policy key |
| `GDPVAL_CONTAINER_PATH` | lit | SIF path — must equal the SIF bind-mount target |
| `GDPVAL_REF_FILES_DIR` | lit:/gdpval_ref_files | shared-FS ref-file staging (node-local /tmp breaks multi-node Ray) |
| `PERSIST_DELIVERABLES_DIR` | lit | where deliverables persist (see MLflow note) |
| `GDPVAL_MAX_TURNS` | lit (optional) | Stirrup turn cap (default 100; golden uses 250) |
| `NEL_INVOCATION_ID` | runtime | stable run id assigned by the validated launcher; do not use `SLURM_JOB_ID` |

`INFERENCE_JUDGE_URL` is the judge host — config (from `.env`), substituted as the
literal `<INFERENCE_JUDGE_URL>` placeholder in `gdpval_judge.base_url`, **not**
`${oc.env:...}`. Judge `model_id` is hardcoded in the config (swap for an equivalent
on your endpoint). The upstream OSS recipe uses a separate `GDPVAL_JUDGE_API_KEY`;
this template reuses the shared `INFERENCE_API_KEY` for the judge.

`GDPVAL_SIF_DIR` (`.env`) is host-side config, **not** a container env var: it's the
persistent SIF cache dir the helper builds into and the config bind-mounts at
`/gdpval/sif`. `gdpval-sif.sh` reads `$GDPVAL_SIF_DIR` directly (its default target);
the config mount is a **literal `<GDPVAL_SIF_DIR>` placeholder** you substitute with
that same path — mount KEYS aren't interpolated, so don't emit `${oc.env:...}` there
(same rule as the judge URLs). One `.env` value feeds both, so the build path and the
run path can't drift.

## MLflow export — the deliverables trap

Deliverables can be large. The mlflow exporter excludes any artifact dir whose
basename matches `*cache*`, so the template sets
`PERSIST_DELIVERABLES_DIR=/results/gdpval/deliverables_cache`: the deliverables stay
under `/results` for inspection but are **not** auto-uploaded. Drop the `_cache`
suffix only if you actually want them uploaded. Everything else about auto-export is
standard (SKILL Step 1 shortcut #4): `auto_export.destinations: [mlflow]` +
`cpu_partition` + a literal-valued `export.mlflow` block (tag `benchmark:
nemo_gym.gdpval`).

## num_repeats

**Use 1.** Both current goldens do, set with a top-level `++num_repeats=1` — it
works, and recent Gym pins already ship `num_repeats: 1` in
`benchmarks/gdpval/config.yaml`, so no `sed` patching is needed.

Historical only: pre-multistage single-reference configs used 2 (220 × 2 = 440
rollouts) and patched it with `sed` because the per-dataset key could not be set
via `++` on those pins. Do not carry a `=2` into a current run.

## Failure modes to check at canary

+ **Silent unsandboxed exec** — grep the eval log for the SIF fallback warning /
  apptainer mount errors; confirm `GDPVAL_CONTAINER_PATH` == the mount target.
+ **Judge 401 / 429** — wrong `INFERENCE_JUDGE_URL` / key, or `max_concurrent_requests`
  too high for the judge endpoint.
+ **Empty reasoning / low win-rate** — thinking mode off. Confirm
  `chat_template_kwargs.enable_thinking: true` (right toggle key for the family) +
  the policy's `--reasoning-parser`.
+ **Run hangs in post-eval** — orphaned Ray/gym processes holding stdout; that's what
  the setsid + process-group reap in the task `command:` prevents.
+ **Multi-node ref-file errors** — `GDPVAL_REF_FILES_DIR` on node-local storage;
  point it at a shared-FS staging dir.
