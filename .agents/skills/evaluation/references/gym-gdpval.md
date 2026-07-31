# GDPVal (NeMo Gym "Stirrup" agent) — reference for the gym / agentic path

GDPVal runs on the **0.2.6 `nel` launcher** as a `nemo_gym` task, but it is
mechanically unlike the `aa/` nemo-skills tasks: the Stirrup agent produces
office/PDF **deliverables** in a per-task **Apptainer** code-exec sandbox, a
pairwise/rubric **judge** (Gemini 3.1 Pro) scores them, and NeMo Gym is pulled and
run **inline in the eval container** (`install_on_the_fly`) via `ng_prepare_benchmark`
+ `ng_e2e_collect_rollouts`. This file is the shared machinery; the config template
is `recipes/examples/gym_gdpval/` and the per-task pointer is
`recipes/tasks/aa_gym/gdpval.md`.

## Where each piece runs

| Component | Where |
|---|---|
| Policy model (under test) | your self-deployed vLLM endpoint (SLURM GPU node) — or an external endpoint |
| NeMo Gym + Stirrup agent orchestration | inside the **eval** container (`nemo_gym` task), pulled via `install_on_the_fly` |
| Per-task code-exec | **Apptainer SIF** launched by the agent inside the eval container |
| Judge (pairwise/rubric) | external OpenAI-compatible endpoint (`gdpval_judge`, e.g. Gemini 3.1 Pro) |
| Agent web search | Tavily (`TAVILY_API_KEY`) |

## Apptainer SIF sandbox (self-contained: build-if-absent, reuse-if-present)

The Stirrup agent runs each task's generated code in an Apptainer SIF, bind-mounted
into the eval container at **exactly** `/gdpval/sif/python-3.12.gdpval.sif` (the path
`GDPVAL_CONTAINER_PATH` points at). If it's missing or at a different path, the agent
**silently falls back to non-sandboxed local exec** — the run "succeeds" but the
numbers aren't comparable, so verify the SIF at canary.

**The skill builds the SIF on the target cluster — it never copies one from
another cluster.** That is the self-contained default and the right behaviour when
you have no prebuilt sandbox.

> **If your site already provides a prebuilt GDPVal SIF, prefer it over building —
> build only when it is absent.** A self-built SIF resolves its pip stack at *your*
> build time, so it can drift from the sandbox a published reference set was
> generated in, which matters for comparison-mode runs. To use a provided one, mount
> its dir at `/gdpval/sif` and point `GDPVAL_CONTAINER_PATH` at the provided
> filename. **NVIDIA-internal users:** `modelopttools:eval-config` (Step 3c) records
> the provided SIF's location and the check-then-fall-back procedure — if you have
> that skill, consult it before building. Without it, building (below) is correct.

Ship the SIF via the idempotent helper, which builds it if absent and reuses it if
already present:

```bash
# GDPVAL_SIF_DIR (.env) — persistent shared-FS dir on the TARGET cluster; the config
# bind-mounts this same dir at /gdpval/sif. Preferred: run the ~30-min build on the
# CPU partition, not a login node. `set -a && source .env` first so it's set.
srun -p cpu -t 01:00:00 --pty \
  .agents/scripts/gdpval-sif.sh          # defaults to $GDPVAL_SIF_DIR (or pass a dir)
```

`gdpval-sif.sh` builds from the NeMo Gym `gdpval.def` at the pinned commit (keep
`GDPVAL_GYM_COMMIT` in sync with the config's `install_on_the_fly.commit`), writes
the SIF into that dir, and is flock-guarded + atomic so concurrent runs never
double-build. Re-running is a no-op once the SIF exists — that's the "reuse the
built one" path. It needs `apptainer`/`singularity` on the build host with
fakeroot/unprivileged build support and network egress; run it on a login or CPU
node (outside enroot, where fakeroot works), **not** inside the eval job.

**Build vs run are separate.** The helper *builds* the SIF (once, off-GPU). The
eval then *runs* the prebuilt SIF inside the eval container — the golden-validated
path. The eval image doesn't ship apptainer, so the config's `pre_cmd` installs the
apptainer **runtime** + squashfuse (FUSE-mounts the SIF; falls back to slower
per-call extraction if `/dev/fuse` is absent). `pre_cmd` runs arbitrary commands →
the launching shell needs `export NEMO_EVALUATOR_TRUST_PRE_CMD=1`. The
apptainer-under-pyxis nesting is the least-validated part of the SLURM path — check
the canary logs for the "falling back to local exec" warning and apptainer mount
errors. (Local/Docker executor instead needs `--privileged` + the SIF bind mount via
`execution.extra_docker_args`; a rootless `--security-opt` + `/dev/fuse` variant is
in the upstream README appendix.)

### Rebuild the SIF when the Gym version changes (SIF ↔ commit coupling)

The SIF is **versioned with the Gym repo**: it's built from `gdpval.def` at
`install_on_the_fly.commit`, and that def changes across commits. Example — between
`2502893977` and the golden `049b1fd0`, the base went **python-3.12 → 3.13** and the
stack gained TeX Live, chromium/playwright, polars/duckdb, xgboost, geospatial
(gdal/proj/geos), and audio/video libs. The newer Stirrup agent's prompt advertises
that richer runtime, so the model's generated code reaches for those libs.

**So whenever you bump `install_on_the_fly.commit`, rebuild the SIF from the matching
commit.** Run the new gym with an old SIF and the generated code fails its imports
*inside the sandbox* — deliverables silently degrade (missing figures/tables/docs) →
junk scores, with no hard error in the eval. Rebuild to a **version-tagged filename**
so the old SIF isn't clobbered (a running job keeps working), then repoint
`GDPVAL_CONTAINER_PATH` + the `/gdpval/sif` mount at the new file:

```bash
# build the SIF for the NEW gym commit under a distinct name (old SIF stays intact)
GDPVAL_SIF_NAME=python-3.13.gdpval.sif \
  .agents/scripts/gdpval-sif.sh --commit <new-install_on_the_fly.commit> "$GDPVAL_SIF_DIR"
# then set GDPVAL_CONTAINER_PATH=/gdpval/sif/python-3.13.gdpval.sif in the config
```

Rule of thumb: **`install_on_the_fly.commit` and the SIF move together.** Any bump
that alters `gdpval.def` (base image or the apt/pip stack) needs a rebuild; a bump
that leaves `gdpval.def` byte-identical does not — diff the def at the two commits
(`raw.githubusercontent.com/NVIDIA-NeMo/Gym/<sha>/responses_api_agents/stirrup_agent/containers/gdpval.def`)
to be sure. Note the def already disables apt's sandbox internally as of `049b1fd0`,
so it builds cleanly under the helper's unprivileged path.

## The `_gym_prepare.yaml` include (why it exists)

`nemo_gym` tasks interpolate two shared snippets into the task `command:`:
`${gym_prepare.prepare}` (activate the baked Gym venv, checkout the
`install_on_the_fly` pin, repair the image's incomplete per-server venvs, front the
main venv on `PYTHONPATH`) and `${gym_prepare.run}` (data prep +
`ng_e2e_collect_rollouts`, run in its own `setsid` session so the whole server/Ray
process tree can be reaped by process group — otherwise orphaned Ray workers hold
the launcher's stdout open and the run **hangs in post-eval**). Both compensate for
the eval image's deployment-oriented packaging and Gym's incomplete shutdown; remove
once the image ships complete ray-consistent venvs.

**The include is co-located, not central.** Hydra resolves `- _gym_prepare` relative
to the run config's directory, so `_gym_prepare.yaml` must sit next to your config —
copy the whole `recipes/examples/gym_gdpval/` dir, not the yaml alone.

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
+ Long runs exceed 4h; rely on NEL's walltime dependency-chain resume
  (`resume_from_cache=true` is already set). See SKILL Step 4 + `run-validation.md`.

## Scoring modes — rubric vs comparison

+ **`rubric`** (template default) — the judge scores each deliverable standalone
  against a rubric; **no reference deliverables** needed.
+ **`comparison`** — pairwise: the judge compares the policy's deliverable to a
  **reference model's** deliverable and the result is an ELO-anchored win-rate.
  Two-step flow:
  1. **Baseline:** run your reference model with `-o gdpval.reward_mode=rubric` to
     generate baseline deliverables (they land under `PERSIST_DELIVERABLES_DIR`).
  2. **Comparison:** run the candidate with `-o gdpval.reward_mode=comparison`, mount
     the baseline deliverables at `/gdpval/refs/test_ref`
     (`execution.mounts.evaluation`), and set `gdpval.reference_elo` to the reference
     model's ELO (golden: Kimi-K2.5-Thinking, elo=1290).

### Comparison mode needs a newer Gym than the public image — override `container:`

The public `nvcr.io/nvidia/eval-factory/nemo-gym:26.05` (== `latest` by digest) runs
**rubric** mode fine, but its Gym predates the multi-reference `reference_models` map.
In comparison mode `gdpval_resources_server` fails validation at startup with
`reward_mode=comparison requires reference_deliverables_dir to be set`, surfacing only
as the unhelpful `Process gdpval_resources_server finished unexpectedly!`.

**You cannot fix this by bumping `install_on_the_fly.commit`.** That image bakes Gym as
a **non-git directory**, so the pin is silently ignored — the prepare step logs
`=== /opt/Gym is not a git repo; using baked-in Gym version ===` and you get the baked
build regardless. Treat the pin as inert unless the log shows `=== NeMo Gym commit ===`
followed by a SHA, and don't attribute run-to-run behaviour changes to it. (In
particular, a head-server `address already in use` on its fixed port 11000 is a
**transient collision** — resubmitting clears it; it is not a Gym-version symptom.)

To run comparison mode you must **override the task's `container:`** with an image
whose Gym has `reference_models`. NVIDIA-internal users: the image path, the canonical
reference set, and the matching gym overrides are in the `modelopttools:eval-config`
skill (Step 3c) — internal cluster paths deliberately live only there.

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
   .agents/scripts/gdpval-sif.sh --check     # uses $GDPVAL_SIF_DIR; exit 1 + lists what IS there
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
| `NEL_INVOCATION_ID` | runtime | run id |

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

## num_repeats workaround (OmegaConf)

`++gdpval.*.num_repeats=N` hits an OmegaConf `ListConfig` merge error, so the count
is patched by editing the checked-out file inside the task `command:`:
`sed -i 's/num_repeats: 2$/num_repeats: 1/' benchmarks/gdpval/config.yaml`. The Gym
default is 2 (the golden). The template seds it to **1** to halve cost; **remove the
sed line to keep 2 for golden-comparable / reported scores.**

## Failure modes to check at canary

+ **Silent unsandboxed exec** — grep the eval log for the SIF fallback warning /
  apptainer mount errors; confirm `GDPVAL_CONTAINER_PATH` == the mount target.
+ **Judge 401 / 429** — wrong `INFERENCE_JUDGE_URL` / key, or `max_concurrent_requests`
  too high for the judge endpoint.
+ **Empty reasoning / low win-rate** — thinking mode off. Confirm
  `chat_template_kwargs.enable_thinking: true` (right toggle key for the family) +
  the policy's `--reasoning-parser`.
+ **Run hangs in post-eval** — orphaned Ray/gym processes holding stdout; that's what
  the `${gym_prepare.run}` setsid + process-group reap prevents. If it still hangs,
  the `_gym_prepare.yaml` include didn't travel with the config.
+ **Multi-node ref-file errors** — `GDPVAL_REF_FILES_DIR` on node-local storage;
  point it at a shared-FS staging dir.
