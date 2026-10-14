# NeMo Gym (`nemo_gym`) — shared reference for the gym path

Gym tasks run on the **0.2.6 `nel` launcher** (not nel-next), so SKILL Steps 1–9
apply — but they are mechanically unlike the `aa/` nemo-skills tasks: NeMo Gym is
pulled and run **inline in the eval container** (`install_on_the_fly`), and each gym
task is **standalone** (one gym eval per config, never merged into a multi-task
`tasks` list). Two config schemas are in use:

| Schema | Tasks | Gym entry points | Image needs |
|---|---|---|---|
| **explicit** (`data_prep_params` / `collect_rollout_params`) | MRCR | `ng_prepare_benchmark` + `ng_e2e_collect_rollouts` in `/opt/Gym` | a **git-backed** `/opt/Gym` |
| **condensed** (`benchmark` / `prepare_args` / `run_args`) | Tau3-Banking | `gym eval prepare` + `gym eval run`, Gym cloned into a temp dir | python + `uv` + `git` + `ray`; **no** `/opt/Gym` |

The condensed `command:` bootstrap must be carried in the config (the template does): the
image does not ship it, and without it nemo-evaluator falls back to `framework.yml`
(`cd /opt/Gym`). It verifies the pin and reaps its own process group; the pin,
prepare/reap and `pre_cmd` sections below describe the explicit schema.

**Serving the policy model:** if it has an upstream fragment (`configs/models/<model>/` in
nvidia-eval-factory-benchmarking), carry its serving layout (TP/DP/expert parallel,
`extra_args`) and `deployment.env_vars` over **as a unit** — env vars there can be
prerequisites of the layout, not tuning. To deviate, change both together.

Per-task recipes: `recipes/tasks/gym/*.md`; self-contained examples:
`recipes/examples/gym/example_<task>.yaml`. A fix here applies to every gym example.

Always invoke a gym task through the pinned wrapper, even if `nel` is already on
PATH:

```bash
"$SKILL_DIR/scripts/nel-gym.sh" --version  # must report nemo_evaluator_launcher: 0.2.6
"$SKILL_DIR/scripts/nel-gym.sh" run --config <gym-config.yaml> --dry-run
"$SKILL_DIR/scripts/nel-gym.sh" run --config <gym-config.yaml>
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
2. Update `NEL_GYM_VERSION` in `scripts/nel-gym.sh` and the expected spec in
   `tests/test_nel_gym.py`.
3. Run the focused test and pre-commit checks. Verify `nel-gym.sh --version`
   reports the candidate version.
4. Dry-run a known gym config and confirm the literal `NEL_INVOCATION_ID`
   assignment still precedes its runtime re-export in every generated `run.sub`.
5. Launch with the candidate version and monitor the first 20–30 minutes for gym
   bootstrap and rollout flow. The launcher-level `limit_samples` reaches the gym
   only if the template gates it into `++limit` — `example_mrcr.yaml` does (see
   `recipes/tasks/gym/mrcr.md` Canary); otherwise add `++limit=N` to
   `collect_rollout_params`. The data-prep pass still runs in full.

Only then update the validated version used for scored runs. Do not mix launcher
versions within a baseline-versus-candidate comparison.

## Where each piece runs

| Component | Where |
|---|---|
| Policy model (under test) | your self-deployed vLLM endpoint (SLURM GPU node) — or an external endpoint |
| NeMo Gym + agent orchestration | inside the **eval** container (`nemo_gym` task), pulled via `install_on_the_fly` |

## Gym pin ↔ container

`install_on_the_fly.commit` is applied by `git checkout` in `/opt/Gym`, so it only
takes effect where that is a git repo. Images that bake Gym as a plain directory
(the public `nvcr.io/nvidia/eval-factory/nemo-gym:*` images) make the pin **inert**.
Whether that is acceptable is per-task: a task whose prepare path only exists after
the pin must fail closed rather than score a different benchmark green, so its
example sets `container: ???` and its bootstrap exits 1 on a non-git `/opt/Gym`.
You cannot check the image's baked Gym from the config — confirm from the client log
(`=== NeMo Gym commit ===` + a SHA, or the "not a git repo" line), and match the
**SHA**, not just the marker: a stale checkout still prints the marker.

## Gym prepare / reap (why the task `command:` is long)

The task `command:` carries two workaround blocks, inlined in each template:

1. **prepare** — activate the baked Gym venv, checkout the `install_on_the_fly` pin
   (only where `/opt/Gym` is a git repo), repair the image's incomplete per-server
   venvs (drop the editable `-e nemo-gym[dev]` line, which forces a ray re-resolve
   that breaks venv-less servers; append `tqdm` and a `ray[default]` pinned to **the
   image's own ray version** — hardcoding a version makes `uv` unsatisfiable when the
   image moves), and front the main venv on `PYTHONPATH`.
2. **run** — data prep, then `ng_e2e_collect_rollouts` executed from a script written
   via a **quoted heredoc** and launched under `setsid`, so the whole server/Ray
   process tree can be reaped by process group. Without that reap, orphaned Ray
   workers hold the launcher's stdout open and the run **hangs in post-eval**; the
   quoted heredoc keeps `$$` and `$*_API_KEY` unexpanded until run time and survives
   params that contain single quotes. The Ray teardown is **uid-scoped** (`pkill -u`)
   — an unscoped `pkill` would kill other jobs' Ray daemons on a shared node.

Both compensate for the eval image's deployment-oriented packaging and Gym's
incomplete shutdown — remove them once the image ships complete ray-consistent venvs
and Gym reaps its own process groups.

Two editing rules for these blocks:

- **Avoid bash `${VAR}`** — OmegaConf parses `${...}`. `$(...)`, `$$` and `$VAR` are
  fine.
- **`set +x` around any param that carries a secret.** `set -x` traces *after*
  expansion, so an `$HF_TOKEN` in `data_prep_params` otherwise lands in the log —
  and in MLflow via `log_logs`.

## Env vars

| Var | Prefix | Purpose |
|---|---|---|
| `HF_TOKEN` | host | model/dataset downloads |
| `DUMMY_API_KEY` | lit:dummy | self-deployed vLLM policy key |
| `NEL_INVOCATION_ID` | runtime | stable run id assigned by the validated launcher; do not use `SLURM_JOB_ID` |

Two launcher-level trust flags gate gym submissions, both set in `.env`:

- `NEMO_EVALUATOR_TRUST_PRE_CMD=1` — for configs that carry a `pre_cmd` (MRCR); prepare
  fails without it.
- `NEMO_EVALUATOR_TRUST_UNLISTED_TASKS=1` — `nemo_gym` is not in the FDF mapping, so
  submission is refused without it.

No gym task here is judge-scored; Tau3-Banking's user simulator adds `INFERENCE_API_KEY`
and `TAU3_USER_*` (its recipe). Endpoint URLs and mount KEYS are substituted as literals,
not `${oc.env:...}`.

## Preflight — what NEL validates, and what it does NOT

NEL validates mount paths at **submit** time (`_collect_mount_paths` +
`_validate_remote_paths_exist`): it ssh's to the cluster, runs `test -d` on every
mount source, and `raise ValueError` listing the missing ones **before** any
`sbatch` — so a missing cache or staging dir costs you nothing. Two gaps to know:

| Artifact | Missing → | Loud? |
|---|---|---|
| mounted dirs (caches, checkpoint, staging) | `ValueError` at submit, no job queued | ✅ pre-allocation |
| task `container:` (image / `.sqsh`) | not collected for validation → pyxis import failure | ⚠️ only after allocation |

1. **`test -d` proves the directory, not its contents.** A mounted dir that exists but
   holds the wrong filename passes validation, and the run then silently degrades.
2. **`--dry-run` skips remote validation entirely** (it never opens the ssh
   connection). A clean dry-run says nothing about whether your mounts exist — run
   the preflight separately.
3. **The container is never checked.** A wrong/rotated image path fails at pyxis
   import, i.e. after the allocation is granted. Verify it with `ls -l` first.

## MLflow export

Standard for the gym path (SKILL Step 1 shortcut #4): `auto_export.destinations:
[mlflow]` + `cpu_partition` + a literal-valued `export.mlflow` block, tagged
`benchmark: nemo_gym.<task>`.

One gym-specific trap: if a task persists large per-rollout output under `/results`,
the mlflow exporter excludes any artifact dir whose **basename matches `*cache*`**.
Naming such a dir `..._cache` keeps the files on disk for inspection without
auto-uploading them; drop the suffix only if you actually want them uploaded.

## num_repeats

- **Explicit schema (MRCR):** `ng_e2e_collect_rollouts` treats a `type: benchmark`
  dataset's declared `num_repeats` as a placeholder, so pin `++num_repeats=1` in
  `common_params` and report `pass@1`.
- **Condensed schema (Tau3-Banking):** `gym eval run` applies the benchmark's declared
  repeats, so set no override and check the count — Tau3's 5 give
  `num_samples_total == 485` (every reviewed upstream run shows 485).

**Do not change repeat counts when aligning to a golden.**

## Failure modes to check at canary

- **Pin not applied** — `grep -A1 "=== NeMo Gym commit ===" $RD/logs/client-*.log`
  and match the SHA; a non-git `/opt/Gym` either exits 1 or silently runs the baked
  version.
- **`pre_cmd` didn't take** — `ModuleNotFoundError` in the client log for anything the
  `pre_cmd` installs.
- **No rollouts flowing** — `wc -l $RD/artifacts/evaluator_rollouts.jsonl`.
- **Run hangs in post-eval** — orphaned Ray/gym processes holding stdout; that's what
  the setsid + process-group reap in the task `command:` prevents.
- **Empty reasoning / collapsed scores on a reasoning model** — thinking mode off, or
  the reasoning trace leaking into the graded answer. Confirm
  `chat_template_kwargs.enable_thinking: true` (right toggle key for the family), the
  policy's `--reasoning-parser`, and `process_reasoning_traces: true` in the adapter.
- **Preempted vs timed out** — long gym runs exceed a 4h walltime. `TIMEOUT`
  auto-resumes from the response cache (`resume_from_cache=true`); `CANCELLED by
  <uid>` (preemption) does not, and its chained job exits in ~20s
  (`…finished with 'CANCELLED…' state. EXIT!`), which is expected, not a bug. Resume
  by hand: `cd <run>/nemo_gym.0 && sbatch run.sub`. Progress is cumulative — check the
  rollout count before assuming loss.
