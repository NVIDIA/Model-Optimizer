# Workflow API v2

Launcher 0.2 adds `api_version: modelopt/v2` through the existing
`modelopt-launcher --yaml` command. Existing unversioned YAMLs continue through
the original parser and preserve their sequential numbered-task behavior.

V2 uses a named task graph, arbitrary parameters and declared task outputs.
NeMo-Run continues to package and execute the jobs.

```yaml
api_version: modelopt/v2
name: quantize_and_evaluate
parameters:
  model: Org/Model
inputs:
  /inputs/base_model:
    ref: ${parameters.model}
    env: BASE_MODEL
    role: base_model
tasks:
  quantize:
    script: common/quantize.sh
    outputs:
      checkpoint:
        path: model
        kind: checkpoint
        env: OUTPUT_MODEL
  evaluate:
    script: common/evaluate.sh
    inputs:
      /inputs/model:
        from: quantize.checkpoint
        env: HF_MODEL_CKPT
    outputs:
      metrics:
        path: metrics.json
        kind: metrics
        env: METRICS_FILE
outputs:
  metrics:
    from: evaluate.metrics
```

This illustrates the contract; supply real scripts for the two script paths.
The executable synthetic example is `examples/smoke/workflow_v2.yaml`.

## Tasks and dependencies

Task names are stable identifiers using letters, digits, underscores and hyphens,
starting with a letter. There is no fixed number of task slots. `depends_on`
lists predecessor names; an input `from: task.output` also adds that dependency.
Declaration order does not impose execution order. Independent tasks can run in
parallel, and a task can wait for multiple predecessors. Duplicate YAML keys,
unknown names, cycles and invalid I/O bindings are rejected before execution.

Tasks accept `script` plus an argument list, or `inline` (including a YAML block
scalar). V2 argument-list items are individual arguments; shell-word splitting
inside an item is a legacy behavior. Inline commands run with Bash `-e`.
`environment`, `reqs` and `reqs_file` remain available. Optional `resources`
contains `nodes`, `ntasks_per_node`, `gpus_per_node`, `time` and `mem`; `container`
can override the profile image. Raw local CPU execution rejects resource and
container requests because it does not enforce them.

## Parameters

Declare defaults in `parameters` and use `${parameters.name}` in values.
An expression occupying a whole value preserves its type; interpolation inside
a string inserts its string representation. Shell variables such as `$RESULT`
and `${SLURM_JOB_ID}` remain shell variables. Overrides use
`--param name=value`, with values parsed as YAML. Unknown parameter names fail.

```bash
modelopt-launcher --yaml workflow.yaml --profile my-profile.yaml \
  --param model=Org/OtherModel --dryrun
```

`--to-yaml resolved.yaml` exports a portable workflow with resolved parameters
and dependencies; execution settings stay in the separately supplied profile.

## Profiles

A profile supplies `executor: slurm`, `docker` or `local`. Slurm and Docker use
the launcher's existing executor builders. Kubernetes is rejected until its
adapter is implemented and verified. A Slurm profile can contain:

```yaml
executor: slurm
user: alice
identity: /home/alice/.ssh/cluster_key
job_dir: /shared/alice/experiments
slurm:
  host: login.example.com
  account: my-account
  container: my-registry/modelopt:version
  container_mounts:
    - /shared/model-cache:/hf-local
asset_resolver: [my-artifact-provider, --profile, my-team]
```

`slurm` accepts the existing SlurmConfig fields; task `resources` override the
profile defaults. A Docker profile needs `hf_local`. A `local` profile runs raw
CPU processes and is intended for development; it uses environment bindings for
task inputs and does not mount catalog inputs. Credentials belong in the runtime
environment, and profiles contain configuration or credential file paths.

## Output materialization

Each task output declares a normalized relative `path`, optional `kind` and
optional `env`. Container execution places outputs below
`/scratchspace/outputs/<task>/`; raw local execution uses the experiment workspace.
The launcher creates parent directories and binds the declared environment
variable to the output path. The task must create that output before succeeding.
`kind` records the intended artifact type; it does not validate the file format.

Consumers receive a read-only container mount at the declared input path; local
CPU consumers receive the producer's path through the required environment
binding. A failed producer or a missing declared output prevents consumers from
running their commands. A task input can override a workflow input at the same
mount path. Other overlapping mounts and conflicting environment bindings fail.

`outputs` at workflow scope exposes selected `task.output` references.
The NeMo experiment's `metadata.json` records the graph, parameters, output paths,
workspace mapping and terminal states. A manifest with `prepared` or `submitted`
status declares intended paths; it does not prove the outputs are ready. A
`submission_error` can mean partial submission and requires reconciliation.

Catalog inputs retain the schema-v2 resolver protocol described in
[configuration](configuration.md#mount-keyed-inputs). Output publication,
registered model creation and automatic MLflow metric association are separate
runtime integrations and are not implied by an output declaration.

## CPU acceptance fixture

```bash
modelopt-launcher --yaml tools/launcher/examples/smoke/workflow_v2.yaml \
  --profile tools/launcher/examples/smoke/local_profile.yaml --param value=11 --yes
```

Run from the repository root. The fixture produces a synthetic JSON checkpoint,
runs two consumers with a bounded concurrency barrier, and joins their outputs.
It exercises real NeMo local execution without GPUs or a cluster. Its values are
synthetic, not accuracy or performance measurements. Slurm/Docker execution of
v2 workflows still requires separate backend validation.
