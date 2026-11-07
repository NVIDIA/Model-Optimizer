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

Catalog inputs use the schema-v2 resolver protocol described in
[configuration](configuration.md#mount-keyed-inputs). The default resolver supports
Hugging Face model and dataset snapshots. A profile can replace it with an
`asset_resolver` command. Input declarations contain a `ref`, optional `env` and
`role`, and `kind: model` (default) or `kind: dataset`; no provider field is needed.
An unversioned HF reference resolves to an exact commit before download. The final
manifest records that commit. For remote Slurm, input preparation uses NeMo's SSH
transport and requires `huggingface_hub` in the cache host's `python3` environment.
HF credentials travel over SSH stdin. No GPU job is submitted until inputs are ready.

## Artifact publication and model lineage

Publication is explicit on a task output:

```yaml
outputs:
  checkpoint:
    path: model
    kind: checkpoint
    publish:
      ref: experiments/quantized-model
      version: trial-001
      model_name: quantized-model
      metadata:
        description: Quantized checkpoint
```

The profile chooses `artifact_store: /shared/artifacts` for verified filesystem
copies, or `artifact_publisher: [my-publisher]` for another storage service. A
publisher reads a schema-version-1 JSON request on stdin and returns JSON with
`status: ready`, `verified: true`, the exact `ref` and `version`, and a durable
`uri`. The request includes `source_path`, `task`, `execution_id`, optional
`run_id` and user metadata. Credentials stay in the environment. A callback must
resume its saved attempt after uncertainty and reject conflicting versions.

Publication runs after all payload ranks succeed and before the producer reports
success. Consumers therefore receive the completed artifact identity. Publication
uses the producer's allocation; include transfer time in its wall-time budget.
Filesystem publication streams hashes, verifies copied bytes, and refuses version
conflicts or changed snapshots. It preserves the source layout.

Add `tracking_uri` to the profile to create one MLflow run per task. Task runtimes
need `mlflow-skinny>=3.11,<4`; install the launch host with
`pip install 'modelopt-launcher[artifacts,tracking]'`. Container images can provide
these dependencies or tasks can use `reqs`. `MLFLOW_RUN_ID` and
`MLFLOW_EXPERIMENT_ID` point to the current task's run, so ordinary MLflow logging
can resume it. Workloads remain responsible for logging their metrics.

`publish.model_name` registers a small artifact manifest and the producer run as
a model version. The manifest and `modelopt.artifact_uri` version tag identify the
exact published checkpoint; the checkpoint stays in its artifact store. This
works with storage schemes the MLflow registry does not accept directly. Each consuming task receives `MODELOPT_INPUTS_MANIFEST`, a JSON list of
exact input references. If it consumes one registered model, `MODELOPT_MODEL_URI`
contains its `models:/name/version` reference. Native MLflow logged-model input
and output relationships connect the producing and consuming runs. Model-version
tags preserve the logged-model ID for stores that omit it on readback.

Later workflows can use `ref: models:/name/version`. The built-in resolver handles
filesystem publications; other storage services use the profile's resolver.
Remote registry resolution needs the launcher and its `tracking` extra installed
in the cache host's `python3` environment, with access to the tracking server.
Registry entries describe raw checkpoints; use the checkpoint's normal loader.
Workflow manifests live at `<workspace>/.artifacts/<task>.json`. They contain
input identities, output URIs, registered model versions and run IDs for tools
and agents. Keep parameters and literal metadata free of credentials. An uncertain
publication or registration requires reconciliation of its saved receipt.

`examples/smoke/artifact_workflow.yaml` exercises publication, registration and two
consumers with the same public workflow spec. Supply a profile containing
`artifact_store` or `artifact_publisher`, plus `tracking_uri`. Choose a fresh
`--param output_version=trial-002` for a new publication. Its checkpoint and metrics
are synthetic.

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
