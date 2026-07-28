# SLURM gotchas (invisible to `--dry-run`; surface at canary)

Use this reference when filling `execution` for a SLURM run (Step 4). These
deploy/eval failures pass `--dry-run` clean and only show up once a real job is
scheduled (canary), so resolve them up front.

## `mount_home: false`

`mount_home: true` mounts the host `~/.cache` into the container. A shared-fs
symlink there (common for `~/.cache/huggingface`) dangles in-container → deploy
dies with `FileNotFoundError /root/.cache/huggingface`. Mount the real cache to
`/hf-cache` and point `HF_HOME` at it instead (see the snippet below).

## `cpu_partition: <cpu-partition>`

**Required for MLflow `auto_export` to work** on clusters with separate GPU/CPU
partitions. If not specified, NEL runs the export (a CPU-only job) on the GPU
partition — it does not auto-route — which gets rejected
(`Cannot find GPU specification`) and fails the task despite `EVAL_EXIT_CODE=0`.
Set it to the CPU partition (e.g. `cpu`).

## Shared vs stage-specific env vars

Shared env vars can go top-level `env_vars:` (merges into both the deployment and
evaluation stages) or per-stage as the example shows; `execution.env_vars`
hard-errors. Stage-specific vars stay under `deployment.env_vars` /
`evaluation.env_vars`.

## Snippet

```yaml
execution:
  cpu_partition: <cpu-partition>
  mounts:
    mount_home: false
    deployment: { <realpath ~/.cache/huggingface>: /hf-cache }   # ssh <host> realpath ~/.cache/huggingface
    evaluation: { <realpath ~/.cache/huggingface>: /hf-cache }
env_vars: { HF_TOKEN: host:HF_TOKEN, HF_HOME: lit:/hf-cache }   # both stages
```
