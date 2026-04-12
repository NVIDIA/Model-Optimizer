# NEL CI Evaluation Guide

NEL CI is the recommended entry point for running evaluations on NVIDIA JET infrastructure. This guide covers patterns for evaluating quantized checkpoints using both the NEL SLURM executor (direct) and the NEL CI GitLab pipeline.

Reference repo: `gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nemo-evaluator-launcher-ci`

---

## 1. Two Execution Paths

| Path | When to use | How it works |
|------|-------------|--------------|
| **NEL SLURM executor** | You have SSH access to the cluster, checkpoint is on cluster storage | `nel run --config config.yaml` from your workstation; NEL SSHes to cluster and submits sbatch jobs |
| **NEL CI GitLab pipeline** | You want managed infrastructure, MLflow export, reproducible configs | Trigger via GitLab API or UI; JET orchestrates everything |

### NEL SLURM executor

Best for iterative development and debugging. Run from any machine with SSH access to the cluster:

```bash
export DUMMY_API_KEY=dummy
export HF_TOKEN=<your_token>

nel run --config eval_config.yaml \
    -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10  # test first
```

### NEL CI trigger

Best for production evaluations with MLflow tracking. See the trigger script pattern in section 4.

---

## 2. Cluster Reference

| Cluster | GPUs/Node | Architecture | Max Walltime | Storage | Notes |
|---------|-----------|-------------|--------------|---------|-------|
| oci-hsg | 4 | GB200 | 4 hours | `/lustre/` | Set `tensor_parallel_size=4` |
| cw | 8 | H100 | — | `/lustre/` | — |
| oci-nrt | 8 | H100 | — | `/lustre/` | Numerics configs |
| dlcluster | 4 (B100 partition) | B100 | 8 hours | `/home/omniml_data_*` | No `/lustre/`; use local NFS paths |

**Important**: `deployment.tensor_parallel_size` determines how many GPUs are requested. If this exceeds the cluster's GPUs per node, the job fails with a memory allocation error.

---

## 3. Checkpoint Availability

The checkpoint must be on a filesystem accessible from the cluster's **compute nodes** (not just login nodes).

| Cluster type | Accessible storage | NOT accessible |
|-------------|-------------------|----------------|
| JET clusters (oci-hsg, cw, oci-nrt) | `/lustre/fsw/...` | Workstation paths (`/home/scratch.*`), NFS mounts from other clusters |
| dlcluster | `/home/omniml_data_*`, `/home/scratch.*` | `/lustre/` (not available) |

If the checkpoint is on a workstation, **copy it to cluster storage first**:

```bash
rsync -av /path/to/local/checkpoint \
    <cluster-login>:/lustre/fsw/portfolios/coreai/users/$USER/checkpoints/
```

For dlcluster, the checkpoint paths are directly accessible since the NFS mounts are shared between login and compute nodes.

---

## 4. NEL CI Trigger Pattern

For JET clusters, trigger evaluations via the GitLab API. Use `NEL_DEPLOYMENT_COMMAND` (not `NEL_OTHER_OVERRIDES` with `deployment.extra_args`) because `NEL_OTHER_OVERRIDES` splits values on spaces, breaking multi-flag commands.

```bash
export GITLAB_TOKEN=<your_gitlab_token>

curl -k --request POST \
  --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "ref": "main",
    "variables": [
      {"key": "NEL_CONFIG_PATH", "value": "configs/AA/minimax_m2_5_lbd_lax.yaml"},
      {"key": "NEL_ACCOUNT", "value": "coreai_dlalgo_modelopt"},
      {"key": "NEL_CLUSTER", "value": "oci-hsg"},
      {"key": "NEL_CHECKPOINT_OR_ARTIFACT", "value": "/lustre/.../checkpoint"},
      {"key": "NEL_DEPLOYMENT_IMAGE", "value": "vllm/vllm-openai:v0.19.0"},
      {"key": "NEL_TASKS", "value": "simple_evals.gpqa_diamond_aa_v3"},
      {"key": "NEL_DEPLOYMENT_COMMAND", "value": "vllm serve /checkpoint --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --quantization modelopt_fp4 --trust-remote-code --served-model-name my-model"},
      {"key": "NEL_OTHER_OVERRIDES", "value": "deployment.tensor_parallel_size=4 execution.walltime=04:00:00"},
      {"key": "NEL_HF_HOME", "value": "/lustre/.../cache/huggingface"},
      {"key": "NEL_VLLM_CACHE", "value": "/lustre/.../cache/vllm"},
      {"key": "NEL_CLUSTER_OUTPUT_DIR", "value": "/lustre/.../nv-eval-rundirs"}
    ]
  }' \
  "https://gitlab-master.nvidia.com/api/v4/projects/221804/pipeline"
```

---

## 5. Environment Variables

### SLURM executor format

Env vars in NEL SLURM configs require explicit prefixes:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `host:VAR_NAME` | Read from the host environment where `nel run` is executed | `host:HF_TOKEN` |
| `lit:value` | Literal string value | `lit:dummy` |

```yaml
evaluation:
  env_vars:
    DUMMY_API_KEY: host:DUMMY_API_KEY
    HF_TOKEN: host:HF_TOKEN
```

### JET executor format

JET configs reference JET secrets with `$SECRET_NAME`:

```yaml
execution:
  env_vars:
    evaluation:
      HF_TOKEN: $COMPEVAL_HF_TOKEN
```

### Gated datasets

Tasks that download gated HuggingFace datasets (e.g., GPQA, HLE) need `HF_TOKEN` passed to the evaluation container.

**NEL CI (JET)**: Handled automatically — the `COMPEVAL_HF_TOKEN` JET secret is pre-configured by the eval platform team. No user action needed; you don't even need personal access to the gated dataset.

**NEL SLURM executor**: You must provide your own HF token, AND your HuggingFace account must have been granted access to the gated dataset (e.g., request access at https://huggingface.co/datasets/Idavidrein/gpqa for GPQA).

```yaml
evaluation:
  env_vars:
    HF_TOKEN: host:HF_TOKEN  # SLURM executor — reads from your local env
  tasks:
    - name: simple_evals.gpqa_diamond
      env_vars:
        HF_TOKEN: host:HF_TOKEN
```

---

## 6. Serving Framework Notes

### vLLM

- Binds to `0.0.0.0` by default — health checks work out of the box
- For NVFP4: `--quantization modelopt_fp4`
- For unsupported models (e.g., ministral3): may need custom `deployment.command` to patch the framework before serving (see `deployment/references/unsupported-models.md`)

### SGLang

- **Must include `--host 0.0.0.0`** — SGLang defaults to `127.0.0.1` which blocks health checks from the eval client
- Must include `--port 8000` to match NEL's expected port
- For NVFP4: `--quantization modelopt_fp4`

---

## 7. Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` pulling eval container | NGC credentials not set on cluster | Set up `~/.config/enroot/.credentials` with NGC API key |
| `PermissionError: /hf-cache/...` | HF cache dir not writable by svc-jet | Set `NEL_HF_HOME` to your own `chmod 777` directory |
| Health check stuck at `000` | Server binding to localhost | Add `--host 0.0.0.0` to deployment command (SGLang) |
| `Memory required by task is not available` | TP size exceeds GPUs/node | Set `tensor_parallel_size` to match cluster (4 for oci-hsg, dlcluster B100) |
| TIMEOUT after eval completes | Walltime too short for eval + MLflow export | Set `execution.walltime=04:00:00` |
| Gated dataset auth failure | `HF_TOKEN` not passed to eval container | Add `env_vars.HF_TOKEN` at evaluation or task level |
| `NEL_OTHER_OVERRIDES` splits `extra_args` | Space-separated parsing breaks multi-flag values | Use `NEL_DEPLOYMENT_COMMAND` instead |
| Checkpoint not found in container | Path not on cluster compute-node filesystem | Copy checkpoint to `/lustre/` (or cluster-accessible path) first |
| `trusted_eval` type mismatch in MLflow export | NEL writes boolean `true` instead of string `"true"` | Fix with `sed -i "s/trusted_eval: true/trusted_eval: 'true'/"` in export config |

---

## 8. Directory Setup for JET Clusters

Before running evaluations on a JET cluster, create writable directories:

```bash
ssh <cluster-login>
mkdir -p /lustre/fsw/portfolios/coreai/users/$USER/cache/huggingface
mkdir -p /lustre/fsw/portfolios/coreai/users/$USER/cache/vllm
mkdir -p /lustre/fsw/portfolios/coreai/users/$USER/nv-eval-rundirs
chmod 777 /lustre/fsw/portfolios/coreai/users/$USER/cache/huggingface
chmod 777 /lustre/fsw/portfolios/coreai/users/$USER/cache/vllm
chmod 777 /lustre/fsw/portfolios/coreai/users/$USER/nv-eval-rundirs
```

`chmod 777` is required because `svc-jet` (JET service account) runs containers and needs write access.
