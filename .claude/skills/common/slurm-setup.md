# SLURM Setup (Common)

Generic SLURM account discovery, job submission, and monitoring patterns.
Skill-specific references (container images, script args) are in each skill's own `references/` directory.

---

## 1. Account and Partition Discovery

```bash
# Accounts available to you
sacctmgr show associations user=$USER format=account%30,cluster%20 -n 2>/dev/null

# GPU partitions and their time/node limits (exclude CPU-only)
sinfo -o "%P %a %l %D %G" 2>/dev/null | grep -v "null\|CPU\|cpu"
```

- One account → use it automatically
- Multiple accounts → show them to the user and ask which to use
- Partition → use the default (marked `*`); report the choice

---

## 2. Job Script Template

**Critical**: container flags (`--container-image`, `--container-mounts`) MUST be on the `srun` line — they do NOT work as `#SBATCH` directives.

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=<N>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<log_dir>/<name>_%j.log

srun \
    --container-image="<path/to/container.sqsh>" \
    --container-mounts="<data_root>:<data_root>" \
    --container-workdir="<workdir>" \
    --no-container-mount-home \
    bash -c "
        # Unset SLURM distributed env vars for single-process scripts (e.g., hf_ptq.py).
        # srun sets WORLD_SIZE/LOCAL_RANK/etc. which cause PyTorch to init a process group
        # and wrap tensors as DTensors, breaking NVFP4 export. Only needed for scripts that
        # use device_map='auto' (not FSDP2/multinode which handle DTensors properly).
        unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS WORLD_SIZE LOCAL_RANK RANK
        <command>
    "
```

### Container registry credentials (pyxis)

If `srun --container-image` uses an image from a private registry (e.g., `nvcr.io/nvidia/...`), pyxis/enroot needs registry credentials on the cluster in `~/.config/enroot/.credentials`. See `skills/common/credentials.md` for the NGC / Docker / HF token setup. Without this, `srun` fails with `401 Unauthorized` when the compute node pulls.

Submit and capture the job ID:

```bash
mkdir -p <log_dir>
JOBID=$(sbatch <script>.sh | awk '{print $4}')
echo "Submitted job $JOBID"
```

### Smoke test pattern

Before a long run, submit a quick smoke test with a short time limit.
Use a comma-separated partition list — SLURM picks whichever allocates first:

```bash
#SBATCH --partition=interactive,batch_short,batch_block1
#SBATCH --time=00:30:00
```

Note: interactive/short partitions may cap node count. If the smoke test needs multiple nodes,
include a multi-node-capable partition as the last fallback.

Only submit the full job after the smoke test exits cleanly.

### Docker (non-pyxis) variant

Some clusters don't have pyxis/enroot installed and instead use plain `docker run` on compute nodes. In this case, replace the `srun --container-image` pattern with `docker run` inside the job script:

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=<N>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<log_dir>/<name>_%j.log

docker run --rm \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --network host \
    -v <data_root>:<data_root> \
    -e CALIB_SIZE="${CALIB_SIZE:-512}" \
    <container_image> \
    bash <path/to/run_script.sh>
```

**Key differences from pyxis**:

- No `srun` wrapper needed — SLURM just allocates the node, Docker runs the container
- Mount paths with `-v` instead of `--container-mounts`
- Pass env vars with `-e` instead of relying on SLURM env propagation
- Use the two-script pattern: SLURM wrapper (sbatch directives + `docker run`) and inner runner (the actual work). The inner runner should unset SLURM env vars and set `HF_HOME`/`HF_DATASETS_OFFLINE` as needed
- **NFS root_squash**: see section 5

**How to detect which pattern to use**: Ask the user how they normally run containers, or check:

```bash
which enroot 2>/dev/null && echo "pyxis/enroot available"
which docker 2>/dev/null && echo "docker available"
```

---

## 3. Monitor Until Completion

After submitting, poll with sleep until done:

```bash
while squeue -j $JOBID -h 2>/dev/null | grep -q .; do
    echo "$(date): job $JOBID still running..."; sleep 60
done
echo "Job $JOBID finished"
sacct -j $JOBID --format=JobID,State,ExitCode,Elapsed
```

**IMPORTANT**: Always use `sleep`-based polling (as above) rather than background tasks or cron.
This keeps output in the current session so the user can see progress.
The sleep loop will wait as long as needed — even hours — until the job completes or fails.

Once the job ends, tail the last 50 lines of the log and verify the output before reporting success.

---

## 4. Multi-node Template

For multi-node jobs, SLURM provides distributed info automatically via env vars:

```bash
#!/bin/bash
#SBATCH --job-name=<name>-multinode
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=<N>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<log_dir>/<name>_%j.log
#SBATCH --exclusive

# SLURM provides: SLURM_NODELIST, SLURM_NNODES, SLURM_PROCID, SLURM_LOCALID
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500
NUM_NODES=$SLURM_NNODES

srun \
    --container-image="<path/to/container.sqsh>" \
    --container-mounts="<data_root>:<data_root>" \
    --container-workdir="<workdir>" \
    --no-container-mount-home \
    bash -c "<distributed_command>"
```

Adjust `--nodes`, `--gpus-per-node`, and the distributed launch command per your workload.

---

## 5. NFS root_squash and Docker Permissions

Docker containers typically run as root, but NFS filesystems with `root_squash` (the default) map root to `nobody`, blocking writes to directories owned by the user. This causes `PermissionError` when creating cache lock files, writing output, or saving logs.

This affects both pyxis/enroot (`srun --container-image`) and plain `docker run` workflows.

**Preferred fix** — run Docker with the host user's UID/GID to match NFS ownership:

```bash
docker run --user $(id -u):$(id -g) ...
```

> Note: `--user` may cause issues if the container expects root for package installation. In that case, fall back to the chmod approach below.

**Fallback fix** — open permissions before submitting the job:

```bash
chmod -R g+rwX /path/to/workspace/
chmod -R g+rwX /path/to/.hf_cache/
```

Scope `chmod` to only the directories the job needs — avoid world-writable paths on shared clusters.

---

## 6. Container Registry Authentication

**Before submitting any SLURM job that pulls a container image**, check that the cluster has credentials for the image's registry. Missing auth causes jobs to fail after waiting in the queue — a costly mistake.

### Step 1: Detect the container runtime

Different clusters use different container runtimes. Detect which is available:

```bash
# On the cluster (or via ssh):
which enroot 2>/dev/null && echo "RUNTIME=enroot"
which docker 2>/dev/null && echo "RUNTIME=docker"
```

| Runtime | Typical clusters | SLURM integration |
| --- | --- | --- |
| **enroot/pyxis** | NVIDIA internal (DGX Cloud, EOS, Selene, GCP-NRT) | `srun --container-image` |
| **Docker** | Bare-metal / on-prem with GPU | `docker run` inside job script |

### Step 2: Check credentials for the image's registry

Determine the registry from the image URI:

| Image pattern | Registry |
| --- | --- |
| `nvcr.io/nvidia/...` | NGC |
| `vllm/vllm-openai:...`, `lmsysorg/sglang:...`, or no registry prefix | DockerHub |
| `ghcr.io/...` | GitHub Container Registry |
| `docker.io/...` | DockerHub (explicit) |

Then check credentials based on the runtime:

#### enroot/pyxis

```bash
grep -E '^\s*machine\s+' ~/.config/enroot/.credentials 2>/dev/null
```

Look for `machine <registry>` lines:
- NGC → `machine nvcr.io`
- DockerHub → `machine auth.docker.io`
- GHCR → `machine ghcr.io`

#### Docker

```bash
cat ~/.docker/config.json 2>/dev/null | python3 -c "import json,sys; print('\n'.join(json.load(sys.stdin).get('auths', {}).keys()))"
```

Look for registry keys (`https://index.docker.io/v1/`, `nvcr.io`, `ghcr.io`).

### Step 3: If credentials are missing

**Do not submit the job.** Instead:

1. Tell the user which registry and runtime need authentication
2. Show the fix for their runtime:

**enroot/pyxis:**

```bash
mkdir -p ~/.config/enroot

# DockerHub (get token from https://hub.docker.com/settings/security)
cat >> ~/.config/enroot/.credentials << 'EOF'
machine auth.docker.io
  login <dockerhub_username>
  password <access_token>
EOF

# NGC (get API key from https://org.ngc.nvidia.com/setup/api-keys)
cat >> ~/.config/enroot/.credentials << 'EOF'
machine nvcr.io
  login $oauthtoken
  password <ngc_api_key>
EOF
```

**Docker:**

```bash
# DockerHub (interactive prompt)
docker login

# NGC (use --password-stdin to avoid exposing secrets in process list)
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

3. **Suggest an alternative image** on an authenticated registry. NVIDIA clusters typically have NGC auth pre-configured, so prefer NGC-hosted images:

| DockerHub image | NGC alternative |
| --- | --- |
| `vllm/vllm-openai:latest` | `nvcr.io/nvidia/vllm:<YY.MM>-py3` (check [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) for latest tag) |
| `nvcr.io/nvidia/tensorrt-llm/release:<tag>` | Already NGC |

> **Note:** NGC image tags follow `YY.MM-py3` format (e.g., `26.03-py3`). Not all DockerHub images have NGC equivalents. If no NGC alternative exists and DockerHub auth is missing, the user must add DockerHub credentials or pre-cache the image as a `.sqsh` file.

4. After the user fixes auth or switches images, verify the image is **actually pullable** before submitting (credentials alone don't guarantee the image exists):

```bash
# enroot — test pull (aborts after manifest fetch)
enroot import --output /dev/null docker://<registry>#<image> 2>&1 | head -10
# Success: shows "Fetching image manifest" + layer info
# Failure: shows "401 Unauthorized" or "404 Not Found"

# docker
docker manifest inspect <image> 2>&1 | head -5

# singularity
singularity pull --dry-run docker://<image> 2>&1 | head -5
```

> **Important**: Credentials existing for a registry does NOT mean a specific image is accessible. The image may not exist, or the credentials may lack permissions for that repository. Always verify the specific image before submitting.

### Common failure modes

| Symptom | Runtime | Cause | Fix |
| --- | --- | --- | --- |
| `curl: (22) ... error: 401` | enroot | No credentials for registry | Add to `~/.config/enroot/.credentials` |
| `pyxis: failed to import docker image` | enroot | Auth failed or rate limit | Check credentials; DockerHub free: 100 pulls/6h per IP |
| `unauthorized: authentication required` | docker | No `docker login` | Run `docker login [registry]` |
| Image pulls on some nodes but not others | any | Cached on one node only | Pre-cache image or ensure auth on all nodes |
