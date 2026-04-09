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
