# SLURM Environment Setup for ModelOpt PTQ

Read this file when running on a SLURM cluster. It covers container setup, job submission, smoke-test strategy, and monitoring.

---

## 1. Container

Get the recommended image version from `examples/llm_ptq/README.md`, then look for a `.sqsh` file in the workspace and common sibling directories:

```bash
ls *.sqsh ../*.sqsh ~/containers/*.sqsh 2>/dev/null
```

If you find a `.sqsh` but aren't sure of its version, check it:

```bash
srun --container-image=<path/to/container.sqsh> --ntasks=1 bash -c \
    "pip show tensorrt-llm 2>/dev/null | grep Version || cat /VERSION 2>/dev/null || echo unknown"
```

If no `.sqsh` exists, import it with enroot. Set writable cache paths first — the default `/raid/containers` is often not writable:

```bash
export ENROOT_CACHE_PATH=/path/to/writable/enroot-cache
export ENROOT_DATA_PATH=/path/to/writable/enroot-data
export TMPDIR=/path/to/writable/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$TMPDIR"

enroot import --output /path/to/container.sqsh \
    docker://nvcr.io#nvidia/tensorrt-llm/release:<version>
```

---

## 2. Account and Partition

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

## 3. Job Script Template

**Critical**: container flags (`--container-image`, `--container-mounts`) MUST be on the `srun` line — they do NOT work as `#SBATCH` directives.

**GPU count**: estimate based on model size. Rough guide: 1 GPU per ~20B params in BF16 (e.g., 0.6B → 1 GPU, 70B → 4 GPUs, 405B → 8 GPUs). `hf_ptq.py` uses `device_map="auto"` so it fills GPUs automatically — request only as many as needed.

```bash
#!/bin/bash
#SBATCH --job-name=ptq
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=<N>   # 1 for small models (<20B), 4-8 for large models
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<log_dir>/ptq_%j.log

srun \
    --container-image="<path/to/container.sqsh>" \
    --container-mounts="<data_root>:<data_root>" \
    --container-workdir="<workdir>" \
    --no-container-mount-home \
    bash -c "pip install -e <modelopt_path>[hf] --quiet && python <ptq_script.py> ..."
```

Submit and capture the job ID:

```bash
mkdir -p <log_dir>
JOBID=$(sbatch <script>.sh | awk '{print $4}')
echo "Submitted job $JOBID"
```

---

## 4. Smoke Test First (Always)

Before the full calibration run, submit a smoke test with `--calib_size 4` and `--time=00:30:00`. This catches script errors cheaply before using GPU quota on a real run.

Use a comma-separated partition list — SLURM picks whichever allocates first. Shorter/interactive partitions queue faster:

```bash
#SBATCH --partition=interactive,batch_short,batch_block1
#SBATCH --time=00:30:00
```

Note: interactive/short partitions may cap node count. If the smoke test needs multiple nodes, include a multi-node-capable partition as the last fallback.

Only submit the full calibration job after the smoke test exits cleanly.

---

## 5. Monitor Until Completion

After submitting the final job, do not stop — the goal is a finished checkpoint, not a submitted job. Poll with sleep until done:

```bash
while squeue -j $JOBID -h 2>/dev/null | grep -q .; do
    echo "$(date): job $JOBID still running..."; sleep 60
done
echo "Job $JOBID finished"
sacct -j $JOBID --format=JobID,State,ExitCode,Elapsed
```

**IMPORTANT**: Always use `sleep`-based polling (as above) rather than `CronCreate` or background tasks. This keeps output in the current session so the user can see progress. The sleep loop will wait as long as needed — even hours — until the job completes or fails.

Once the job ends, tail the last 50 lines of the log and verify the export directory before reporting success.

---

## 6. Multi-node PTQ (FSDP2)

For models too large for a single node (rough guide: 200B+ params), use `examples/llm_ptq/multinode_ptq.py` with FSDP2.

Edit `examples/llm_ptq/fsdp2.yaml`:

- `num_machines` and `num_processes` → match SLURM allocation
- `fsdp_transformer_layer_cls_to_wrap` → model's decoder layer class name

```bash
accelerate launch --config_file fsdp2.yaml multinode_ptq.py \
    --pyt_ckpt_path <model> --qformat <format> --export_path <output>
```

Set `--nodes=N` and `--ntasks-per-node=8` in the SLURM script accordingly.
