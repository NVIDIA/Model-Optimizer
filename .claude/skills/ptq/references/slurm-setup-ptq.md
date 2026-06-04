# SLURM Setup for PTQ

PTQ-specific SLURM details. For generic SLURM patterns (account discovery, job template,
monitoring), see `skills/common/slurm-setup.md`.

---

## 1. Container

Get the recommended image version from `examples/llm_ptq/README.md`, then look for an existing `.sqsh` file:

```bash
ls *.sqsh ../*.sqsh ~/containers/*.sqsh 2>/dev/null
```

**If a `.sqsh` exists**, use it directly with `--container-image=<path>`. Skip import.

**If no `.sqsh` exists**, import with enroot (caches for subsequent smoke tests and reruns):

```bash
export ENROOT_CACHE_PATH=/path/to/writable/enroot-cache
export ENROOT_DATA_PATH=/path/to/writable/enroot-data
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
enroot import --output /path/to/container.sqsh docker://nvcr.io#nvidia/tensorrt-llm/release:<version>
```

If enroot import fails (e.g., permission errors on lustre), use pyxis inline pull as fallback — pass the NGC URI directly to `--container-image="nvcr.io/nvidia/tensorrt-llm/release:<version>"`. Note this re-pulls on every job.

### Container dependency pitfalls

**New models may need newer transformers** than what's in the container:

```bash
pip install -U transformers
```

For unlisted models that need unreleased transformers (e.g., from git), see `references/unsupported-models.md` Step A.

**Prefer `PYTHONPATH`** to use the synced ModelOpt source instead of installing inside the container — this avoids risking dependency conflicts (e.g., `pip install -U nvidia-modelopt[hf]` can upgrade PyTorch and break other packages):

```bash
export PYTHONPATH=/path/to/Model-Optimizer:$PYTHONPATH
```

If `PYTHONPATH` doesn't work due to missing compiled extensions, fall back to `pip install -e ".[hf]" --no-build-isolation` (run from the Model-Optimizer repo root).

**Watch for pip dependency conflicts** — NGC containers set `PIP_CONSTRAINT` to pin versions, causing `ResolutionImpossible` errors. Unset it first so pip can resolve freely:

```bash
unset PIP_CONSTRAINT
pip install -U transformers   # now upgrades and resolves with new deps included
```

If that still conflicts, fall back to `--no-deps` (skips new deps — may need to add missing ones manually):

```bash
pip install -U transformers --no-deps
```

---

## 2. GPU Sizing

Estimate GPU count from model size and available GPU memory. `hf_ptq.py` uses `device_map="auto"` so it fills GPUs automatically — request only as many as needed.

For multi-node PTQ (200B+ params), use `hf_ptq.py --use_fsdp2` launched via `torchrun`:

```bash
torchrun \
    --nnodes=$NUM_NODES \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/llm_ptq/hf_ptq.py \
        --pyt_ckpt_path <model> \
        --qformat <format> \
        --export_path <output> \
        --use_fsdp2
```

Add `--cpu_offload` when the per-rank decoder shard approaches GPU capacity (200B+ at low rank count). Layer detection is automatic; no YAML config needed.

Use the multi-node template from `skills/common/slurm-setup.md` section 4 as the job script wrapper.

---

## 3. Smoke Test

Before the full calibration run, submit a smoke test with `--calib_size 4` and `--time=00:30:00`.
This catches script errors cheaply before using GPU quota on a real run.

See `skills/common/slurm-setup.md` section 2 for the smoke test partition pattern.

Only submit the full calibration job after the smoke test exits cleanly.
