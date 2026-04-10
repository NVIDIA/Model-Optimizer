# SLURM Setup for PTQ

PTQ-specific SLURM details. For generic SLURM patterns (account discovery, job template,
monitoring), see `skills/common/slurm-setup.md`.

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

## 2. GPU Sizing

Estimate GPU count from model size and available GPU memory. `hf_ptq.py` uses `device_map="auto"` so it fills GPUs automatically — request only as many as needed.

For multi-node PTQ (200B+ params), use `examples/llm_ptq/multinode_ptq.py` with FSDP2 and accelerate:

```bash
accelerate launch \
    --config_file examples/llm_ptq/fsdp2.yaml \
    --num_machines $NUM_NODES \
    --num_processes $((NUM_NODES * GPUS_PER_NODE)) \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    examples/llm_ptq/multinode_ptq.py \
        --pyt_ckpt_path <model> \
        --qformat <format> \
        --export_path <output>
```

The `num_machines`, `num_processes`, `main_process_ip`, and `machine_rank` are overridden on the command line — no need to edit `fsdp2.yaml`. Only update `fsdp_transformer_layer_cls_to_wrap` in the YAML if the model uses a non-default decoder layer class.

Use the multi-node template from `skills/common/slurm-setup.md` section 4 as the job script wrapper.

---

## 3. Smoke Test

Before the full calibration run, submit a smoke test with `--calib_size 4` and `--time=00:30:00`.
This catches script errors cheaply before using GPU quota on a real run.

See `skills/common/slurm-setup.md` section 2 for the smoke test partition pattern.

Only submit the full calibration job after the smoke test exits cleanly.

---

## 4. PTQ-Specific Notes

- **Gated datasets**: Some calibration datasets (e.g., `nvidia/Nemotron-Post-Training-Dataset-v2`) require HF authentication. Set `HF_TOKEN` in the job environment, or use `--dataset cnn_dailymail` to use a non-gated alternative.
- **NFS permissions**: Docker + NFS root_squash causes `PermissionError` on output/cache dirs. See `skills/common/slurm-setup.md` section 5 for fixes.
