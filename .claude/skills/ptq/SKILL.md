---
name: ptq
description: This skill should be used when the user asks to "quantize a model", "run PTQ", "post-training quantization", "NVFP4 quantization", "FP8 quantization", "INT8 quantization", "INT4 AWQ", "quantize LLM", "quantize MoE", "quantize VLM", or needs to produce a quantized HuggingFace or TensorRT-LLM checkpoint from a pretrained model using ModelOpt.
---

# ModelOpt Post-Training Quantization

Produce a quantized checkpoint from a pretrained model. **Read `examples/llm_ptq/README.md` first** — it has the support matrix, CLI flags, and accuracy guidance.

## Step 1 — Environment

Read `skills/common/environment-setup.md` and `skills/common/workspace-management.md`. After completing them you should know:

- ModelOpt source is available
- Local or remote (+ cluster config if remote)
- SLURM / Docker+GPU / bare GPU
- Launcher available?
- Which workspace to use

## Step 2 — Is the model supported?

Check `modelopt/torch/export/model_utils.py` for `MODEL_NAME_TO_TYPE`. If the model class matches → **supported**. Otherwise → **unsupported**.

## Step 3 — Choose quantization format

**First**, check for a pre-built recipe:

```bash
ls modelopt_recipes/models/ modelopt_recipes/general/ptq/ 2>/dev/null
```

If a recipe matches, use `--recipe <path>`.

**If no recipe**, recommend based on GPU (details in `examples/llm_ptq/README.md`):

- **Blackwell** (B100/B200/GB200): `nvfp4` variants
- **Hopper** (H100/H200) or older: `fp8` or `int4_awq`

All format definitions: `modelopt/torch/quantization/config.py`.

> NVFP4 can be calibrated on Hopper but requires Blackwell for inference.

## Step 4 — Run PTQ

**Goal: checkpoint on disk** (`.safetensors` + `config.json`).

**IMPORTANT**: Run a smoke test first (`--calib_size 4`). Wait for it to succeed. Then run full calibration (`--calib_size 512`).

### Which path?

```text
Supported? ──→ YES ──→ SLURM (local or remote)? ──→ LAUNCHER (4B)
                  │     Local Docker + GPU? ────────→ LAUNCHER (4B)
                  │     Remote Docker (no SLURM)? ──→ MANUAL via remote_run (4A)
                  │     Bare GPU (local or remote)? → MANUAL (4A)
                  │
                  └→ NO (unsupported) ──→ Any env ──→ CUSTOM SCRIPT (4C)
```

### 4A — Direct: supported model, manual execution

```bash
pip install --no-build-isolation "nvidia-modelopt[hf]"

python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path <model> \
    --qformat <format> \
    --calib_size 512 \
    --export_path <output>
```

Run `--help` for all options.

For remote: use `remote_run` from `remote_exec.sh` (see `skills/common/remote-execution.md`).

### 4B — Launcher: supported model on SLURM or local Docker

Write a YAML config using `common/hf_ptq/hf_ptq.sh`. See `references/launcher-guide.md` for the full template.

```bash
cd tools/launcher
# SLURM (remote or local):
SLURM_HOST=<host> SLURM_ACCOUNT=<acct> uv run launch.py --yaml <config.yaml> user=<ssh_user> identity=<ssh_key> --yes
# Local Docker:
uv run launch.py --yaml <config.yaml> hf_local=<hf_cache> --yes
```

The launcher blocks and tails logs until the job completes.

### 4C — Custom script: unsupported model

Follow `references/unsupported-models.md`. Core steps:

1. Load model (dequantize FP8 if needed)
2. Monkey-patch unsupported layers, register with `mtq.register()`
3. Create calibration dataloader
4. `mtq.quantize(model, config, forward_loop)`
5. `export_hf_checkpoint(model, export_dir)`

Run directly (local) or via `remote_run` (remote). For SLURM environments, use a job script (see `references/slurm-setup.md`).

### Monitoring

- **Launcher**: blocks and tails logs automatically
- **SLURM (manual)**: poll with `squeue -u $USER` + `sleep` (not cron or background tasks)
- **Local**: watch stdout

## Step 5 — Verify output

```bash
ls -lh <output_path>/
# Expect: config.json, tokenizer files, model-*.safetensors
```

Report the path and size to the user.

## Key API Rules

- `mtq.register()` classes **must** define `_setup()` and call it from `__init__`
- Call `mto.enable_huggingface_checkpointing()` **before** quantization
- Wildcard `*gate*` matches too broadly — use `*mlp.gate*` or `*router*`
- VLMs need `AutoModel`, not `AutoModelForCausalLM`
- FP8 loading: `FineGrainedFP8Config(dequantize=True)`, not a dict
- Custom quantizer names must end with `_input_quantizer` or `_weight_quantizer`
- Never modify files under `modelopt/` — custom code goes in your own script

## References

| Reference | When to read |
| --- | --- |
| `skills/common/environment-setup.md` | Step 1: detect env |
| `skills/common/workspace-management.md` | Step 1: organize work by model |
| `references/launcher-guide.md` | Step 4B: launcher YAML config |
| `references/unsupported-models.md` | Step 2/4C: unsupported model |
| `skills/common/remote-execution.md` | Remote execution via SSH |
| `references/slurm-setup.md` | Manual SLURM job scripts |
| `tools/launcher/CLAUDE.md` | Launcher full docs |
| `examples/llm_ptq/README.md` | Support matrix, CLI, accuracy |
| `modelopt/torch/quantization/config.py` | Format definitions |
| `modelopt/torch/export/model_utils.py` | Supported architectures |
| `modelopt_recipes/` | Pre-built recipes |
