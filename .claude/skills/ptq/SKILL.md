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

Check the support table in `examples/llm_ptq/README.md` for verified HF models.

- **Listed** → supported, use `hf_ptq.py` (step 4A/4B)
- **Not listed** → read `references/unsupported-models.md` to determine if `hf_ptq.py` can still work or if a custom script is needed (step 4C)

## Step 2.5 — Check for model-specific dependencies

If the model uses `trust_remote_code` (check `config.json` for `auto_map`), inspect its custom Python files for imports not present in the container:

```bash
grep -h "^from \|^import " <model_path>/modeling_*.py | sort -u
```

**Known dependency patterns:**

| Import found | Packages to install |
| --- | --- |
| `from mamba_ssm` / `from causal_conv1d` | `mamba-ssm causal-conv1d` (Mamba/hybrid models: NemotronH, Jamba) |

If extra deps are needed:
- **Launcher (4B)**: set `EXTRA_PIP_DEPS` in the task's `environment` section — `ptq.sh` installs them automatically
- **Manual (4A)**: `unset PIP_CONSTRAINT && pip install <deps>` before running `hf_ptq.py`

## Step 3 — Choose quantization format

**First**, check for a model-specific recipe:

```bash
ls modelopt_recipes/models/ 2>/dev/null
```

If a model-specific recipe exists, use `--recipe <path>` — it may contain tuned settings.

**If no model-specific recipe**, choose a format based on GPU (details in `examples/llm_ptq/README.md`):

- **Blackwell** (B100/B200/GB200): `nvfp4` variants
- **Hopper** (H100/H200) or older: `fp8` or `int4_awq`

Use `--qformat <name>` (e.g., `--qformat nvfp4`). Format definitions: `modelopt/torch/quantization/config.py`. General PTQ recipes in `modelopt_recipes/general/ptq/` correspond to the same formats — `--qformat` is the simpler way to use them.

> NVFP4 can be calibrated on Hopper but requires Blackwell for inference.

## Step 4 — Run PTQ

**Goal: checkpoint on disk** (`.safetensors` + `config.json`).

For **listed models** (4A/4B): run full calibration directly (`--calib_size 512`).
For **unlisted models** (4C): run a smoke test first (`--calib_size 4`), wait for success, then full calibration.

### Which path?

```text
In README table? ─→ YES ──→ SLURM (local or remote)? ──→ LAUNCHER (4B)
                  │          Local Docker + GPU? ────────→ LAUNCHER (4B)
                  │          Remote Docker (no SLURM)? ──→ MANUAL (4A)
                  │          Bare GPU (local or remote)? → MANUAL (4A)
                  │
                  └→ NOT LISTED ──→ UNLISTED MODEL (4C)
```

### 4A — Direct: supported model, manual execution

```bash
pip install --no-build-isolation "nvidia-modelopt[hf]"
pip install -r examples/llm_ptq/requirements.txt

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

The launcher blocks and tails logs until the job completes. If the launcher fails (missing deps, config errors), fall back to path 4A (manual execution).

### 4C — Unlisted model

Follow `references/unsupported-models.md`. It walks through investigating the model, patching ModelOpt if needed, and running `hf_ptq.py`. Run manually (like 4A) for easier monitoring and debugging.

For SLURM, see `skills/common/slurm-setup.md` and `references/slurm-setup-ptq.md`.

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

### Post-quantization validation

Validate the exported checkpoint's quantization pattern matches the recipe. Quantization config patterns can silently miss layers if the model uses non-standard naming (e.g., Gemma4 `experts.*` missed by `*mlp*` patterns) — this only surfaces later as deployment failures. Read `references/checkpoint-validation.md` for the validation script, expected patterns per recipe, and common pattern gaps.

## Key API Rules

- `mtq.register()` classes **must** define `_setup()` and call it from `__init__`
- Call `mto.enable_huggingface_checkpointing()` **before** quantization
- Wildcard `*gate*` matches too broadly — use `*mlp.gate*` or `*router*`
- VLMs: `hf_ptq.py` auto-extracts the language model via `extract_and_prepare_language_model_from_vl()` — no manual VLM handling needed in most cases
- FP8 checkpoints: prefer `_QuantFP8Linear` (lazy dequant) over `FineGrainedFP8Config(dequantize=True)` which wastes ~2x memory. See `references/unsupported-models.md` for details
- Custom quantizer names must end with `_input_quantizer` or `_weight_quantizer`

## Common Pitfalls

- **Model-specific dependencies**: Models with `trust_remote_code` may import packages not in the container (e.g., `mamba-ssm` for hybrid Mamba models). See Step 2.5. Use `EXTRA_PIP_DEPS` env var with the launcher, or install manually before running `hf_ptq.py`
- **Transformers version**: New models may need a newer version of transformers than what's installed. Check `config.json` for `transformers_version`. In containers, beware of `PIP_CONSTRAINT` blocking upgrades — see `references/slurm-setup-ptq.md` for workarounds
- **Gated datasets**: Some calibration datasets require HF authentication. Ensure `HF_TOKEN` is set in the job environment, or use `--dataset cnn_dailymail` as a non-gated alternative
- **NFS root_squash + Docker**: See `skills/common/slurm-setup.md` section 5

## References

| Reference | When to read |
| --- | --- |
| `skills/common/environment-setup.md` | Step 1: always |
| `skills/common/workspace-management.md` | Step 1: always |
| `references/launcher-guide.md` | Step 4B only (launcher path) |
| `tools/launcher/CLAUDE.md` | Step 4B only, if you need more launcher detail |
| `references/unsupported-models.md` | Step 4C only (unlisted model) |
| `references/checkpoint-validation.md` | Step 5: validate quantization pattern matches recipe |
| `skills/common/remote-execution.md` | Step 4A/4C only, if target is remote |
| `skills/common/slurm-setup.md` | Step 4A/4C only, if using SLURM manually (not launcher) |
| `references/slurm-setup-ptq.md` | Step 4A/4C only, PTQ-specific SLURM (container, GPU sizing, FSDP2) |
| `examples/llm_ptq/README.md` | Step 3: support matrix, CLI flags, accuracy |
| `modelopt/torch/quantization/config.py` | Step 3: format definitions |
| `modelopt/torch/export/model_utils.py` | Step 4C: TRT-LLM export type mapping |
| `modelopt_recipes/` | Step 3: pre-built recipes |
