---
name: ptq
description: This skill should be used when the user asks to "quantize a model", "run PTQ", "post-training quantization", "NVFP4 quantization", "FP8 quantization", "INT8 quantization", "INT4 AWQ", "quantize LLM", "quantize MoE", "quantize VLM", or needs to produce a quantized HuggingFace or TensorRT-LLM checkpoint from a pretrained model using ModelOpt.
---

# ModelOpt Post-Training Quantization

Produce a quantized checkpoint from a pretrained HuggingFace model using NVIDIA Model Optimizer. The output is ready for TensorRT-LLM deployment or HuggingFace-compatible inference.

## Decision Process

### 0. Check the execution environment

Do this first — the environment determines how to run the job and which formats are viable.

**Step 1 — Is this a SLURM cluster?**
```bash
which srun squeue sbatch 2>/dev/null | head -1
```
If any of those exist, you're on SLURM. Query accounts and partitions:

```bash
# Get user's accounts and cluster
sacctmgr show associations user=$USER format=account%30,partition%20,cluster%20 -n 2>/dev/null

# List partitions with time limits
sinfo -o "%P %a %l %G" 2>/dev/null | grep -v "^PARTITION"
```

- If the user has **one account**: use it automatically.
- If the user has **multiple accounts**: show them and ask which to use. Default to the account whose name most closely matches the project or working directory.
- For partition, use the default (marked with `*` in `sinfo` output). Report the choice.

**Step 2 — If not SLURM, check for a local GPU:**
```bash
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('no-gpu')"
```

| Result | Action |
|--------|--------|
| SLURM detected | Proceed — GPU will be allocated via `srun`. Infer GPU type from `sinfo` node features. |
| Local GPU found | Proceed — report the GPU model(s) to the user. |
| Neither found | **Stop and report**: "No GPU found and this doesn't appear to be a SLURM cluster. PTQ calibration requires a CUDA GPU. Please confirm the target environment." |

The GPU model feeds directly into format recommendation in Step 2.

### 1. Is the model architecture supported?

**Read `examples/llm_ptq/README.md` first.** It is the authoritative reference for this workflow and contains information that isn't duplicated here: the full support matrix, correct CLI flag names, accuracy guidance, and hardware requirements. Key sections to check:
- Support matrix (~line 100) — which architectures and formats are supported
- Correct flags `--pyt_ckpt_path` / `--export_path` (~line 149)
- Accuracy note: prefer `nvfp4_mlp_only` or `nvfp4_omlp_only` for NVFP4 (~line 131)
- Blackwell GPU requirement for NVFP4 inference (~line 126, footnote 5)

After reading the README, check `modelopt/torch/export/model_utils.py` for `MODEL_NAME_TO_TYPE`. If the model's class name substring-matches a key in that dict, it is supported.

**Supported** → Use the existing `examples/llm_ptq/hf_ptq.py` script directly. No custom code needed.

**Unsupported** → Follow the investigation steps below before writing any custom code.

### 1b. Unsupported model — investigate before writing code

The goal is to get the model's source code and understand the checkpoint format before writing anything.

**Step A — Is it a HuggingFace checkpoint?**

Check for `config.json`. If present, try loading with `AutoConfig.from_pretrained()` directly:

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<ckpt_path>')
print(type(cfg).__name__)
"
```

- **Succeeds** → transformers knows the architecture. Find the source file:
  ```bash
  python -c "
  import inspect
  from transformers import AutoConfig, AutoModel
  cfg = AutoConfig.from_pretrained('<ckpt_path>')
  cls = AutoModel._model_type_to_module_name.get(cfg.model_type)
  import transformers; mod = getattr(transformers, cls, None)
  print(inspect.getfile(mod) if mod else 'not found')
  "
  ```
  Read the relevant modeling file and proceed to Step B.

- **Raises `ValueError` / `OSError` (unknown architecture)** → ask the user:
  > "The checkpoint uses `<ArchName>` which isn't in the installed transformers. Can you provide the model source code (a file path or the modeling `.py` file)?"

- **No `config.json`** → look for docs or scripts inside the checkpoint directory:
  ```bash
  ls <ckpt_path>/
  ```
  - **README or `.py` files found** → read them to understand how to load the model.
  - **Nothing useful** → ask the user:
    > "This doesn't look like a standard HuggingFace checkpoint and has no README. Can you point me to the modeling code?"

**Step B — Is the checkpoint already FP8-quantized?**

Check `config.json` for `"quantization_config"` or look for `*_scale_inv*` tensors in the weight files. If found, the model must be dequantized before re-quantizing with ModelOpt. Read **Pattern 5 (FP8 Checkpoint Dequantization)** in `references/unsupported-models.md` — HuggingFace's `WeightConverter` only handles standard `weight` / `weight_scale_inv` names and will silently miss non-standard parameter names (e.g., 3D expert tensors in MoE layers).

**Step C — Analyze the source for quantization targets:**
1. Which linear layers need quantization (`nn.Linear`, `F.linear`, or custom weight tensors)
2. Non-standard weight storage (e.g., 3D expert weight tensors in MoE layers) → Pattern 1
3. VLM structure (vision encoder + language model) → Pattern 4
4. MoE routing (needs all-expert calibration wrapper) → Pattern 2

**Step D — Check weight names against ModelOpt's config patterns:**

Scan actual parameter names in the checkpoint and compare them against the wildcard patterns in the chosen quant config (`modelopt/torch/quantization/config.py`). If a module has a weight that would be quantized but uses a non-standard name (e.g., `gate_up_proj` instead of `gate_proj`/`up_proj`, or `experts.w1` instead of `experts.*.w1`), the wildcard will silently miss it.

```python
import json
idx = json.load(open('<ckpt_path>/model.safetensors.index.json'))
# Print all unique weight base names (strip layer index)
import re
names = set(re.sub(r'\.\d+\.', '.N.', k) for k in idx['weight_map'])
for n in sorted(names): print(n)
```

Compare these against the `enable`/`disable` patterns in the config. Add custom overrides to the config dict if needed.

Then read `references/unsupported-models.md` for the full patterns: `mtq.register()`, `_setup()`, TensorQuantizer injection, and calibration routing.

### 2. Choose the quantization format

If the user has not specified a format, **recommend one based on the GPU detected in Step 0**:

| GPU generation | Memory priority | Accuracy priority |
|----------------|-----------------|-------------------|
| **Blackwell** (B100, B200, GB200) | `nvfp4_mlp_only` | `nvfp4_omlp_only` |
| **Hopper** (H100, H200) or older | `int4_awq` | `fp8` |

Tell the user which GPU was detected and which format you are recommending, and why.

For reference, all available configs are in `modelopt/torch/quantization/config.py`:

| Format | Config | Notes |
|--------|--------|-------|
| NVFP4 MLP-only | `NVFP4_MLP_ONLY_CFG` | Recommended for Blackwell; best accuracy/throughput tradeoff |
| NVFP4 output+MLP | `NVFP4_OMLP_ONLY_CFG` | Slightly higher accuracy than MLP-only |
| NVFP4 all layers | `NVFP4_DEFAULT_CFG` | May reduce accuracy; see README |
| NVFP4 + AWQ calibration | `NVFP4_AWQ_LITE_CFG` | Best NVFP4 accuracy, slower calibration |
| FP8 per-tensor | `FP8_DEFAULT_CFG` | Accuracy-first for Hopper |
| INT4 weight-only | `INT4_AWQ_CFG` | Memory-first for Hopper/older |
| INT8 + SmoothQuant | `INT8_SMOOTHQUANT_CFG` | Older GPUs, activation quantization |

> **NVFP4 requires Blackwell GPUs** for inference. H100 can run NVFP4 calibration but not inference.

For MLP-only quantization (skipping attention), use configs with `MLP_ONLY` in the name, or create a custom config by disabling `*self_attn*`.

### 3. Set up the environment

- **SLURM**: You MUST use a container — running without one will fail due to missing dependencies. Get the recommended container image version from `examples/llm_ptq/README.md`, then find the matching `.sqsh` file (search near the working directory, or sibling paths). Save the sqsh path — it becomes the `--container-image` value in the `srun` command. If no sqsh found, read `references/environment-setup.md` for enroot import instructions.
- **Local GPU**: `pip install nvidia-modelopt[hf]`

**GPU memory**: Estimate `num_params × 2 bytes` for BF16. Use `device_map="auto"` for multi-GPU. For models exceeding single-node memory, read `references/environment-setup.md` for FSDP2 multi-node setup.

### 4. Write, submit, and monitor

**The goal is a quantized checkpoint on disk — not a script handed to the user.** Write the script, submit it, follow the logs, fix errors, and resubmit until the export directory contains `.safetensors` shards and a `config.json`.

#### Smoke test first (always)

Before a full calibration run, submit a smoke-test job with `--calib_size 4` and `--time=00:30:00`. This catches script errors before wasting GPU quota on a full run. Only request a longer time limit after the smoke test passes.

#### Supported models — command to run

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path <model_path> \
    --qformat <format_name> \
    --calib_size 512 \
    --export_path <output_path>
```

Run `python examples/llm_ptq/hf_ptq.py --help` to see all options.

#### Unsupported models — custom script

After Step 1b, write a custom PTQ script following `references/unsupported-models.md`. Core steps:
1. Load model (handle FP8 dequantization if needed)
2. Register monkey-patched modules via `mtq.register()`
3. Create calibration dataloader
4. Call `mtq.quantize(model, config, forward_loop)`
5. Export with `export_hf_checkpoint(model, export_dir)`

#### Submit the job

**SLURM** — container flags (`--container-image`, `--container-mounts`) MUST be on the `srun` line, not as `#SBATCH` directives. Use this structure every time:

```bash
#!/bin/bash
#SBATCH --job-name=ptq
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<log_dir>/ptq_%j.log

srun \
    --container-image="<path/to/container.sqsh>" \
    --container-mounts="<data_root>:<data_root>" \
    --container-workdir="<workdir>" \
    --no-container-mount-home \
    bash -c "pip install -e <modelopt_path>[hf] --quiet && python <ptq_script.py> ..."
```

Then submit and save the job ID:
```bash
mkdir -p <log_dir>
JOBID=$(sbatch <script>.sh | awk '{print $4}')
echo "Submitted job $JOBID"
```

**Local GPU:**
```bash
nohup python ptq_script.py ... > <log_file> 2>&1 &
echo $!  # save PID
```

#### Monitor and debug

Tail the job log and resubmit with fixes until the export directory contains `.safetensors` shards and `config.json`. Two PTQ-specific failure modes to check via `mtq.print_quant_summary()`:
- **Quantizers not enabled**: wildcard missed modules — check `*gate*` vs `*mlp.gate*`
- **FP8 tensors still present after dequant**: missed a non-standard param name — inspect `model.named_parameters()` for `float8_e4m3fn` dtypes

### 5. Verify the output checkpoint

Once the job succeeds, confirm the export is valid:

```bash
# Check export directory has model shards and config
ls -lh <output_path>/
# Expect: config.json, tokenizer files, model-*.safetensors

# Verify no unexpected FP8 tensors remain
python -c "
from safetensors import safe_open
import glob, os
for f in sorted(glob.glob('<output_path>/model*.safetensors'))[:1]:
    with safe_open(f, framework='pt') as sf:
        for k in list(sf.keys())[:5]:
            t = sf.get_tensor(k)
            print(k, t.dtype, t.shape)
"
```

Report the output path and checkpoint size to the user.

## Key API Rules

These are non-obvious requirements that cause hard-to-debug failures:

- **`mtq.register()` requires `_setup` method**: Any class registered with `mtq.register(original_cls=X, quantized_cls=Y)` MUST define a method named exactly `_setup()`. Not `_init_quantizers`, not `setup` — exactly `_setup`. Also, the `__init__` must call `self._setup()` — if you forget this, TensorQuantizers are never instantiated and quantization silently does nothing.

- **Call `mto.enable_huggingface_checkpointing()` before quantization**: Required for HF checkpoint export to work.

- **Wildcard pattern `*gate*` is dangerously broad**: It matches both MoE router gates AND any quantizer with "gate" in the name (e.g., `gate_up_weight_quantizer`). Use `*mlp.gate*` or `*router*` to target router gates specifically. Always verify with `mtq.print_quant_summary()`.

- **VLMs need `AutoModel`**: Vision-Language Models (e.g., `Mistral3ForConditionalGeneration`, `Mllama`) are NOT registered under `AutoModelForCausalLM`. Use `AutoModel.from_pretrained()`.

- **FP8 checkpoints need the config class**: When loading an FP8-quantized checkpoint with `dequantize=True`, pass `FineGrainedFP8Config(dequantize=True)` — not a plain dict. HF validates the config type matches.

- **Quantizer naming convention**: Custom `TensorQuantizer` modules must end with `_input_quantizer` or `_weight_quantizer` for ModelOpt's wildcard matching.

- **Do not modify ModelOpt core source**: All custom code (monkey-patching, `mtq.register()` wrappers, dequantization helpers) must live in your own script or under `examples/`. Never edit files under `modelopt/torch/` unless there is no easy way to patch from outside — and if you must, note it explicitly so it can be upstreamed.

## Additional Resources

### Reference Files

- **`references/unsupported-models.md`** — Patterns for extending ModelOpt to new architectures: MoE expert quantization, VLM language model extraction, FP8 dequantization, calibration routing
- **`references/environment-setup.md`** — Container selection, SLURM/enroot/pyxis setup, common environment errors

### ModelOpt Examples

- **`examples/llm_ptq/README.md`** ← **read this first** — support matrix, correct flag names, accuracy guidance, hardware requirements
- **`examples/llm_ptq/hf_ptq.py`** — Main PTQ script for supported models
- **`examples/llm_ptq/multinode_ptq.py`** — Multi-node PTQ with FSDP2
- **`examples/deepseek/ptq.py`** — Custom PTQ for DeepSeek MoE (reference for MoE monkey-patching)

### Source Code

- **`modelopt/torch/quantization/config.py`** — All quantization configs and format definitions
- **`modelopt/torch/export/model_utils.py`** — `MODEL_NAME_TO_TYPE` (supported architectures), `get_model_type()`, `is_multimodal_model()`
- **`modelopt/torch/quantization/conversion.py`** — `mtq.register()` implementation (see `_setup` requirement)
- **`modelopt/torch/utils/dataset_utils.py`** — `get_dataset_dataloader()`, `get_supported_datasets()`
