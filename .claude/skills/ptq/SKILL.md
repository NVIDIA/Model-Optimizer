---
name: ptq
description: This skill should be used when the user asks to "quantize a model", "run PTQ", "post-training quantization", "NVFP4 quantization", "FP8 quantization", "INT8 quantization", "INT4 AWQ", "quantize LLM", "quantize MoE", "quantize VLM", or needs to produce a quantized HuggingFace or TensorRT-LLM checkpoint from a pretrained model using ModelOpt.
---

# ModelOpt Post-Training Quantization

Produce a quantized checkpoint from a pretrained HuggingFace model using NVIDIA Model Optimizer. The output is ready for TensorRT-LLM deployment or HuggingFace-compatible inference.

## Decision Process

### 0. Check the execution environment

Do this first — the environment determines how to run the job and which formats are viable.

**Multi-user / Slack bot mode?**

If `MODELOPT_WORKSPACE_ROOT` is set, you are running in a multi-user environment. Read `skills/common/workspace-management.md` and check for an existing workspace for this model before proceeding. If you create or switch to a model-specific workspace, all subsequent steps run there.

**Is this a remote execution?**

Check if a remote cluster config exists or the user mentioned running on a remote machine:

```bash
cat ~/.config/modelopt/clusters.yaml 2>/dev/null || cat .claude/clusters.yaml 2>/dev/null
```

**Case A — config found, or user says "run on [cluster]" / "run remotely" / "use SSH":**
Switch to remote execution mode — read `references/remote-execution.md` now. All subsequent steps apply whether local or remote.

**Case B — no config, user hasn't mentioned a cluster:**
Skip remote mode and proceed with local execution below.

**Case C — no config, but user clearly wants remote (e.g. "run on the cluster", "use SSH", mentions a hostname):**
Ask the user for the following info, then create `~/.config/modelopt/clusters.yaml` before proceeding:

```text
I need a few details to set up the remote cluster. Please provide:
1. Login node hostname (e.g. cluster-login.example.com)
2. SSH username
3. SSH key path (default: ~/.ssh/id_rsa) — press Enter to use default
4. Remote working directory (e.g. /lustre/fs1/username/modelopt or ~/modelopt)
5. Cluster name/alias for future reference (e.g. "selene", "cw-dfw")
```

Once you have the answers, write `~/.config/modelopt/clusters.yaml`:

```yaml
clusters:
  <alias>:
    login_node: <hostname>
    user: <username>
    ssh_key: <ssh_key_path>
    workspace: <remote_workdir>

default_cluster: <alias>
```

Then read `references/remote-execution.md` and continue.

**Is this a SLURM cluster?**

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

**If not SLURM, check for a local GPU:**

```bash
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('no-gpu')"
```

| Result | Action |
|--------|--------|
| SLURM detected | Proceed — GPU will be allocated via `srun`. Infer GPU type from `sinfo` node features. |
| Local GPU found | Proceed — report the GPU model(s) to the user. |
| Neither found | **Stop and report**: "No GPU found and this doesn't appear to be a SLURM cluster. PTQ calibration requires a CUDA GPU. Please confirm the target environment." |

The GPU model feeds directly into format recommendation in the next step.

### 1. Is the model architecture supported?

**Read `examples/llm_ptq/README.md` first.** It is the authoritative reference for this workflow and contains information that isn't duplicated here: the full support matrix, correct CLI flag names, accuracy guidance, and hardware requirements. Key sections to check:

- Support matrix (~line 100) — which architectures and formats are supported
- Correct flags `--pyt_ckpt_path` / `--export_path` (~line 149)
- Accuracy note: prefer `nvfp4_mlp_only` or `nvfp4_omlp_only` for NVFP4 (~line 131)
- Blackwell GPU requirement for NVFP4 inference (~line 126, footnote 5)

After reading the README, check `modelopt/torch/export/model_utils.py` for `MODEL_NAME_TO_TYPE`. If the model's class name substring-matches a key in that dict, it is supported.

**Supported** → Use the existing `examples/llm_ptq/hf_ptq.py` script directly. No custom code needed.

**Unsupported** → **Read `references/unsupported-models.md` now.** It covers model source investigation, FP8 detection, patch assessment, weight name verification, and all implementation patterns.

### 2. Choose the quantization format

If the user has not specified a format, **recommend one based on the GPU detected above**:

| GPU generation | Memory priority | Accuracy priority |
|----------------|-----------------|-------------------|
| **Blackwell** (B100, B200, GB200) | `nvfp4_mlp_only` | `nvfp4_awq_lite` |
| **Hopper** (H100, H200) or older | `int4_awq` | `fp8` |

Tell the user which GPU was detected and which format you are recommending, and why.

> **If the user explicitly requests `nvfp4` on a Hopper GPU**: proceed — H100/H200 can *calibrate* NVFP4 checkpoints fine. Just note: "NVFP4 inference requires Blackwell GPUs; this checkpoint will be calibrated on H100 but must be deployed on Blackwell."

For reference, all available configs are in `modelopt/torch/quantization/config.py`:

| Format | Config | Notes |
|--------|--------|-------|
| NVFP4 MLP-only | `NVFP4_MLP_ONLY_CFG` | Recommended for Blackwell; best accuracy/throughput tradeoff |
| NVFP4 MLP weight-only | `NVFP4_MLP_WEIGHT_ONLY_CFG` | Quantize MLP weights only (no activations) |
| NVFP4 all layers | `NVFP4_DEFAULT_CFG` | May reduce accuracy; see README |
| NVFP4 + AWQ calibration | `NVFP4_AWQ_LITE_CFG` | Best NVFP4 accuracy, slower calibration |
| FP8 per-tensor | `FP8_DEFAULT_CFG` | Accuracy-first for Hopper |
| INT4 weight-only | `INT4_AWQ_CFG` | Memory-first for Hopper/older |
| INT8 + SmoothQuant | `INT8_SMOOTHQUANT_CFG` | Older GPUs, activation quantization |

> **NVFP4 requires Blackwell GPUs** for inference. H100 can run NVFP4 calibration but not inference.

For MLP-only quantization (skipping attention), use configs with `MLP_ONLY` in the name, or create a custom config by disabling `*self_attn*`.

### 3. Set up the environment

- **SLURM**: Read `references/slurm-setup.md` — it has container setup, account/partition selection, the job script template, smoke-test strategy, and monitoring instructions.
- **Local GPU**: Check if Docker is available first — it's the cleanest isolation:
  - **Docker available**: use the TRT-LLM NGC container (version from `examples/llm_ptq/README.md`):

    ```bash
    docker run --gpus all -v <model_path>:<model_path> -v <output_path>:<output_path> \
        nvcr.io/nvidia/tensorrt-llm/release:<version> bash -c "pip install --no-build-isolation -e <modelopt_path>[hf] --quiet && python <ptq_script.py> ..."
    ```

  - **No Docker**: set up a virtual environment with conda (preferred) or venv:

    ```bash
    # conda
    conda create -n modelopt python=3.10 -y && conda activate modelopt
    # or venv
    python -m venv modelopt-env && source modelopt-env/bin/activate

    pip install --no-build-isolation nvidia-modelopt[hf]
    ```

**GPU memory**: Estimate `num_params × 2 bytes` for BF16. Use `device_map="auto"` for multi-GPU. If the model exceeds single-node memory, see the FSDP2 section in `references/slurm-setup.md`.

### 4. Write and run

**The goal is a quantized checkpoint on disk — not a script handed to the user.** Write the script, run it (or submit it), follow the logs, fix errors, and rerun until the export directory contains `.safetensors` shards and a `config.json`.

#### Supported models

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path <model_path> \
    --qformat <format_name> \
    --export_fmt hf \
    --calib_size 512 \
    --export_path <output_path>
```

Always pass `--export_fmt hf` explicitly — older versions of the script default to `tensorrt_llm` which produces TRT-LLM format instead of a HuggingFace checkpoint.

Run `python examples/llm_ptq/hf_ptq.py --help` to see all options.

#### Unsupported models

Write a custom script following `references/unsupported-models.md`. Core steps:

1. Load model (dequantize FP8 if needed)
2. Register monkey-patched modules via `mtq.register()`
3. Create calibration dataloader
4. Call `mtq.quantize(model, config, forward_loop)`
5. Export with `export_hf_checkpoint(model, export_dir)`

#### Local GPU — run and monitor

```bash
nohup python ptq_script.py ... > <log_file> 2>&1 &
tail -f <log_file>
```

PTQ-specific failure modes to check via `mtq.print_quant_summary()`:

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
- **`references/slurm-setup.md`** — SLURM job script template, container/enroot setup, partition selection, smoke-test strategy, monitoring, multi-node FSDP2
- **`references/remote-execution.md`** — **Read this when running PTQ on a remote machine/cluster via SSH.** Covers cluster config, persistent SSH sessions, SLURM container jobs, the two-script pattern, and troubleshooting.
- **`skills/common/workspace-management.md`** — **Read this when `MODELOPT_WORKSPACE_ROOT` is set (Slack bot / multi-user).** Covers when to create vs reuse workspaces, naming conventions, and cross-task workspace sharing (PTQ → deploy → eval).

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
