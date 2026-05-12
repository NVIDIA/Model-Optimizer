---
name: deployment
description: Serve a quantized or unquantized LLM checkpoint as an OpenAI-compatible API endpoint using vLLM, SGLang, or TRT-LLM. Use when user says "deploy model", "serve model", "start vLLM server", "launch SGLang", "TRT-LLM deploy", "AutoDeploy", "benchmark throughput", "serve checkpoint", or needs an inference endpoint from a HuggingFace or ModelOpt-quantized checkpoint. Do NOT use for quantizing models (use ptq) or evaluating accuracy (use evaluation).
license: Apache-2.0
---

# Deployment Skill

Serve a model checkpoint as an OpenAI-compatible inference endpoint. Supports vLLM, SGLang, and TRT-LLM (including AutoDeploy).

## Quick Start

Prefer `scripts/deploy.sh` for standard local deployments — it handles quant detection, health checks, and server lifecycle. Use the raw framework commands in Step 4 when you need flags the script doesn't support, or for remote deployment.

```bash
# Start vLLM server with a ModelOpt checkpoint
scripts/deploy.sh start --model ./qwen3-0.6b-fp8

# Start with SGLang and tensor parallelism
scripts/deploy.sh start --model ./llama-70b-nvfp4 --framework sglang --tp 4

# Start from HuggingFace hub
scripts/deploy.sh start --model nvidia/Llama-3.1-8B-Instruct-FP8

# Test the API
scripts/deploy.sh test

# Check status
scripts/deploy.sh status

# Stop
scripts/deploy.sh stop
```

The script handles: GPU detection, quantization flag auto-detection (FP8 vs FP4), server lifecycle (start/stop/restart/status), health check polling, and API testing.

## Decision Flow

### 0. Check workspace (multi-user / Slack bot)

If `MODELOPT_WORKSPACE_ROOT` is set, read `skills/common/workspace-management.md`. Before creating a new workspace, check for existing ones — especially if deploying a checkpoint from a prior PTQ run:

```bash
ls "$MODELOPT_WORKSPACE_ROOT/" 2>/dev/null
```

If the user says "deploy the model I just quantized" or references a previous PTQ, find the matching workspace and `cd` into it. The checkpoint should be in that workspace's output directory.

### 1. Identify the checkpoint

Determine what the user wants to deploy:

- **Local quantized checkpoint** (from ptq skill or manual export): look for `hf_quant_config.json` in the directory. If coming from a prior PTQ run in the same workspace, check common output locations: `output/`, `outputs/`, `exported_model/`, or the `--export_path` used in the PTQ command.
- **HuggingFace model hub** (e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`): use directly
- **Unquantized model**: deploy as-is (BF16) or suggest quantizing first with the ptq skill

> **Note:** This skill expects HF-format checkpoints (from PTQ with `--export_fmt hf`). TRT-LLM format checkpoints should be deployed directly with TRT-LLM — see `references/trtllm.md`.

Check the quantization format if applicable:

```bash
cat <checkpoint_path>/hf_quant_config.json 2>/dev/null || echo "No hf_quant_config.json"
```

If not found, also check `config.json` for a `quantization_config` section with `quant_method: "modelopt"`. If neither exists, the checkpoint is unquantized.

### 2. Choose the framework

If the user hasn't specified a framework, recommend based on this priority:

| Situation | Recommended | Why |
|-----------|-------------|-----|
| General use | **vLLM** | Widest ecosystem, easy setup, OpenAI-compatible |
| Best SGLang model support | **SGLang** | Strong DeepSeek/Llama 4 support |
| Maximum optimization | **TRT-LLM** | Best throughput via engine compilation |
| Mixed-precision / AutoQuant | **TRT-LLM AutoDeploy** | Only option for AutoQuant checkpoints |

Check the support matrix in `references/support-matrix.md` to confirm the model + format + framework combination is supported.

### 3. Check the environment

Read `skills/common/environment-setup.md` for GPU detection, local vs remote, and SLURM/Docker/bare metal detection. After completing it you should know: GPU model/count, local or remote, and execution environment.

Then check the **deployment framework** is installed:

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || echo "vLLM not installed"
python -c "import sglang; print(f'SGLang {sglang.__version__}')" 2>/dev/null || echo "SGLang not installed"
python -c "import tensorrt_llm; print(f'TRT-LLM {tensorrt_llm.__version__}')" 2>/dev/null || echo "TRT-LLM not installed"
```

If not installed, consult `references/setup.md`.

**GPU memory estimate** (to determine tensor parallelism):

- BF16: `params × 2 bytes` (8B ≈ 16 GB)
- FP8: `params × 1 byte` (8B ≈ 8 GB)
- FP4: `params × 0.5 bytes` (8B ≈ 4 GB)
- Add ~2-4 GB for KV cache and framework overhead

If the model exceeds single GPU memory, use tensor parallelism (`-tp <num_gpus>`).

### 4. Deploy

Read the framework-specific reference for detailed instructions:

| Framework | Reference file |
|-----------|---------------|
| vLLM | `references/vllm.md` |
| SGLang | `references/sglang.md` |
| TRT-LLM | `references/trtllm.md` |

**Quick-start commands** (for common cases):

#### vLLM

```bash
# Serve as OpenAI-compatible endpoint
python -m vllm.entrypoints.openai.api_server \
    --model <checkpoint_path> \
    --quantization modelopt \
    --tensor-parallel-size <num_gpus> \
    --host 0.0.0.0 --port 8000
```

For NVFP4 checkpoints, use `--quantization modelopt_fp4`.

#### SGLang

```bash
python -m sglang.launch_server \
    --model-path <checkpoint_path> \
    --quantization modelopt \
    --tp <num_gpus> \
    --host 0.0.0.0 --port 8000
```

#### TRT-LLM (direct)

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="<checkpoint_path>")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8, top_p=0.95))
```

#### TRT-LLM AutoDeploy

For AutoQuant or mixed-precision checkpoints, see `references/trtllm.md`.

### 5. Verify the deployment

After the server starts, verify it's healthy:

```bash
# Health check
curl -s http://localhost:8000/health

# List models
curl -s http://localhost:8000/v1/models | python -m json.tool

# Test generation
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<model_name>",
        "prompt": "The capital of France is",
        "max_tokens": 32
    }' | python -m json.tool
```

All checks must pass before reporting success to the user.

### 6. Remote deployment (SSH/SLURM)

If a cluster config exists (`~/.config/modelopt/clusters.yaml` or `.claude/clusters.yaml`), or the user mentions running on a remote machine:

0. **Check container registry auth** — before submitting any SLURM job with a container image, verify credentials exist on the cluster per `skills/common/slurm-setup.md` section 6. If credentials are missing for the image's registry, ask the user to fix auth or switch to an image on an authenticated registry (e.g., NGC). **Do not submit until auth is confirmed.**

1. **Source remote utilities:**

   ```bash
   source .claude/skills/common/remote_exec.sh
   remote_load_cluster
   remote_check_ssh
   remote_detect_env
   ```

2. **Sync the checkpoint** (only if it was produced locally):

   If the checkpoint path is a remote/absolute path (e.g., from a prior PTQ run on the cluster), skip sync — it's already there. Verify with `remote_run "ls <checkpoint_path>/config.json"`. Only sync if the checkpoint is local:

   ```bash
   remote_sync_to <local_checkpoint_path> checkpoints/
   ```

3. **Deploy based on remote environment:**

   - **SLURM** — see `skills/common/slurm-setup.md` for job script templates (container setup, account/partition discovery). The server command inside the container is the same as Step 4 (e.g., `python -m vllm.entrypoints.openai.api_server --model <path> --quantization modelopt`). After submitting, register the job and set up monitoring per the **monitor skill**. Get the node hostname from `squeue -j $JOBID -o %N`.

   - **Bare metal / Docker** — use `remote_run` to start the server directly:

     ```bash
     remote_run "nohup python -m vllm.entrypoints.openai.api_server --model <path> --port 8000 > deploy.log 2>&1 &"
     ```

4. **Verify remotely:**

   ```bash
   remote_run "curl -s http://localhost:8000/health"
   remote_run "curl -s http://localhost:8000/v1/models"
   ```

5. **Report the endpoint** — include the remote hostname and port so the user can connect (e.g., `http://<node_hostname>:8000`). For SLURM, note that the port is only reachable from within the cluster network.

For NEL-managed deployment (evaluation with self-deployment), use the evaluation skill instead — NEL handles SLURM container deployment, health checks, and teardown automatically.

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Model too large for GPU(s) | Increase `--tensor-parallel-size` or use a smaller model |
| `quantization="modelopt" not recognized` | vLLM/SGLang version too old | Upgrade: vLLM >= 0.10.1, SGLang >= 0.4.10 |
| `hf_quant_config.json not found` | Not a ModelOpt-exported checkpoint | Re-export with `export_hf_checkpoint()`, or remove `--quantization` flag |
| `Connection refused` on health check | Server still starting | Wait 30-60s for large models; check logs for errors |
| `modelopt_fp4 not supported` | Framework doesn't support FP4 for this model | Check support matrix in `references/support-matrix.md` |

## Unsupported Models

If the model is not in the validated support matrix (`references/support-matrix.md`), deployment may fail due to weight key mismatches, missing architecture mappings, or quantized/unquantized layer confusion. Read `references/unsupported-models.md` for the iterative debug loop: **run → read error → diagnose → patch framework source → re-run**. For kernel-level issues, escalate to the framework team rather than attempting fixes.

## Success Criteria

1. Server process is running and healthy (`/health` returns 200)
2. Model is listed at `/v1/models`
3. Test generation produces coherent output
4. Server URL and port are reported to the user
5. If benchmarking was requested, throughput/latency numbers are reported
