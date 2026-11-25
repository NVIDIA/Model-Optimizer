# Evaluation

**Experimental:** This NeMo Gym evaluation workflow currently supports MMLU-Pro and is intended to replace the [NEL evaluation](../evaluation/) workflow in the future.

## Results

Qwen3-8B was evaluated using NeMo Gym with vLLM on one H100 per evaluation. QAD starts from the PTQ checkpoint and uses the BF16 model as teacher.

**Temporary values:** the numbers below are from eight questions per checkpoint and will be replaced after full MMLU-Pro evaluation (12,032 questions).

| Checkpoint | Deployed precision | MMLU-Pro accuracy ↑ |
|---|---|---:|
| Teacher (BF16) | BF16 | 75.0% |
| PTQ (NVFP4) | NVFP4 weights / BF16 activations | 62.5% |
| PTQ → QAD (NVFP4) | NVFP4 weights / BF16 activations | 75.0% |

The quantized checkpoints use Marlin's W4A16 fallback on H100; these results do not measure native NVFP4 W4A4 execution on Blackwell. KV cache is BF16 for all checkpoints.

## Run evaluations

- `gym_config.yaml`: entry point containing the MMLU-Pro recipe and driver settings.
- `gym_configs/`: shared Slurm execution and vLLM deployment settings.

Install the pinned Gym checkout with a recent `uv` release:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
git checkout 06f5f8966
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --no-dev
```

The driver container already includes `$HOME/.local/bin` in `PATH`, so the `uv` installer skips creating `$HOME/.local/bin/env`. Gym still tries to source that file and fails with `No such file or directory`, even though `uv` is installed correctly. In the Gym checkout, open `nemo_gym/orchestration/executors/script_templates.py` and replace `source "$HOME/.local/bin/env"` with `export PATH="$HOME/.local/bin:$PATH"` to avoid relying on the missing file.

From `examples/llm_qat/evaluation_nemogym/`, configure access to your Slurm cluster:

```bash
export GYM_SLURM_HOST="<cluster-login-host>"
export GYM_SLURM_ACCOUNT="<slurm-account>"
export GYM_SLURM_PARTITION="<slurm-partition>"
export GYM_GPUS_PER_NODE=1
export GYM_OUTPUT_DIR="/shared/path/to/evaluation-results"
export GYM_CACHE_DIR="/shared/path/to/cache"
export MODEL_CHECKPOINT="/shared/path/to/checkpoint"
export HF_TOKEN="<huggingface-token>"

# Optional: use local SQSH images instead of registry pulls.
# export GYM_VLLM_IMAGE="/shared/path/to/vllm-v0.29.0.sqsh"
# export GYM_DRIVER_IMAGE="/shared/path/to/nemo-gym-v0.5.1.sqsh"

# Run MMLU-Pro.
gym eval submit --config gym_config.yaml \
    services.policy.served_model_name=qwen3-8b-ptq
```

Use the BF16 teacher or an exported Hugging Face PTQ/QAD checkpoint. The checkpoint directory must be accessible on the cluster and contain the model files, without symlinks to unmounted locations.

For the teacher, download a self-contained directory on the cluster; no additional `/blobs` mounts are needed:

```bash
hf download Qwen/Qwen3-8B --local-dir /shared/path/to/qwen3-8b-teacher
export MODEL_CHECKPOINT="/shared/path/to/qwen3-8b-teacher"
```

Repeat for teacher and QAD by changing `MODEL_CHECKPOINT` and `services.policy.served_model_name`. Add `--dry-run` to preview. Gym launches a Slurm job for MMLU-Pro; results are saved under `GYM_OUTPUT_DIR`. Increase `GYM_GPUS_PER_NODE` to serve one tensor-parallel-size-1 replica per GPU.

The cluster needs Pyxis/Enroot and shared storage. The driver installs Gym at revision `6bbc05ed9d1664dc695faaf2b0e71fc9863e08ca` and downloads `TIGER-Lab/MMLU-Pro`, so network access is required even when using SQSH images.

Results are saved under `$GYM_OUTPUT_DIR/gym-job-<timestamp>-<id>/mmlu_pro/`:

- `artifacts/rollouts.jsonl`: raw responses and scores for each question.
- `artifacts/rollouts_aggregate_metrics.json`: MMLU-Pro accuracy under `pass@1/accuracy`.
- `artifacts/rollouts_failures.jsonl`: failed rollouts.

## MMLU-Pro accuracy by subject

<details>
<summary>Show per-subject results</summary>

Temporary results cover eight business questions only; N/A means the subject has not yet been evaluated.

| Subject | Questions evaluated | Teacher (BF16) | PTQ (NVFP4) | PTQ → QAD (NVFP4) |
|---|---:|---:|---:|---:|
| Biology | 0 | N/A | N/A | N/A |
| Business | 8 | 75.00% | 62.50% | 75.00% |
| Chemistry | 0 | N/A | N/A | N/A |
| Computer science | 0 | N/A | N/A | N/A |
| Economics | 0 | N/A | N/A | N/A |
| Engineering | 0 | N/A | N/A | N/A |
| Health | 0 | N/A | N/A | N/A |
| History | 0 | N/A | N/A | N/A |
| Law | 0 | N/A | N/A | N/A |
| Mathematics | 0 | N/A | N/A | N/A |
| Other | 0 | N/A | N/A | N/A |
| Philosophy | 0 | N/A | N/A | N/A |
| Physics | 0 | N/A | N/A | N/A |
| Psychology | 0 | N/A | N/A | N/A |

</details>
