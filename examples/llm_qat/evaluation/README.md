# Evaluation

## Results

Qwen3-8B was evaluated using NEL with vLLM on one H100 per evaluation. QAD starts from the PTQ checkpoint and uses the BF16 model as teacher.

| Checkpoint | Deployed precision | MMLU accuracy ↑ (14,042 questions) | WikiText-2 perplexity ↓ (241,335 words) |
|---|---|---:|---:|
| Teacher (BF16) | BF16 | 74.94% | 13.485 |
| PTQ (NVFP4) | NVFP4 weights / BF16 activations | 73.59% | 13.883 |
| PTQ → QAD (NVFP4) | NVFP4 weights / BF16 activations | 73.76% | 13.684 |

QAD improves both metrics over PTQ but remains below the BF16 teacher. The quantized checkpoints use Marlin's W4A16 fallback on H100; these results do not measure native NVFP4 W4A4 execution on Blackwell. KV cache is BF16 for all checkpoints.

## Run evaluations

- `nel_config.yaml`: entry point containing the MMLU and WikiText recipes.
- `nel_configs/`: shared Slurm execution and vLLM deployment settings.

From `examples/llm_qat/evaluation/`, configure access to your Slurm cluster:

```bash
pip install "nemo-evaluator-launcher[all]==0.2.6"

export NEL_SLURM_HOST="<cluster-login-host>"
export SLURM_USER="<cluster-username>"
export NEL_SLURM_ACCOUNT="<slurm-account>"
export NEL_SLURM_PARTITION="<slurm-partition>"
export NEL_OUTPUT_DIR="/shared/path/to/evaluation-results"
export MODEL_CHECKPOINT="/shared/path/to/checkpoint"

# See execution.mounts.evaluation in nel_configs/h100_slurm_execution.yaml
# for why the checkpoint mount is passed via -o.
# Run MMLU and WikiText.
nemo-evaluator-launcher run --config nel_config.yaml \
    -o deployment.served_model_name=qwen3-8b-ptq \
    -o "++execution.mounts.evaluation={$MODEL_CHECKPOINT: /checkpoint}"
```

Use the BF16 teacher or an exported Hugging Face PTQ/QAD checkpoint. The checkpoint directory must be accessible on the cluster and contain the model files, without symlinks to unmounted locations.

For the teacher, download a self-contained directory on the cluster; no additional `/blobs` mounts are needed:

```bash
hf download Qwen/Qwen3-8B --local-dir /shared/path/to/qwen3-8b-teacher
export MODEL_CHECKPOINT="/shared/path/to/qwen3-8b-teacher"
```

Add `--dry-run` to preview, or `-t lm-evaluation-harness.mmlu` / `-t lm-evaluation-harness.wikitext` to select one benchmark. NEL 0.2.6 launches one Slurm job per benchmark; results are saved under `NEL_OUTPUT_DIR`.

## MMLU accuracy by subject

<details>
<summary>Show per-subject results</summary>

| Subject | Questions | Teacher (BF16) | PTQ (NVFP4) | PTQ → QAD (NVFP4) |
|---|---:|---:|---:|---:|
| Abstract algebra | 100 | 55.00% | 56.00% | 56.00% |
| Anatomy | 135 | 70.37% | 68.89% | 68.89% |
| Astronomy | 152 | 90.13% | 88.82% | 90.13% |
| Business ethics | 100 | 78.00% | 77.00% | 79.00% |
| Clinical knowledge | 265 | 78.49% | 80.38% | 79.62% |
| College biology | 144 | 87.50% | 86.81% | 87.50% |
| College chemistry | 100 | 56.00% | 59.00% | 56.00% |
| College computer science | 100 | 73.00% | 72.00% | 68.00% |
| College mathematics | 100 | 58.00% | 59.00% | 58.00% |
| College medicine | 173 | 79.19% | 78.03% | 78.03% |
| College physics | 102 | 64.71% | 59.80% | 63.73% |
| Computer security | 100 | 81.00% | 81.00% | 80.00% |
| Conceptual physics | 235 | 84.68% | 85.11% | 84.68% |
| Econometrics | 114 | 65.79% | 64.91% | 68.42% |
| Electrical engineering | 145 | 80.69% | 80.69% | 82.76% |
| Elementary mathematics | 378 | 78.84% | 75.40% | 76.19% |
| Formal logic | 126 | 67.46% | 66.67% | 60.32% |
| Global facts | 100 | 52.00% | 44.00% | 43.00% |
| High school biology | 310 | 90.97% | 91.29% | 91.61% |
| High school chemistry | 203 | 75.86% | 72.91% | 71.92% |
| High school computer science | 100 | 86.00% | 86.00% | 86.00% |
| High school European history | 165 | 85.45% | 85.45% | 87.27% |
| High school geography | 198 | 86.87% | 86.36% | 86.87% |
| High school government and politics | 193 | 93.26% | 92.23% | 93.26% |
| High school macroeconomics | 390 | 81.03% | 79.49% | 80.00% |
| High school mathematics | 270 | 53.70% | 50.00% | 51.85% |
| High school microeconomics | 238 | 93.70% | 91.60% | 92.02% |
| High school physics | 151 | 72.85% | 69.54% | 70.20% |
| High school psychology | 545 | 91.19% | 90.46% | 90.83% |
| High school statistics | 216 | 78.24% | 75.93% | 75.46% |
| High school US history | 204 | 86.27% | 85.29% | 83.82% |
| High school world history | 237 | 86.50% | 85.65% | 85.65% |
| Human aging | 223 | 74.44% | 75.34% | 73.99% |
| Human sexuality | 131 | 83.21% | 88.55% | 84.73% |
| International law | 121 | 73.55% | 73.55% | 73.55% |
| Jurisprudence | 108 | 81.48% | 83.33% | 79.63% |
| Logical fallacies | 163 | 82.21% | 80.37% | 82.21% |
| Machine learning | 112 | 64.29% | 64.29% | 65.18% |
| Management | 103 | 87.38% | 89.32% | 87.38% |
| Marketing | 234 | 93.59% | 92.74% | 91.88% |
| Medical genetics | 100 | 83.00% | 82.00% | 82.00% |
| Miscellaneous | 783 | 85.82% | 83.78% | 83.40% |
| Moral disputes | 346 | 76.01% | 72.25% | 73.99% |
| Moral scenarios | 895 | 55.42% | 53.85% | 53.85% |
| Nutrition | 306 | 81.05% | 81.70% | 80.07% |
| Philosophy | 311 | 77.81% | 74.28% | 76.85% |
| Prehistory | 324 | 83.95% | 82.72% | 83.02% |
| Professional accounting | 282 | 59.22% | 58.51% | 57.09% |
| Professional law | 1,534 | 51.17% | 49.02% | 50.26% |
| Professional medicine | 272 | 81.99% | 80.51% | 80.88% |
| Professional psychology | 612 | 78.59% | 76.31% | 76.14% |
| Public relations | 110 | 70.00% | 68.18% | 70.00% |
| Security studies | 245 | 79.59% | 80.00% | 80.41% |
| Sociology | 201 | 86.57% | 84.58% | 84.08% |
| US foreign policy | 100 | 84.00% | 83.00% | 85.00% |
| Virology | 166 | 56.02% | 52.41% | 53.61% |
| World religions | 171 | 87.13% | 85.38% | 84.21% |

</details>
