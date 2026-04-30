# Puzzletron Distillation Results

The following MMLU results demonstrate knowledge distillation on student models that were first compressed using [Puzzletron](../../puzzletron/README.md). The original (uncompressed) model serves as the teacher, and distillation recovers accuracy lost during compression.

## Qwen3-8B compressed to 80% of original

The student was created by compressing Qwen3-8B to 80% of its original size using Puzzletron.

| Model | MMLU | Humanities | Other | Social Sci | STEM |
|-------|------|------------|-------|------------|------|
| Student (before distillation) | 0.5910 | 0.5046 | 0.6363 | 0.6831 | 0.5855 |
| Student (after distillation) | 0.6921 | 0.5906 | 0.7316 | 0.7975 | 0.7016 |
| Teacher (original Qwen3-8B) | 0.7493 | 0.6648 | 0.7856 | 0.8385 | 0.7526 |

MMLU accuracy improved from 59.10% to 69.21% (+10.11 pp) after distillation with just 100 iterations on WikiText-103, recovering 64% of the gap to the teacher model.

## Llama-3.1-8B-Instruct compressed to 50% of original

The student was created by compressing Llama-3.1-8B-Instruct to 50% of its original size using Puzzletron.

| Model | MMLU | Humanities | Other | Social Sciences | STEM |
|-------|------|------------|-------|-----------------|------|
| Student (before distillation) | 0.2316 | 0.2462 | 0.2292 | 0.2250 | 0.2274 |
| Student (after distillation) | 0.2960 | 0.3146 | 0.3085 | 0.2925 | 0.2768 |
| Teacher (original Llama-3.1-8B-Instruct) | 0.6839 | 0.7231 | 0.7038 | 0.7667 | 0.5911 |

## Llama-3.1-8B-Instruct compressed to 69% of original (regression)

The student was created by compressing Llama-3.1-8B-Instruct to ~69% of its original size using Puzzletron. This example shows regression due to overfitting on the small WikiText-103 dataset (100 iterations). MMLU was evaluated on a subset of 100 samples per task:

| Model | MMLU | Humanities | Other | Social Sciences | STEM |
|-------|------|------------|-------|-----------------|------|
| Student (before distillation) | 0.6626 | 0.7069 | 0.6892 | 0.7525 | 0.5574 |
| Student (after distillation) | 0.6496 | 0.6862 | 0.6677 | 0.7433 | 0.5532 |
| Teacher (original Llama-3.1-8B-Instruct) | 0.6839 | 0.7231 | 0.7038 | 0.7667 | 0.5911 |

MMLU decreased from 66.26% to 64.96% (-1.30 pp) -- the model overfitted to WikiText-103. This highlights the importance of using larger, more diverse datasets for distillation.

## Recommendations

- **Use larger datasets** for production distillation (e.g., [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1)) to avoid overfitting as shown in the regression case above.
- **Train for more iterations** to ensure proper convergence.
