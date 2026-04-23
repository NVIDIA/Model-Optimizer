# Megatron-Bridge Distillation Results

Experiment results for knowledge distillation via Megatron-Bridge after structured pruning. Two pruning algorithms are covered:

| Algorithm | Experiment | Summary |
| --- | --- | --- |
| [Minitron](minitron/NVIDIA-Nemotron-Nano-9B-v2/README.md) | Nemotron-Nano-9B-v2 → Pruned 7B | Structured pruning of Nemotron-Nano-9B-v2 to 7B, distilled up to 80B tokens. Achieves near-parity with the official 9B across MMLU, MMLU Pro, GPQA, LCB, AIME, Math 500, IFEval, and SciCode. |
| [Puzzletron](puzzletron.md) | Qwen3-8B → 80%; Llama-3.1-8B-Instruct → 50%/69% | Compression using Puzzletron followed by short distillation runs (100 iters on WikiText-103). Shows MMLU recovery and illustrates dataset size requirements — small datasets can cause overfitting and regression. |

## Shared Resources

- **[MEGATRON_DATA_PREP.md](../../dataset/MEGATRON_DATA_PREP.md)** — Tokenization commands for all datasets used in distillation experiments (Nemotron Pre/Post-Training collections).
