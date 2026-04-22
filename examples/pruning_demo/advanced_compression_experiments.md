# Advanced Compression Experiments: Results & Insights

This document extends the [main tutorial](README.md) with results and insights from more sophisticated experiments, addressing the open questions raised in Section 9.

---

## 1. Extended Distillation: WikiText vs. Nemotron-v2 at 80% Memory

The main tutorial uses a deliberately minimal distillation setup (100 iterations on WikiText-103, ~1.6M tokens). Here we investigate what happens when we scale up distillation significantly (using the higher-quality [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) with 1000x more tokens) on Qwen3-8B models compressed to ~80% of the original memory footprint.

### 1.1 Results across all benchmarks

| Model | Params | Distillation | Tokens | MMLU | HellaSwag acc_norm | GSM8K flex |
|---|---|---|---|---|---|---|
| Original Qwen3-8B | 8B | — | — | 0.7493 | 0.7653 | 0.8749 |
| | | | | | | |
| **Puzzletron 80%** | | | | | | |
| Puzzletron — pruned | 7.75B | — | — | 0.5910 | 0.6863 | 0.5762 |
| Puzzletron + WikiText | 7.75B | gbs=4, seq=4096, 100 iters | 1.6M | 0.6921 | 0.7390 | 0.7612 |
| **Puzzletron + Nemotron-v2** | **7.75B** | **gbs=768, seq=8192, 300 iters** | **1.9B** | **0.7186** | **0.7381** | **0.8378** |
| | | | | | | |
| **Minitron 80% (36→28 layers)** | | | | | | |
| Minitron — pruned | 6.65B | — | — | 0.5084 | 0.5295 | 0.0114 |
| Minitron + WikiText | 6.65B | gbs=4, seq=4096, 100 iters | 1.6M | 0.7302 | 0.6166 | 0.4761 |
| **Minitron + Nemotron-v2** | **6.65B** | **gbs=768, seq=8192, 300 iters** | **1.9B** | **0.7394** | **0.6357** | **0.7453** |

### 1.2 Key takeaways

**Nemotron-v2 improves both methods, but the gains are benchmark-dependent.** MMLU improvements are modest (+2.65pp for Puzzletron, +0.92pp for Minitron). The real payoff is on reasoning: Puzzletron's GSM8K jumps +7.66pp, Minitron's +26.92pp. Higher-quality distillation disproportionately recovers reasoning capabilities.

**Minitron + WikiText (1.6M tokens) still beats Puzzletron + Nemotron-v2 (1.9B tokens) on MMLU.** Minitron recovers to 97.5% of the teacher with minimal distillation, while Puzzletron needs 1000x more compute to reach 95.9%.

**On reasoning (GSM8K), Puzzletron leads regardless of distillation recipe.** With Nemotron-v2, Puzzletron retains 96.0% of the teacher vs. Minitron's 85.4%. Depth pruning's impact on reasoning can be partially compensated by better distillation, but heterogeneous pruning still preserves reasoning structure better.

**Distillation loss still doesn't predict downstream accuracy.** Minitron's final loss (5.59e-1) is 10x higher than Puzzletron + Nemotron-v2 (5.63e-2), yet Minitron scores better on MMLU.

---

## 2. Chaining Minitron Depth Pruning with Puzzletron

### 2.1 Motivation

The main tutorial uses Minitron and Puzzletron independently. A natural question is: can we combine them?

This is motivated by a limitation in Puzzletron's scoring for full layer removal: its independent block-level scoring does not account for inter-block dependencies when multiple layers are removed simultaneously, leading to poor layer selection and degraded quality.

| Method | Layers dropped (1-indexed) | Pre-distill MMLU | Post-distill MMLU |
|---|---|---|---|
| Minitron (BI scoring) | L27–L34 | 0.5084 | 0.7302 |
| Puzzletron (Cosine Embedding Loss) | L3–L4, L8–L9, L15, L19, L21, L27 | 0.2949 | 0.4993 |

> **Note:** To isolate depth pruning behavior, Puzzletron was configured to only allow full layer removal.

Minitron's BI scoring concentrates drops in late layers, producing a far better model. This motivates a chained approach: Minitron for depth pruning, then Puzzletron for heterogeneous width pruning on the surviving layers.

### 2.2 Experiment: Minitron 36→32L + Puzzletron → 80% memory

We first prune Qwen3-8B from 36 to 32 layers using Minitron (~10% reduction), then apply Puzzletron to the 32-layer model to reach the 80% memory target (~10% further reduction). We compare this chained approach against using each method alone at the same 80% memory target.

**Intermediate step — Minitron 36→32L (~90% memory)**

| Model | Params | Distillation | Tokens | MMLU | HellaSwag acc_norm | GSM8K flex |
|---|---|---|---|---|---|---|
| Qwen3-8B (teacher) | 8.19B | — | — | 0.7493 | 0.7653 | 0.8749 |
| Minitron 36→32L — pruned | 7.42B | — | — | 0.7396 | 0.6671 | 0.2873 |
| Minitron 36→32L + WikiText | 7.42B | gbs=4, seq=4096, 100 iters | 1.6M | 0.7421 | 0.6987 | 0.7604 |

Minitron's depth pruning retains 98.7% of MMLU with no distillation at all (0.7396), confirming that the 4 dropped late layers contribute little to general knowledge. GSM8K drops sharply (0.2873) but recovers well with minimal distillation (0.7604).

**80% memory target — all three approaches compared**

| Model | Params | Distillation | Tokens | MMLU | HellaSwag acc_norm | GSM8K flex |
|---|---|---|---|---|---|---|
| Qwen3-8B (teacher) | 8.19B | — | — | 0.7493 | 0.7653 | 0.8749 |
| | | | | | | |
| **Chained: Minitron 36→32L + Puzzletron** | | | | | | |
| Pruned | 7.42B | — | — | 0.6674 | 0.6698 | 0.6331 |
| + WikiText | 7.42B | gbs=4, seq=4096, 100 iters | 1.6M | 0.7074 | 0.6874 | 0.7081 |
| **+ Nemotron-v2** | **7.42B** | **gbs=768, seq=8192, 300 iters** | **1.9B** | **0.7332** | **0.7126** | **0.8499** |
| | | | | | | |
| **Puzzletron only** | | | | | | |
| Pruned | 7.75B | — | — | 0.5910 | 0.6863 | 0.5762 |
| + WikiText | 7.75B | gbs=4, seq=4096, 100 iters | 1.6M | 0.6921 | 0.7390 | 0.7612 |
| **+ Nemotron-v2** | **7.75B** | **gbs=768, seq=8192, 300 iters** | **1.9B** | **0.7186** | **0.7381** | **0.8378** |
| | | | | | | |
| **Minitron depth only (36→28L)** | | | | | | |
| Pruned | 6.65B | — | — | 0.5084 | 0.5295 | 0.0114 |
| + WikiText | 6.65B | gbs=4, seq=4096, 100 iters | 1.6M | 0.7302 | 0.6166 | 0.4761 |
| **+ Nemotron-v2** | **6.65B** | **gbs=768, seq=8192, 300 iters** | **1.9B** | **0.7394** | **0.6357** | **0.7453** |

### 2.3 Key takeaways

**The chained approach gives the best balanced results with extended distillation.** With Nemotron-v2, Minitron+Puzzletron achieves 0.7332 MMLU, 0.7126 HellaSwag, and 0.8499 GSM8K. No single method matches this balance: Minitron alone leads on MMLU (0.7394) but lags on HellaSwag (0.6357) and GSM8K (0.7453); Puzzletron alone leads on HellaSwag (0.7381) but trails on MMLU (0.7186).

**Chaining leverages each method's strength.** Minitron handles depth pruning cleanly (BI scoring correctly identifies which late layers to drop), then Puzzletron applies surgical per-layer width optimization on the surviving 32-layer model. The result is a model that preserves both general knowledge and reasoning better than either method alone.

**Pre-distillation quality is much higher for the chained approach.** The chained model starts at 0.6674 MMLU before any distillation — well above Puzzletron alone (0.5910) and Minitron alone (0.5084). This gives distillation more structure to work with.

**Conclusion:** On Qwen3-8B, for an 80% memory target, pruning ~10% with Minitron depth (36→32L) followed by ~10% with Puzzletron width, then applying extended distillation with Nemotron-v2, gives the best balanced trade-off across all benchmarks tested.

---

## 3. Blockwise Local Distillation (BLD)

BLD (bypass) locally trains block variants before the MIP assembly step, so the search prefers blocks that recover well after distillation rather than blocks that merely look good as immediate swaps.

### 3.1 At moderate compression (7B target): marginal impact

We tested BLD on the Scenario 1 setup (Qwen3-8B → 7B), applying it to FFN subblock variants pruned below 50% of the original intermediate size.

| Model | Parameters | MMLU (pruned) | MMLU (distilled) | % of Teacher |
|---|---|---|---|---|
| Minitron 7B | 6.96B | 0.7038 | 0.7166 | 95.6% |
| Puzzletron 7B | 6.99B | 0.6621 | 0.6823 | 91.1% |
| Puzzletron 7B + BLD | 6.99B | 0.6696 | 0.6867 | 91.6% |

BLD provides a marginal improvement over standard Puzzletron (+0.44pp post-distillation), and the MIP selects a very similar architecture. At this moderate compression level, the gain appears insufficient to justify the added complexity, and Minitron still leads on MMLU by a wide margin.

### 3.2 At aggressive compression (80% memory target): significant impact

A recurring pattern when optimizing for memory is that the MIP solver drops full attention blocks from many layers (since KV cache dominates memory). This means the FFN part of those attention-less variants becomes critical and is exactly where BLD can have the most impact. Here we apply BLD to train the FFN part of block variants that drop attention (`no_op`).

**Results (% of teacher, post-distillation with WikiText)**

| Benchmark | Puzzletron 80% | Puzzletron 80% + BLD | Minitron 80% |
|---|---|---|---|
| MMLU | 92.4% | **98.0%** | 97.5% |
| HellaSwag acc_norm | **96.6%** | 95.6% | 80.6% |
| GSM8K flex | 87.0% | **92.0%** | 54.4% |

**Full results**

| Model | MMLU (pruned) | MMLU (distilled) | HellaSwag acc_norm (pruned) | HellaSwag acc_norm (distilled) | GSM8K flex (pruned) | GSM8K flex (distilled) |
|---|---|---|---|---|---|---|
| Qwen3-8B (teacher) | 0.7493 | — | 0.7653 | — | 0.8749 | — |
| Puzzletron 80% | 0.5910 | 0.6921 | 0.6863 | 0.7390 | 0.5762 | 0.7612 |
| **Puzzletron 80% + BLD** | **0.7277** | **0.7341** | **0.7097** | **0.7317** | **0.7331** | **0.8044** |
| Minitron 80% | 0.5084 | 0.7302 | 0.5295 | 0.6166 | 0.0114 | 0.4761 |

BLD transforms Puzzletron's results at this compression level. The pre-distillation MMLU jumps from 0.5910 to 0.7277. After WikiText distillation, Puzzletron + BLD reaches 0.7341 MMLU, beating both standard Puzzletron (0.6921) and Minitron (0.7302) — flipping the Puzzletron vs. Minitron ranking on MMLU, where without BLD Minitron was ahead. The improvement is consistent across all benchmarks, with GSM8K showing a particularly strong gain (0.8044 vs. 0.7612 without BLD).

Unlike the moderate compression case where BLD had negligible impact, at aggressive compression BLD substantially changes the architecture the MIP selects and the quality of the resulting model.

---

## 4. Beyond Dense Transformers: Compressing a Mamba-Transformer Hybrid

All experiments so far use Qwen3-8B, a dense Transformer-only model. Here we test both methods on [Nemotron-Nano-12B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2), a **Mamba-Transformer hybrid** with 12.3B parameters and 62 layers (alternating Mamba and attention blocks). This is an early exploration; results are pre-distillation only (MMLU).

### 4.1 Results

| Model | MMLU | % of Teacher |
|---|---|---|
| Nemotron-Nano-12B-v2 (baseline, 49k MiB) | 78.6 | 100% |
| | | |
| **~10B parameter target** | | |
| Minitron 10B | **73.7** | **93.8%** |
| Puzzletron 10B | 48.9 | 62.2% |
| | | |
| **~34k MiB memory target** | | |
| Minitron 34k | 51.8 | 65.9% |
| Puzzletron 34k | **54.3** | **69.1%** |

### 4.2 Observations

**Puzzletron never removes Mamba blocks.** Across all Puzzletron runs (both 10B and 34k MiB targets), every Mamba block is kept intact: the MIP solver exclusively targets attention blocks and FFN layers for pruning. This suggests that removing a single Mamba block is too costly for model quality.

**At moderate compression (~10B), Minitron dominates.** Minitron retains 93.8% of teacher MMLU vs. Puzzletron's 62.2%. This is consistent with the Qwen3-8B pattern where Minitron wins at moderate compression, but the gap is much larger here.

**At aggressive compression (~34k MiB), Puzzletron slightly leads.** Puzzletron edges ahead (54.3 vs. 51.8 MMLU), similarly to the pattern observed on Qwen3-8B. 

**Hybrid architectures present unique challenges for Puzzletron.** On dense Transformers, Puzzletron's strength is heterogeneous per-layer optimization. On hybrids, the Mamba blocks are effectively frozen — Puzzletron can only optimize the attention/FFN half of the model. This reduces its effective search space and may explain why Minitron's simpler uniform approach outperforms at moderate compression levels.

---
