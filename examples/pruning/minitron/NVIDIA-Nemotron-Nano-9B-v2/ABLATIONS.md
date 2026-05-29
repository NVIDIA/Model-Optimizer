# Distillation Blend Ablations

All experiments prune Nemotron-Nano-9B-v2 → 7B and distill with teacher = Nemotron-Nano-9B-v2 (official). The final chosen blend (**30pre_70post_v1v3**) is in [README.md](README.md).

---

## Baseline: Pre-SFT-v1 Only (no post-training data)

Pure Nemotron-Pretraining-SFT-v1 data only (no post-training reasoning traces).

| Tokens | MMLU | MMLU Pro | GPQA Diamond | LCB v6 | AIME 2025 | Math 500 | IFEval | SciCode |
|---|---|---|---|---|---|---|---|---|
| 19B | 72.7 | 70.5 | 53.9 | 58.8 | 63.4 | 94.4 | 57.9 | 19.2 |
| 56B | 73.3 | 71.9 | 54.3 | 62.0 | 63.8 | 95.0 | 58.7 | 17.9 |

**Notes:** Highest MMLU of any blend, but AIME stagnates and LCB lags. Pretraining data alone insufficient for reasoning benchmarks.

---

## Baseline: Pure Post-Training Data (pt-v1v2)

100% post-training data (no pretraining data), Nemotron-v1/v2 blend.

| Tokens | MMLU | MMLU Pro | GPQA Diamond | LCB v6 | AIME 2025 | Math 500 | IFEval | SciCode |
|---|---|---|---|---|---|---|---|---|
| 2.5B | 71.0 | 69.3 | 52.6 | 54.8 | 58.2 | 94.1 | 51.7 | 14.4 |
| 5B | 70.8 | 70.7 | 53.6 | 57.2 | 63.8 | 94.1 | 50.5 | 14.2 |
| 20B | 69.8 | 71.7 | 54.7 | 57.5 | 64.7 | 94.6 | 41.9 | 13.4 |
| 40B | 70.0 | 71.7 | 53.2 | 57.4 | 67.6 | 95.2 | 43.3 | 16.2 |

**Notes:** IFEval degrades badly at longer training (41.9 at 20B). LCB lags behind other blends.

---

## 30% Pretraining / 70% Post-Training: v1v2 Blend

30% Nemotron-Pretraining-SFT-v1 + 70% Nemotron-v1/v2 post-training data.

| Tokens | MMLU | MMLU Pro | GPQA Diamond | LCB v6 | AIME 2025 | Math 500 | IFEval | SciCode |
|---|---|---|---|---|---|---|---|---|
| 2.5B | 71.9 | 68.9 | 49.8 | 56.4 | 55.3 | 93.3 | 58.2 | 14.6 |
| 5B | — | — | — | — | — | — | — | — |
| 20B | 71.6 | 71.2 | 52.7 | 58.0 | 65.1 | 94.0 | 55.7 | 14.2 |
| 40B | 72.7 | 71.1 | 54.0 | 59.7 | 65.5 | 95.2 | 53.8 | 19.2 |
| 60B | 73.0 | 71.9 | 55.9 | 60.0 | 67.8 | 95.4 | 56.4 | 21.7 |
| 80B | 73.4 | 72.7 | 54.7 | 61.8 | 70.7 | 95.3 | 57.8 | 19.9 |
| 100B | 73.5 | 72.8 | 56.4 | 62.4 | 71.9 | 95.8 | 59.1 | 19.4 |

**Notes:** Best MMLU of the 30/70 blends (~1% above v3 blends). IFEval ~56–59 (lower than v3 blends). GPQA shows instability at longer runs.

---

## 30% Pretraining / 70% Post-Training: v3 Blend

Refined v3 blend: dropped exercism/text2sql, added Nemotron-Math-v2 part01, boosted Math to 30% total.

| Tokens | MMLU | MMLU Pro | GPQA Diamond | LCB v6 | AIME 2025 | Math 500 | IFEval | SciCode |
|---|---|---|---|---|---|---|---|---|
| 2.5B | 70.5 | 69.0 | 51.2 | 59.1 | 62.9 | 94.3 | 62.2 | 11.6 |
| 5B | 71.0 | 69.8 | 53.0 | 59.4 | 65.0 | 94.4 | 66.8 | 20.3 |
| 20B | 71.2 | 70.8 | 53.3 | 60.0 | 69.1 | 95.3 | 63.8 | 22.6 |
| 40B | 71.0 | 71.7 | 54.0 | 62.3 | 71.3 | 95.3 | 66.8 | 17.9 |
| 60B | 72.0 | 72.3 | 56.3 | 62.0 | 71.6 | 95.6 | 65.5 | 21.5 |
| 80B | 72.3 | 73.0 | 53.9 | 63.0 | 72.4 | 96.2 | 65.5 | 21.3 |

**Notes:** Better AIME and LCB than blend 1 at 40B+. GPQA still unstable (53.9 at 80B). MMLU ~1% below v1v2 blend.

---

## Blend Design Notes

**Why MMLU is ~1% lower with v3 blends:** The heavy reasoning-trace format (chain-of-thought, TIR) in v3 data suppresses general knowledge recall measured by MMLU. This is structural — v1v2 post-training data has a more knowledge-dense format. Upweighting Pretraining-SFT-v1 General (to 20%) partially mitigates this. Given that MMLU Pro is better with v3 blends, lower MMLU is acceptable.

**Why GPQA is unstable in blend 1:** Science-v1 MCQ (497M tokens) and RQA (278M tokens) are repeated ~14× over 100B training steps, causing overfitting to MCQ format. Fix in v1v3: add Nemotron-Post-Training-Dataset-v1 STEM (~60B tokens, ~0.13 epochs at 80B) as primary science source; reduce Science-v1 to low weights (3+2) for format alignment only.

**Why 80B is the recommended stopping point:** SciCode degrades or crashes at 100B (blend2: 1.6; AIME also degrades). Best overall profile is at 60–80B tokens.
