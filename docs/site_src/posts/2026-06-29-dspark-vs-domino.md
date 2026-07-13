---
title: "DSpark vs Domino: Same DFlash Backbone, Different Correction Heads"
author: ModelOpt Team
date: 2026-06-29
tags: [speculative-decoding, dflash, dspark, domino, architecture]
summary: "DSpark (DeepSpec) and Domino both build on block-parallel DFlash draft generation but diverge sharply in their token-level correction heads. DSpark uses a stateless VanillaMarkov head (fast, parallelizable); Domino uses a GRU (expressive, sequential). This post walks through the architecture diagram, the checkpoint weight anatomy, and when each design wins."
highlights:
  - "Both systems share the DFlash block-parallel backbone — nearly identical parallel draft throughput"
  - "DSpark defaults to VanillaMarkov: stateless W1/W2 embedding lookups, no hidden state to thread through"
  - "Domino uses nn.GRU: token embeddings as input, draft hidden z_i concatenated at readout — true RNN"
  - "Both correction heads are sequential at inference (x_{k-1} must be sampled before step k); DSpark's per-step cost is an O(1) embedding lookup vs Domino's full GRU cell"
  - "DSpark adds a hardware-aware prefix scheduler (confidence_head) absent from Domino; vLLM does not plan to support this feature"
  - "For dense models, DSpark simplifies to backbone + Markov head only — no mHC, no hc_head"
---

> **New to DFlash?** This post assumes familiarity with the DFlash block-parallel draft backbone. For a thorough introduction see [**DFlash: Block-Parallel Speculative Decoding for Nemotron**](/blog/dflash-presentation-for-nemotron-summit-diffusion-llm-sectio/) — a good starting point before diving into the comparison below.

Both [DSpark (DeepSpec)](https://github.com/deepseek-ai/DeepSpec) and [Domino](https://github.com/jianuo-huang/Domino) are speculative decoding systems that pair a block-parallel draft backbone with a sequential correction head. From a distance they look identical. Up close, the correction heads are fundamentally different — and that difference drives the latency, complexity, and hardware tradeoffs.

<img src="/tools/dspark_fig1.png" alt="DSpark Figure 1: overall architecture and decoding cycle" style="width:100%; margin-top:1.2em; margin-bottom:1.2em" />

## Shared foundation: DFlash block-parallel backbone

Both systems use **DFlash**: a draft backbone that runs a single causal attention forward pass over all γ draft positions in parallel, producing:

- **h₁…h_γ** — per-position backbone hidden states
- **U₁…U_γ** — base draft logits

This is the expensive step. Everything below is cheap correction on top of these parallel outputs.

## Where they diverge: the correction head

```
                          ┌──────────────────────────────────────────────────────────────────────────────────────────┐
                          │                                  DFlash Parallel Pass                                    │
                          │   x_0 ──► backbone ──► h_1, U_1                                                        │
                          │   x_0 ──► backbone ──► h_2, U_2                                                        │
                          │    …                    …                                                               │
                          │   x_0 ──► backbone ──► h_γ, U_γ                                                        │
                          └──────────────────────────────────────────────────────────────────────────────────────────┘
                                                                       │
                                       ┌───────────────────────────────┴───────────────────────────────┐
                                       │                                                               │
              ┌────────────────────────▼───────────────────────┐              ┌────────────────────────▼───────────────────────┐
              │         DSpark — VanillaMarkov                  │              │              Domino — GRU                      │
              │                                                 │              │                                                │
              │  For k = 1..γ:                                  │              │  For k = 1..γ:                                 │
              │                                                 │              │                                                │
              │  e_{k-1} = W1[x_{k-1}]                         │              │  e_{k-1} = Emb[x_{k-1}]                       │
              │                                                 │              │                                                │
              │  bias_k  = W2 · e_{k-1}                        │              │  gru_h_k = GRU(e_{k-1},                       │
              │                                                 │              │                gru_h_{k-1})                   │
              │  p_k = softmax(U_k + bias_k)                   │              │                                                │
              │  x_k ~ p_k                                      │              │  p_k = softmax(U_k + W·[h_k;                  │
              │                                                 │              │                        gru_h_k])              │
              │  (depends on x_{k-1} only —                    │              │  x_k ~ p_k                                    │
              │   no hidden state carried forward)              │              │                                                │
              │                                                 │              │  (full prefix history                         │
              │                                                 │              │   accumulated in gru_h_{k-1})                 │
              └─────────────────────────────────────────────────┘              └────────────────────────────────────────────────┘
```

### DSpark — VanillaMarkov (first-order Markov head)

The correction is a **stateless** first-order Markov transition. For each draft position k:

```
e_{k-1} = W1[x_{k-1}]          # embedding lookup: previous token only
bias_k   = W2 · e_{k-1}        # transition bias (no hidden state)
p_k      = softmax(U_k + bias_k)
x_k      ~ p_k
```

`W1 ∈ ℝ^{V×r}` and `W2 ∈ ℝ^{V×r}` (r=512, V=129280 for DeepSeek-V4-Pro). The correction at position k depends **only on x_{k-1}** — no RNN hidden state threads across steps. This means the dominant computation (W1 embedding lookup + W2 projection) is a simple table lookup, not a recurrent rollout.

DSpark also defines `GatedMarkov` and `RNNHead` as drop-in alternatives, but the released production checkpoint uses VanillaMarkov (confirmed in DSpark Section 4.3.2: *"we use the Markov head as the default"*).

### Domino — GRU correction head

Domino uses `nn.GRU` with **token embeddings as input** and the backbone draft hidden state z_i concatenated at readout:

```
gru_h_k  = GRU(Emb[x_{k-1}], gru_h_{k-1})   # GRU: carries hidden state
p_k      = softmax(U_k + W · [h_k; gru_h_k])
x_k      ~ p_k
```

The GRU accumulates a hidden state `gru_h_{k-1}` that carries information about the full prefix x_0…x_{k-1}. This is more expressive than VanillaMarkov — the correction can condition on the entire draft history, not just the immediately preceding token.

<img src="/tools/domino_fig.png" alt="Domino pipeline: DFlash backbone + GRU causal correction head" style="width:100%; margin-top:1.2em; margin-bottom:1.2em" />

The figure below shows training acceptance length (AL) on Qwen3-8B across three systems — DFlash baseline, Domino GRU, and our DSpark implementation. Both correction heads lift AL above the parallel backbone baseline, with the causal dependency injected by the correction head becoming visible as training progresses.

<img src="/tools/dspark_domino_al_qwen3_8b.png" alt="Training acceptance length on Qwen3-8B: DFlash baseline vs Domino GRU vs DSpark" style="width:100%; margin-top:1.2em; margin-bottom:1.2em" />

## Correction head overhead: per-step cost

Both correction heads are **strictly sequential at inference** — computing step k requires sampling x_{k-1} first, regardless of whether there is a recurrent hidden state. The difference is how expensive each step is:

| System | Per-step compute | State carried |
|---|---|---|
| DSpark VanillaMarkov | Two embedding lookups: `W1[x_{k-1}]` + `W2 · e_{k-1}` | None — stateless |
| Domino GRU | Full GRU cell: input gate + reset gate + new gate over 4096-dim input, 1024-dim hidden | `gru_h_{k-1}` — 1024-dim hidden state |

DSpark's Markov head adds only **0.2–1.3% latency** over DFlash at batch=128 (DSpark paper Figure 4) precisely because each step is an O(1) table lookup. Domino's GRU cell is heavier per step, though it remains small relative to the backbone cost. During **training** both can be parallelized via teacher forcing; the sequential constraint only bites at inference.

## Additional DSpark machinery

DSpark ships two components that Domino omits entirely:

### Hardware-aware prefix scheduler (confidence_head)

Unique to the **final MTP layer (mtp.2)**:

```
c_k = σ(w^T [hc_head(h_k); W1[x_{k-1}]])    # per-position acceptance probability
```

`hc_head` is a small 4-component calibration module that transforms `h_k` before the scalar confidence projection. The scheduler (Algorithm 1) uses `c_k` to dynamically pick how many draft tokens to submit for verification per request, maximizing throughput `Θ = τ · SPS(B)` across all concurrent requests.

**If you disable the prefix scheduler, `confidence_head` and `hc_head` are never called.** Offline benchmarking always submits all γ tokens — the scheduler is a serving-time optimization. Notably, vLLM does not plan to support the hardware-aware prefix scheduler, so `confidence_head` and `hc_head` are effectively unused in vLLM-based deployments.

### Manifold-constrained Hyper-Connections (mHC)

For MoE models, the backbone transformer layers carry `hc_attn_*` and `hc_ffn_*` weights — the mHC parameters (Xie et al., 2026) applied to attention and FFN sublayers respectively. These are always active in the backbone forward pass.

**For dense models, mHC is absent.** The `hc_attn_*` / `hc_ffn_*` weights are MoE-specific — they constrain connections across expert routing paths. A dense draft backbone (standard FFN layers) has no analog and would not carry these weights.

## Checkpoint anatomy

### Domino — `Qwen3-8B-Domino-b16`

Reading the [released checkpoint](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16) reveals the exact forward pass (DFlash backbone layers excluded):

| Weight | Shape | Derivation |
|---|---|---|
| `fc.weight` | [4096, 20480] | 20480 = 5 × 4096: fuses all 5 backbone layer outputs (concatenated) → z_k ∈ ℝ^{4096} |
| `hidden_norm.weight` | [4096] | LayerNorm on z_k before GRU |
| `prefix_gru.weight_ih_l0` | [3072, 4096] | GRU input gate: input_size=4096 (model dim), 3×hidden=3072 → hidden_size=1024 |
| `prefix_gru.weight_hh_l0` | [3072, 1024] | GRU hidden gate: confirms hidden_size=1024 |
| `embed_proj.0.weight` | [256, 5120] | 5120 = 4096 + 1024 = [z_k ; gru_h_k], projects to bottleneck 256 |
| `embed_proj.2.weight` | [151936, 256] | bottleneck → vocab (151936 = Qwen3-8B vocab), produces Δlogit_k |
| `norm.weight` | [4096] | LayerNorm on final output |

The full correction pass for each draft position k:

```
z_k      = fc(concat(h_k^{(1)}, ..., h_k^{(5)}))   # fuse 5 backbone layers
z_k      = hidden_norm(z_k)
gru_h_k  = GRU(z_k, gru_h_{k-1})                   # hidden_size=1024, carries prefix history
Δlogit_k = embed_proj([z_k; gru_h_k])               # [5120] → 256 → 151936
p_k      = softmax(U_k + Δlogit_k)
x_k      ~ p_k
```

Two details are worth noting. First, the GRU input is `z_k` (the fused backbone hidden state), **not** a token embedding — which differs from the simplified description in the Domino README. Second, the `fc` fusion layer is what gives the GRU richer per-position context than a single layer's output.

### DSpark — `DeepSeek-V4-Pro-DSpark`

The DeepSeek-V4-Pro-DSpark checkpoint illustrates the full weight structure:

| Weight | Shape | Role | Required? |
|---|---|---|---|
| `mtp.{i}.hc_attn_fn` | [24, 28672] | mHC on attention (MoE only) | Always (MoE) |
| `mtp.{i}.hc_ffn_fn` | [24, 28672] | mHC on FFN (MoE only) | Always (MoE) |
| `mtp.{i}.markov_head.markov_w1` | [129280, 512] | Markov embedding W1 | Always |
| `mtp.{i}.markov_head.markov_w2` | [129280, 512] | Markov projection W2 | Always |
| `mtp.2.hc_head_fn` | [4, 28672] | Calibration for confidence (last layer only) | Scheduler only |
| `mtp.2.confidence_head.proj` | [1, 7680] | Scalar confidence predictor | Scheduler only |

For a **dense model without the prefix scheduler**, only `markov_w1` and `markov_w2` are active.

## Production results (DSpark)

<img src="/tools/dspark_fig7.png" alt="DSpark Figure 7: throughput vs. TPS Pareto frontier" style="width:100%; margin-top:1.2em; margin-bottom:1.2em" />

## Takeaways

1. **Same backbone, different correction heads.** DFlash draft generation is shared — if your baseline is the backbone, both systems are comparable. The correction head is the differentiator.

2. **VanillaMarkov is cheaper per step than GRU — both are sequential.** Both heads must unroll left-to-right at inference (each step needs the previous sample). The difference is per-step cost: VanillaMarkov is two embedding lookups; Domino's GRU runs a full recurrent cell over a 4096-dim input with 1024-dim hidden state. At scale this per-step gap compounds across the draft block.

3. **GRU is more expressive but the gain is marginal in practice.** DSpark's ablation (Section 4.3.2) explicitly validates that VanillaMarkov matches or beats RNNHead at the same draft quality. The extra expressiveness of a full RNN rarely translates to meaningfully higher acceptance rates on standard benchmarks.

4. **DSpark's design is more general.** The pluggable Markov-head family (VanillaMarkov → GatedMarkov → RNNHead) means Domino's GRU approach is essentially one point in DSpark's design space — a specific `RNNHead` variant with embedding-only input. DSpark explored and explicitly rejected it as the default.

5. **The prefix scheduler is a serving-time feature, not a draft-quality feature — and vLLM won't support it.** `confidence_head` and `hc_head` have zero effect on draft acceptance rate — they route how many tokens you *submit* for verification. vLLM does not plan to implement the hardware-aware prefix scheduler, so in practice these weights are dead weight for vLLM-based deployments. Disable the scheduler and you lose throughput optimization, not correctness.

6. **Dense models get the simple version.** No mHC, no `hc_head` — just backbone + Markov head. The MoE-specific weights (`hc_attn_*`, `hc_ffn_*`, `hc_head_*`) are architectural additions for a specific deployment that happen to be visible in the DeepSeek-V4-Pro checkpoint.

## Links

- [DeepSpec / DSpark repo](https://github.com/deepseek-ai/DeepSpec) — model code + paper
- [DeepSeek-V4-Pro-DSpark checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) — released MoE checkpoint with VanillaMarkov + mHC weights
- [Domino repo](https://github.com/jianuo-huang/Domino) — GRU-based correction head implementation
- [Domino checkpoint: Qwen3-8B-Domino-b16](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16) — SpecForge drafter format
- [ModelOpt PR #1710](https://github.com/NVIDIA/Model-Optimizer/pull/1710) — Domino training implementation (DFlash backbone + GRU correction head, dual loss curriculum)
- [Our DFlash reproduction on Qwen3.5-4B](/blog/dflash-qwen35-4b-reproduction) — training results and AL curves

## Citations

```bibtex
@misc{dspark2026,
  title  = {DSpark: Accelerating Large Language Model Inference with Speculative Decoding},
  author = {DeepSeek-AI},
  year   = {2026},
  url    = {https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf}
}

@misc{domino2026,
  title  = {Domino: Accelerating Speculative Decoding with a Block-Parallel DFlash Backbone and GRU Correction Head},
  author = {Huang, Jianuo and others},
  year   = {2026},
  url    = {https://github.com/jianuo-huang/Domino}
}

@misc{xie2026mhc,
  title  = {Manifold-Constrained Hyper-Connections},
  author = {Xie and others},
  year   = {2026}
}
```
