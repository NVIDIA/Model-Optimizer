# Ablations: Nemotron-3-Nano-30B-A3B-BF16

> [!NOTE]
> Every benchmark table in this document is from evaluations run **without** eval-time tool calling (no Python sandbox). The [main README](README.md#results) reports both with-tools and no-tools numbers for GPQA and AIME. Consequently, the gains attributed to the `tool_calling` training data in the [data-blend ablation](#effect-of-data-blend-tool_calling) reflect **training-time** exposure to agentic/function-call reasoning, not tools invoked during evaluation.

## Pruning

> [!NOTE]
> The search space analysis below is specific to Nemotron hybrid (Mamba + Attention + MoE) models. Standard transformers expose only layers/hidden/attention/FFN dimensions, but these models add Mamba-specific dimensions (`mamba_num_heads`, `mamba_head_dim`) and MoE dimensions (`num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`). The resulting search space is significantly larger, and the default 40% width / 20% depth constraints include many dead-zone architectures that waste scoring compute. The tighter constraints recommended here were derived from this analysis.

- Score function: `mmlu_10pct_bs32` (zero-shot MMLU on 10% subset, no distillation applied)
- Candidates analyzed: ~50 across multiple NAS runs, all with `--prune_target_active_params 3e9`
- Random baseline (5-way MMLU): ~0.25
- Search space note: `num_attention_heads` was skipped in all runs.

### Key Findings Summary

| Dimension | Good range | Avoid | Key finding |
| --- | --- | --- | --- |
| `num_layers` | ≥ 48 | ≤ 46 | 42L fails universally; 46L avg MMLU 0.261 |
| `mamba_state_dim` | ≥ 3072 (56×56 or 64×64) | < 3072; asymmetric pairs (56×64) | Symmetric reduction — both heads and head_dim must shrink together |
| `hidden_size` | 2304–2560 | 2688 (original); 2048 | Original hidden_size is actively harmful when other dims are pruned |
| MoE dims | experts 96–128; shared 3072–3712 | experts = 88 | Weak independent signal after controlling for dominant dims |

---

### Original Model Dimensions

Params: 31.6B, Active: 3.6B

| Dimension | Value |
| --- | --- |
| `num_hidden_layers` | 52 |
| `hidden_size` | 2688 |
| `mamba_num_heads` | 64 |
| `mamba_head_dim` | 64 |
| `num_moe_experts` | 128 |
| `moe_ffn_hidden_size` | 1856 |
| `moe_shared_expert_intermediate_size` | 3712 |

---

### Top Candidates (best seen across all runs)

All candidates below have `active_params = 3.00B`.

| Score | Layers | Hidden | Heads | HeadDim | Experts | FFN | Shared | Total Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0.4783** | 52 | 2304 | 64 | 64 | 104 | 1856 | 3072 | 22.3B |
| **0.4727** | 52 | 2560 | 64 | 64 | 80 | 1280 | 3712 | 14.2B |
| **0.4650** | 48 | 2560 | 56 | 56 | 112 | 1792 | 3072 | 25.4B |
| **0.4608** | 52 | 2304 | 64 | 64 | 96 | 1856 | 3072 | 20.7B |
| **0.4329** | 48 | 2560 | 56 | 56 | 104 | 1792 | 3072 | 23.7B |
| 0.4119 | 50 | 2304 | 64 | 64 | 80 | 1856 | 3712 | 17.6B |
| 0.3762 | 52 | 2560 | 48 | 64 | 96 | 1536 | 3712 | 19.3B |

Two architecture families dominate the top results — see [Design Recipe](#design-recipe) below.

---

### Dimension Sensitivity

Sensitivity is ranked by strength of signal across all candidates.

#### 1. `mamba_state_dim` = `mamba_num_heads × mamba_head_dim` — strongest width signal

<details>
<summary>mamba_state_dim sensitivity: threshold ≥3072, best families 56×56 and 64×64 (click to expand)</summary>

Analyzing num_heads and head_dim jointly is more predictive than either alone:

| state_dim | Formula | Avg MMLU | Max MMLU | n |
| --- | --- | --- | --- | --- |
| 1920 | 48×40 | 0.254 | 0.257 | 2 |
| 2304 | 48×48 | 0.249 | 0.260 | 8 |
| 2688 | 48×56 or 56×48 | 0.264 | 0.310 | 7 |
| 3072 | 48×64 | 0.342 | 0.376 | 3 |
| **3136** | **56×56** | **0.400** | **0.465** | 4 |
| 3584 | 56×64 or 64×56 | 0.261 | 0.340 | 9 |
| **4096** | **64×64** | **0.399** | **0.478** | 7 |

**Threshold:** `state_dim ≥ 3072` to escape the near-random zone. Below 3072, no candidate has ever exceeded 0.31 regardless of other settings. The two reliable good families are symmetric configurations: **56×56 = 3136** and **64×64 = 4096** (original).

**Why 3584 has a poor average despite being large:** All 3584 candidates in the data are at 46L (depth-limited). The low average is a depth confound, not an inherent failure of 3584.

**Asymmetric reductions hurt:** Reducing only one of {num_heads, head_dim} while keeping the other at 64 (giving 3584) performs worse than symmetric reduction of both to 56 (giving 3136). The 56×56 pattern is consistently more reliable.

**`head_dim=48` is uniformly bad:** Across all candidates with head_dim=48, every single one scored 0.240–0.260. This holds across varying layers (48L–52L) and hidden sizes. head_dim=48 is the effective lower bound under 30% width pruning and it is never viable.

</details>

---

#### 2. `num_layers` — hard lower bound

<details>
<summary>num_layers sensitivity: hard floor at 48L, 42L universally fails (click to expand)</summary>

| Layers | Avg MMLU | Max MMLU | n |
| --- | --- | --- | --- |
| 42 | 0.232 | 0.234 | 7 |
| 46 | 0.261 | 0.340 | 12 |
| 48 | 0.409 | 0.465 | 4 |
| 50 | 0.257 | 0.412 | 9 |
| 52 | 0.336 | 0.478 | 19 |

**42L is a universal failure** — 7/7 candidates at 42L scored near-random, with no other dimension able to compensate. Eliminated by the 15% depth constraint.

**46L is still suboptimal** — avg 0.261, no candidate above 0.340. The effective floor for good performance is **48L**. The 15% depth constraint (min 45L) is correct but 46L candidates still appear in results and are reliably mediocre.

**50L avg is pulled down by head_dim=48 candidates** — when controlling for head_dim≥56, 50L performs comparably to 52L. The 50L failure is a head_dim confound, not a genuine depth issue.

</details>

---

#### 3. `hidden_size` — bad at both extremes, joint constraint with depth

<details>
<summary>hidden_size sensitivity: 2304–2560 good, original 2688 consistently bad (click to expand)</summary>

| hidden_size | Avg MMLU | Max MMLU | n |
| --- | --- | --- | --- |
| 2304 | 0.445 | 0.478 | 4 |
| 2560 | 0.307 | 0.473 | 26 |
| **2688** | **0.258** | 0.268 | 10 |

**`hidden_size=2688` (the original) is definitively bad in pruned configurations.** This is confirmed by candidates at 52L with hidden=2688 — sufficient depth — all scoring 0.255–0.260. It is not a depth confound. Keeping hidden size un-pruned while reducing other dimensions means the active param budget is consumed inefficiently, leaving too little capacity in the MoE and Mamba layers.

**`hidden_size=2304` requires `num_layers ≥ 48`.** When paired with 42L, it scores 0.232. Paired with 48–52L, it produces the best candidates. This is a joint constraint.

**`hidden_size=2048`** (seen in runs without an active param constraint): consistently undershoots the 3B active target, capping MMLU at ~0.30. Not a viable option.

</details>

---

#### 4. `num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size` — weak independent signal

<details>
<summary>MoE dimension sensitivity: weak signal, experts=88 bad, shared 3072–3712 preferred (click to expand)</summary>

No strong monotonic pattern after controlling for the dominant dimensions above.

- `experts=88`: consistently bad (avg 0.260)
- `moe_shared_expert_intermediate_size` in 3072–3712: preferred; 2560 and 3328 tend to be worse
- `moe_ffn_hidden_size=1280`: produced the most parameter-efficient good architecture (0.4727, 14.15B total) but is a 31% reduction from the original 1856 — just outside the 30% width constraint. Use **32% width** to include this family if total-param efficiency matters.

</details>

---

### Pruning Constraint Recommendations

```bash
--max_depth_pruning 0.15          # eliminates 42L dead zone; no good arch needs >7 layers removed
--max_width_pruning 0.30          # eliminates head_dim≤40, shared=2560; 0.32 to also include ffn=1280
--prune_target_params 28e9        # not very critical constraint as active params are the primary target
--prune_target_active_params 3e9  # required; omitting this causes active params to undershoot, capping MMLU at ~0.30
--hparams_to_skip num_attention_heads
```

**Remaining dead zones within 15%/30% search space** — the following are still reachable by the NAS but consistently fail:

- `hidden_size=2688` — all 4 candidates at 52L scored 0.255–0.260; consider hardcoding min hidden to 2304
- `num_layers=46` — avg 0.261 across 12 candidates, none above 0.340
- `mamba_head_dim=48` — all 7 candidates scored 0.240–0.260; consider hardcoding min head_dim to 56

---

### Design Recipe

Two confirmed high-quality architecture families based on all candidates:

**Family 1 — Best MMLU** (`hidden=2304`, `state_dim=4096`):

```text
52L | hidden=2304 | mamba_num_heads=64 | mamba_head_dim=64 | num_moe_experts=96–104 | moe_ffn_hidden_size=1856 | shared=3072
active=3.00B, total=20.7–22.3B, MMLU=0.461–0.478
```

**Family 2 — Good MMLU, larger total params** (`hidden=2560`, `state_dim=3136`):

```text
48L | hidden=2560 | mamba_num_heads=56 | mamba_head_dim=56 | num_moe_experts=104–112 | moe_ffn_hidden_size=1792 | shared=3072
active=3.00B, total=23.7–25.4B, MMLU=0.433–0.465
```

**Required conditions for any good candidate:**

| Condition | Threshold | Failure rate below threshold |
| --- | --- | --- |
| `num_layers` | ≥ 48 | 42L: 7/7 fail; 46L: 12/12 below 0.35 |
| `mamba_state_dim` | ≥ 3072 | 0/24 candidates below 3072 exceed 0.31 |
| `hidden_size` | 2304–2560 | 2688: 10/10 below 0.27 |
| `mamba_head_dim` | ≥ 56 | 15/15 candidates with head_dim≤48 score 0.24–0.26 |

---

### Distillation results: 1st best vs 2nd best pruning candidate

To show that the pre-distillation MMLU score is a reasonable proxy for downstream quality, we also distilled the 2nd-best candidate by MMLU score (within the 30% width / 15% depth constraints) and evaluated at 2.5B and 20B tokens.

| Rank | Score | Layers | Hidden | Heads × HeadDim | Experts | FFN | Shared | Total | Family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1st | 0.4783 | 52 | 2304 | 64 × 64 | 104 | 1856 | 3072 | 22.3B | Family 1 |
| 2nd | 0.4650 | 48 | 2560 | 56 × 56 | 112 | 1792 | 3072 | 25.4B | Family 2 |

Evaluation after distillation (same blend, same training schedule):

| Checkpoint | Candidate | MMLU | MMLU Pro | GPQA Diamond | LiveCodeBench v6 | AIME 2025 | IFBench | SciCode (Subtask) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5B tokens | 1st best | **68.6** | **73.6** | **62.5** | **57.5** | **79.1** | **58.3** | **22.6** |
| 2.5B tokens | 2nd best | 68.1 | 72.6 | 60.0 | 48.1 | 63.0 | 50.7 | 15.2 |
| 20B tokens | 1st best | 70.8 | 74.6 | **65.3** | **61.0** | **79.8** | **63.6** | **22.5** |
| 20B tokens | 2nd best | **71.0** | 74.6 | 61.6 | 49.6 | 54.0 | 57.3 | 20.5 |

**Observations:**

- In this run the 1st-best candidate wins on most benchmarks at both checkpoints. The pre-distillation MMLU proxy score gap (0.4783 vs 0.4650) translated to meaningful downstream gaps on reasoning benchmarks here (AIME: 79.1 vs 63.0 at 2.5B; 79.8 vs 54.0 at 20B).
- **The NAS proxy score is a useful signal but not a guarantee.** We have seen cases in other experiments where ranks flip after distillation — the proxy is computed without any recovery training, and small differences in architecture can interact non-trivially with the distillation blend. When the top few candidates have similar scores (e.g. within ~2-3% MMLU), running short (~2B-token) distillation on each before committing to the full run is the safer practice. Picking only the 1st-best blindly works often; but may not always be the best choice.

---

## Distillation

### Effect of data blend (tool_calling)

Two distillation data blends were tried on the same pruned 22B/A3.0B candidate with identical hyperparameters during the 8K-seq-length phase.

- **Old blend (no tool_calling):** 30% Pretraining (Code 5, General 20, MATH 5) + 70% Post-training v1/v3 (Math 30, Coding 20, Science 15, IF 5). Schedule: 100B @ 8K then +25B @ 32K (125B total).
- **New blend (used in [main README](README.md)):** adds 5% `Nemotron-Agentic-v1 / tool_calling`, reducing Math 30→27 and Science 15→13 to make room. Same pretraining split. Schedule: 80B @ 8K then +20B @ 32K (100B total).

The blend comparison below is **capped at 80B tokens** — beyond that the two runs diverge in `seq_length` (new blend switches to 32K at 80B; old blend continues at 8K until 100B), so per-checkpoint deltas at higher token counts would conflate blend differences with long-context-phase differences.

Old blend results (8K seq length only). Average is over the 6 reasoning benchmarks (excluding MMLU as we already have MMLU Pro), matching the main README convention:

| Tokens (iters at 8K) | MMLU | MMLU Pro | GPQA Diamond | LiveCodeBench v6 | AIME 2025 | IFBench | SciCode (Subtask) | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5B (100 iters) | 68.6 | 73.6 | 62.5 | 57.5 | 79.1 | 58.3 | 22.6 | 58.9 |
| 20B  (800 iters) | 70.8 | 74.6 | 65.3 | 61.0 | 79.8 | 63.6 | 22.5 | 61.1 |
| 40B  (1600 iters) | 71.6 | 75.6 | 64.5 | 61.6 | 76.8 | 67.4 | 28.3 | 62.4 |
| 60B  (2400 iters) | 71.5 | 76.0 | 67.5 | 63.0 | 77.5 | 68.4 | 29.4 | 63.6 |
| 80B  (3200 iters) | 71.7 | 76.5 | 68.4 | 64.2 | 80.2 | 66.5 | 28.7 | 64.1 |

Per-benchmark difference (Δ = new − old), at matching iter counts during the 8K phase. Average Δ uses the 6-benchmark average (no MMLU):

| Benchmark | 2.5B Δ | 20B Δ | 40B Δ | 60B Δ | 80B Δ |
| --- | --- | --- | --- | --- | --- |
| MMLU | +0.1 | 0.0 | -0.3 | +0.2 | 0.0 |
| MMLU Pro | -0.3 | +0.2 | **+0.8** | +0.1 | 0.0 |
| GPQA Diamond | **+1.2** | **+0.7** | **+2.7** | **+0.6** | **+0.7** |
| LiveCodeBench v6 | -2.2 | **+1.3** | +0.7 | +0.6 | -0.3 |
| AIME 2025 | -1.5 | -0.2 | **+3.0** | **+1.3** | +0.5 |
| IFBench | **+0.8** | **+1.8** | -1.4 | -1.1 | 0.0 |
| SciCode (Subtask) | **+2.5** | **+3.6** | -1.7 | -2.4 | +0.3 |
| **Average** | +0.1 | **+1.3** | **+0.7** | -0.1 | +0.2 |

**Summary — new blend is preferred:**

- **GPQA Diamond** is the most consistent winner — positive Δ at every single checkpoint (+1.2/+0.7/+2.7/+0.6/+0.7). The tool_calling allocation helps throughout training, not just at the end. Since these evals run **without** eval-time tools, the gain comes from training-time exposure to agentic/function-call reasoning traces transferring to GPQA — not from the model invoking a tool during evaluation.
- **SciCode** trends positive early (+2.5/+3.6 at 2.5B/20B) but neutral-to-slightly-negative mid-training (-1.7/-2.4 at 40B/60B) and positive again at 80B (+0.3). Even averaged-of-8, SciCode has the largest residual checkpoint-to-checkpoint noise of any benchmark.
- **AIME 2025** trends positive from 40B onward (+3.0/+1.3/+0.5). The early dip is consistent with the math share dropping from 30%→27%; the model catches up once enough tokens have been seen.
- **IFBench** is small-positive early (+0.8/+1.8) then dips mid-training (-1.4/-1.1) before recovering to neutral at 80B (0.0). Net roughly flat over training.
- **MMLU / MMLU Pro / LiveCodeBench v6** are essentially flat — the upweighted General pretraining split keeps knowledge metrics steady despite the math/science share decrease.
- **Average** is non-negative at 4 of 5 checkpoints (+0.1/+1.3/+0.7/-0.1/+0.2). Net slightly positive overall, with most of the win coming from GPQA and AIME at later checkpoints.

### Effect of long context training

After the 8K-seq-length phase of the old-blend run, training was continued with the same blend but with `seq_length` increased from 8192 to 32768. The longer-context phase is short (200–1000 additional iters) but disproportionately impactful. Numbers below use the old-blend run because it has the longer LC sweep (1000 iters / +25B tokens). The new-blend run's shorter LC phase (+800 iters / +20B tokens, ending at 100B) is in the [main README](README.md) and shows the same qualitative finding (large AIME jump immediately at the start of the LC phase) — though at higher absolute AIME scores there, since the README evals enable tools.

Average is over the 6 reasoning benchmarks (excluding MMLU), matching the main README:

| Tokens (additional iters at 32K) | MMLU | MMLU Pro | GPQA Diamond | LiveCodeBench v6 | AIME 2025 | IFBench | SciCode (Subtask) | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100B (end of 8K phase) | 71.8 | 76.6 | 68.4 | 64.5 | 81.0 | 68.4 | 28.0 | 64.5 |
| 105B (+200 iters at 32K) | 71.9 | 76.8 | 69.7 | 65.6 | **87.3** | 69.3 | 29.5 | 66.4 |
| 125B (+1000 iters at 32K) | 71.9 | 76.7 | 70.5 | 65.7 | **88.0** | 68.5 | 27.6 | 66.2 |

**Summary — high-leverage, low-cost addition:**

- **AIME 2025** sees the largest jump: +6.3 points from 100B → 105B after only 200 additional iters at 32K. AIME chains-of-thought routinely exceed 8K tokens, so the 8K-trained student was being truncated mid-reasoning; allowing 32K immediately unlocks the latent capability.
- **GPQA Diamond** (+2.1) and **LiveCodeBench v6** (+1.2) also benefit meaningfully — both produce long reasoning traces.
- **MMLU / MMLU Pro / IFBench** are flat — these benchmarks comfortably fit within 8K and gain nothing from longer context.
- Overall Average lifts from 64.5 → 66.4 (+1.9) after the first 200 iters at 32K. Going further to 1000 iters slightly regresses (66.2), so most of the LC benefit is captured early.
