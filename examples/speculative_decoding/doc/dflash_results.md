# DFlash Block Diffusion — ModelOpt Training Results

Qwen3-8B target model, trained on nvidia/Nemotron-Post-Training-Dataset-v2 (2M samples)

## Key Metrics

| Benchmark | Acceptance Rate |
|-----------|----------------|
| **gsm8k** | **5.19** |
| **MT-Bench** | **4.36** |

> Online validation, block_size=8, osl=512

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Target Model | Qwen3-8B |
| Draft Layers | 5 |
| Block Size | 8 |
| Sequence Length | 4096 |
| Anchors per Sample | 512 |
| Loss | KD (logit distillation) + exponential decay (gamma=4) |
| Learning Rate | 6e-4 (linear decay) |
| Epochs | 10 |
| GPUs | 64 (8 nodes x 8 H100) |
| Total Steps | 306,620 |
| Final Loss | 1.129 |
| Final Per-Token Acc | 67.0% |

## MT-Bench Per-Category AR (Online Validation)

80 prompts, block_size=8, osl=512, steps=7

| Category | 80K | 150K | 306K (final) |
|----------|-----|------|-------------|
| math | 5.44 | 5.54 | **5.52** |
| extraction | 4.81 | 4.82 | **4.88** |
| coding | 4.40 | 4.53 | **4.60** |
| reasoning | 4.34 | 4.41 | **4.44** |
| stem | 4.05 | 4.15 | **4.17** |
| writing | 3.76 | 3.79 | **3.84** |
| roleplay | 3.58 | 3.73 | **3.78** |
| humanities | 3.55 | 3.62 | **3.65** |
| **ALL** | **4.24** | **4.32** | **4.36** |

## Comparison with z-lab/Qwen3-8B-DFlash-b16

### ModelOpt Eval (online validation, osl=512)

| Dataset | z-lab | ModelOpt (306K) | Diff |
|---------|-------|-----------------|------|
| gsm8k | 4.10 | **5.19** | **+1.09** |
| MT-Bench | 3.58 | **4.36** | **+0.78** |

### z-lab Official Eval (dflash.benchmark, osl=512)

| Dataset | z-lab | ModelOpt (306K) | Diff |
|---------|-------|-----------------|------|
| gsm8k | **5.00** | 4.08 | -0.92 |
| MT-Bench | **3.28** | 2.99 | -0.29 |

> z-lab model trained with block_size=16. ModelOpt trained with block_size=8.

## Evaluation Method Impact (gsm8k)

| Eval Method | z-lab checkpoint | ModelOpt (306K) |
|-------------|-----------------|-----------------|
| Fixed GT (ModelOpt eval) | 2.95 | 4.23 |
| Online GT (ModelOpt eval) | 4.10 | **5.19** |
| z-lab official eval | **5.00** | 4.08 |

- **Fixed GT**: pre-compute greedy ground truth, check draft against it.
- **Online GT**: recompute ground truth after each accepted draft (context-dependent).
- **z-lab official**: actual speculative decoding with draft KV cache.

## Key Findings

| Finding | Evidence |
|---------|----------|
| Loss decay boosts AR | +0.12 AR at 55K steps (gamma=7, bs16); consistent across all checkpoints |
| Longer sequences help | seq=4096 vs 512: +0.49 AR on AA-Synthetic at same checkpoint |
| Online validation essential | Fixed GT underestimates by ~1.0 AR; context-dependent GT matches actual spec-decode |
| Forward pass identical to z-lab | Max diff 0.5 (bf16 noise) on same mask_token_id; 6/7 draft tokens match |
| sdpa vs flash_attn: negligible | Overall AR 3.31 vs 3.31; hidden states identical, logits differ <2% |
