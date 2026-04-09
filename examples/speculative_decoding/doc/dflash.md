# DFlash — Block Diffusion for Speculative Decoding

DFlash predicts an entire block of tokens in a single forward pass using masked parallel
prediction with KV injection from the target model's hidden states.

Reference: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) |
[SpecForge](https://github.com/sgl-project/SpecForge) |
[z-lab](https://github.com/z-lab/dflash)

## Architecture

```
Target Model (frozen)
  │
  ├─ hidden_states[layer 1, 9, 17, 25, 33]  ──► concat ──► FC + RMSNorm ──► target_hidden
  │                                                                              │
  │                                                                    K/V injection
  │                                                                              │
  └─ embed([anchor, mask, mask, ...])  ──► noise_embedding ──► DFlash Decoder (5 layers)
                                                                         │
                                                               lm_head ──► draft tokens
```

**Key components:**
- **Feature Fusion**: Multi-layer hidden states → Linear(num_layers × hidden_size, hidden_size) + RMSNorm
- **KV Injection**: In each draft decoder layer, K/V = concat(k_proj(target_hidden), k_proj(noise))
  with QK-norm. Q comes from noise only.
- **Parallel Drafting**: Position 0 is the anchor (known token), positions 1..B-1 are mask tokens
  predicted in parallel. Bidirectional attention within the block.
- **Random Anchor Sampling**: During training, anchor positions are sampled randomly from
  valid (assistant response) positions, not uniformly spaced.

**Draft model components** (Qwen3-based):
- `Qwen3MLP`, `Qwen3RMSNorm`, `Qwen3RotaryEmbedding` from transformers
- Sliding window attention supported via `config.layer_types`
- Independent of target model architecture

## Training

### Quick Start

```bash
uv run launch.py --yaml examples/Qwen/Qwen3-8B/hf_online_dflash.yaml --yes
```

### Recipe

See [`modelopt_recipes/general/speculative_decoding/dflash.yaml`](../../../modelopt_recipes/general/speculative_decoding/dflash.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dflash.dflash_block_size` | 8 | Block size for parallel prediction |
| `dflash.dflash_num_anchors` | 512 | Random anchor positions per sample |
| `dflash.dflash_loss_decay_factor` | 4.0 | Exponential decay gamma (0 disables) |
| `dflash.dflash_self_logit_distillation` | true | Logit distillation from target |
| `dflash.dflash_architecture_config.num_hidden_layers` | 5 | Draft decoder layers |
| `dflash.dflash_architecture_config.mask_token_id` | auto | Token ID for masked positions |
| `training.answer_only_loss` | false | Mask loss on non-assistant tokens |

### Loss Decay

The exponential decay factor (gamma) weights early block positions higher than later ones.
If position 0 in a block is wrong, all subsequent positions are rejected in speculative
decoding. Decay aligns the training loss with what matters for acceptance rate.

```
weight[k] = exp(-k / gamma)    for k = 0..B-1
```

Paper recommendation: gamma=7 for block_size=16, gamma=4 for block_size=8.

### Checkpoint Resume

DFlash supports checkpoint resume transparently. The `DFlashModule._apply()` method
handles meta-tensor rotary buffers that arise during ModelOpt checkpoint restore — no
special resume logic needed in the training script.

### Export

```bash
python scripts/export_hf_checkpoint.py \
    --model_path /path/to/training/output \
    --export_path /path/to/exported/model
```

Exports to z-lab compatible HF format (`config.json` + `model.safetensors`).

## Results (Qwen3-8B)

Trained on nvidia/Nemotron-Post-Training-Dataset-v2 (2M samples), 64 GPUs, 10 epochs.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Block Size | 8 |
| Sequence Length | 4096 |
| Anchors | 512 |
| Loss | KD + decay (gamma=4) |
| Total Steps | 306,620 |
| Final Per-Token Acc | 67.0% |

### MT-Bench Per-Category AR (Online Validation, osl=512)

| Category | 80K | 150K | 306K |
|----------|-----|------|------|
| math | 5.44 | 5.54 | **5.52** |
| extraction | 4.81 | 4.82 | **4.88** |
| coding | 4.40 | 4.53 | **4.60** |
| reasoning | 4.34 | 4.41 | **4.44** |
| stem | 4.05 | 4.15 | **4.17** |
| writing | 3.76 | 3.79 | **3.84** |
| roleplay | 3.58 | 3.73 | **3.78** |
| humanities | 3.55 | 3.62 | **3.65** |
| **ALL** | **4.24** | **4.32** | **4.36** |

### Comparison with z-lab/Qwen3-8B-DFlash-b16

**ModelOpt eval (online validation, osl=512):**

| Dataset | z-lab | ModelOpt | Diff |
|---------|-------|----------|------|
| gsm8k | 4.10 | **5.19** | **+1.09** |
| MT-Bench | 3.58 | **4.36** | **+0.78** |

**z-lab official eval (dflash.benchmark, osl=512):**

| Dataset | z-lab | ModelOpt | Diff |
|---------|-------|----------|------|
| gsm8k | **5.00** | 4.08 | -0.92 |
| MT-Bench | **3.28** | 2.99 | -0.29 |

> z-lab trained with block_size=16; ModelOpt trained with block_size=8.

### Evaluation Methods

| Method | Description |
|--------|-------------|
| **Fixed GT** | Pre-compute greedy ground truth, check draft against it |
| **Online GT** | Recompute ground truth after each accepted draft (context-dependent) |
| **z-lab official** | Actual speculative decoding with draft KV cache |

Online GT is more accurate than Fixed GT (~+1.0 AR) because speculative decoding
acceptance depends on context-dependent verification, not a fixed reference sequence.

### Key Findings

| Finding | Evidence |
|---------|----------|
| Loss decay boosts AR | +0.12 AR at 55K (gamma=7, bs16); consistent across checkpoints |
| Longer sequences help | seq=4096 vs 512: +0.49 AR on AA-Synthetic |
| Online validation essential | Fixed GT underestimates by ~1.0 AR |
| Forward pass identical to z-lab | Max diff 0.5 (bf16); 6/7 draft tokens match |
| sdpa vs flash_attn: negligible | AR 3.31 vs 3.31; hidden states identical |

## Open Items

### Offline Training

Online training requires the full target model in GPU memory alongside the draft model.
Offline training would pre-compute target hidden states and train the draft model separately.

**Challenge**: DFlash uses random anchor sampling over full sequences, requiring hidden states
at ALL positions. For Qwen3-8B with 5 target layers and seq_len=4096, this is ~160MB per sample
in bf16. With 2M samples, full pre-computation would require ~320TB — not feasible.

**Potential approaches:**
- Pre-sample anchor positions and store only relevant slices (limits randomness)
- Stream hidden states from disk with chunked loading
- Hybrid: quantized base model on CPU computes hidden states on-the-fly, draft on GPU
- Logit distillation adds another dimension: teacher logits at anchor+k-1 positions
  need `[seq_len, vocab_size]` per sample (~600MB in bf16)

### z-lab Eval Gap

ModelOpt eval (online GT) gives higher AR than z-lab's official eval on our checkpoint
(5.19 vs 4.08 on gsm8k). The gap is likely from:
- z-lab uses draft KV cache (accumulates context across blocks); our eval re-runs from scratch
- z-lab's `acceptance_length + 1` counting (minimum 1 per step)
- `rope_theta` mismatch in exported config (was 10000 instead of 1000000 — now fixed)

### Model Support Expansion

Currently supports Qwen3 draft architecture. See `hf_dflash.py` module docstring for
instructions on adding:
- **Qwen3MoE**: Replace MLP with `Qwen3MoeMLP` via config flag
- **MLA (DeepseekV3/Kimi-K2)**: Requires MLA-aware KV injection with compressed K/V

### FP8 / NVFP4 Quantization

The DFlash export pipeline supports quantized checkpoints via ModelOpt PTQ, following
the same flow as EAGLE3:

1. Train draft model (bf16)
2. Apply PTQ: `mtq.quantize(model, quant_cfg)` with `FP8_DEFAULT_CFG` or `NVFP4_DEFAULT_CFG`
3. Export: `export_hf_checkpoint.py` auto-detects quantization and writes scales + `quantization_config`

The exporter's `has_quant_opt()` check and `_export_transformers_checkpoint()` handle
quantized weights transparently. No DFlash-specific quantization code is needed.

TODO: Add a quantization recipe/script and validate FP8/NVFP4 AR impact.

### vLLM Deployment

DFlash speculative decoding is supported in vLLM nightly (v0.19.1+):

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}' \
    --attention-backend flash_attn \
    --max-num-batched-tokens 32768
```

Validated: **386 tok/s** on single H100 with Qwen3-8B + DFlash-b16 (15 spec tokens).

Note: requires `vllm/vllm-openai:nightly` — the `latest` tag (v0.19.0) does not include DFlash.
See [`tools/launcher/common/dflash/vllm_serve.sh`](../../../tools/launcher/common/dflash/vllm_serve.sh)
for a complete serve + benchmark script.

### Docker Local Testing

The launcher example currently requires Slurm cluster access. A local Docker example
with `hf_local=` path mapping would enable development without cluster access.
