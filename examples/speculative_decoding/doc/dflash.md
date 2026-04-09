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
| `dflash.dflash_num_anchors` | 512 | Random anchor positions per sample (see below) |
| `dflash.dflash_loss_decay_factor` | 4.0 | Exponential decay gamma (0 disables, see below) |
| `dflash.dflash_self_logit_distillation` | true | Use target model logits as soft labels (vs hard CE) |
| `dflash.dflash_architecture_config.num_hidden_layers` | 5 | Draft decoder layers |
| `dflash.dflash_architecture_config.mask_token_id` | auto | Token ID for masked positions |
| `training.answer_only_loss` | false | Mask loss on non-assistant tokens |

### Random Anchor Sampling (`num_anchors`)

During training, anchor positions are sampled randomly from valid (assistant response)
tokens in each batch, rather than dividing the sequence into fixed blocks. Each anchor
starts a block of `block_size` tokens where the draft model predicts positions 1..B-1.

**Tradeoff:** Higher `num_anchors` = more training signal per sample but more compute.
Lower = faster iteration but less data efficiency. With `seq_len=4096` and `block_size=8`,
`num_anchors=512` means the model sees ~512 blocks per sample (covering ~4096 positions).
Scale proportionally: `num_anchors ≈ seq_len / block_size` gives full coverage.

### Loss Decay

The exponential decay factor (gamma) weights early block positions higher than later ones.
If position 1 in a block is wrong, all subsequent positions are rejected in speculative
decoding. Decay aligns the training loss with what matters for acceptance rate.

```
weight[k] = exp(-(k-1).clamp(min=0) / gamma)    for k = 0..B-1
```

Positions 0 (anchor, excluded by loss mask) and 1 get full weight (1.0). Later positions
decay: e.g., with `gamma=4` and `block_size=8`, position 7 contributes only 22% as
much as position 1. Paper recommendation: gamma=7 for block_size=16, gamma=4 for block_size=8.

Note: this is different from EAGLE3's `eagle_loss_decay_factor` which multiplies loss by
`alpha^step` across TTT steps. DFlash decay operates within a single block, weighting
early positions higher because they gate acceptance of all later positions.

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

### AR Evaluation

AR is evaluated using `ar_validate.py` which calls `pseudo_speculative_generate`
with online (context-dependent) ground truth:

1. Run base model on `input_ids` → get base token + hidden states
2. Build draft block: `[base_token, MASK, MASK, ...]`
3. Run DFlash draft forward → get `block_size-1` draft tokens
4. Verify each draft token against the base model's prediction **given the
   accepted sequence so far** (not a pre-computed fixed reference)
5. Accept consecutive matches, append target's correction on first mismatch
6. AR = total accepted tokens / number of speculative steps

```bash
python scripts/ar_validate.py --model_path /path/to/checkpoint --per_category --osl 512 --steps 7
```

### vLLM Deployment Results

vLLM nightly (v0.19.1+), H100, MT-Bench 80 prompts, 1024 max tokens:

| | Baseline | z-lab (bs16) | **ModelOpt (bs8)** |
|---|---------|-------------|-------------------|
| TP=1 tok/s | 145 | 422 | **443** |
| TP=8 tok/s | 377 | 919 | **1053** |
| Speedup (TP=1) | 1.0x | 2.9x | **3.1x** |

**Per-Category (TP=8):**

| Category | ModelOpt Accept | z-lab Accept | ModelOpt TPS | z-lab TPS |
|----------|----------------|-------------|-------------|-----------|
| math | **5.14** | 4.24 | **1238** | 1098 |
| coding | **4.03** | 3.52 | **1299** | 1269 |
| writing | **3.99** | 3.97 | **1002** | 903 |
| reasoning | **3.89** | 3.49 | **1188** | 1020 |
| roleplay | **3.88** | 3.37 | **1069** | 923 |
| extraction | **3.60** | 3.02 | **1002** | 789 |
| stem | 3.55 | **3.63** | **1027** | 914 |
| humanities | **3.05** | 2.68 | **786** | 672 |
| **ALL** | | | **1053** | 919 |

ModelOpt wins acceptance length on 7/8 categories and TPS on 8/8 categories.

### Key Findings

| Finding | Evidence |
|---------|----------|
| 3.1x speedup over baseline (TP=1) | 443 vs 145 tok/s on vLLM |
| 15% faster than z-lab | TP=1: 443 vs 422; TP=8: 1053 vs 919 |
| More efficient drafting | 44% vs 16.5% draft acceptance; fewer tokens drafted, more accepted |
| Loss decay boosts AR | +0.12 AR at 55K (gamma=7, bs16); consistent across checkpoints |
| Longer sequences help | seq=4096 vs 512: +0.49 AR on AA-Synthetic |

## Open Items

### Offline Training

Online training requires the full target model in GPU memory alongside the draft model.
Offline training would pre-compute target hidden states and train the draft model separately.

**Challenge**: DFlash needs hidden states from multiple target layers (not just the last)
at all positions for KV injection. EAGLE3 offline only stores last-layer hidden states
and reruns `lm_head` during training, but DFlash's feature fusion concatenates hidden
states from layers [1, 9, 17, 25, 33] — 5x the storage per position.

**Potential approaches:**
- Store only the fused (post-FC) target hidden states instead of raw multi-layer states
- Pre-sample anchor positions and store only relevant slices
- Hybrid: quantized base model on CPU computes hidden states on-the-fly, draft on GPU

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
    --speculative-config '{"method": "dflash", "model": "path/to/dflash-checkpoint", "num_speculative_tokens": 7}' \
    --max-num-batched-tokens 32768
```

Note: requires `vllm/vllm-openai:nightly` — the `latest` tag (v0.19.0) does not include DFlash.
See [`tools/launcher/common/dflash/vllm_serve.sh`](../../../tools/launcher/common/dflash/vllm_serve.sh)
for serve + benchmark scripts.

### Docker Local Testing

The launcher example currently requires Slurm cluster access. A local Docker example
with `hf_local=` path mapping would enable development without cluster access.
