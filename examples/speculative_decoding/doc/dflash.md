# DFlash — Block Diffusion for Speculative Decoding

DFlash predicts an entire block of tokens in a single forward pass using masked parallel
prediction with KV injection from the target model's hidden states.

Reference: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) |
[SpecForge](https://github.com/sgl-project/SpecForge) |
[z-lab](https://github.com/z-lab/dflash)

## Architecture

```text
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
- **Parallel Drafting**: Position 0 is the anchor (the last accepted token — known and correct),
  positions 1..B-1 are filled with a special mask token (similar to BERT's `[MASK]`). The draft
  model predicts all B-1 unknown positions in a single forward pass, unlike autoregressive drafters
  (EAGLE3) which predict one token at a time. Benefit: one forward pass produces B-1 draft tokens.
- **Feature Fusion**: Multi-layer hidden states → Linear(num_layers × hidden_size, hidden_size) + RMSNorm.
  Concatenates hidden states from multiple target layers (e.g., layers 1, 9, 17, 25, 33) to give
  the draft model richer context. Similar to EAGLE3's fused feature approach.
- **KV Injection**: A design choice for how the draft model conditions on the target model's
  context. In each draft decoder layer, K and V are projected from the concatenation of target
  hidden states and the block's own embeddings, while Q is projected from the block embeddings
  only. This is one approach to parallel prediction — alternatives include cross-attention or
  prefix conditioning.
- **Random Anchor Sampling**: During training, anchors are sampled randomly from assistant response
  positions (where `loss_mask=1`), not placed at fixed intervals. The anchor is the starting token
  of each training block — it's always correct (from the ground truth) and the model learns to
  predict the next B-1 tokens given this anchor and the target's hidden states. See the
  [illustrated example](#random-anchor-sampling-num_anchors) below for why this improves efficiency.

**KV Injection (token-level example):**

Given context `"The answer is"` and block_size=4 with anchor `"is"`:

```text
Target model hidden states (from frozen base model):
  h["The"]  h["answer"]  h["is"]     ← target_hidden (ctx_len=3)
     │          │           │
     └──── FC + RMSNorm ────┘
              │
     fused context features

Block input (draft token embeddings):
  embed("is")  embed(MASK)  embed(MASK)  embed(MASK)     ← noise_embedding (block_size=4)
     pos=3       pos=4        pos=5        pos=6

In each DFlash decoder layer:
  Q = q_proj(noise_embedding)                             ← shape [4, head_dim]
      only the block tokens generate queries

  K = concat(                                             ← shape [7, head_dim]
        k_proj(fused_context),   ← from target hidden     [3 positions: "The","answer","is"]
        k_proj(noise_embedding)  ← from block tokens      [4 positions: "is",MASK,MASK,MASK]
      )

  V = concat(v_proj(fused_context), v_proj(noise_embedding))  ← same shape as K

  Attention: Q (4 tokens) attends to K/V (7 tokens)

              K/V:  "The" "answer" "is"  │  "is"  MASK  MASK  MASK
                     pos0   pos1   pos2  │  pos3  pos4  pos5  pos6
              ───────────────────────────┼──────────────────────────
  Q pos=3 "is"  :    ✓      ✓      ✓    │   ✓     ✓     ✓     ✓
  Q pos=4 MASK  :    ✓      ✓      ✓    │   ✓     ✓     ✓     ✓
  Q pos=5 MASK  :    ✓      ✓      ✓    │   ✓     ✓     ✓     ✓
  Q pos=6 MASK  :    ✓      ✓      ✓    │   ✓     ✓     ✓     ✓
                     ─── context ───     │  ──── block ────────────
  (bidirectional within block, no attention mask at inference)

  Output → lm_head → predictions:
    pos=3: skip (anchor, already known)
    pos=4: predict token after "is"      → "5"
    pos=5: predict token after "is 5"    → "."
    pos=6: predict token after "is 5."   → "[EOS]"
```

**Training vs Inference:**

```text
TRAINING (2 anchors, block_size=4):

  Context tokens:  "The"  "answer"  "is"   "5"    "."
  Block 0 (anchor="The"):  [The,  MASK, MASK, MASK]
  Block 1 (anchor="is"):   [is,   MASK, MASK, MASK]

  All blocks processed in ONE forward pass. Attention mask controls visibility:

              K/V (context)          K/V (block 0)        K/V (block 1)
              "The" "ans" "is" "5" "."  The  M    M    M    is   M    M    M
               c0    c1   c2   c3  c4   b0   b1   b2   b3   b4   b5   b6   b7
  Q ─────────────────────────────────────────────────────────────────────────
  b0 "The"  :  ✗     ✗    ✗    ✗   ✗    ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗
  b1  MASK  :  ✗     ✗    ✗    ✗   ✗    ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗
  b2  MASK  :  ✗     ✗    ✗    ✗   ✗    ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗
  b3  MASK  :  ✗     ✗    ✗    ✗   ✗    ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗
  b4  "is"  :  ✓     ✓    ✗    ✗   ✗    ✗    ✗    ✗    ✗    ✓    ✓    ✓    ✓
  b5  MASK  :  ✓     ✓    ✗    ✗   ✗    ✗    ✗    ✗    ✗    ✓    ✓    ✓    ✓
  b6  MASK  :  ✓     ✓    ✗    ✗   ✗    ✗    ✗    ✗    ✗    ✓    ✓    ✓    ✓
  b7  MASK  :  ✓     ✓    ✗    ✗   ✗    ✗    ✗    ✗    ✗    ✓    ✓    ✓    ✓
               ── context ──────    ── block 0 ──────    ── block 1 ──────

  Block 0: first block sees NO context (✗), only its own block (bidirectional ✓)
  Block 1: sees context before anchor "is" (c0,c1 ✓), NOT its own anchor or later
           plus its own block (bidirectional ✓)

  Loss: computed on all non-anchor positions simultaneously.
  No verification — ground truth labels known from training data.

INFERENCE (one block at a time, NO attention mask):

  Step 1: target forward("The answer is") → base_token = "5"
          block = [5, MASK, MASK, MASK]

              K/V:  "The" "ans" "is"  │  "5"  MASK  MASK  MASK
  Q ─────────────────────────────────┼──────────────────────────
  "5"   :            ✓     ✓     ✓   │   ✓     ✓     ✓     ✓
  MASK  :            ✓     ✓     ✓   │   ✓     ✓     ✓     ✓
  MASK  :            ✓     ✓     ✓   │   ✓     ✓     ✓     ✓
  MASK  :            ✓     ✓     ✓   │   ✓     ✓     ✓     ✓

  All ✓ — no mask at inference. Block sees full context freely.
  Target verifies → accept 3 → sequence: "The answer is 5 . [EOS]"

  Step 2: next block with grown context (5 tokens) ...
```

The draft model sees the target's internal representation of the context (via KV injection)
without re-running the target model for drafting. The expensive target forward pass is
only needed for verification — the lightweight draft model reuses the target's hidden states.

**Draft model components** (Qwen3-based):
- `Qwen3MLP`, `Qwen3RMSNorm`, `Qwen3RotaryEmbedding` from transformers
- Sliding window attention supported via `config.layer_types` *(implemented, not yet validated end-to-end)*
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

> **Note on `answer_only_loss` and chat templates:** When `answer_only_loss=true`, the
> tokenizer's chat template must include `{% generation %}` / `{% endgeneration %}` tags
> around assistant content. HuggingFace uses these tags to produce `assistant_masks` via
> `return_assistant_tokens_mask=True` in `apply_chat_template()`.
>
> Most model tokenizers (Qwen3, Llama3, etc.) do **not** ship with these tags by default.
> You must provide a custom template via `data.chat_template=path/to/template.jinja` in
> the training config. A Qwen3 template is provided at
> `tools/launcher/examples/Qwen/Qwen3-8B/chat_template_train.jinja`.
>
> To create a template for other models:
> 1. Copy the model's original `chat_template` from `tokenizer_config.json`
> 2. Add `{% generation %}` before and `{% endgeneration %}` after assistant content
> 3. Test that tokenization matches the original template (no extra/missing tokens)
>
> See the [HuggingFace guide on train-on-completions-only](https://huggingface.co/docs/transformers/en/chat_templating#train-on-completions-only)
> for details on generation tags.

### Random Anchor Sampling (`num_anchors`)

During training, anchor positions are sampled randomly from valid (assistant response)
tokens in each batch, rather than dividing the sequence into fixed blocks. Each anchor
starts a block of `block_size` tokens where the draft model predicts positions 1..B-1.

```text
Sequence:  [SYS] You helpful [USR] What 2+3? [AST] The answer is 5
Position:    0    1     2      3     4    5     6    7    8    9  10
loss_mask:   0    0     0      0     0    0     0    1    1    1   1
                                                   ^^^^^^^^^^^^^^^^
                                                   assistant response

Fixed blocks (block_size=4):
Block 0: pos [0,1,2,3]   anchor=0  → predict 1,2,3   → loss_mask=0,0,0   → ZERO LOSS
Block 1: pos [4,5,6,7]   anchor=4  → predict 5,6,7   → loss_mask=0,0,1   → 1/3 useful
Block 2: pos [8,9,10,—]  anchor=8  → predict 9,10,—  → loss_mask=1,1,—   → 2/2 useful

Efficiency: 3/8 = 38%

Random anchors (num_anchors=3, sampled from loss_mask=1):
Anchor 7:  pos [7,8,9,10]   → predict 8,9,10   → loss_mask=1,1,1   → 3/3 useful
Anchor 9:  pos [9,10,—,—]   → predict 10,—,—   → loss_mask=1,—,—   → 1/1 useful
Anchor 8:  pos [8,9,10,—]   → predict 9,10,—   → loss_mask=1,1,—   → 2/2 useful

Efficiency: 6/6 = 100%
```

Random anchors guarantee every prediction is on assistant tokens.
Fixed blocks waste compute on prompt tokens where loss_mask=0.

**Tradeoff:** Higher `num_anchors` = more training signal per sample but more compute.
Lower = faster iteration but less data efficiency. With `seq_len=4096` and `block_size=8`,
`num_anchors=512` means the model sees ~512 blocks per sample (covering ~4096 positions).
Scale proportionally: `num_anchors ≈ seq_len / block_size` gives full coverage.

### Loss Decay

The exponential decay factor (gamma) weights early block positions higher than later ones.
If position 1 in a block is wrong, all subsequent positions are rejected in speculative
decoding. Decay aligns the training loss with what matters for acceptance rate.

```text
weight[k] = exp(-(k-1).clamp(min=0) / gamma)    for k = 0..B-1
```

Positions 0 (anchor, excluded by loss mask) and 1 get full weight (1.0). Later positions
decay: e.g., with `gamma=4` and `block_size=8`, position 7 contributes only 22% as
much as position 1. Paper recommendation: gamma=7 for block_size=16, gamma=4 for block_size=8.

Note: this is different from EAGLE3's `eagle_loss_decay_factor` which multiplies loss by
`alpha^step` across TTT steps. DFlash decay operates within a single block, weighting
early positions higher because they gate acceptance of all later positions.

### Checkpoint Resume

DFlash supports checkpoint resume transparently. Rotary embeddings are lazily
initialized on first forward (matching EAGLE3's `_maybe_init_rope` pattern),
avoiding meta-tensor issues during `from_pretrained` model construction.

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

### HuggingFace AR Evaluation

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

### Not Yet Implemented

- **Offline training**: DFlash needs multi-layer hidden states at all positions for KV
  injection (5x storage vs EAGLE3's single-layer approach). Possible approaches: store
  fused hidden states, pre-sample anchors, or hybrid CPU base + GPU draft.
- **Qwen3MoE draft**: Replace `Qwen3MLP` with `Qwen3MoeMLP` via config flag. See
  `hf_dflash.py` module docstring for instructions.
- **MLA support (DeepseekV3/Kimi-K2)**: Requires MLA-aware KV injection with compressed K/V.
- **Docker local testing**: Launcher example requires Slurm. Need a local Docker example
  with `hf_local=` path mapping.

### Implemented but Not Yet Validated End-to-End

- **Sliding window attention**: Code reads `config.layer_types` and sets `sliding_window`
  per layer. Unit tested but not validated in a full training run with sliding window models.
- **FP8 / NVFP4 quantization**: Export pipeline supports quantized checkpoints via
  `hf_ptq.py` (PTQ succeeded in testing). AR impact of quantization not yet measured.
  The flow: train (bf16) → `mtq.quantize(model, quant_cfg)` → `export_hf_checkpoint.py`.
- **Checkpoint resume**: Lazy rotary embedding init (matching EAGLE3 pattern).
  Validated in train+resume E2E tests on DDP and FSDP2.

### Validated

- **Online training**: E2E pipeline (train → export → eval) on sample-1K and sample-10K.
- **Multi-node DDP**: 8-node (64 GPU) training on full dataset, 10 epochs.
- **AR evaluation**: `ar_validate.py` with online GT, per-category MT-Bench.
- **vLLM deployment**: Speculative decoding with `vllm/vllm-openai:nightly` (v0.19.1+).
  3.1x speedup over baseline. Per-category benchmarks on MT-Bench.

  ```bash
  vllm serve Qwen/Qwen3-8B \
      --speculative-config '{"method": "dflash", "model": "path/to/checkpoint", "num_speculative_tokens": 7}' \
      --max-num-batched-tokens 32768
  ```

- **FSDP2 training**: `dp_shard_size=8` with checkpoint resume validated.
- **Export**: z-lab compatible HF format, loadable by vLLM and z-lab benchmark.
- **Loss decay**: Validated +0.12 AR improvement with gamma=7 (bs16).
- **Qwen3.5-4B**: E2E pipeline validated (train → export → vLLM serve).

### vLLM Deployment Notes for Qwen3.5

Qwen3.5 models have non-standard architecture dimensions (`head_dim=160`, `num_attention_heads=16`)
that cause vLLM KV cache page size errors when inherited by the DFlash draft model. Key findings:

1. **M-RoPE incompatibility**: Qwen3.5 uses M-RoPE (multimodal RoPE with `mrope_section`).
   DFlash draft models use standard Qwen3 RoPE. Inheriting `rope_scaling` from the base model
   causes `NotImplementedError: Speculative Decoding does not support M-RoPE yet` in vLLM.
   **Fix**: `rope_scaling` is not inherited from base model; the draft uses standard RoPE.

2. **Head dimension mismatch**: Qwen3.5-4B has `head_dim=160` (2560/16 heads) which is
   non-standard and causes `page size not divisible` errors in vLLM's KV cache.
   **Fix**: Override draft architecture to use z-lab's dimensions: `num_attention_heads=32`,
   `num_key_value_heads=8`, `head_dim=128`, `intermediate_size=9728`, `rope_theta=10000000`.
   See `tools/launcher/examples/Qwen/Qwen3.5-4B/hf_online_dflash.yaml`.

3. **MTP alternative**: Qwen3.5 has built-in MTP (Multi-Token Prediction) heads that work
   with vLLM natively (no draft model needed): `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`.
   MTP-1 achieves 91% acceptance rate on Qwen3.5-4B. DFlash provides higher speedup when
   trained sufficiently (z-lab's checkpoint: 3.84 mean acceptance length).
