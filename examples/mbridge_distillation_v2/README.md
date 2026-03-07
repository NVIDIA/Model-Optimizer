# mbridge_distillation_v2

Knowledge distillation (KD) with heterogeneous AnyModel/Puzzletron models via Megatron Bridge — without modifying Bridge or MCore.

Supports any combination of homogeneous and heterogeneous student/teacher models. A *heterogeneous* model is an AnyModel/Puzzletron checkpoint in which each decoder layer can have a different architecture (Mamba, MoE, dense attention, etc.), encoded as `block_configs` in its `config.json`.

## Contents

| File | Purpose |
|------|---------|
| `distill.py` | End-to-end KD training script |
| `kd.yaml` | Default KD configuration overrides |
| `generate_text.py` | Inference sanity-check script |
| `block_config_utils.py` | Load/translate `block_configs` → MCore per-layer overrides |
| `layer_patchers.py` | `mbridge_patcher` context manager (patches MCore `build_module`) |
| `provider_patch.py` | Provider-level patches for standard and distillation providers |

## Prerequisites

- **Megatron Bridge** cloned at the same level as `Model-Optimizer`
- **NVIDIA ModelOpt** (`nvidia-modelopt`) installed, with the `torch.puzzletron` extra for AnyModel support
- A Slurm cluster or workstation with one or more NVIDIA GPUs

```bash
# Install from the repo root
pip install -e ".[torch,puzzletron]"
```

---

## Sanity check — text generation

Before running a multi-day distillation job, verify the model loads and runs correctly with a quick generation test. `generate_text.py` loads a single model and does greedy decoding — no training, no optimizer, minimal setup.

```bash
# Homogeneous model (loads from HuggingFace Hub):
torchrun --nproc_per_node=1 generate_text.py --model llama --prompt "Hello, how are you?"

# Heterogeneous checkpoint saved by AnyModel (loads from local directory):
torchrun --nproc_per_node=1 generate_text.py \
    --model nemo \
    --checkpoint /path/to/nemotronh_checkpoint \
    --trust-remote-code \
    --prompt "Explain attention in one sentence."

# Multi-GPU with tensor parallelism:
torchrun --nproc_per_node=4 generate_text.py --model llama --tp 4 --prompt "Hello"
```

**What to check:**

- No shape/weight-loading errors → `block_configs` are being read and applied correctly.
- Generated text is coherent (not random garbage) → weights loaded successfully.
- Top-5 token log at steps 0–2 shows reasonable candidates → logit all-gather across TP works.
- No CUDA graph crashes → `is_moe_layer` flag reset correctly after `NoOpWithBias` replacement.

Supported `--model` values: `gpt` (GPT-OSS-20B), `nemo` (Nemotron-H 30B), `llama` (Llama-3.2-3B), `qwen` (Qwen3-8B).

---

## Knowledge distillation

### Quickstart

```bash
# Minimal: homogeneous student ← heterogeneous teacher (single GPU, mock data)
torchrun --nproc_per_node=1 distill.py \
    --student llama \
    --teacher nemo \
    --teacher-checkpoint /path/to/nemotronh_checkpoint \
    --trust-remote-code

# From local checkpoints with a YAML config:
torchrun --nproc_per_node=8 distill.py \
    --student llama --student-checkpoint /path/to/student \
    --teacher nemo  --teacher-checkpoint /path/to/teacher \
    --config-file kd.yaml

# All heterogeneous — GPT-OSS teacher → Nemotron-H student:
torchrun --nproc_per_node=8 distill.py \
    --student nemo  --student-checkpoint /path/to/student \
    --teacher gpt   --teacher-checkpoint /path/to/teacher \
    model.tensor_model_parallel_size=4 \
    model.expert_model_parallel_size=2
```

### Configuration

Settings are resolved in this priority order (highest last wins):

1. **Base defaults** from `_pretrain_common()` — training iterations, optimizer, DDP, etc.
2. **YAML overrides** from `--config-file` (defaults to `kd.yaml` if it exists in the script directory).
3. **CLI overrides** — Hydra-style dotlist, e.g. `model.tensor_model_parallel_size=4`.

```bash
# Combine YAML + CLI overrides:
torchrun --nproc_per_node=8 distill.py \
    --student llama --teacher nemo \
    --config-file kd.yaml \
    train.train_iters=100000 \
    optimizer.lr=5e-5 \
    model.tensor_model_parallel_size=4 \
    model.expert_model_parallel_size=4 \
    checkpoint.save=/checkpoints/llama_kd
```

### Key KD settings (`model.kd_config.*`)

| Field | Default | Description |
|-------|---------|-------------|
| `logit_layers` | `["output_layer", "output_layer"]` | Student/teacher output-logit module names |
| `intermediate_layer_pairs` | `[["decoder.final_layernorm", "decoder.final_layernorm"]]` | Pairs for auxiliary cosine-similarity loss. Must be sequence-parallel layers. Set to `[]` for logit-only KD. |
| `skip_lm_loss` | `true` | Skip the standard LM cross-entropy loss (lower memory) |
| `kd_loss_scale` | `1.0` | Scale of KD loss relative to LM loss (only used when `skip_lm_loss: false`) |
| `logit_kl_temperature` | `1.0` | Temperature for KL divergence over output logits |

Override in `kd.yaml`:

```yaml
model:
  kd_config:
    intermediate_layer_pairs: []   # logit-only KD, lower memory
    logit_kl_temperature: 2.0      # softer teacher distribution
    skip_lm_loss: true
```

### Parallelism

Student and teacher **must share the same parallelism** — Bridge validates this. Setting `model.tensor_model_parallel_size` automatically mirrors to the teacher via `DistillationProvider.__setattr__`.

```bash
# TP=4, EP=4 (for MoE models like GPT-OSS or Nemotron-H):
torchrun --nproc_per_node=16 distill.py \
    --student llama --teacher nemo \
    model.tensor_model_parallel_size=4 \
    model.expert_model_parallel_size=4 \
    model.pipeline_model_parallel_size=1
```

### Real data

`kd.yaml` uses mock (random) data by default. To use a real pre-tokenized dataset:

```yaml
dataset:
  seq_length: 4096
  blend:
    - ["/path/to/my_data_text_document", 1.0]

tokenizer:
  tokenizer_type: NullTokenizer
  vocab_size: 131072    # must match student model vocab size
```

### Checkpointing

Only the **student** model is saved. Checkpoints are compatible with all standard Bridge workflows (inference, further fine-tuning, evaluation).

```yaml
checkpoint:
  save: /checkpoints/my_kd_student
  save_interval: 500
  # Optional: resume from a prior KD run or pre-trained student:
  # load: /checkpoints/previous_run
```

---

## How it works

### Normal Bridge KD (homogeneous models)

```
distill(cfg)
  → pretrain(cfg, forward_step_modelopt)
    → DistillationProvider.provide()
      → _super_class.provide(self, ...)    # builds student MCore model
      → teacher.provide_distributed_model()  # builds teacher MCore model
      → mtd.convert(student, kd_loss_mode)   # wraps in DistillationModel
```

### Heterogeneous extension (this package)

Two additional patches are applied before training starts:

**`apply_patch()`** patches `ModelProviderMixin.provide` at the class level. When `teacher.provide()` is called (via `teacher.provide_distributed_model()`), it activates `mbridge_patcher(teacher_block_configs)` which injects per-layer config overrides into MCore's `build_module` during teacher construction.

**`apply_distillation_patch()`** patches `DistillationProvider.provide` itself. The student is built via `self._super_class.provide(self, ...)`, which calls the concrete provider class directly and bypasses the `ModelProviderMixin` patch. This patch temporarily replaces `_super_class.provide` with a wrapper that activates `mbridge_patcher(student_block_configs)`, restoring the original in a `finally` block.

```
apply_distillation_patch()
  → DistillationProvider.provide() [patched]
      → _super_class.provide wrapped in mbridge_patcher(student_block_configs)
          → student MCore model with per-layer config ✓
      → teacher.provide_distributed_model()
          → teacher.provide() [instance-patched by set_provider_block_configs]
              → mbridge_patcher(teacher_block_configs)
                  → teacher MCore model with per-layer config ✓
```

`mbridge_patcher` uses thread-local state, so nested activations (student outer, teacher inner) are safe and independent.

---

## Troubleshooting

**`TypeError: cannot unpack non-sequence Tensor`**
MCore's bias-dropout-add path unpacks `(output, bias)` tuples. This occurs when `IdentityOp` is used for no-op layers (returns a plain tensor). This package uses `NoOpWithBias` (returns `(zeros_like(x), None)`) which satisfies the contract.

**CUDA graph crash with MoE layers**
When a MoE layer's MLP is replaced with `NoOpWithBias`, `layer.is_moe_layer` must be reset to `False`. The patcher does this automatically. If you see this, check that `apply_distillation_patch()` and `apply_patch()` are both called before `distill()`.

**Wrong weight shapes / `size mismatch` errors**
`block_configs` are not being read correctly. Check that `hf_config.block_configs` is populated (it should be if the checkpoint was saved by AnyModel). Run `generate_text.py` on the checkpoint first to verify block_configs load cleanly.

**`num_attention_heads` or `hidden_size` missing on provider**
The patcher falls back to the homogeneous global config for that model. This is a warning, not an error. It means the provider class doesn't expose these as dataclass fields — check the provider's field names.

**OOM during distillation**
- Set `model.kd_config.intermediate_layer_pairs: []` to use logit-only KD.
- Reduce `train.micro_batch_size`.
- Enable activation recompute: `model.recompute_granularity: full`.
- Increase TP/EP/PP to spread the models across more GPUs.
