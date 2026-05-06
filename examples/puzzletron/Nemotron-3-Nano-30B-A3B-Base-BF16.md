# Bypass Distillation Tutorial: Nemotron-3-Nano-30B-A3B (KV-heads-only)

A minimal end-to-end demonstration that **bypass distillation improves quality** at the same compression budget. The setup is a **toy pruning task on a real production model** — we compress only KV heads (12 → 9, a modest 25% reduction) so a single comparison surfaces the bypass benefit cleanly without needing extensive downstream evaluation. The model itself ([Nemotron-3-Nano-30B-A3B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16)) is a real 30B-A3B MoE-Mamba hybrid, not a tiny stand-in.

## What this tutorial does

The teacher has 6 attention layers (each with `num_key_value_heads=2`) interleaved between Mamba and MoE-FFN blocks — **12 KV heads total** across the whole model. We compress to **9 KV heads (75% of teacher)** in two ways and compare:

1. **Without bypass** — replacement library uses Truncate-init weights (KV heads sliced from teacher; no further training).
2. **With bypass** — the bypass step runs ~10M tokens of per-block knowledge distillation, training a 1-KV-head variant per attention layer against the teacher.

Both runs use the same MIP solver and the same constraint (`target_num_kv_heads: 9`), so MIP picks per attention layer from `{teacher 2-head, 1-head, no_op}` (the no_op variant lets the solver drop attention entirely on a layer if doing so is cheap enough). FFN/MoE/Mamba blocks are copied verbatim from the teacher in both runs — only attention weights change.

**Metrics:** `lm_loss` and `token_accuracy_top_1` measured against the same held-out dataset by the realize-model step (printed automatically to `puzzle_dir/log.txt`).

## Hardware & install

- 8×H100 80GB (the teacher needs ≥60 GiB for activation scoring on a 4096 context).
- Container: `nvcr.io/nvidia/nemo:26.04` or later.
- `pip install -e ".[dev]"` from the modelopt repo root.
- Mamba kernels (required by Nemotron-3-Nano's hybrid backbone):
  ```bash
  pip install mamba-ssm[causal-conv1d] --no-build-isolation
  ```
- HF auth set up so the model is downloadable: `huggingface-cli login`.

## Step A — pipeline without bypass

Edit `examples/puzzletron/configs/nemotron-3-nano-30b-a3b/nemotron-3-nano-30b-a3b.yaml` to point `puzzle_dir` and `dataset_path` at writable locations, then:

```bash
torchrun --nproc_per_node=8 examples/puzzletron/main.py \
    --config examples/puzzletron/configs/nemotron-3-nano-30b-a3b/nemotron-3-nano-30b-a3b.yaml
```

This runs the 8-step puzzletron pipeline (convert → score pruning activations → prune → build replacement library → score replacements → MIP → realize). With `bypass:` added in Step B the pipeline grows to 9 steps; without it, the bypass step is skipped and progress prints `N/8`. Wall-clock: roughly **1h on 8×H100** for this KV-heads-only task (KV-head importance scoring is one forward pass via `IndependentKvHeadContributionHook`, much cheaper than iterative FFN-channel scoring).

When the realize-model step finishes, the log lines at `${puzzle_dir}/log.txt` contain:

```
validate_model_with_kl_div(model_name='teacher', ...)
Average losses = {'lm_loss': ..., 'token_accuracy_top_1': ..., 'token_accuracy_top_5': ..., 'token_accuracy_top_10': ...}
...
validate_model_with_kl_div(model_name='solution_0', ...)
Average losses = {..., 'token_accuracy_top_1': ..., ...}
```

Record the teacher's `token_accuracy_top_1` and `solution_0`'s `token_accuracy_top_1`. **Move or rename `${puzzle_dir}/single_sequence_replacement_solutions--validation/` and `${puzzle_dir}/mip/` aside** before Step B if you want to keep the no-bypass artifacts — Step B reuses the same `puzzle_dir` and the library/scoring/MIP outputs will be overwritten.

## Step B — pipeline with bypass

Add `bypass: defaults` to the `defaults:` list of `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16.yaml` (replace the existing empty `- bypass:` entry):

```yaml
defaults:
  - pruning: kv_heads_pruning
  - scoring: ../validate_solutions_defaults
  - realize_model: ../validate_solutions_defaults
  - bypass: defaults                    # <-- changed from `bypass:`
  - override hydra/hydra_logging: disabled
  - _self_
```

Re-run the same command:

```bash
torchrun --nproc_per_node=8 examples/puzzletron/main.py \
    --config examples/puzzletron/configs/nemotron-3-nano-30b-a3b/nemotron-3-nano-30b-a3b.yaml
```

Skip-if-done caching reuses Step A's converted teacher checkpoint, activation scores, and pruned checkpoints. Only Step 5 (bypass distillation, ~60 min for 10M tokens) and the downstream library/scoring/MIP rerun. Wall-clock: roughly **+1.5 h** on top of Step A.

Bypass writes its outputs under `${puzzle_dir}/bypass/bypass_runs/bypass_heads_1/` and creates a symlink `${puzzle_dir}/ckpts/bypass_heads_1` that the replacement library builder picks up automatically.

Capture `solution_0`'s `token_accuracy_top_1` from the new realize-model log section.

## Results

Reducing total KV heads from 12 → 9 (75% of teacher) at fixed FFN/MoE/Mamba on Nemotron-3-Nano-30B-A3B-Base-BF16:

| Run | `target_num_kv_heads` | `lm_loss` | `token_accuracy_top_1` |
|------------------------------|----------------------:|----------:|-----------------------:|
| Teacher                      | 12                    | 0.5950    | 0.8468                 |
| Pruned, **no bypass** (Truncate-init) | 9            | 0.6347    | 0.8373                 |
| Pruned, **with bypass** (10M-token BLD) | 9          | **0.6055**| **0.8441**             |

**Bypass closes ~74% of the regression gap** at this compression budget:

- `lm_loss` gap to teacher: `0.0397` without bypass → `0.0105` with bypass — bypass recovers **74%**.
- `token_accuracy_top_1` gap to teacher: `0.0095` without bypass → `0.0027` with bypass — bypass recovers **72%**.

For 10M tokens of per-block KD, that's a substantial lift on a real 30B-A3B teacher.

## Going further: full accuracy recovery

Bypass distillation is Stage 1 of the PUZZLE pipeline — local, per-block KD that tightens the replacement library. For larger compression targets (or more aggressive KV pruning) you'll want Stage 2: **global knowledge distillation** on the realized student. See [`examples/pruning/puzzletron/`](../pruning/puzzletron/) for the Megatron-Bridge recipe and concrete MMLU recovery numbers.
