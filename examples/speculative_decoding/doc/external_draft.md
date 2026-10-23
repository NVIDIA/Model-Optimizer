# External Draft — Training a Standalone Draft Model

An external draft is an ordinary pretrained causal LM trained to propose tokens for a
target model. Unlike the head-based modes (EAGLE, Medusa, DFlash) nothing is grafted onto
the target: it contributes its vocabulary and, on the dense path, its `lm_head`. The result
exports as a plain HuggingFace checkpoint.

Recipe: [`modelopt_recipes/general/speculative_decoding/external_draft.yaml`](../../../modelopt_recipes/general/speculative_decoding/external_draft.yaml) |
Swap implementation: [`vocab_swap.py`](../../../modelopt/torch/speculative/external/vocab_swap.py)

## Architecture

```text
Target (frozen, never modified)
  │
  ├─ sparse path:  top-k deployment policy  ──► dumped to disk ──┐
  │                                                              │
  └─ dense path:   hidden states ──► target lm_head ──► logits ──┤
                                                                 │
                                                         TVD / soft_ce
                                                                 │
Draft (a complete causal LM, trained)  ──► its own lm_head ──► draft logits
```

**Key components:**

- **No grafted head.** The draft owns its embeddings and `lm_head` and runs standalone at
  serving time. Engines that already support a separate draft model load it with no
  mode-specific support.
- **Acceptance-shaped objectives.** Under rejection sampling the probability a proposed
  token survives verification is `sum(min(p, q))`, so total-variation distance is
  `1 - acceptance`. Training on TVD optimises the serving metric directly.
- **Two data formats.** The TVD objectives need only the target's truncated next-token
  distribution, which is far smaller than a hidden state per token and keeps the target off
  the GPU at training time. `soft_ce` needs full logits and so needs dumped hidden states.
- **Vocabulary coupling.** The draft must index the target's vocabulary. A draft from
  another family can be re-indexed onto it, at the cost of replacing a large fraction of a
  small model.

## Training

### Quick Start

```bash
# Every command in this section reads $SEQ_LEN; change it here and they stay consistent.
SEQ_LEN=4096

torchrun --nproc_per_node 8 main.py \
  --config ../../modelopt_recipes/general/speculative_decoding/external_draft.yaml \
  model.model_name_or_path=$TARGET \
  draft_model_name_or_path=$DRAFT \
  data.sparse_data_path=$POLICY_DIR \
  data.sparse_cache_dir=$CACHE_DIR \
  training.training_seq_len=$SEQ_LEN \
  training.output_dir=$OUTPUT_DIR
```

### Recipe

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.model_name_or_path` | — | The **target**, used for its vocabulary and `lm_head` |
| `draft_model_name_or_path` | — | The draft, the model actually trained |
| `external.external_loss` | `tvd` | `soft_ce`, `tvd`, `tvd_deploy` or `tvd_ce` (see below) |
| `external.external_top_k` | 20 | Serving top-k; must match the dump |
| `external.external_top_p` | 0.95 | Serving top-p; must match the dump |
| `external.external_tvd_alpha` | 0.9 | Weight on the TVD term under `tvd_ce` |
| `external.external_ce_alpha` | 0.1 | Weight on the CE term under `tvd_ce` |
| `external.external_vocab_swap` | false | Re-index the draft onto the target's vocabulary |
| `external.external_report_acc` | true | Log top-1 agreement during training |
| `data.sparse_data_path` | — | Dumped top-k policy; selects `data.mode=sparse` |
| `data.sparse_cache_dir` | — | Where the memmapped cache is built |
| `data.offline_data_path` | — | Dumped hidden states; the alternative to the above |
| `model.use_fake_base_for_offline` | true | Load only the target's `lm_head`, not its layers |

> **Note on `external_top_k` / `external_top_p`:** the stored policy *is* the teacher
> distribution, so a dump narrower than the objective expects charges the draft for mass on
> tokens the teacher actually supported. A top-k mismatch draws a warning under
> `tvd_deploy`; the dump does not record its top-p, so a top-p mismatch goes undetected.

### Objectives

| `external_loss` | Scores | Needs |
|-----------------|--------|-------|
| `soft_ce` | Cross-entropy against the target's soft distribution (the EAGLE offline objective) | hidden states |
| `tvd` | Total-variation distance between the two distributions | either format |
| `tvd_deploy` | The same, with both sides put through the serving filter first | either format |
| `tvd_ce` | `tvd` plus a cross-entropy term on the target's argmax token | either format |

On the sparse path the target side is truncated by the dump whichever objective runs, since
that is all the dump holds; `tvd` and `tvd_deploy` then differ only in whether the *draft*
is put through the filter too. Only `tvd_deploy` applies the same nucleus to both sides.
Pairing `soft_ce` with `data.sparse_data_path` is a hard error, not a quiet degradation.

Report which truncation you trained under. `top_k` and `top_p` settings are not
interchangeable, so a number quoted without them is not comparable to anything.

`tvd_ce` exists because TVD is bounded and so ranks weakly among tokens inside the target's
support; cross-entropy's log penalty has no such ceiling.

### Teacher policy data

`$CONVERSATIONS` should carry responses the **target generated itself**, not the responses
that shipped with the prompts. The objective trains the draft to match the target at the
positions it will meet when serving, and those lie along the target's own samples; scoring
it against another model's text trains it on positions the target would rarely produce.
[Data Synthesis](../README.md#data-synthesis) covers generating them.

```bash
python collect_hidden_states/compute_sparse_policy_hf.py \
  --model $TARGET --input-data $CONVERSATIONS --output-dir $POLICY_DIR \
  --top-k 20 --top-p 0.95 --max-seq-len $SEQ_LEN \
  --answer-only-loss --chat-template ./template.jinja
```

Set `--max-seq-len` to the `training.training_seq_len` you will train at. The producer skips
whole conversations longer than it, scored turns included, and prints how many it skipped. The
collator pads every sample to `training_seq_len`, so over-sizing spends the difference on
padding and under-sizing drops the long conversations from the dump.

On a large corpus the first run spends a long time building the memmapped cache beside the
shards. Under `torchrun` only rank 0 builds it and the others wait, so build it once in a
single process first and point every run at the same `data.sparse_cache_dir`:

```bash
python - <<PY
from modelopt.torch.speculative.external.sparse_data import (
    SparsePolicyDataset,
    discover_shards,
)

SparsePolicyDataset(
    discover_shards("$POLICY_DIR"), max_length=$SEQ_LEN, cache_dir="$CACHE_DIR"
)
PY
```

Use `discover_shards` and the same `cache_dir` the training run will use: a cache keyed on a
different shard list is ignored, and `torchrun` rebuilds it from scratch. A ragged
third-party dump also needs `top_k=<its top-k>` here, since the cache width is otherwise
inferred from the first record and the recipe has no field to set it.

> **Note on `answer_only_loss` and chat templates:** when `--answer-only-loss` is set, the
> chat template must wrap assistant content in `{% generation %}` / `{% endgeneration %}`
> tags. Most published templates do not. Copy the model's own template and add them:
>
> 1. Print `AutoTokenizer.from_pretrained(TARGET).chat_template` to a file
> 2. Wrap the assistant content each branch emits, including a reasoning branch if present
> 3. Pass it with `--chat-template` to **every** dump command for that target
>
> Verify by decoding, not by counting: `tokenizer.decode(record["gen_ids"])` should read as
> the assistant's reply alone. A misplaced wrap still cuts at the wrong boundary, so a
> length check passes while the span is wrong.

The dense alternative dumps hidden states instead. Pass `--no-aux-hidden-states`: this mode
never reads the aux planes and they dominate the dump size.

```bash
python collect_hidden_states/compute_hidden_states_hf.py \
  --model $TARGET --input-data $CONVERSATIONS --output-dir $HIDDEN_DIR \
  --max-seq-len $SEQ_LEN --no-aux-hidden-states --answer-only-loss --chat-template ./template.jinja
```

### Cross-tokenizer vocabulary swap (`external_vocab_swap`)

Speculative decoding needs the draft to propose tokens the target can verify, so the two
must index the same vocabulary. When the target's family has no small member to draft with,
`external.external_vocab_swap=true` rebuilds the draft's embeddings and `lm_head` against
the target's vocabulary before conversion. Rows for tokens both tokenizers spell identically
carry over; the rest are drawn from the draft's own embedding statistics.

The run reports what carried over, typically a minority of the table and nearly all of the
most frequent ids, since the rare tail is where tokenizers disagree.

> **Note on judging a swap:** the match percentage describes the swap, not the draft.
> Overlap figures far apart have produced similar acceptance. Measure acceptance on held-out
> data before committing to a swapped draft, and prefer a vocabulary-matched draft where one
> exists: the replacement matrix is `base_vocab_size × hidden_size` parameters, a large
> fraction of a small draft, and a swapped draft stays behind a matched one of similar size.

A swapped draft starts with a large untrained fraction and wants a higher learning rate than
a matched draft of comparable size.

### Checkpoint resume

A resumed draft keeps the embeddings it was saved with. The swap reads
`external_vocab_swap` out of the checkpoint's own ModelOpt state, so re-running with the
flag on does not rewrite rows that are already base-indexed.

### Export

Training writes an ordinary HuggingFace checkpoint, shipped with the **target's** tokenizer
when the vocabulary was swapped, since the draft's rows are indexed by the target's ids from
that point on. Nothing in the saved artifact depends on ModelOpt at load time.

## Reference Configuration (Llama-3.2-1B drafting for Qwen3.8-27B)

A cross-tokenizer draft: Llama-3.2-1B-Instruct re-indexed onto Qwen3.8-27B's
248,320-token vocabulary. Trained on a sparse teacher policy dumped once from the target,
8 GPUs, one epoch.

### Teacher policy

The [teacher policy data](#teacher-policy-data) dump, with `--max-seq-len 12288` to match
`training.training_seq_len` below.

### Training

```bash
torchrun --nproc_per_node 8 main.py \
  --config ../../modelopt_recipes/general/speculative_decoding/external_draft.yaml \
  model.model_name_or_path=$TARGET \
  draft_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  external.external_vocab_swap=true \
  external.external_loss=tvd \
  external.external_top_k=20 external.external_top_p=0.95 \
  data.sparse_data_path=$POLICY_DIR \
  data.sparse_cache_dir=$CACHE_DIR \
  training.training_seq_len=12288 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=2 \
  training.learning_rate=2.0e-5 \
  training.warmup_steps=250 \
  training.max_grad_norm=1.0 \
  training.num_train_epochs=1 \
  training.seed=42 \
  training.output_dir=$OUTPUT_DIR
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Objective | `tvd` |
| Vocabulary swap | yes (Llama tokenizer onto Qwen's) |
| Sequence length | 12,288 |
| Effective batch | 16 (1 per device × 2 accumulation × 8 GPUs) |
| Learning rate | 2.0e-5 |
| Warmup steps | 250 |
| Epochs | 1 |
| Serving policy | top-k 20, top-p 0.95, temperature 1.0 |

> **Note on the learning rate:** we recommend sweeping against held-out acceptance to find
> the best rate for your own draft and target.

## Open Items

### Validated end-to-end

- Sparse teacher-policy training with `tvd`, from dump through training to an exported
  checkpoint, with acceptance measured on held-out conversations.
- The cross-tokenizer vocabulary swap, on a draft trained and measured this way.

### Covered by tests, not yet exercised in a full run

- `soft_ce`, `tvd_deploy` and `tvd_ce`, and the dense hidden-states data path.
- Resume, with and without a swap.
- Untied output heads, and heads carrying a bias.

### Not yet validated

- Streaming the teacher data from a live `vllm serve`. It is accepted for this mode but has
  not been exercised. Online training is rejected outright, since it produces no cached
  target outputs for the draft to be scored against.
- VLM targets. `main.py` routes through `load_vlm_or_llm`, so the path exists, but no
  external draft has been trained against one.
