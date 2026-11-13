# External Draft — Training a Standalone Draft Model

An external draft is an ordinary pretrained causal LM trained to propose tokens for a
target model. Unlike the head-based modes (EAGLE, Medusa, DFlash) nothing is grafted onto
the target, and it has no dependence on target hidden states. Such drafters can be served efficiently on decode-efficient hardware such as the LPU.

This document serves as a "quick start" guide to get users up and running with training drafts.

Recipe: [`modelopt_recipes/general/speculative_decoding/external_draft.yaml`](../../../modelopt_recipes/general/speculative_decoding/external_draft.yaml)
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

## Training

We generally start draft training with an existing pre-trained small model. We then perform a "vocabulary" or "tokenizer swap" (see [`vocab_swap.py`](../../../modelopt/torch/speculative/external/vocab_swap.py)) to transform the existing small model into the target's embedding space. This allows for the small draft to be (1) directly optimized via logit distillation, (2) used directly as a draft, without the need for any transformation between the draft and target token spaces during inference.

The tokenizer-swap does not completely preserve the small draft model's performance, and on top of this, we need to align the draft. As our intended deployment setting is always speculative decoding at non-zero temperatures with rejection sampling, we tend to opt for TVD loss, or combinations of TVD loss and e.g. cross entropy. These losses directly minimise the rejection probability under rejection sampling and as such are advantageous here.

Furthermore, in search for a good drafter we recommend sweeping other hyperparameters, primarily the training loss, batch size, and potentially even teacher deployment settings.

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

We currently support the following objectives for external draft training though we intend to add more in the future:

| `external_loss` | Scores | Needs |
|-----------------|--------|-------|
| `soft_ce` | Cross-entropy against the target's soft distribution (the EAGLE offline objective) | hidden states |
| `tvd` | Total-variation distance between the two distributions | either format |
| `tvd_deploy` | The same, with both sides put through the serving filter first | either format |
| `tvd_ce` | `tvd` plus a cross-entropy term on the target's argmax token | either format |

We generally recommend using "TVD deploy" as it matches the deployment conditions for a draft model. On the sparse path the target side is truncated by the dump whichever objective runs, since that is all the saved dataset holds. The `tvd` and `tvd_deploy` losses then differ only in whether the *draft* is put through the filter too. Only `tvd_deploy` applies the same nucleus to both sides.

### Teacher policy data

Standalone draft models are aligned via logit distillation against the target model. The required
training data is a set of target model prompts and responses, with saved top-k logprobs for each
response token. Training data is teacher generated responses. Responses should be generated at the deployment sampling parameters the draft model is intended to be used at, usually for this we reference a target models recommended sampling parameters. This is so that the draft learns the teacher distribution under the intended deployment settings.

For a simple first attempt at draft training we tend to use 100-200k training samples taken from `nvidia/Nemotron-Post-Training-Dataset-v2`, we recommend initially trying an even split of datasets. If a user has a specific deployment use case then training data should be specialized toward that, however, we do still recommend maintaining a mix of subjects in the training data. Ideally some component of the training data should also match the context size that you intend to deploy the draft-target pair at.

The data format:

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

In general, standalone draft models have to share the same vocabulary/embedding space as their
target model. This allows for both the proposal of tokens in the target's native vocabulary, and
for direct alignment of the draft via logit distillation. However, a target does not always have a
small model in its own family, and sometimes a standalone draft must be produced from another
source. Instead of a costly process of training a standalone draft from scratch, we use a
"vocab-swap": taking an existing draft model candidate and transplanting the target model's
vocabulary and embeddings onto it, in a way that is not totally destructive. Set
`external.external_vocab_swap=true` to enable it. On a high level the process is to:

1. **Create new embedding tables** for the draft at the target's vocabulary size, filled with
   random rows drawn from the draft's original embedding statistics.
2. **Carry over every token that both tokenizers spell identically**, copying the draft's trained
   embedding row to the token's position in the target's vocabulary.
3. **Leave the remaining rows at those random values.** Draft-target alignment then trains them.
4. **Rebuild the output head the same way** when it is not tied to the embeddings.
5. **Point the special tokens** (BOS, EOS, padding) at the target tokenizer's ids.

After the swap the new resulting draft model undergoes the same draft-target alignment process as
an in-family draft would. The run reports what carried over, typically a minority of the table and
nearly all of the most frequent ids, since the rare tail is where tokenizers disagree.

> **Note on judging a swap:** the match percentage describes the swap, not the draft.
> Overlap figures far apart have behaved similarly, so do not read the coverage number as a
> quality score. Prefer a vocabulary-matched draft where one exists: the replacement matrix
> is `base_vocab_size × hidden_size` parameters, a large fraction of a small draft, and a
> swapped draft stays behind a matched one of similar size.

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
8 GPUs, two epochs.

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
  external.external_loss=tvd_ce \
  external.external_top_k=20 external.external_top_p=0.95 \
  data.sparse_data_path=$POLICY_DIR \
  data.sparse_cache_dir=$CACHE_DIR \
  training.training_seq_len=12288 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=2 \
  training.learning_rate=5.0e-5 \
  training.warmup_steps=250 \
  training.max_grad_norm=1.0 \
  training.num_train_epochs=2 \
  training.seed=42 \
  training.output_dir=$OUTPUT_DIR
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Objective | `tvd_ce` |
| Vocabulary swap | yes (Llama tokenizer onto Qwen's) |
| Sequence length | 12,288 |
| Effective batch | 16 (1 per device × 2 accumulation × 8 GPUs) |
| Learning rate | 5.0e-5 |
| Warmup steps | 250 |
| Epochs | 2 |
| Serving policy | top-k 20, top-p 0.95, temperature 1.0 |

> **Note on the learning rate:** we recommend sweeping to find the best rate for your own
> draft and target. The hyperparameters chosen here came from a sweep over learning rate and
> loss, scored on acceptance length.
