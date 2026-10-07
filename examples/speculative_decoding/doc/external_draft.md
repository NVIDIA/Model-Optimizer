# External Draft — Training a Standalone Draft Model

The head-based modes (EAGLE, Medusa, DFlash) graft a draft head onto the target and train
it against the target's hidden states. The `external` mode instead trains an **ordinary
pretrained causal LM** as the draft. The target is never modified, and the result exports as
a plain HuggingFace checkpoint that TRT-LLM, vLLM and SGLang load with no mode-specific
support.

This guide walks the whole workflow: dump a teacher policy once, then train several drafts
against it — including a draft whose tokenizer differs from the target's.

## Contents

1. [Choosing a draft](#1-choosing-a-draft)
2. [Generating the teacher dataset](#2-generating-the-teacher-dataset)
3. [Training the drafts](#3-training-the-drafts)

## 1. Choosing a draft

Any pretrained causal LM can be the draft. Two properties matter:

- **Vocabulary.** The draft must propose tokens the target can verify. A draft that already
  shares the target's tokenizer needs nothing special. One that does not requires
  `external.external_vocab_swap=true`, which rebuilds its embeddings and lm_head against the
  target's vocabulary — a large change to a small draft, so prefer a matched draft where one
  exists.
- **Size.** The draft runs autoregressively at every speculated position, so its cost is
  paid on every proposal. Smaller is cheaper per token but proposes worse.

This guide trains three, which between them cover the cases you are likely to hit:

| draft | relationship to target | needs swap |
|---|---|---|
| mid-size, same family | shares the tokenizer | no |
| small, same family | shares the tokenizer | no |
| small, different family | different tokenizer | **yes** |

## 2. Generating the teacher dataset

Two formats are supported. Both are produced once and reused across every draft.

### Sparse teacher policy (recommended)

The TVD objectives only ever need the target's *truncated* next-token distribution, not a
full hidden state per token. Storing that directly is dramatically smaller and needs no
target lm_head at training time:

Supervising only the response requires a chat template that marks where the response
starts. Most published templates do not carry those markers, so prepare one first — copy the
model's own template and wrap the assistant content it emits:

```bash
cd examples/speculative_decoding
cp $TARGET_MODEL/chat_template.jinja ./template.jinja
# then edit the assistant branch so the content it emits is wrapped:
#     {{- '<|im_start|>' + message.role + '\n' }}{% generation %}{{- content }}{% endgeneration %}
# A template may emit assistant content in more than one branch (for example with and
# without a reasoning block); wrap every one of them.
```

```bash
python collect_hidden_states/compute_sparse_policy_hf.py \
  --model $TARGET_MODEL \
  --input-data $CONVERSATIONS_JSONL \
  --output-dir $POLICY_DIR \
  --top-k 20 --top-p 0.95 \
  --answer-only-loss --chat-template ./template.jinja
```

Check the split in the output before training on it: `prompt_ids` should hold the context and
`gen_ids` only the response. If `prompt_ids` has a single entry then nothing was masked and
you are about to supervise the prompt as well as the answer.

`--top-k` / `--top-p` must match `external.external_top_k` / `external.external_top_p` at
training time. The stored policy *is* the teacher distribution, so a dump narrower than the
objective expects charges the draft for mass on tokens the teacher actually supported. The
loader warns if the stored rows do not sum to ~1, which is the symptom.

Shard the work with `--dp-rank` / `--dp-world-size` across however many GPUs you have.

### Dumped hidden states (alternative)

The offline EAGLE flow also works, and is the only way to train `soft_ce` or to measure TVD
against the *untruncated* target distribution:

```bash
python collect_hidden_states/compute_hidden_states_hf.py \
  --model $TARGET_MODEL --input-data $CONVERSATIONS_JSONL \
  --output-dir $HIDDEN_DIR --no-aux-hidden-states
```

Pass `--no-aux-hidden-states`: this mode never reads the aux planes and they dominate the
dump size. Only modes that ignore them accept such a dump; EAGLE and DFlash still fail
loudly on one.

## 3. Training the drafts

All three drafts train from the same `$POLICY_DIR` with the same recipe. Only
`draft_model_name_or_path` — and, for the cross-tokenizer draft, one extra flag — changes.

### Vocabulary-matched drafts

```bash
torchrun --nproc_per_node 8 main.py \
  --config ../../modelopt_recipes/general/speculative_decoding/external_draft.yaml \
  model.model_name_or_path=$TARGET_MODEL \
  draft_model_name_or_path=$DRAFT_MODEL \
  data.sparse_data_path=$POLICY_DIR \
  external.external_loss=tvd \
  external.external_top_k=20 external.external_top_p=0.95 \
  training.output_dir=$OUTPUT_DIR \
  training.training_seq_len=12288 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=2 \
  training.learning_rate=1.0e-5 \
  training.warmup_steps=250 \
  training.num_train_epochs=1 \
  training.seed=42
```

Repeat with a different `draft_model_name_or_path` for each matched draft. Nothing else
changes — the dataset, the target and the objective are shared.

Because the sparse path never places the target on the GPU, memory is dominated by the draft
and its optimizer state, not by the target. A small draft trains comfortably on far fewer
GPUs than the target itself would need.

### Cross-tokenizer draft

Add one flag. The swap runs before conversion, rebuilds the draft's embeddings and lm_head
against the target's vocabulary, and reports what carried over:

```bash
torchrun --nproc_per_node 8 main.py \
  --config ../../modelopt_recipes/general/speculative_decoding/external_draft.yaml \
  model.model_name_or_path=$TARGET_MODEL \
  draft_model_name_or_path=$CROSS_TOKENIZER_DRAFT \
  external.external_vocab_swap=true \
  data.sparse_data_path=$POLICY_DIR \
  external.external_loss=tvd \
  training.output_dir=$OUTPUT_DIR \
  training.learning_rate=5.0e-5 \
  training.num_train_epochs=1
```

A swapped draft starts with a large untrained fraction, so it needs a higher learning rate
and more data than a matched one. Budget well beyond a single epoch.

### Choosing the objective

| `external_loss` | scores |
|---|---|
| `soft_ce` | cross-entropy against the target's soft distribution (the EAGLE offline objective); needs dumped hidden states |
| `tvd` | total-variation distance between the two distributions |
| `tvd_deploy` | the same, with both sides put through the serving filter first |
| `tvd_ce` | `tvd` plus a cross-entropy ranking term |

Under rejection sampling the acceptance probability is exactly `sum(min(p, q))`, so
`tvd_deploy` is `1 - acceptance` — the deployment metric itself rather than a proxy for it.
`tvd_ce` exists because TVD is bounded and so discriminates weakly between tokens inside the
target's support; the log penalty of cross-entropy does not have that ceiling.

Report which truncation you trained under. `top_k` and `top_p` settings are not
interchangeable, and a number quoted without them is not comparable to anything.
