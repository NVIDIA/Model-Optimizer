# Draft train-vs-serve activation diff

Tools for answering "the drafter trains fine but acceptance is ~1.0 — where do
training and serving disagree?" by running ONE fixed sample through both paths
and diffing every intermediate tensor in forward order.

This found a real defect on Gemma-4-E4B: `FakeBaseModel` rebuilt the base
embedding as a plain `nn.Embedding` and dropped Gemma's `sqrt(hidden_size)`
forward-time scale, so the draft trained on inputs 50.6x smaller than vLLM feeds
it at serve time. Training loss fell normally and `train_acc` reached 31.5%,
while MT-Bench AL sat at 1.0040. Nothing errored anywhere.

## Why activation diffing and not more AL runs

AL is an end-to-end scalar. It cannot tell you *where* the two paths diverge,
and it is easily confounded by a second unrelated defect. Five config-level
hypotheses (chat template, `target_layer_ids`, weight loading, embedding scale
tested on the *serving* side, RoPE theta) were each tested against AL and each
gave a wrong answer. The activation diff localized the bug in one run.

## Usage

    # 1. serving-side reference (also produces the aux hidden states)
    python dump_serve.py --base <base> --draft <vllm-format drafter> \
        --corpus <jsonl> --row 0 --anchor 64 --seqlen 256 \
        --capture-ids 6,12,18,24,36,42 --chat-template <jinja> --out OUT

    # 2. training side, consuming the SAME aux tensors
    python dump_train.py --ckpt <training checkpoint> --base <base> \
        --corpus <jsonl> --row 0 --anchor 64 --seqlen 256 \
        --chat-template <jinja> --aux-from OUT/eval_dump.pt --out OUT

    # 3. diff, in forward order -- the FIRST mismatch localizes the bug
    python compare_activations.py OUT/train_dump.pt OUT/eval_dump.pt

Both sides must consume identical aux hidden states (`--aux-from`) and the same
fixed anchor, or the comparison is meaningless: anchors are normally sampled
with `torch.rand`, so `dump_train.py` monkeypatches `_sample_anchor_positions`.

## Scope

`dump_serve.py` is a faithful reimplementation of vLLM's Gemma4DSpark math, not
a wrapper around vLLM's kernels. On the validation sample it agrees with the
ModelOpt training path to <2% (bf16 accumulation) with identical top-1 tokens.
That is enough to localize logic bugs — wrong scale, wrong mask, wrong layer
wiring — but it does NOT validate vLLM's paged-attention kernel, which needs a
populated KV cache and attention metadata to exercise. Treat "serving" here as
"the serving math", not "the serving kernel".

`probe_norm_grads.py` runs one real backward and prints per-parameter gradient
norms — use it to tell "gradients are zero" (broken graph) apart from
"gradients flow but updates round away" (bf16 resolution).

## Traps, all of which cost real debugging time

- **Report a norm RATIO, not just cosine.** Cosine is blind to pure scale
  errors: the 50.6x embedding bug showed `cos = 1.000003`. Cosine over a
  flattened multi-million-element bf16 tensor is also unreliable near 1.0 — it
  read `1.0035` on bit-identical inputs. Trust `max|d|` and relative error.
- **Offline/streaming checkpoints refuse an eval forward** (base layers are
  deleted). Run `model.train()` under `no_grad` and pass `base_model_outputs`.
- **`mto.enable_huggingface_checkpointing()` is required** before
  `from_pretrained`, or you get a bare `FakeBaseModel` and every
  `dflash_module.*` tensor is silently reported UNEXPECTED and dropped.
- **ModelOpt has its own `apply_rotary_pos_emb`** (Q takes the LAST `q_len`
  positions, K takes all). The Qwen3 one assumes `len(Q) == len(K)` and raises.
- **vLLM's `_kv_proj` returns K *after* `k_norm`.** Tapping raw `k_proj` output
  on the training side compares different tensors (looked like 16%, was 0%).
- **The draft attention mask is not a per-query causal ramp.** Measured from the
  tensor training actually passes to SDPA: all draft queries in a block share
  ONE window over context `[0, anchor)` — position `anchor` itself is excluded —
  plus the full block. A block is predicted in one shot.
- **HF `sdpa_attention_forward` returns `.transpose(1, 2)`** → `[B, q, H, D]`.
  Reconstructing with `einsum(...->qhd)` gives different head packing.
- **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** is required to reach a constructed vLLM
  module from the driver process; otherwise EngineCore spawns elsewhere.
