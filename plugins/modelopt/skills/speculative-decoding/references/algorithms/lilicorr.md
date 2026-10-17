# LiLiCorr

**A DFlash variant, not a separate pipeline.** LiLiCorr is the DFlash draft backbone
plus a **reranker over the candidate lattice the backbone already produces**, selected
with `dflash_architecture_config.projector_type=lilicorr`. The backbone keeps its
top-k tokens per slot; a small two-layer transformer scores transitions between
adjacent slots' candidates, and serving commits a path greedily, left to right. It is
trained jointly with the backbone, so the drafter learns to propose candidates that
correlate into longer accepted sequences.

Read `dflash.md` first — the pipeline, dump flags, export behaviour, and generic
failure modes are all shared. This sheet covers only the delta.

Recipes — **two files, one algorithm**:

| Recipe | What it is |
| --- | --- |
| `modelopt_recipes/general/speculative_decoding/lilicorr.yaml` | The published `base` variant. Trains today with no extra dependency |
| `modelopt_recipes/general/speculative_decoding/lilicorr_conv.yaml` | The same, plus DFlash2's grouped sublayer convolutions. See *The conv recipe* below |

Recipes do not compose, so the second is a standalone file rather than an overlay —
**keep shared fields in the two in sync when editing either.**

Example: `tools/launcher/examples/Qwen/Qwen3-8B/hf_online_lilicorr.yaml`.

Paper: <https://arxiv.org/abs/2608.20530>. Serving support is
[sgl-project/sglang#37462](https://github.com/sgl-project/sglang/pull/37462) — the
reranker's serving path is in SGLang, not vLLM.

## Pipeline tasks

One committed shape, 2 tasks — identical in layout to DFlash's online example, only
the `--config` recipe differs.

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/eagle3/make_dataset.sh` | Build training conversations (Daring-Anteater multi-turn SFT, 50K, `--full-conversations`) | `/scratchspace/data/train.jsonl` |
| task_1 | `common/specdec/dflash_online_training.sh` | Online train against a live base model, then export | `/scratchspace/lilicorr_bs16` |

**Online only, and not by convention.** `data.mode: online` is a hard requirement:
the distractor penalty weights every competing candidate by the target model's own
logit gap, so a target model has to be in the process. An offline or streaming run
fails at loss construction rather than training a weaker model.

As committed the example is a short convergence check (`max_steps=2000`,
1 node x 8 GPUs). The published Qwen3-8B numbers come from 6 epochs at
8 nodes x 8 H100, global batch 64.

No inference path is wired in. The reranker is not applied in
`pseudo_speculative_generate`, so acceptance measured from this repo reports the
backbone alone.

## Recipe and training knobs

Everything in `dflash.md` applies. LiLiCorr adds a three-term head objective on top
of the DFlash block loss:

```text
loss = dflash_loss + w_ce*CE + w_margin*hinge + w_pen*penalty
```

No outer multiplier, so `loss == origin_loss + lilicorr_loss` holds exactly.

| Override | Recipe default | Note |
| --- | --- | --- |
| `dflash.dflash_architecture_config.projector_type` | `lilicorr` | Selects the variant |
| `dflash.dflash_lilicorr_w_ce` | 0.25 | Cross-entropy term |
| `dflash.dflash_lilicorr_w_margin` | 0.0 | Hinge term. **0 in the `base` variant** |
| `dflash.dflash_lilicorr_w_pen` | 0.25 | Distractor penalty. Needs the target model's logits — hence online-only |
| `dflash.dflash_lilicorr_margin` | 2.0 | Hinge width in log-potential units. Unused while `w_margin` is 0; must be > 0 when it isn't |
| `dflash.dflash_fp32_master_weights` | true | The published numbers were trained with this on. Turning it off changes the optimizer's arithmetic, not just its memory |

**Two published variants**, differing only in how the cross-entropy block is split:

| Variant | `w_ce` | `w_margin` | `w_pen` |
| --- | --- | --- | --- |
| `base` (the shipped recipe) | 0.25 | 0.0 | 0.25 |
| `margin` | 0.125 | 0.125 | 0.25 |

The head's total weight is 0.50 either way. The weights are **absolute and validated
all-or-nothing** — at least one of the three must be above 0.

Reranker geometry, under `dflash_architecture_config`. **Every field is required and
never defaulted**, which matters more here than elsewhere: `lilicorr_candidate_topk`
sets the lattice width and the shape of `rank_embedding`, while `lilicorr_logit_scale`
and `lilicorr_vector_eps` change the score **without changing any tensor shape**. A
guessed value for those two builds a head that loads cleanly and scores a different
function.

| Field | Default |
| --- | --- |
| `lilicorr_candidate_topk` | 8 |
| `lilicorr_hidden_size` | 1024 |
| `lilicorr_factor_dim` | 1024 |
| `lilicorr_num_layers` | 2 |
| `lilicorr_num_heads` | 8 |
| `lilicorr_mlp_ratio` | 2.0 |
| `lilicorr_logit_scale` | 8.0 |
| `lilicorr_vector_eps` | 1.0e-4 |

`dflash_init_checkpoint` restores **weights only** and reads geometry from the config,
so a warm start reproduces a head only if every field above matches the one the
checkpoint was trained with.

Other recipe defaults that differ from DFlash's: `block_size` 16, `num_anchors` 512,
`loss_objective` `decay` with `decay_factor` 7.0 (not `dpace` — the published variants
were trained on the static decay), `lr_scheduler_type` **cosine** (not linear — the
published variants used a linear warmup into cosine decay, and the schedule is part of
the recipe those numbers came from), `num_train_epochs` 6, `training_seq_len` 3072.

`dflash_self_logit_distillation` is **false**: the reranker's terms are added to the
plain weighted cross-entropy, and turning KD on would replace that base term and
change the objective the published checkpoints were trained under.

## The conv recipe

`lilicorr_conv.yaml` is `lilicorr.yaml` with DFlash2's grouped dynamic depthwise
convolution wrapped around every draft sublayer. The convolution is **shared code** —
`DFlashGroupedConv` imported from the DFlash2 plugin, installed on the no-op sublayer
seam `DFlashDecoderLayer` already exposes — so the two variants cannot drift apart
arithmetically.

It adds three keys to `dflash_architecture_config`:

| Field | Default | Note |
| --- | --- | --- |
| `conv_kernel_size` | 2 | Tap count; must not exceed `dflash_block_size` |
| `conv_group_size` | 16 | Must divide the draft's `hidden_size` |
| `conv_projection_init_std` | 0.0 | **Zero means exact identity at init** — see below |

The two conv geometry keys are **all-or-nothing**: one alone is rejected rather than
silently building a draft without convolutions.

`conv_projection_init_std: 0.0` zeroes `kernel_projection`, and `base_kernel` is
identity at tap 0, so the wrapper is an *exact* identity at step 0. A run from this
recipe begins as the plain reranker, and any difference is attributable to the
convolutions rather than to a perturbed start. Raise the key only if you want a
perturbed start deliberately. It is a separate key from `initializer_range`, which
also seeds the reranker.

**Memory is the binding constraint.** The convolutions add ~42M trainable parameters
(20 tensors for a 5-layer draft), and it is the activations they hold that bind. At an
8B target, combined with `dflash_fp32_master_weights`, this is the memory worst case
and may need `training.gradient_checkpointing: true` to fit on 80 GiB; at a 4B target
it fits without. Checkpointing is mathematically neutral — same objective, same data
order, same resulting model — but it trades step time for memory, so a run using it is
not step-time-comparable with one that does not.

**Historical note:** this recipe used to be unusable on `main` because the DFlash2
plugin lived on a branch. `modelopt/torch/speculative/plugins/modeling_dflash2.py` is
now on `main`, so it trains today. If you hit the `ImportError` below, the install is
older than that merge.

## Per-model adjustments

Everything in `dflash.md`'s table applies, plus the DFlash-family items:

| Situation | What to change |
| --- | --- |
| Any model | **The draft does not inherit the base model's GQA/FFN dims.** Set `num_attention_heads`, `num_key_value_heads`, `head_dim` and `intermediate_size` in `dflash_architecture_config` explicitly |
| Any non-Qwen3 base | Both recipes hardcode `dflash_mask_token_id: 151669`, a Qwen3-specific unused id. Override it |
| Reproducing published numbers | Use `tools/launcher/examples/Qwen/Qwen3-8B/chat_template_train.jinja`, as the other speculative recipes do. It differs from the reference mask by 6 tokens of supervision per record (the empty `<think>` preamble and the end-of-turn token); token ids are identical either way |
| Warm start from a published head | Every reranker geometry field must match the checkpoint's, including `lilicorr_logit_scale` and `lilicorr_vector_eps`, which affect no shape and so fail silently |

## Success markers

Same as `dflash.md`. The example ships no smoke test or AR eval, so the in-pipeline
evidence is training progress plus the export landing in `training.output_dir`.

## Quality gate

**Do not trust in-training AR for LiLiCorr.** The recipes pin `estimate_ar: false`
and `ar_validate_steps: 0` deliberately: eval runs the DFlash backbone only, with the
reranker not applied in `pseudo_speculative_generate`, so a reported AR describes the
backbone alone and understates the trained model.

Evaluate by exporting the drafter and benchmarking it on SGLang
(sgl-project/sglang#37462). Published Qwen3-8B acceptance lengths for context, all
trained on one matched contract and served through SGLang on a single H100 at
concurrency 1: LiLiCorr+conv 7.715 on gsm8k and 4.014 on mtbench, plain LiLiCorr
7.557 / 3.939, against DFlash's head-free control at 6.341 / 3.478. Treat these as
reference points for a reproduction, not as a pass threshold — see
`../stages/validate.md` on why no threshold is enforced anywhere in this repo.

The training-regression gate from `dflash.md` (`MAX_FINAL_LOSS`, `MIN_FINAL_ACC` via
`check_regression.py`) applies; the online example sets `MAX_FINAL_LOSS=5.0` and
`MIN_FINAL_ACC=0.15` for its 2000-step convergence check.

## Known failures

Generic infrastructure failures are in `../stages/triage.md`; shared block-diffusion
failures (`seq_len` divisibility, offline eval, mask token, chat template) are in
`dflash.md`. LiLiCorr-specific:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| `All three LiLiCorr objective weights are 0, so the reranker would never ...` | `w_ce`, `w_margin` and `w_pen` all 0 | Set at least one above 0 |
| `dflash_lilicorr_w_pen > 0 requires the target model's logits, which are ...` | Offline or streaming mode with the penalty enabled | Use `data.mode: online`, or set `w_pen: 0` |
| `dflash_lilicorr_w_margin > 0 requires a positive dflash_lilicorr_margin` | Hinge enabled with a zero/negative width | Set `dflash_lilicorr_margin` above 0 |
| `LiLiCorr block_size must be >= 2, got N` | Block too small for a lattice | Raise `dflash_block_size` |
| `LiLiCorr candidate_topk must be >= 1` / `lilicorr_candidate_topk=K exceeds the vocabulary size V` | Lattice width out of range | Set `1 <= lilicorr_candidate_topk <= vocab_size` |
| `... DFlashGroupedConv, which is not available in this installation. Remove conv_kernel_size and conv_group_size from dflash_architecture_config to ...` | `lilicorr_conv.yaml` on an install predating the DFlash2 merge | Update the install, or drop the two conv keys to train plain LiLiCorr |
| `... requires conv_kernel_size and conv_group_size ... The grouped convolution needs the tap count ...` | Only one of the two conv keys supplied | They are all-or-nothing — set both or neither |
| Warm-started head scores differently despite loading cleanly | `lilicorr_logit_scale` or `lilicorr_vector_eps` differ from the trained head; neither affects any tensor shape, so nothing errors | Transcribe every reranker field from the checkpoint's config |
| Conv run OOMs at an 8B target | ~42M extra params plus their activations, on top of fp32 master weights | `training.gradient_checkpointing: true` — neutral to the result, costs step time |
| Benchmark reports a poor acceptance length | `--draft_length` was passed, or the drafter was benchmarked on vLLM | The DFlash family reads `--block_size`; the reranker's serving path is SGLang |
