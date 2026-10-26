# DFlash2

**A DFlash variant, not a separate pipeline.** DFlash2 is the DFlash draft backbone
plus two additions, selected with `dflash_architecture_config.projector_type=dflash2`:

- a **grouped dynamic depthwise convolution** wrapped around every attention and MLP
  sublayer, giving each block position a view of its predecessors inside the block.
  Taps are clipped at block boundaries and the wrapper is identity-initialized, so a
  fresh DFlash2 draft computes exactly what its DFlash backbone would;
- a **low-rank candidate selector** that scores transitions between adjacent block
  positions' top-k candidates, so serving walks one coherent path instead of taking
  the per-slot argmax. Trained by an extra cross-entropy term weighted by
  `dflash_selector_loss_alpha`.

Read `dflash.md` first — the pipeline, dump flags, export behaviour, and generic
failure modes are all shared. This sheet covers only the delta.

Recipe: `modelopt_recipes/general/speculative_decoding/dflash2.yaml` (its
`metadata.recipe_type` is `speculative_dflash`, and every knob lives in the `dflash.*`
namespace).

Examples: `tools/launcher/examples/Qwen/Qwen3-8B/hf_online_dflash2.yaml`,
`hf_streaming_dflash2.yaml`.

Reference: <https://inco.ai/blog/dflash2>. Serving support is vLLM PR #52816.

## Pipeline tasks

Two committed shapes. They differ in task count and in where the export lands, so
read the one you are running.

**Online** (`hf_online_dflash2.yaml`) — 2 tasks:

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/eagle3/make_dataset.sh` | Build training conversations (Daring-Anteater multi-turn SFT, 50K, `--full-conversations`) | `/scratchspace/data/train.jsonl` |
| task_1 | `common/specdec/dflash_online_training.sh` | Online train against a live base model, then export | `<training.output_dir>/exported-checkpoint-<step>` and `-final` |

The export goes under `training.output_dir` (`/scratchspace/dflash2_bs16` in the
example), **not** `/scratchspace/export` — that path belongs to the streaming flow.
Checking the wrong one reports a successful run as missing its export.

`data.mode=online` is the recipe default. As committed the example is a short
convergence check (`max_steps=2000`, 1 node x 8 GPUs). The header documents the full
published A/B run: 3 epochs over the 1.96M-conversation Spec-Decoding-Dataset-v1,
8 nodes x 8 H100, global batch 64, `save_steps=4000`.

**Streaming** (`hf_streaming_dflash2.yaml`) — 3 tasks:

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/eagle3/make_dataset.sh` | Build training conversations | `/scratchspace/data/train.jsonl` |
| task_1 | `common/eagle3/train_eagle_streaming.sh` | Streaming train (serve replicas + DDP trainers), then export | `/scratchspace/export` |
| task_2 | `common/specdec/vllm_smoke_test.sh` | Serve the exported drafter and measure acceptance length | gate: `MIN_ACCEPTANCE_LENGTH=1.2` |

See `dspark.md` for the streaming environment variables, which are shared across the
whole DFlash family. task_2 keeps `SPEC_METHOD: "dflash"` — vLLM has no separate
dflash2 method and selects the DFlash2 path from the checkpoint's
`architectures: ["DFlash2DraftModel"]`.

**Scope of the selector caveat.** The candidate selector is not applied in
`pseudo_speculative_generate`, so acceptance measured through the *local* AR path
reports the backbone plus convolutions alone. That does **not** apply to the
streaming example's vLLM smoke test, which serves the exported checkpoint and
measures real acceptance with the selector in play.

## Recipe and training knobs

Everything in `dflash.md` applies. DFlash2 adds:

| Override | Recipe default | Note |
| --- | --- | --- |
| `dflash.dflash_architecture_config.projector_type` | `dflash2` | Selects the variant |
| `dflash.dflash_architecture_config.conv_kernel_size` | 2 | Convolution taps; 2 = each position also sees its predecessor. Must be >= 1 and must not exceed `dflash_block_size` |
| `dflash.dflash_architecture_config.conv_group_size` | 16 | Must be >= 1 and must divide the draft's `hidden_size` |
| `dflash.dflash_architecture_config.selector_rank` | 256 | Rank of the transition codebooks. Must be >= 1 |
| `dflash.dflash_architecture_config.selector_top_k` | 16 | How many backbone candidates per position the selector re-ranks. Must be in `[1, vocab_size]` |
| `dflash.dflash_selector_loss_alpha` | 1.0 | Selector cross-entropy weight. **0 disables the selector** and trains backbone + convolutions only |
| `dflash.dflash_lk_loss_type` | `lambda` | One of `ce` / `tv` / `lambda` (config default `ce`). `lambda` anneals the block objective from cross-entropy toward acceptance as acceptance rises, and requires `dflash_self_logit_distillation: false`; `ce` is the only value compatible with distillation |
| `dflash.dflash_lk_ce_scale` / `dflash_lk_ce_decay` | 1.0 / 1.0 | Shape of that anneal |

`dflash_self_logit_distillation` is **false**, and not merely as a default: the
`lk_loss_type` anneal needs the draft's probability of the gold token, which the KD
path never forms.

Recipe defaults that differ from DFlash's: `block_size` 16, `num_anchors` 256,
`num_train_epochs` 6, `training_seq_len` 3072, `warmup_ratio` 0.04,
`learning_rate` 6.0e-4, `loss_decay_factor` 7.0.

`training.ddp_find_unused_parameters` is `true` by design — the selector parameters
are unused when `dflash_selector_loss_alpha == 0`, which would otherwise trip DDP.

## Per-model adjustments

Everything in `dflash.md`'s table applies, plus the DFlash-family items:

| Situation | What to change |
| --- | --- |
| Any model | **The draft does not inherit the base model's GQA/FFN dims.** Set `num_attention_heads`, `num_key_value_heads`, `head_dim` and `intermediate_size` in `dflash_architecture_config` explicitly, or you get a silently wrong-shaped draft |
| Any non-Qwen3 base | `dflash2.yaml` hardcodes `dflash_mask_token_id: 151669`, a Qwen3-specific unused id. The pin is inherited, so a different base silently trains against a token that means something else. Override it |
| Changing `dflash_block_size` | `conv_kernel_size` must not exceed it, and the convolution needs a sequence length divisible by it |
| Changing the draft's `hidden_size` | `conv_group_size` must still divide it |

## Success markers

Same as `dflash.md`, with the export path differing by mode — check the one the run
actually uses:

| Mode | Export lands in | Further evidence |
| --- | --- | --- |
| Online | `<training.output_dir>/exported-checkpoint-{<step>,final}` | training progress only |
| Streaming | `/scratchspace/export` | task_2's acceptance length against `MIN_ACCEPTANCE_LENGTH` |

## Quality gate

**Do not trust in-training AR for DFlash2.** The recipe pins `estimate_ar: false`
and `ar_validate_steps: 0` deliberately: eval takes a plain per-position argmax, so
the candidate selector is not applied. The convolutions *are* — they run inside the
backbone layers — so a reported AR measures backbone + convolutions and understates
the trained model.

Evaluate by serving the export instead. The streaming example already does this in
task_2 (`vllm_smoke_test.sh`, gate `MIN_ACCEPTANCE_LENGTH=1.2`); the online example
ships no such step, so benchmark its export separately on the serving stack
(vLLM PR #52816) or via the offline acceptance-length harness.

The training-regression gate from `dflash.md` applies. The online example sets
`MAX_FINAL_LOSS=7.0` (DFlash2's loss runs higher than DFlash's, so 5.0 fails healthy
runs) and `MIN_FINAL_ACC=0.15` for its 2000-step convergence check, and runs it strict
(`REGRESSION_STRICT=1`): a missed threshold, or no metrics to check, fails the
training task.

## Known failures

Generic infrastructure failures are in `../stages/triage.md`; shared block-diffusion
failures (`seq_len` divisibility, offline eval, mask token, chat template) are in
`dflash.md`. DFlash2-specific:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| `DFlash2 (projector_type='dflash2') requires <keys> in dflash_architecture_config` | Convolution or selector geometry missing | Supply all four: `conv_kernel_size`, `conv_group_size`, `selector_rank`, `selector_top_k` |
| `DFlash2 (projector_type='dflash2') requires an integer '<name>' in dflash_architecture_config` | A geometry key is present but not an int | Fix the type — a YAML string will not be coerced |
| `DFlash2 conv_kernel_size must be >= 1` / `must not exceed dflash_block_size (N)` | Tap count out of range | Set `1 <= conv_kernel_size <= dflash_block_size` |
| `DFlash2 conv_group_size (N) must be >= 1 and divide hidden_size (H)` | Group size does not divide the draft hidden size | Pick a divisor of the draft's `hidden_size` |
| `DFlash2 convolution needs a sequence length divisible by block_size (N), got M` | `training_seq_len` not a multiple of `dflash_block_size` | Same divisibility rule as DFlash — round `training_seq_len` to a multiple |
| `DFlash2 selector_rank must be >= 1` / `selector_top_k must be in [1, vocab_size=V]` | Selector geometry out of range | Correct the value |
| `dflash_lk_loss_type=... needs the draft's probability of the gold token, which the KD path never forms` | A non-`ce` `dflash_lk_loss_type` alongside `dflash_self_logit_distillation: true` | Keep `dflash_self_logit_distillation: false`, or set `dflash_lk_loss_type: ce` to keep distillation. **Not `null`** — the field is `Literal["ce", "tv", "lambda"]` and rejects it |
| DDP hangs or complains about unused parameters | `dflash_selector_loss_alpha` set to 0, leaving the selector untrained | Keep `training.ddp_find_unused_parameters: true`, as the recipe does |
| Benchmark reports a poor acceptance length | `--draft_length` was passed. The DFlash family reads `--block_size`, and it must match the drafter's | Pass `--block_size <N>` matching the drafter |
