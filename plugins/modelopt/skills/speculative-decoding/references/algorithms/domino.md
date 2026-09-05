# Domino

**A DFlash variant, not a separate pipeline.** Domino is the DFlash draft backbone
plus a lightweight causal correction head — a GRU over the block's previously decoded
tokens producing a logit correction on the block suffix — selected with
`dflash_architecture_config.projector_type=domino`. It trains with a base/final dual
loss whose `lambda_base` weight decays from 1 to 0 over training (curriculum).

Read `dflash.md` first — the pipeline, dump flags, export behaviour, and generic
failure modes are all shared. This sheet covers only the delta.

Recipe: `modelopt_recipes/general/speculative_decoding/domino.yaml` (its
`metadata.recipe_type` is `speculative_dflash`, and every knob lives in the `dflash.*`
namespace).

Example: `tools/launcher/examples/Qwen/Qwen3-8B/hf_online_domino.yaml`. Reference:
SpecForge PR #571 (z-lab); drafter format `huggingface.co/Huang2020/Qwen3-8B-Domino-b16`.

## Pipeline tasks

The committed example is **online**, 2 tasks:

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/eagle3/make_dataset.sh` | Build training conversations (Daring-Anteater multi-turn SFT, 50K, `--full-conversations`) | `/scratchspace/data/train.jsonl` |
| task_1 | `common/specdec/dflash_online_training.sh` | Train the draft, then export | `<output_dir>/exported-checkpoint-*` |

`--full-conversations` matters: it keeps real assistant completions so
`answer_only_loss` has assistant spans to mask.

**The inference side is intentionally not wired up yet.** The Domino correction head
is not applied in `pseudo_speculative_generate` or in the serving stack, so the
example ships no vLLM smoke test and no MT-Bench AR eval. Do not treat their absence
as a broken config.

When that path lands, copy the smoke-test and AR-eval steps from
`hf_online_dflash.yaml` (its `task_1` and `task_2`) and append them to the Domino
config as **`task_2` and `task_3`** — Domino's `task_0`/`task_1` are already the
dataset build and training, so keep the source file's numbering and the destination's
distinct.

## Recipe and training knobs

Everything in `dflash.md` applies. Domino adds:

| Override | Recipe default | Note |
| --- | --- | --- |
| `dflash.dflash_architecture_config.projector_type` | `domino` | Selects the variant |
| `dflash.dflash_architecture_config.emb_dim` | 256 | GRU head embedding dim. **Required** |
| `dflash.dflash_architecture_config.gru_hidden_dim` | 1024 | GRU hidden dim. **Required** |
| `dflash.dflash_architecture_config.pure_draft_prefix_len` | 1 | Positions at block start kept as base logits only (no causal correction). Must be in `[0, block_size-1]` |
| `dflash.dflash_architecture_config.shift_label` | true | Next-token alignment — **only `true` is supported** |
| `dflash.dflash_lambda_base_start` | 1.0 | Curriculum start weight on the base loss |
| `dflash.dflash_lambda_base_decay_ratio` | 1.0 | Fraction of training over which `lambda_base` decays to 0 |

`dflash_self_logit_distillation` is **false** — Domino trains its own base/final CE
losses rather than distilling target logits. Recipe defaults also differ from
DFlash's: `block_size` 16, `num_anchors` 256, `num_train_epochs` 6,
`training_seq_len` 3072, `warmup_ratio` 0.04, `max_grad_norm` 1.0.

`ddp_find_unused_parameters: true` is **required**, not incidental: while
`lambda_base == 1` the head params are absent from the backward graph and DDP would
otherwise fail.

## Per-model adjustments

Everything in `dflash.md`'s table applies. Additionally:

| Situation | What to change |
| --- | --- |
| Any model | **The Domino draft does not inherit the base model's GQA/FFN dims** — a fresh `Qwen3Config` already carries defaults, so `modify()`'s inherit-if-missing guard is a no-op. Set `num_attention_heads`, `num_key_value_heads`, `head_dim`, and `intermediate_size` explicitly. The Qwen3-8B reference drafter uses `32 / 8 / 128 / 12288`. |
| Any run | **Set `training.max_steps`.** The `lambda_base` curriculum is scheduled against `state.max_steps`; if it's unset the decay window collapses to one step and the curriculum is disabled (`lambda_base` 0 from the start). This warns rather than errors. The Qwen3-8B example sets `max_steps=2000`. |

## Success markers

Same as `dflash.md`: `Training time: N seconds`, then the `=== Exporting: ... ===`
and `=== Regression Check (...) ===` blocks, with `exported-checkpoint-*` on disk.

Because there is no smoke test or AR eval step, training completion plus a clean
export is the whole in-pipeline signal.

## Quality gate

**Do not trust in-training AR for Domino.** The recipe pins `estimate_ar: false` and
`ar_validate_steps: 0` deliberately: eval delegates to the DFlash backbone with the
correction head not applied, so reported acceptance rates are backbone-only. The code
logs this once as a warning — treat that warning as expected, not as a defect.

The training-regression gate from `dflash.md` applies and the Qwen3-8B example sets
it: `MAX_FINAL_LOSS=5.0`, `MIN_FINAL_ACC=0.15`, checked by `check_regression.py`
against `trainer_state.json`. Since no inference metric is produced, this is currently
the only automatic gate — and per `dflash.md` it is invoked with `|| true`, so confirm
the `=== Regression Check ===` block actually appears in the log.

## Known failures

Generic infrastructure failures are in `../stages/triage.md`; shared block-diffusion
failures (`seq_len` divisibility, offline eval, mask token, chat template) are in
`dflash.md`. Domino-specific:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| `Domino (projector_type='domino') requires ['emb_dim', 'gru_hidden_dim'] in dflash_architecture_config` | GRU head dims missing | Set both in `dflash_architecture_config` |
| `Domino currently supports shift_label=True (next-token alignment) only` | `shift_label=false` | Leave it at `true` |
| `pure_draft_prefix_len must be in [0, N] (block_size=B), got X` | Prefix length >= block size | Lower it below `block_size` |
| `DominoLambdaCallback: state.max_steps unset (<=0); lambda_base curriculum disabled` (warning) | `training.max_steps` not set | Set `training.max_steps`, else the curriculum never runs |
| `Domino eval uses the DFlash backbone only ...` (warning) | Correction head not applied at eval | Expected — do not chase it; evaluate after export once the inference path lands |
| DDP error about unused parameters | `ddp_find_unused_parameters` turned off | Keep it `true` |
| Draft trains but quality is poor | Draft dims left at `Qwen3Config` defaults instead of matching the base | Set the GQA/FFN dims explicitly |
