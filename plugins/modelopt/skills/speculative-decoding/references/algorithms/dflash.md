# DFlash

Block-diffusion draft: predicts a whole block of `block_size` tokens in one forward
pass instead of autoregressively. Design details, results, and open items are in
`examples/speculative_decoding/doc/dflash.md`; the paper is arXiv:2602.06036.

Examples: `tools/launcher/examples/*/*/hf_online_dflash.yaml`,
`hf_offline_dflash.yaml`, `hf_streaming_dflash_multi_node.yaml`,
`specdec_bench_dflash_vllm.yaml`.

## Pipeline tasks

DFlash has three variants, and none uses EAGLE3 offline's 4-task shape: **online is 3
tasks, offline is 2, streaming is 3.** Read the task count off the config you're
using rather than assuming one.

**Online** (`hf_online_dflash.yaml`) — base model forwards during training:

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/specdec/dflash_online_training.sh` | Train the draft, then export | `<output_dir>/checkpoint-*`, `<output_dir>/exported-checkpoint-*` |
| task_1 | `common/specdec/vllm_smoke_test.sh` | Serve target + draft, verify responses | Smoke-test log |
| task_2 | `common/specdec/ar_eval_mtbench.sh` | MT-Bench per-category AR evaluation (1 GPU) | AR per category |

**Offline** (`hf_offline_dflash.yaml`) — for base models too large to forward
alongside training:

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/eagle3/dump_offline_data_vllm.sh` | Dump base hidden states via vLLM (TP-sharded) | Hidden-state dump directory |
| task_1 | `common/specdec/dflash_online_training.sh` | Train on the dump, then export | `<output_dir>/exported-checkpoint-*` |

The dump script is shared with EAGLE3 — DFlash selects its own captured layers with
`--aux-layers dflash`. Two flags matter for the dump:

- `--aux-layers dflash` — **must match the draft's `num_hidden_layers`** (recipe
  default 5). `build_target_layer_ids(num_target_layers, num_draft_layers)` picks the
  captured layers, so a mismatch here silently produces the wrong hidden states.
- `--answer-only-loss` and `--chat-template` — must agree with the training task's
  `training.answer_only_loss` and `data.chat_template`.

Offline training additionally needs `data.mode=offline`,
`model.use_fake_base_for_offline=true` (loads only `lm_head` + `embed_tokens` rather
than the full base), and `data.offline_data_path` pointing at the dump.

**Streaming** (`hf_streaming_dflash_multi_node.yaml`) — same NIXL RDMA transport as
streaming EAGLE3, splitting nodes into serve replicas plus DDP trainers. See
`common/eagle3/train_eagle_streaming.sh` for dispatch and sharding.

**Benchmark** (`specdec_bench_dflash_vllm.yaml`) — `common/specdec_bench/run.sh` with
`--speculative_algorithm DFLASH` and `--block_size`.

## Recipe and training knobs

`modelopt_recipes/general/speculative_decoding/dflash.yaml`, passed to
`dflash_online_training.sh` via `--config` with OmegaConf dotted overrides. Full table
in `examples/speculative_decoding/README.md#dflash-block-diffusion-for-speculative-decoding`.

| Override | Default | Note |
| --- | --- | --- |
| `dflash.dflash_block_size` | 8 | Tokens predicted per block. `training.training_seq_len` **must** be divisible by it. |
| `dflash.dflash_num_anchors` | 512 | Random anchor positions sampled per sequence |
| `dflash.dflash_loss_decay_factor` | 4.0 | Exponential decay gamma; 0 disables |
| `dflash.dflash_self_logit_distillation` | true | Logit distillation from the target |
| `dflash.dflash_architecture_config.num_hidden_layers` | 5 | Draft decoder layers — keep in sync with `--aux-layers` |
| `dflash.dflash_mask_token_id` | auto | See *Per-model adjustments* |
| `dflash.dflash_swa_window_size` | unset | Sliding-window attention for the draft; must be >= `dflash_block_size` |
| `dflash.dflash_export_rope_scaling` | `{}` | YaRN config injected at export so a short-window draft can serve long context |
| `training.learning_rate` | 6.0e-4 | |
| `training.training_seq_len` | 4096 | |
| `data.chat_template` | — | Required when `answer_only_loss=true` |

Export is automatic: after training, rank 0 exports every `checkpoint-<step>` to
`exported-checkpoint-<step>`, plus `exported-checkpoint-final` when
`modelopt_state.pth` sits directly in `output_dir`.

## Per-model adjustments

| Situation | What to change |
| --- | --- |
| Any model | Pin `dflash.dflash_mask_token_id` to a token that **already exists in the target's embedding** — the draft reuses the target's `embed_tokens`. Unset falls back to `tokenizer.mask_token_id`, which many tokenizers lack. MiniMax-M2.7 uses a reserved row (200054); Qwen3-8B uses 151669. |
| `answer_only_loss=true` (recipe default) | The chat template must contain `{% generation %}` / `{% endgeneration %}` tags. Most stock templates don't — supply one via `data.chat_template=<path>.jinja`. Each model keeps its own next to its example YAML (`examples/<Org>/<Model>/chat_template_train.jinja`); copy the closest one. Note `dflash.yaml`'s comment points at a `chat_templates/` recipe directory that does not exist — ignore it. |
| `trust_remote_code` MoE with an older transformers pin | Set `OVERRIDE_TRANSFORMERS` in the task environment (MiniMax-M2.7 needs 4.57.1). Set `ACCELERATE_CONFIG` when the model needs FSDP2 via accelerate config rather than transformers-native `ParallelismConfig`. |
| Very large MoE base | Use the offline variant with `model.use_fake_base_for_offline=true`; plain DDP suffices, so no FSDP2 patches. Set `MIXED_PRECISION: "no"` with `training.bf16=false` if the model requires it. |
| Draft trained at short context, served long | Set `dflash.dflash_export_rope_scaling` (YaRN); factor = target context / `training_seq_len`. |
| Multi-node | Set `NUM_NODES` in the environment; `HEAD_NODE_IP` is auto-detected from Slurm. |

## Success markers

| Task | Log evidence | Artifact |
| --- | --- | --- |
| Hidden-state dump (offline) | vLLM extraction completes over the input data | Dump directory populated |
| Training | `Training time: N seconds`, then `=== Exporting: <ckpt> → <export_dir> ===` and `=== Regression Check (...) ===` | `<output_dir>/checkpoint-*/trainer_state.json`, `<output_dir>/exported-checkpoint-*` |
| Smoke test | `Auto-detected draft model: ...`, `Server ready after Ns` | Smoke-test log with responses |
| AR eval | Per-category MT-Bench AR output | AR results |
| Benchmark | `Average_AL` in the saved results | JSON under `--save_dir` |

## Quality gate

DFlash gates in three places rather than on one acceptance-rate number.

**1. Training regression** — `common/check_regression.py` reads the latest
`trainer_state.json` and compares against env thresholds set in the YAML:

| Env var | Meaning |
| --- | --- |
| `MAX_FINAL_LOSS` | Final loss must be below this |
| `MIN_FINAL_ACC` | Final accuracy must be above this (any log key containing `acc`) |

Qwen3-8B online reference uses `MAX_FINAL_LOSS=5.0`, `MIN_FINAL_ACC=0.15`. Its
convergence baseline (8×B200, bs=1, seq_len=4096, 5-layer draft, block_size=16, 100K
samples, 1 epoch ≈ 12,500 steps) is in the YAML header — compare against it when
judging whether a run under-trained.

Note: `check_regression.py` is invoked with `|| true`, and it only warns when no
`trainer_state.json` exists. A green Slurm exit is not proof the gate ran — confirm
the `=== Regression Check ===` block is present in the log.

**2. Smoke test** — `MIN_ACCEPTANCE_LENGTH` env var (Qwen3-8B online uses 1.4) with
`NUM_SPEC_TOKENS` speculative tokens.

**3. Benchmark** — `Average_AL` (average acceptance length) from
`common/specdec_bench/run.sh`. Acceptance length is concurrency-independent, so it is
the primary metric even when the run trades timing fidelity for wall clock.

## Known failures

Generic infrastructure failures are in `../stages/triage.md`. These are
DFlash-specific:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| `seq_len (N) must be divisible by block_size (B)` | `training.training_seq_len` not a multiple of `dflash_block_size` | Adjust either value, or pad |
| `DFlash offline model cannot run eval/inference forward` | Offline conversion deletes base-model layers to save memory | Don't run eval on the offline model; reload the full base first |
| `DFlash offline model cannot run AR validation / pseudo_speculative_generate` | Same cause, hit via AR validation | Keep `training.estimate_ar=false` and `training.ar_validate_steps=0` in offline runs |
| `dflash_swa_window_size (N) must be >= dflash_block_size (B)` | Config validation | Raise the window or lower the block size |
| `The base model did not return hidden states required for DFlash training` | Base model's top-level forward ignores `output_hidden_states=True` | Usually a multimodal wrapper — needs a model-side fix |
| `ERROR: DRAFT_CKPT_DIR=... contains no exported-checkpoint-* directory` | Upstream training produced no draft | Fix training; do not chase the smoke test |
| vLLM rejects the speculative config / no DFlash method | DFlash landed in vLLM v0.22.0 (`vllm/v1/spec_decode/dflash.py`) | Use `vllm/vllm-openai:v0.22.1` or newer |
| Draft quality plateaus despite clean training | `--aux-layers` count and `num_hidden_layers` disagree, so the dump captured the wrong layers | Re-dump with matching values |
| Loss stalls high with `answer_only_loss=true` | Chat template lacks `{% generation %}` tags, so no positions contribute loss | Supply a template with generation tags |
| `dflash_dpace_alpha must be in (0, 1]` | Invalid D-PACE alpha | Correct the value |
| Qwen3-VL mRoPE / `mm_token_type_ids` errors | Qwen3-VL DFlash needs Transformers 5.3.0 or >=5.4.0 and the AutoProcessor's `mm_token_type_ids` | Match the version; don't drop processor outputs |
