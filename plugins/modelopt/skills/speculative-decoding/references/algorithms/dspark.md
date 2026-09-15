# DSpark

**A DFlash variant, not a separate pipeline.** DSpark is the DFlash draft backbone
plus a lightweight sequential (Markov) head and an optional confidence head, selected
with `dflash_architecture_config.projector_type=dspark`. The Markov head adds a
prefix-dependent transition bias to the base logits, inducing a causal block
distribution (semi-autoregressive generation).

Read `dflash.md` first — the pipeline, dump flags, export behaviour, and generic
failure modes are all shared. This sheet covers only the delta.

Recipe: `modelopt_recipes/general/speculative_decoding/dspark.yaml` (its
`metadata.recipe_type` is `speculative_dflash`, and every knob lives in the `dflash.*`
namespace).

Examples: `tools/launcher/examples/*/*/hf_streaming_dspark_multi_node.yaml`,
`hf_streaming_dspark_warmstart.yaml`, `hf_dspark_ptq_*.yaml`.

## Pipeline tasks

Three committed shapes, all 2 tasks. No committed online or offline DSpark example;
the recipe defaults to `data.mode=online`, so an online run would use
`common/specdec/dflash_online_training.sh` exactly as DFlash does.

**Streaming, multi-node, from scratch** (Kimi-K2.6, MiniMax-M3):

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/eagle3/make_dataset.sh` | Build training conversations | `/scratchspace/data/train.jsonl` |
| task_1 | `common/eagle3/train_eagle_streaming.sh` | Streaming train (serve replicas + DDP trainers over NIXL RDMA), then export | `/scratchspace/export` |

`data.mode=streaming` with `model.use_fake_base_for_offline=true`.

**Streaming warm-start** (Nemotron-3.5-Lightning) — same two scripts, but continues a
**released** drafter from its published weights instead of training from scratch. The
key difference: the drafter architecture is **not** overridden in the YAML. It lives in
a per-model recipe under `modelopt_recipes/models/<org>/<model>/speculative_decoding/`,
because every field there is transcribed from the released checkpoint's own
`config.json` and must match it exactly. Keep drafter shape changes in that recipe and
cluster/serve knobs in the YAML.

**Drafter PTQ** (Qwen3-8B nvfp4; Nemotron-3.5-Lightning fp8 and nvfp4) — quantizes an
already-exported drafter. Calibration-free: every scale comes from the weights.

| Task | Script | Purpose | Output |
| --- | --- | --- | --- |
| task_0 | `common/specdec/quantize_drafter.sh` | Quantize the drafter (CPU-only, ~1 min for an 8B-class draft) | `/scratchspace/export_quantized` |
| task_1 | `common/specdec_bench/quick_check.sh` | Acceptance length on MT-Bench, so the cost is visible | JSON results |

`DRAFTER_CKPT` accepts an exported checkpoint, an HF repo id, or a training
`output_dir` — for a directory it auto-selects the newest `exported-checkpoint-*`.
`w4a16_nvfp4` keeps activations in bf16; `nvfp4` does weight+activation with a fixed
`input_scale` of 1.0. To measure what quantization cost, re-run task_1 against the
unquantized drafter and compare acceptance length.

Streaming-specific environment (set in `task_1`, see
`common/eagle3/train_eagle_streaming.sh` for dispatch and rendezvous):

| Env var | Meaning |
| --- | --- |
| `EAGLE_CAPTURE_IDS` | Which base layers the serve side captures — the draft's target layer ids **+1**, plus the true final hidden layer |
| `SERVE_NODES` / `SERVE_TP` | How the node pool splits into serve replicas vs DDP trainers |
| `STREAMING_NUM_WORKERS` | Trainer-side streaming workers |
| `SERVE_MAX_MODEL_LEN`, `SERVE_MAX_NUM_SEQS`, `SERVE_GPU_MEM_UTIL`, `SERVE_READY_TIMEOUT` | Serve-replica limits |
| `EXPORT_EXTRA_ARGS` | Extra args at export (e.g. `--trust_remote_code`) |

No inference path is wired into these examples — neither ships a vLLM smoke test or
AR eval step.

## Recipe and training knobs

The general recipe is `modelopt_recipes/general/speculative_decoding/dspark.yaml`. A
warm-start run instead points `--config` at a **per-model** recipe under
`modelopt_recipes/models/<org>/<model>/speculative_decoding/`, which pins the drafter
shape to the released checkpoint.

Everything in `dflash.md` applies. DSpark adds:

| Override | Recipe default | Note |
| --- | --- | --- |
| `dflash.dflash_architecture_config.projector_type` | `dspark` | Selects the variant |
| `dflash.dflash_architecture_config.markov_rank` | 256 | Markov head low-rank dimension. **Required** and must be > 0 |
| `dflash.dflash_architecture_config.markov_head_type` | `vanilla` | `vanilla` (memoryless), `gated` (hidden-gated), or `rnn` (recurrent, closest to Domino's GRU) |
| `dflash.dflash_architecture_config.use_confidence_head` | true | Builds the per-position acceptance predictor |
| `dflash.dflash_ce_loss_alpha` | 0.1 | Cross-entropy term |
| `dflash.dflash_l1_loss_alpha` | 0.9 | TVD term — the DeepSpec defaults are L1/TVD-dominant |
| `dflash.dflash_confidence_head_alpha` | 1.0 | Confidence BCE term; requires `use_confidence_head=true` when > 0 |

Total loss is `ce_alpha*CE + l1_alpha*TVD + conf_alpha*confidence_BCE`.

`dflash_self_logit_distillation` is **false** for DSpark — it computes the target
distribution internally for the TVD and confidence terms, so the DFlash KD path is
unused. Recipe defaults also differ from DFlash's: `block_size` 16, `num_anchors` 256,
`num_train_epochs` 6, `training_seq_len` 3072, `warmup_ratio` 0.04.

## Per-model adjustments

Everything in `dflash.md`'s table applies. Additionally:

| Situation | What to change |
| --- | --- |
| Any model | **The DSpark draft does not inherit the base model's GQA/FFN dims.** Set `num_attention_heads`, `num_key_value_heads`, `head_dim`, and `intermediate_size` in `dflash_architecture_config` explicitly, or you get a silently wrong-shaped draft. Kimi-K2.6 uses `num_hidden_layers=6, num_key_value_heads=8, intermediate_size=18432`; MiniMax-M3 uses `intermediate_size=12288`. |
| Streaming | `EAGLE_CAPTURE_IDS` must be the draft's target layer ids +1 plus the final hidden layer. Kimi-K2.6: `[2,13,25,36,48,59,61]` for a 6-layer draft. Getting the final layer wrong caps acceptance length rather than erroring. |
| Sparse-attention base (e.g. MiniMax-M3 MSA) | Set `SERVE_BLOCK_SIZE` to the base's `sparse_block_size` (M3: 128) |
| Serve container lacks tensorboard | `training.report_to=none`, else trainer init crashes |
| Tokenizer can't emit assistant masks (e.g. Kimi slow tokenizer) | `training.answer_only_loss=true` still works — masks are recovered from token ids |

## Success markers

Same as `dflash.md`. Because the streaming examples have no smoke test or AR eval,
the only in-pipeline evidence is training progress plus the export landing in
`/scratchspace/export`.

## Quality gate

**Do not trust in-training AR for DSpark.** The recipe pins `estimate_ar: false` and
`ar_validate_steps: 0` deliberately: eval runs the DFlash backbone only, with the
Markov head not applied, so any reported AR reflects the backbone alone rather than
the trained model.

`pseudo_speculative_generate` *is* overridden for DSpark (unlike Domino), so a
non-offline model can generate correctly — but the offline/streaming path deletes base
layers and refuses. Evaluate by exporting and running the offline acceptance-length
harness separately.

Otherwise the training-regression gate from `dflash.md` (`MAX_FINAL_LOSS`,
`MIN_FINAL_ACC` via `check_regression.py`) applies; none of the streaming examples set
those thresholds.

The **PTQ** examples are the exception — they do produce a real inference metric, via
`quick_check.sh` on MT-Bench with `--speculative_algorithm DSPARK`. There is no fixed
pass threshold; the gate is relative. Run the same benchmark against the unquantized
drafter and compare acceptance length, so the quantization cost is explicit rather than
assumed. Published reference points on Qwen3-8B: the weight+activation `nvfp4` format
costs roughly 3.9% acceptance length, which is why the examples default to
`w4a16_nvfp4` (weights only, activations in bf16). Adding `fc` to the quantized set
costs about 0.7% acceptance length (3.0186 vs 3.0392) for ~3% more size saved.

## Known failures

Generic infrastructure failures are in `../stages/triage.md`; shared block-diffusion
failures (`seq_len` divisibility, offline eval, mask token, chat template) are in
`dflash.md`. DSpark-specific:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| `DSpark (projector_type='dspark') requires 'markov_rank' (> 0) in dflash_architecture_config` | Markov head dimension missing | Set `dflash_architecture_config.markov_rank` |
| `DSpark requires markov_rank > 0, got N` | Non-positive value | Set a positive rank |
| `Unsupported markov_head_type: '...'. Expected 'vanilla', 'gated' or 'rnn'` | Typo or unsupported head | Use one of the three |
| `dflash_confidence_head_alpha > 0 but the confidence head was not built` | Loss term enabled without the head | Set `dflash_architecture_config.use_confidence_head=true`, or set the alpha to 0 |
| `DSpark offline model cannot run AR validation / pseudo_speculative_generate` | Offline/streaming conversion deleted base layers | Keep `estimate_ar=false` and `ar_validate_steps=0`; evaluate after export |
| Draft trains but acceptance length is poor | Draft dims left at defaults instead of matching the base | Set the GQA/FFN dims explicitly (see *Per-model adjustments*) |
| Acceptance length capped despite clean training (streaming) | `EAGLE_CAPTURE_IDS` final layer wrong, or the vLLM aux-capture fix (vllm#46788) missing | Correct the ids; use a container with the fix |
| Trainer init crash on a serve container | tensorboard absent | `training.report_to=none` |
| Drafter PTQ produces a broken draft, or export fails on a packed tensor | q/k/v were quantized. DFlash-family drafters build their fused context-KV projection by reading `qkv_proj.weight` raw, which cannot be a packed tensor | The exclusions are **mandatory**, not tuning: `--exclude '*q_proj*' '*k_proj*' '*v_proj*' '*qkv_proj*'`. `o_proj` and the MLP still quantize; `*fc*` is optional (~3% more size saved for ~0.7% acceptance length on Qwen3-8B) |
| Benchmark reports a poor or nonsensical acceptance length | `--draft_length` was passed. DSPARK and DFLASH read `--block_size` instead, and it must match the drafter's block size | Pass `--block_size <N>` matching the drafter, plus `--speculative_algorithm DSPARK` |
| Warm-start run diverges or the drafter shape mismatches the released checkpoint | Drafter architecture overridden in the launcher YAML | Leave shape fields in the per-model recipe — they are transcribed from the released `config.json` and must match exactly |
