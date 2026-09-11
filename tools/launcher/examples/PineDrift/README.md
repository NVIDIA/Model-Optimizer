# DFlash2 drafter for PineDrift-820B-A42B-NVFP4

Training harness for a DFlash2 draft head on PineDrift (`musespark1x_omni`), by
**streaming** against a vLLM TP8 serve of the base. Everything here runs on
[AWS PDX](../../../../CLAUDE.md) (B300, 8 GPU/node, x86_64).

    repo      ~/lustre/modelopt-pinedrift        branch pinedrift/dflash2
    entry     ~/pinedrift_dflash2.sh {smoke|full|status|logs}
    recipe    modelopt_recipes/general/speculative_decoding/dflash2_pinedrift.yaml
    pipeline  tools/launcher/examples/PineDrift/hf_streaming_dflash2_pinedrift.yaml
    base      ~/lustre/hf-local/pinedrift-820b-a42b-nvfp4_vv3          (496 GiB)
    corpus    ~/lustre/hf-local/Speculative-Decoding-Dataset-v1-Qwen3-8B/
                default-msgs.jsonl            1.96 M records, 31.65 GB  (full)
                default-msgs-smoke20k.jsonl   20 k records,  327 MB     (smoke)
    container ~/lustre/containers/vllm-pinedrift-x86-20260911-bda0fc.sqsh

## Why streaming and not online

The base is 820 B NVFP4 and **has no transformers implementation at all** —
`musespark1x_omni` lives only in vLLM. It cannot be loaded in-process, so the
trainer runs against a `FakeBaseModel` (embed_tokens + lm_head + final norm) while
a vLLM serve supplies hidden states over NIXL RDMA.

DFlash2 permits this. **LiLiCorr does not** — `hf_lilicorr.py` refuses
`dflash_offline` (streaming counts as offline) because its distractor penalty needs
the target's logits in-process. DFlash2 and LiLiCorr are *different algorithms* that
both ride the `dflash` mode; do not conflate them.

## What had to be added to modelopt

modelopt had zero `musespark1x_omni` support. Four things, all verified against the
real checkpoint (`verify_musespark_fakebase.py`):

| what | value | why it matters |
|---|---|---|
| final norm | **plain** `rmsnorm` | `output_norm_gain_center_type: "zero"` → gain_center 0. Only the *post*-norms are zero-centred gamma. Registering it as `gemma_rmsnorm` adds a spurious +1. |
| embed norm | weightless RMSNorm, eps 1e-5 | The table is row-normalized (every row L2 = √32, RMS 0.0625). The norm rescales to RMS 1, i.e. **16×**. Dropping it feeds the draft embeddings 16× too small. |
| embed multiplier | `metap.m_emb` = 1.0 | No-op for vv3, but read from config so a future release that changes it is picked up. |
| logits | `20·tanh(x · 0.176777 / 20)` in fp32 | multiplier = `(hidden/metap.base_width)^-0.5` = `(8192/256)^-0.5` for `metap_mode: sp`. Dropping the cap or the multiplier gives a completely different KD target. |

Implemented as `_WeightlessRMSNorm` / `_TransformedEmbedding` / `_SoftCappedLMHead` /
`_select_base_transforms` in `modeling_fakebase.py`, plus the `modeling_final_norm.py`
table entry. The transforms live *inside* the modules because callers resolve
`embed_tokens` / `lm_head` and call them directly.

`FakeBaseModel.from_source` also needed a **raw `config.json` fallback**: `AutoConfig`
raises on an architecture transformers does not know, before FakeBase gets to read the
dozen scalars it actually needs.

## Verified numbers

    embedding RMS after the weightless norm : 1.0000
    raw logits  min/max                     : -25.125 / 28.000
    capped logits min/max                   :  -4.375 /  4.844
    vs 20·tanh(raw·0.176777/20) cast to bf16 : 0.000e+00   (bit-exact)
    vs the same reference in fp32            : 1.530e-02   (bf16 ulp = 3.125e-02)

The fp32 gap is half an ulp — `_SoftCappedLMHead` computes in fp32 and casts back to
the head dtype on purpose, so a `[seq, 202048]` fp32 logit tensor does not dominate
activation memory.

## Configuration decisions, and which are measured

| decision | value | basis |
|---|---|---|
| mask token | `201817` | **measured** — highest `<|reserved_special_token_*|>`; inside vocab_size, never emitted, absent from the chat template, embedding row real (L2 5.6568) |
| capture ids | `[3,19,31,47,59]` + final | **inferred, not measured** — all ≡3 (mod 4), the last layer of each `[SWA2048 ×3, global+NoPE]` block, so all five aux features come from the same phase. The default `build_target_layer_ids(62,5)` = `[1,16,30,44,59]` lands on phases `[1,0,2,0,3]`. Worth an A/B. |
| draft attention dims | 128 heads × head_dim 64, 16 kv | mirrors the base, as the Qwen3-8B reference recipe mirrors its own base |
| draft `intermediate_size` | 24576 | **inferred** — 3× hidden, matching the reference's 12288/4096 ratio. The base has no single intermediate (every layer is MoE over a compressed 4096-dim expert space). |
| `answer_only_loss` | true | **measured** — PineDrift's own `chat_template.jinja` carries `{% generation %}` at lines 192/222, unlike M3's |
| loss objective | dpace (default) | DFlash2 was developed with dpace. `dflash_loss_decay_factor` is silently ignored under it — deliberately omitted rather than set to a discarded value. |

## Traps this harness already encodes

1. **Corpus.** The published `default.jsonl` is **gzip despite the name** and stores
   the reply under `conversations`. `hf_streaming_dataset` prefers `messages`, so an
   unconverted file gives an empty answer span and a **silent hang**. `convert_specdec.py`
   rebuilds a `messages`-only file and refuses to emit one without assistant turns.
   Failure signature to watch for anyway: `train_acc` → exactly 1.0 within ~20 steps
   plus an impossibly high step rate = the loss mask is empty, not convergence.
2. **`SERVE_EXTRA_ARGS` is exported unquoted** by nemo_run, so its value must be
   space-free. Only `--language-model-only` is passed; `--enable-expert-parallel` and
   `--tokenizer-mode hf` are dropped (neither is needed for correctness). Add a
   dedicated knob, as `SERVE_BLOCK_SIZE` did, if EP is wanted.
3. **`TRITON_CACHE_DIR` must be node-local.** `dflash_use_flex_attention` compiles
   Triton templates on the first backward; ranks sharing a lustre cache die with
   `OSError: [Errno 14] Bad address`. `/raid/scratch` is the only user-writable
   node-local NVMe on PDX (`/raid` and `/raid/enroot` are root-only).
4. **NIXL must use LIBFABRIC on PDX.** UCX segfaults at agent init on EFA nodes.
   `NCCL_IB_DISABLE=1` for the trainer's DDP for the same reason.
5. **`report_to=none`** — the trainer runs in the serve container, which has no
   tensorboard.
6. **Explicit `time:`** — the launcher asks for 4 h otherwise and a long run dies as a
   bare Slurm TIMEOUT with no traceback.
7. **`global_vars` keys must name real `launch()` parameters.** Only `hf_model` does;
   a custom key is rejected at argument-parse time.
8. **The serve container downgrades `nvidia-nccl-cu13` 2.30.7 → 2.29.7** when modelopt
   is pip-installed, disabling DeepEP v2. Harmless for the trainer; do not reuse that
   env for a standalone serve.
9. **Use `bash -c`, not `bash -lc`**, in any hand-rolled srun with `~/lustre` mounted:
   `~/.bashrc` activates conda and shadows the container's python.

## BLOCKER (diagnosed, fix applied, not yet observed to pass) — KV-cache sizing

Job 405516 (first smoke) failed at engine init:

    File "vllm/v1/kv_cache_interface.py", line 429, in page_size_bytes
      assert self.page_size_padded >= self.unpadded_page_size_bytes
    AssertionError

Same class as the gpt-oss DFlash serve crash (ModelOpt PR #1692), which was fixed
with `--block-size=32`.

Root cause (this is the same one as gpt-oss, and it is *not* the hybrid attention —
that was a red herring there too). vLLM's `extract_hidden_states` packs **all** captured
layers into a single `HiddenStateCacheSpec` with `num_kv_heads = len(capture_ids)` and
`head_size = hidden_size`, then re-adds it to the KV groups. Its per-token cost is
therefore **independent of block size**, while the attention page scales with it:

    hidden-state spec : N x hidden x 2 B            = 6 x 8192 x 2   =  98304 B/token
    attention page    : 2 x B x kv_heads x head_dim x 2 B
                        = 2 x B x 16 x 64 x 2       = 4096 x B       B=16 ->  65536
                                                                     B=32 -> 131072

`page_size_padded` is the max over specs, so the assert fires exactly when the
hidden-state spec exceeds the attention page. At the default block size 16 it does, by
1.5x. **`--block-size 32` clears it** with 33 % headroom; that is the whole fix, and it
is safe because this is a throwaway producer serve where block size only changes KV
paging granularity. Encoded as `SERVE_BLOCK_SIZE: "32"` in the pipeline yaml.

The crossover in capture count at B=16 is N=4 (65536 fits exactly); at B=32 it is N=8.
So the alternative lever — fewer capture ids — would also work but costs drafter depth,
and is not needed.

(An earlier version of this section guessed that both pages scale with block size and
that `--block-size` might not move the ratio. That was wrong: only the attention page
scales.)

## Serve port

`SERVE_PORT`/`HS_SIDECAR_PORT` are pinned to 27650/27651, not the script defaults
8765/18999. Job 405564 died with `OSError: [Errno 98] Address already in use` on 8765 —
those defaults are shared by every modelopt streaming run on the cluster, and PDX
hands out nodes that other people's serves are still holding.

## Running it

    ~/pinedrift_dflash2.sh smoke     # 2 nodes, 20k records, 1 epoch
    ~/pinedrift_dflash2.sh full      # 4 nodes, full corpus, with the reaper exemption
    ~/pinedrift_dflash2.sh status
    ~/pinedrift_dflash2.sh logs

`full` resubmits the launcher-generated sbatch with the
`OccupiedIdleGPUsJobReaper` exemption, because tokenizing 1.96 M records before step 1
sits past the 30 min of idle GPU that PDX allows and nemo_run has no `--comment` field.

## After training

The export lands in `/scratchspace/export` inside the job. There is **no vLLM serve
path for a DFlash2 drafter on MuseSpark** — vLLM ships `MuseSparkDSparkForCausalLM`
(DSpark) only. The plan is to measure a first AL with the offline harness
(`$U/domino_eval`, see the DFlash2 eval runbook) and hand the checkpoint over for the
vLLM side to be pipe-cleaned separately.
