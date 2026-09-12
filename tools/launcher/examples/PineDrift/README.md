# DFlash2 drafter for PineDrift-820B-A42B-NVFP4

Training harness for a DFlash2 draft head on PineDrift (`musespark1x_omni`), by
**streaming** against a vLLM TP8 serve of the base. Everything here runs on
[AWS PDX](../../../../CLAUDE.md) (B300, 8 GPU/node, x86_64).

    repo      ~/lustre/modelopt-pinedrift        branch pinedrift/dflash2
    entry     ~/pinedrift_dflash2.sh {smoke|full|status|logs}
    recipe    modelopt_recipes/general/speculative_decoding/dflash2_pinedrift.yaml
    pipeline  tools/launcher/examples/PineDrift/hf_streaming_dflash2_pinedrift.yaml
    base      ~/lustre/hf-local/pinedrift-820b-a42b-nvfp4_vv3          (496 GiB)
    corpus    ~/lustre/hf-local/pinedrift-xhigh-split/        583 shards
                (split from ~/lustre/pinedrift_xhigh_snapshot_583shards_20260911_0908)
              ~/lustre/hf-local/pinedrift-xhigh-split-smoke8/  8 shards (smoke)
    template  ~/lustre/hf-local/pinedrift-templates/chat_template_train_xhigh.jinja
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

## The capture bug this harness was built blind to (fixed 2026-09-12, `33bea53ac` + `18a650988`)

For every run before this commit, the drafter trained on hidden states belonging to
**other requests**. `save_kv_layer` walked the batch's `slot_mapping` with a running
offset, assuming the newly scheduled requests sat at its front; vLLM appends them at the
back, so any request already decoding shifted the window.

Nothing errored. Shapes, pool slot, the `/done` generation check and the recorded
`token_ids` were all correct — only the bytes belonged to a different prompt. The
drafter therefore had nothing but its anchor token to go on and learned a bigram model,
which is exactly what its position-1 accuracy of 0.217 was.

Measured on one smoke, changing only the serve's concurrency:

| serve `max_num_seqs` | per-sequence hidden/token match | `teacher_top1_match` |
|---|---|---|
| 32, before | 0.00 - 0.02 (8x8 pairing matrix all dark) | 0.0103 |
| 1, before | 0.94 - 1.00 (clean identity) | — |
| **32, after** | **0.94 - 1.00 (clean identity)** | **0.9714** |

Three things worth keeping:

- **`SERVE_MAX_NUM_SEQS=1` is the fastest test for this whole class of bug.** If a
  metric jumps when the serve can only hold one request per batch, the problem is batch
  bookkeeping, not the model or the data.
- **The 8x8 pairing matrix** (`DFLASH_TEACHER_SWEEP=1`, `HFDFlashModel._teacher_sweep`)
  distinguishes the cases that matter: a bright diagonal means correct, a bright
  off-diagonal means a permutation, all dark means the hidden is not from this batch at
  all. Per-plane and per-offset sweeps could not tell these apart — they all read ~0.00.
- **`teacher_top1_match` is what surfaced this.** Without a target-side ceiling to read
  the draft against, the only symptom is "accuracy seems a bit low", which looks exactly
  like a hyperparameter problem. Loss goes down the whole time.

### The second bug, introduced by the first fix (`18a650988`)

Addressing each request's own blocks means building index tensors. The first version
built them on the default stream and consumed them inside `with torch.cuda.stream(cs)`.
Nothing orders the copy stream against that allocation, so the caching allocator is free
to hand the memory to another consumer while the copy stream is still reading it. The
garbage indices then trip an out-of-bounds assert **inside whatever kernel runs next** —
tens of minutes into a run, not reproducibly, and never pointing at the line responsible.

Two submissions died this way. What identified it was setting `CUDA_LAUNCH_BLOCKING=1`,
which serialises the streams: the same code then ran clean for 45 minutes. Building the
tensors inside the stream context fixes it.

The code this replaced was safe only incidentally — it sliced vLLM's long-lived
`slot_mapping` instead of allocating anything of its own.

### `logger.info` from this connector goes nowhere

vLLM never configures this module's logger. Every `logger.info` in
`rdma_hidden_states_connector.py` has always been invisible, including the line that
reports the cache layout. Three debugging rounds were spent reasoning about numbers that
were being printed to nobody. The diagnostics now use `print(..., flush=True)`.

The first thing they revealed, which no amount of reading had:

    [HSCONN] kv-cache group 0: spec=SlidingWindowSpec   layers=16 spec_block_size=256
    [HSCONN] kv-cache group 1: spec=SlidingWindowSpec   layers=15 spec_block_size=256
    [HSCONN] kv-cache group 2: spec=SlidingWindowSpec   layers=15 spec_block_size=256
    [HSCONN] kv-cache group 3: spec=FullAttentionSpec   layers=16 spec_block_size=256
    [HSCONN] kv-cache group 4: spec=HiddenStateCacheSpec layers=1 spec_block_size=1  <-- ours
    [HSCONN] capture cache view (76920, 6, 1, 8192) planes_dim=1 blocks=76920 block_size=1

**The capture cache has `block_size=1`** — one block per token, so a slot id *is* a block
id. That also rules out the group-mismatch theory this was chasing at the time: there is
exactly one `HiddenStateCacheSpec` group and it is the one being used.

### What the drafter was actually worth, before the fix

The `ckpt33500` export was served under vLLM (`num_speculative_tokens=4`, 1129 drafts):

    ACCEPTANCE LENGTH        1.8131
    per-position acceptance  0.4632 / 0.2037 / 0.1001 / 0.0461

Two calibration facts fall out. The training per-position accuracy tracks the served
acceptance from position 2 on, so it is a usable proxy. But the "implied AL" read off it
(1 + the running product of the marginals) gives 1.26 against a measured 1.81: the
selector, not the backbone argmax, picks the token at serve time, and acceptance at
successive positions is positively correlated. Treat implied AL as a lower bound.

## Teacher-side diagnostics

Every log step also reports two numbers about the BASE, not the draft, over exactly the
block positions the draft is scored on:

    teacher_top1_match       how often PineDrift's own top-1 is the corpus's next token
    teacher_top1_confidence  the mean probability it puts on that top-1

They are the ceiling the draft is being read against. A draft position-1 accuracy of
0.20 means one thing if the teacher matches 0.90 and something completely different if
the teacher itself only matches 0.45 — the latter says the corpus is off-distribution
for the target (or the mask/template is wrong), not that the draft is undertrained.
Both are flat by construction, so drift in them means the data mix moved.

Getting them was NOT a copy of the DSpark implementation, which reads a
`flat_teacher` that its KD term had already gathered. This run has
`dflash_self_logit_distillation: false`, so no teacher distribution exists anywhere in
the step. `HFDFlashModel._teacher_top1` therefore asks `from_offline_dict` for the
post-final-norm base hidden with `defer_lm_head=True` and projects only the supervised
rows, in chunks of 256, reducing each to `(argmax, max - logsumexp)` immediately. The
obvious version -- gather then softmax -- costs 6.6 GB at this vocabulary, and the
full-sequence version 10 GB, next to draft logits that are already 6.6 GB.

`selector_accuracy` / `selector_coverage` ride the same channel. DFlash2 had been
computing them into `self._selector_metrics` since the port; nothing ever read that
attribute, so they had never once been logged.

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

1. **Corpus — the assistant content is a raw Harmony stream and must be split.**
   The corpus is PineDrift's own xhigh self-synthesis (see the snapshot's README for
   provenance). Each reply is stored as one raw stream in `content`:

       " to=self<|message|>{CoT}<|eom|><|start|>assistant to=user<|message|>{answer}"

   Fed to the stock chat template that renders as

       <|start|>assistant to=user<|message|> to=self< |message|>{CoT}< |eom|>...

   — two independent corruptions, neither of which errors:
   - the template adds **its own** `<|start|>assistant to=user<|message|>` header, so
     the reply is double-wrapped and the channel header is wrong;
   - the template runs content through `esc()`, a deliberate injection defence that
     **inserts a space into every control token**, so `<|message|>` (id 200023)
     becomes ordinary BPE pieces. Same family as the K3 split-tag bug.

   The result is a target sequence that cannot occur at inference. The template
   already knows the right rendering — its assistant branch reads `reasoning` and
   `content` separately and emits `to=self … <|eom|>` then `to=user … <|eot|>` — so
   `tools/pinedrift_split_harmony.py` splits the stream into those two fields.
   `--verify` renders a converted row back and requires
   `"<|start|>assistant" + original_content + "<|eot|>"` to appear **verbatim**, and
   checks the four control ids survive as single tokens. Measured: exact rebuild.

   Drop rules (both on): rows with no `to=user` (hit the 8192 cap mid-CoT, 15.6%) and
   degenerate tail loops (6.0%, detector from `pinedrift_filter.py`). Truncated rows
   are dropped *here* although the snapshot README argues they are valid next-token
   data — that argument holds for the verbatim stream, but once the template
   synthesises a terminator, a cut CoT teaches a false `<|eom|>`.

   **`reasoning_effort` must be `xhigh`.** The synthesis ran at xhigh, which the
   system turn carries as `Reasoning strength: 512`; the stock template defaults to
   `medium` (64) and the trainer does not pass the parameter. So the pipeline uses a
   one-line copy of the template with the default changed. Training against the
   medium system prompt would condition the drafter on a prompt that never produced
   these replies.

   Also still true of any corpus here: `_tokenize_entry` starts with
   `cid = entry.get("conversation_id") or entry.get("uuid")` and drops the entry when
   that is None — silently, as an "unfit entry" — so a corpus with no id field dies
   ~25 min in with `no fetchable sample found in the entire corpus` and no hint about
   why (job 405859). This corpus has `uuid`.

   Failure signature to watch for separately: `train_acc` → exactly 1.0 within ~20
   steps plus an impossibly high step rate = the loss mask is empty, not convergence.
   Measured on 300 rows of this corpus: 299 tokenize with a non-empty mask (median
   2246 supervised tokens of 3072), 1 had no assistant turn.

2. **`SERVE_EXTRA_ARGS` is exported unquoted** by nemo_run, so its value must be
   space-free. Only `--language-model-only` is passed; `--enable-expert-parallel` and
   `--tokenizer-mode hf` are dropped (neither is needed for correctness). Add a
   dedicated knob, as `SERVE_BLOCK_SIZE` did, if EP is wanted.
3. **`TRITON_CACHE_DIR` must be node-local.** `dflash_use_flex_attention` compiles
   Triton templates on the first backward; ranks sharing a lustre cache die with
   `OSError: [Errno 14] Bad address`. `/raid/scratch` is the only user-writable
   node-local NVMe on PDX (`/raid` and `/raid/enroot` are root-only).
4. **The image has no `nixl`.** It is a serving image; the serve dies at connector
   init with `ModuleNotFoundError: No module named 'nixl'` (job 405777).
   `PIP_EXTRA_PACKAGES: "nixl==1.3.0"` installs it on every node. Install **plain
   `nixl`**, not `nixl-cu13` — the latter installs the module as `nixl_cu13` and the
   connector imports `nixl`. Plain `nixl` pulls both cu12 and cu13 builds. It also
   downgrades `nvidia-nccl-cu13` (see trap 9).
5. **NIXL transport: LIBFABRIC over EFA, which needs an rdma-core graft.**
   The stock container's rdma-core is too old for the plugin — `libefa` lacks
   `EFA_1.2`, there is no `libfabric.so.1` exporting `FABRIC_1.7`, no
   `libhwloc.so.15` — so `createBackend("LIBFABRIC")` returns `NIXL_ERR_NOT_FOUND`.
   Ubuntu/apt libfabric (1.14/1.20) and the host's own do not substitute: wrong
   symbol version and/or a glibc mismatch. Only the AWS `/opt/amazon/efa` set works,
   and the whole set has to be grafted together. `EFA_GRAFT_DIR=/efa-rdma-core`
   (copied from the K3 experiment's `efa701`, ~4 MB, now owned by this harness)
   turns on the graft block in `train_eagle_streaming.sh`, which also asserts the
   backend actually instantiates rather than letting the run fall back silently.
   Measured working in *this* image by probe job 405849: `Backend LIBFABRIC was
   instantiated` plus a successful VRAM `registerMem`. The `libionic-rdmav59.so`
   load warning under the graft is benign.

   The alternative is UCX, and it works, but only over plain TCP:
   `UCX_TLS=tcp,sm,self,cuda_copy`. `cuda_copy` is load-bearing — without it UCX
   logs `8 NVIDIA GPU(s) were detected, but UCX CUDA support was not found` and
   `registerMem` fails on the VRAM buffers with `NIXL_ERR_BACKEND`, even though
   `libuct_cuda.so` is on disk the whole time. And `UCX_TLS` cannot be left to
   probe: UCX then finds the node's 16 EFA NICs and segfaults the trainer's forked
   dataloader workers at agent init.

   **This harness briefly shipped the UCX/TCP path.** The reason given was a K3
   A/B measuring LIBFABRIC at 24 s/step against TCP's 15 s — but that measurement
   is recorded with an explicit "do not treat as final, re-measure clean" caveat
   (it was taken during warmup, and the two arms captured different numbers of
   layers), and every K3 and M3 production run since uses LIBFABRIC. Citing a
   number its own author flagged as unreliable was the error.

   `NCCL_IB_DISABLE=1` is separate and still set: it is about NCCL's own ibverbs
   path for the trainer's DDP, not about NIXL.

6. **`container_mounts` replaces the default, it does not extend it.** The default
   is exactly `["$SLURM_HF_LOCAL:/hf-local"]` and the Slurm executor never re-adds
   it, so adding one mount for the EFA graft silently unmounted the model. vLLM
   then fell through to the Hub and died in `validate_repo_id` on a local path
   (job 406717). Any yaml here that sets `container_mounts` must list `/hf-local`
   itself.
7. **The hidden-state KV cache view changed shape in this vLLM.** `create_kv_cache_views`
   now hands out one 4D **`[B, H, N, C]`** view per layer unconditionally — the
   `KVCacheLayout` permutes strides, not axes. `RdmaHiddenStatesConnector` was written
   against the older `[B, N, H, C]`, so it read `shape[1]` as the block size (really the
   plane count) and `shape[2:]` as the per-token feature (really `(N, C)`).

   With our hidden group at `block_size=1` that yields `feat=(1, 8192)`: the trainer
   receives **one** plane, `_format` peels it off as the base hidden, and the draft gets
   an empty aux tensor — `mat1 and mat2 shapes cannot be multiplied (3072x0 and
   40960x8192)` (job 406838). Nothing upstream of that errors.

   Fixed in `rdma_hidden_states_connector.py` by locating the plane axis by its length
   rather than by position, and refusing to guess when both candidates match. The serve
   now logs the view it found; `tools/test_hidden_cache_gather.py` checks both layouts
   by value, since a transposed gather would train silently.

   Watch for `RdmaHiddenStatesConnector: hidden-state cache view (..., 6, 1, 8192) ->
   6 planes x 8192 hidden` in the serve log.
8. **`report_to=none`** — the trainer runs in the serve container, which has no
   tensorboard.
9. **Explicit `time:`** — the launcher asks for 4 h otherwise and a long run dies as a
   bare Slurm TIMEOUT with no traceback.
10. **`global_vars` keys must name real `launch()` parameters.** Only `hf_model` does;
   a custom key is rejected at argument-parse time.
11. **The serve container downgrades `nvidia-nccl-cu13` 2.30.7 → 2.29.7** when modelopt
   is pip-installed, disabling DeepEP v2. Harmless for the trainer; do not reuse that
   env for a standalone serve.
12. **Use `bash -c`, not `bash -lc`**, in any hand-rolled srun with `~/lustre` mounted:
   `~/.bashrc` activates conda and shadows the container's python.

## BLOCKER — KV-cache page sizing (root-caused; fix = block_size >= 32*N)

Job 405516 (first smoke) failed at engine init:

    File "vllm/v1/kv_cache_interface.py", line 429, in page_size_bytes
      assert self.page_size_padded >= self.unpadded_page_size_bytes
    AssertionError

Same class as the gpt-oss DFlash serve crash (ModelOpt PR #1692), which was fixed
with `--block-size=32`.

Root cause. vLLM's `extract_hidden_states` packs **all** captured layers into a single
`HiddenStateCacheSpec` with `num_kv_heads = len(capture_ids)` and `head_size =
hidden_size`, neither of which is TP-sharded. `HiddenStateCacheSpec` subclasses
`MLAAttentionSpec`, so `head_size_v = 0` — there is no K/V doubling. That spec is then
re-added to the KV groups with its page **padded up to the attention page**
(`kv_cache_utils.py:2244-2262`):

    per_token      = num_kv_heads * head_size * dtype_size
    max_block_size = max(common_page // per_token, 1)        # <-- clamps to 1
    new_bs         = _largest_divisor_at_most(group_block_size, max_block_size)
    aligned        = replace(spec, block_size=new_bs, page_size_padded=common_page)

When `common_page < per_token` the floor division gives 0, the `max(..., 1)` clamps it
to 1, and the resulting spec has `unpadded_page_size_bytes = per_token > common_page =
page_size_padded`. That is the assert.

For PineDrift at TP8 — and the TP sharding is the part that is easy to get wrong, since
only the attention side is sharded:

    hidden-state page : N * hidden_size * 2 B          = N * 16384       (no /TP, no K+V)
    attention page    : block * (kv_heads/TP) * (head_dim + head_dim) * 2 B
                        = block * (16/8) * 128 * 2     = block * 512

so the requirement is simply

    block_size >= 32 * N_capture

N = 6 needs block_size >= 192; 256 is the next power of two. **Measured**: block 16
(job 405516) and block 32 (job 405733) both assert, which is what this predicts.

The other lever is N. At block 128 the ceiling is N = 4, i.e. 3 aux planes + final,
which costs drafter depth; we would rather pay block size. Block size on a throwaway
producer serve only changes KV paging granularity.

The cost of a large block is *not* wasted KV memory but coarse paging: the hidden-state
group ends up at `new_bs = 1`, one block per token, while the attention groups page at
256. All groups share one block pool, so the hidden group is what bounds concurrency.
With ~180 GB of KV headroom and ~2 pages per group-block this still leaves several
hundred thousand blocks, far past what `max_num_seqs 32` x `max_model_len 4096` needs.

(Two earlier versions of this section were wrong: the first guessed that both pages
scale with block size, the second used the *unsharded* 16 kv heads and so predicted
block 32 would be enough. Only the attention page scales, and it scales off the
per-rank head count.)

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
