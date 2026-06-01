# Parallelism: GPU topology (TP / DP / PP / EP) and concurrency (`parallelism` / `--max-num-seqs`)

Two layers of decisions, made in order:

1. **GPU topology** — how the model is laid out across the GPUs you have
   (`--tensor-parallel-size`, `--data-parallel-size`, `--pipeline-parallel-size`,
   `--enable-expert-parallel`). Decide this first; it determines how many
   independent replicas exist.
2. **Concurrency** — how many requests are in flight (`parallelism`) and how many
   sequences each replica decodes at once (`--max-num-seqs`). Sized on top of the
   topology.

Both layers affect **throughput only** — never scores.

---

## Layer 1 — GPU topology: TP / DP / PP

| Dim | What it shards | What it buys | Cost |
| --- | --- | --- | --- |
| **TP** (tensor parallel) | Each layer's weights + KV across GPUs **within one replica** | Lets a model that doesn't fit on one GPU run; splits KV so longer context fits | All-reduce **every layer** → latency + needs fast interconnect (NVLink). Keep **within one node**. |
| **DP** (data parallel) | Nothing — **replicates** the (TP-sharded) model | Throughput: N independent replicas serve N× the concurrent requests | N× the weight memory (one full copy per replica) |
| **PP** (pipeline parallel) | Contiguous **layer ranges** across GPUs | Fits a model too big for intra-node TP; cheaper cross-node than TP | Pipeline bubbles → lower utilization. Mostly for very large / multi-node (see `multi-node.md`). |

**Decision procedure (single node, G GPUs):**

1. **TP = the smallest value that makes the model fit with KV headroom.**
   Estimate weight memory ≈ `params × bytes_per_param`:
   - NVFP4 ≈ 0.5–0.6 B/param (incl. scales) · FP8 ≈ 1 B · BF16/FP16 ≈ 2 B.
   - Need `weights/TP + KV cache + activations + CUDA-graph/overhead` to fit in
     `GPU_mem × gpu_memory_utilization`. If it fits on one GPU → **TP = 1**.
   - Constraints: TP must **divide `num_attention_heads`** (and ideally
     `num_key_value_heads` for GQA, else KV heads get replicated and waste memory);
     use a **power of 2**; never span nodes with TP.
   - Smaller TP = less communication = higher efficiency. Don't over-shard "to be
     safe" — it slows decode.
2. **DP = floor(G / (TP × PP))** — use the leftover GPUs as replicas. For
   throughput-bound evals (the common case), **maximize DP**: a model that fits on
   one GPU should run `TP=1, DP=G`, not `TP=G, DP=1`.
3. **PP only if** the model can't fit even at the largest sensible intra-node TP,
   or you're going multi-node — then see `multi-node.md`.

DP is the lever that grows **serving capacity** (Layer 2): more replicas → more
concurrent sequences.

> **Gotcha — bit-width sets the topology, not the model name.** The weight estimate
> hinges on `bytes_per_param`, so **read the actual precision from `config.json`
> (`quantization_config` / `quant_algo` / dtype) before sizing** — do not infer it
> from the org/handle. Two checkpoints of the *same architecture at the same
> bit-width* have the same footprint → the same TP/DP/EP, regardless of vendor or
> quant scheme (INT4 vs NVFP4 differ only in kernel/quant-method flags, which vLLM
> auto-detects — and a negligible effective-bit difference). The split only changes
> when the bit-width changes the *size* (see the Kimi example below).

### Choosing the TP/DP split (when more than one layout fits)

A fixed GPU count usually admits several valid splits — on 8 GPUs a MoE could run
any factorization with `TP×DP=8`: `TP=1/DP=8`, `TP=2/DP=4`, `TP=4/DP=2`, or
`TP=8/DP=1` (all with EP=8, see Layer 1b). They are **not** equally good. Default to
**smallest TP, largest DP**, because:

- **DP scales throughput ~linearly with no extra communication** — attention lanes
  are independent; only the MoE all-to-all couples ranks, and that's EP=`TP×DP`
  regardless of the split, so it's identical across them.
- **TP adds an all-reduce on every attention layer** and scales sublinearly. Each
  step up in TP buys KV/weight room at an efficiency cost.

Raise TP above 1 **only** to relieve a memory constraint DP cannot fix:

1. **A single request's KV won't fit one replica's pool.** A request runs entirely
   on one replica, so its longest (context + generation) KV must fit in that
   replica's free HBM: TP=1 → one GPU's HBM, TP=2 → two GPUs' (KV is TP-sharded).
   Long-context tasks (AA-LCR ~120K, full 262K) on memory-tight models force this.
2. **You need more concurrent seqs per replica than one GPU's KV allows** — i.e.
   you see preemption on TP=1 at your target per-replica `max-num-seqs`. TP=2
   doubles the KV blocks per replica.
3. **Weights don't fit one GPU** even after EP-sharding (dense models, or a MoE
   whose replicated attention + expert shard exceeds one GPU).

If none bite, higher TP just wastes the extra KV and gives up replicas → net slower.
**How to know which wins:** deploy the candidate and read the startup line
`Maximum concurrency for <max-model-len> tokens per request: X.XX×`; if it's
comfortably above `parallelism / DP` with zero preemption in the canary, the smaller
TP wins. Step up TP only when the canary proves TP=1 is KV-bound.

---

## Layer 1b — Expert parallelism (EP), MoE only

`--enable-expert-parallel` changes how the **MoE expert (FFN) layers** are parallelized:

- **Off (default):** every expert is tensor-sharded across the TP ranks (each rank
  holds a slice of *all* experts).
- **On:** whole experts are **partitioned across ranks** (each rank owns a subset of
  experts). Less per-FFN communication and better expert batching — the efficient
  choice for many-expert MoE.

### EP size is derived, not set

`--enable-expert-parallel` is a **boolean** — there is **no `--expert-parallel-size`**
in vLLM. The EP degree is always:

```text
EP = tensor_parallel_size × data_parallel_size      (the full world size)
```

So **EP equals TP only when DP=1.** On a fixed 8-GPU node every fitting split gives
EP=8 — you don't tune EP, you tune the TP/DP split, which only changes the
*attention* side:

| Layout (8 GPUs) | EP | Attention | Best when |
| --- | :--: | --- | --- |
| `TP=1 DP=8 --enable-expert-parallel` | 8 | 8 replicas, comm-free | throughput; one request's KV fits 1 GPU (**default**) |
| `TP=2 DP=4 --enable-expert-parallel` | 8 | 4 replicas, TP=2 | need ~2× per-replica KV pool (long ctx) |
| `TP=4 DP=2 --enable-expert-parallel` | 8 | 2 replicas, TP=4 | ~4× per-replica KV pool, or weights too big for TP≤2, but still want >1 replica |
| `TP=8 DP=1 --enable-expert-parallel` | 8 | 1 replica, TP=8 | trillion-scale weights / one huge KV pool |

Going down the table trades replicas (throughput) for a bigger per-replica KV pool
and more weight-fit room; the all-reduce cost rises with TP. Pick the **topmost row
that satisfies the memory constraints** (the TP-up triggers above).

### How DP-attention connects to the experts (the dataflow)

The DP group and the EP group are the **same physical GPUs** — rank `r` is both DP
lane `r` (a full attention replica, since TP=1) *and* the owner of expert-shard `r`.
Per MoE decoder layer:

1. **Attention** runs DP-local — each rank on its own tokens + its own KV, **no
   cross-rank comm**.
2. **Router** picks top-k experts per token; those experts may live on any rank.
3. **Dispatch all-to-all** sends each token's hidden vector to the rank that owns its
   expert(s). *This is the only coupling between the DP lanes.*
4. **Experts** compute locally on the tokens they received (gathered from all ranks).
5. **Combine all-to-all** returns outputs to each token's home rank → top-k weighted
   sum → next layer.

Consequences:

- **Comm profile differs from TP.** TP = all-reduce *every* layer (incl. attention);
  DP+EP = all-to-all *only at MoE layers*, attention is comm-free. Keep the EP
  all-to-all **intra-node (NVLink)** — cross-node EP is far slower (see `multi-node.md`).
- **Experts still see a global batch** (tokens gathered from all DP lanes) → better
  utilization than TP-sharded experts.
- **Routing is data-dependent → load can be uneven** across ranks; vLLM makes idle
  ranks run dummy forward passes while any rank is busy, so DP+EP works best with load
  spread evenly across replicas (normal under steady eval concurrency).

### When to enable

- **Yes — any MoE**, especially large / many-expert (DeepSeek, large Qwen MoE, GLM
  MoE). EP powers the standard high-throughput **DP-attention + EP-MoE** layout.
- **No — dense models** (no experts). Also a **no-op at `TP=DP=1`** (nothing to
  distribute), so it's safe-but-pointless on a single GPU.

**Detecting MoE:** handle suffix encoding active params (`-A10B`, `-A3B`, `-A22B`),
`num_experts` / `num_local_experts` / `n_routed_experts` in `config.json`, or the
card. (`-A10B` etc. = *active* params of an MoE — a strong MoE signal.)

> Cross-check `recipes.vllm.ai` for the family's validated TP/DP/EP layout and GPU
> count, then adapt to your GPUs with the fit math (e.g. recipe TP=2 on 2×H200 → on
> an 8-GPU node, TP=2/DP=4).

---

## Layer 2 — Concurrency knobs and how they relate

- **`parallelism`** — requests the eval **client** keeps in flight *per benchmark*.
  Continuous batching holds `parallelism` open at all times, dispatching a new one
  the instant another finishes (a sliding window, not discrete "waves").
- **`--max-num-seqs`** — sequences a single vLLM **replica** decodes concurrently.
  Total server capacity:

  ```text
  serving_capacity = max-num-seqs × data_parallel_size × num_instances
  ```

  (TP and PP shard *one* replica, so they don't add capacity; replicas = DP, times
  `num_instances` for HAProxy multi-instance — see `multi-node.md`.)

Keep them matched: **`max-num-seqs = ceil(parallelism / (DP × num_instances))`**.
If `parallelism` exceeds `serving_capacity`, the surplus just queues in vLLM — no
speedup, and deep queues can trip `request_timeout`.

## The binding constraint flips with run size

`parallelism` is useful only up to the smaller of two ceilings:

1. **Total requests for the task** = `dataset_size × repeats` (`repeats` =
   `n_samples` for simple-evals / tau2-bench, `num_repeats` for nemo-skills — see
   `quantization-benchmarks.md`). You can't have more in flight than exist.
2. **Sustainable serving capacity** = `max-num-seqs × DP × num_instances`, bounded
   by KV-cache memory (below).

| Situation | Set `parallelism` to | Why |
| --- | --- | --- |
| `total_requests ≤ serving_capacity` (small run) | `total_requests` (round up a little for uneven DP routing) | All requests dispatch at once → one wave → finishes in ~one generation-time. Higher is wasted. |
| `total_requests ≫ serving_capacity` (large run) | the **preemption-free** capacity at the *task's* context — often *below* nominal `serving_capacity` (see Balanced sizing) | Throughput-bound: keep decode slots full *without thrashing*. Request count no longer matters; KV headroom does. |

So "set it higher" is right **only up to the request count** for small runs; for
large runs it's right **only up to the preemption-free point** — past that you don't
just over-reserve KV, you *regress* (next section).

## Sizing `--max-num-seqs` against KV cache

For throughput-bound runs `max-num-seqs` is capped by KV memory, driven by
**context length × concurrent sequences**. High `max_new_tokens` (e.g. 81920) makes
each sequence's KV large, shrinking the sustainable batch. **Read the ceiling from
vLLM's startup log** rather than guessing:

- `# GPU blocks: N` — total KV blocks.
- `Maximum concurrency for <max-model-len> tokens per request: X.XX×` — how many
  full-context sequences fit (more at shorter effective context).

During the canary, watch:

- **Preemption** (`Preempted N requests` / recompute) ⇒ `max-num-seqs` above what
  KV sustains; lower it (preemption wastes work).
- **`GPU KV cache usage`** well below 100% with zero preemption ⇒ headroom; raise.

Factors that **relax** the KV limit: small / low-precision weights (more HBM for
KV); **KV-cache quantization** — either baked into the checkpoint (`kv_cache_scheme`
in `config.json`) **or enabled at serve time** with vLLM's `--kv-cache-dtype fp8`
(or `fp8_e4m3` / `fp8_e5m2`), which roughly halves KV vs bf16/fp16 → ~2× the
sustainable concurrency / context; and **hybrid / linear-attention** layers
(near-constant state instead of growing KV). Serve-time KV quantization is a knob
you can turn in `deployment.command` to fit more sequences when KV is the bottleneck
(verify the model/recipe supports it — it can slightly affect accuracy).

## Balanced sizing: bigger is not always faster (especially long context)

Decode throughput first saturates HBM bandwidth (more sequences stop adding
tokens/sec) and then, past the KV-fit point, **regresses** — and the regression is
worst for long-context / long-output tasks. Three mechanisms:

1. **Preemption thrash.** When admitted sequences exceed what KV holds, vLLM preempts
   (recompute or swap). Recompute discards a partially-finished decode — and
   re-running a ~120K-token prefill is enormous wasted work. A modest,
   preemption-free concurrency finishes *sooner* than a high one that thrashes.
2. **Prefill/decode contention.** Long inputs = huge prefills. With
   `--max-num-batched-tokens` fixed, many concurrent long prefills split that budget
   and starve decode — everything crawls.
3. **Latency → timeout → retry cascade.** Too many in-flight requests shrink each
   one's compute share; p99 latency climbs past `request_timeout`, triggering
   `max_retries` resubmissions that pile *more* load onto an already-saturated server.

**Sustainable concurrency is context-dependent.** vLLM's startup
`Maximum concurrency for <max-model-len> tokens` is the *full-length floor*; at a
task's actual working length you fit more (short tasks) — but for long-context tasks
only a handful. So a `parallelism` that's ideal for GPQA (short prompt) will thrash
AA-LCR (~120K input). **Never inherit a short task's `parallelism` for a long one.**

**Balanced rule:**

- Target `parallelism` ≈ **70–80% of the preemption-free KV-fit concurrency at the
  task's working context** (prompt + expected generation) × DP — not the model's
  nominal max. The 20–30% margin absorbs length variance and uneven DP routing.
- **Per-task override for long-context / long-output tasks** (AA-LCR, big
  `max_new_tokens` reasoning): set a *lower* `parallelism` under that task's `params`;
  don't let the higher top-level value apply.
- **Tune empirically (canary), raising only while ALL THREE hold:** throughput
  (req/s) rises, preemption ≈ 0, and p99 latency stays within `request_timeout`. Stop
  at the first that breaks — that's the knee; back off ~20%.
- **When unsure, err low for long context.** A slightly-too-small `parallelism` only
  mildly underutilizes the GPUs; a too-large one thrashes and can be *multiples*
  slower. Goal = **largest batch with ~zero preemption**, not the max the config accepts.

## Non-GPU caps

- **Judge / user-sim tasks** (HLE, AA-LCR, Tau2-Bench Telecom): `parallelism` is
  often capped by the **judge's rate limit**, not the served model. Start
  conservative; raise only after judge logs are clean. Use a per-task `parallelism`
  override when its ceiling differs (e.g. Tau2 cap 512).
- **Context length is itself a per-task cap.** Long-context / long-output tasks need
  a *lower* `parallelism` than short ones on the same deployment — give them an
  explicit per-task override (see Balanced sizing), don't reuse the top-level value.
- **Per-task overrides:** size `--max-num-seqs` off the **max** `parallelism` across
  the top-level and all per-task overrides (the deployment must support the busiest
  task), even though long-context tasks themselves run at a lower `parallelism`.

## Running a suite: `parallelism` is per-task, not per-run

A benchmark suite (e.g. AA) runs tasks with **different bottlenecks** against one
deployment, so a single suite-wide `parallelism` is wrong. Set a top-level **default**
for the model/GPU-bound short tasks, then **override the outliers**:

| Bottleneck | Example AA tasks | Set `parallelism` by |
| --- | --- | --- |
| Model / GPU KV (short in) | `gpqa_diamond_aa_v3`, `ns_ifbench` | top-level default (preemption-free KV-fit) |
| **Long-context KV** (~120K in) | `ns_aa_lcr` | **LOW** per-task override — prefill thrash; model-dependent (MLA ≫ GQA) |
| **Judge / user-sim rate limit** | `ns_hle_aa`, `ns_aa_lcr`, `tau2_bench_telecom` | the judge endpoint (429s), **not** your model |
| **Sandbox execution** | `ns_scicode` | concurrent sandbox slots |

Rules:

- **Judge/sandbox-bound tasks bottleneck *before* the model** — over-parallelizing
  them yields 429s/retries (the timeout cascade), not speed. Cap to the endpoint and
  tune by *its* errors, independent of GPU KV.
- **Long-context tasks (AA-LCR) are KV-bound** — give them an explicit **low**
  per-task `parallelism` (see Balanced sizing); never inherit the short-task default.
- `--max-num-seqs` is sized off the **max** `parallelism` across all tasks (the
  deployment must serve the busiest one), even though the long-context / judge-bound
  tasks themselves run lower.
- **Canary each bottleneck class separately** (model-only / judge-scored / sandbox)
  and tune that task by its own signal — preemption, judge 429s, or sandbox saturation.

Tasks whose `parallelism` is endpoint- or context-dependent ship with the field as
`???` in their recipe (`ns_aa_lcr`, `tau2_bench_telecom`) so it's a conscious choice.

---

## Worked examples

**Dense 9B NVFP4, 8×B200 (this skill's GPQA run).** Weights ~5–6 GB → fits one GPU
with huge KV headroom → **TP=1, DP=8, no EP.** Concurrency: GPQA Diamond = 198
questions; `n_samples=1` → 198 requests (request-bound) → `parallelism=256`,
`max-num-seqs=ceil(256/8)=32`. `n_samples=8` → 1,584 requests (capacity-bound) →
start `parallelism=512` (`max-num-seqs=64`), then tune up **only while preemption
stays ≈ 0** — GPQA's reasoning outputs run to ~82K tokens, so the knee may sit well
below 1024; watch the preemption counter rather than assuming KV headroom.

**Dense ~70B BF16, 8×H100 (80 GB).** ~140 GB weights → won't fit one GPU; TP=2
(~70 GB/GPU + KV) fits → **TP=2, DP=4, no EP.** `serving_capacity = max-num-seqs ×
4`.

**Large MoE ~235B-A22B, 8×H200.** MoE (the `-A22B` = active params) → enable EP.
Throughput layout: **`--data-parallel-size 8 --enable-expert-parallel`**
(DP-attention + EP-MoE, EP size 8), or TP=8 + EP if a single replica's attention/KV
needs the full node. Pick per `recipes.vllm.ai` and the fit math.

**Trillion-scale MoE (Kimi-K2-class, ~1T/32B, MLA), 8×B200 — bit-width flips the
split.** Same architecture, same node; only the precision differs:

- **FP8 (~1040 GB):** weight-bound. Experts alone are ~124 GB/GPU after EP-sharding;
  add replicated non-expert weights and TP=1 overflows 173 GB → **forced to
  `TP=8, DP=1, EP on`** (one replica across the node, ~43 GB/GPU KV — fine for MLA).
- **4-bit — INT4 or NVFP4 (~520–572 GB):** experts drop to ~62–68 GB/GPU, leaving
  room to replicate attention 8× → **`TP=1, DP=8, EP on`** (8 lanes, max
  throughput); step to `TP=2/DP=4` only if a long-context canary shows KV preemption.

INT4 and NVFP4 here are ~the same size → **same layout** — don't let the differing
handles (`moonshotai/…` vs `nvidia/…-NVFP4`) suggest otherwise. The only real
divider is FP8 (weight-bound, TP=8) vs 4-bit (DP-capable, TP=1). This is also why a
4-bit Kimi that needed `TP=8/DP=1` on a tighter 8×H200/640 GB node can switch to
`TP=1/DP=8` on the larger 8×B200 node — adapt the layout to the GPUs you actually
have.
