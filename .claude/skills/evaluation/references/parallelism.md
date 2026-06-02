# Parallelism: topology (TP/DP/PP/EP) + concurrency (`parallelism` / `--max-num-seqs`)

Two decisions, in order — both affect **throughput only, never scores**:

1. **Topology** — how the model is laid out across GPUs (sets the replica count).
2. **Concurrency** — requests in flight (`parallelism`) and per replica
   (`--max-num-seqs`), sized on top of the topology.

## Layer 1 — topology (TP / DP / PP)

- **TP** shards each layer (weights+KV) within one replica → fits a too-big model /
  splits KV for long context; costs an all-reduce **every layer** (keep intra-node).
- **DP** replicates the model → N independent replicas = N× concurrency; N× weight memory.
- **PP** shards layer ranges → very large / multi-node; pipeline bubbles. See `multi-node.md`.

**Decide (single node, G GPUs):**

1. **TP = smallest that fits** with KV headroom. Weights ≈ `params × bytes/param`
   (NVFP4 ≈0.5–0.6, FP8 ≈1, BF16 ≈2); need
   `weights/TP + KV + activations + overhead < GPU_mem × util`. Fits on one GPU → TP=1.
   TP must divide `num_attention_heads` (ideally `num_key_value_heads`), be a power of
   2, and never cross nodes.
2. **DP = floor(G / (TP×PP))** — maximize for throughput (a 1-GPU-fit model runs
   `TP=1,DP=G`, not `TP=G,DP=1`).
3. **PP** only if it won't fit at max intra-node TP, or multi-node.

> **Gotcha — bit-width sets the topology, not the model name.** Read precision from
> `config.json` (`quantization_config`/`quant_algo`/dtype); don't infer from the
> handle. Same arch + same bit-width → same TP/DP/EP regardless of vendor (INT4 vs
> NVFP4 differ only in auto-detected kernel flags). The split changes only when
> bit-width changes *size*.

**Choosing the TP/DP split** (e.g. on 8 GPUs: `1/8`, `2/4`, `4/2`, `8/1`, all EP=8):
default **smallest TP, largest DP** — DP scales throughput ~linearly with no extra
comm; TP adds an all-reduce per attention layer. Raise TP **only** to relieve memory
DP can't:

1. a single request's KV won't fit one replica's HBM (long context — AA-LCR ~120K / 262K);
2. preemption at your target per-replica `max-num-seqs` (TP=2 doubles per-replica KV);
3. weights don't fit one GPU even after EP-sharding.

Else higher TP wastes KV and gives up replicas. **Verify:** vLLM startup
`Maximum concurrency for <max-model-len> tokens` ≳ `parallelism/DP` with no canary
preemption → smaller TP wins.

## Layer 1b — Expert parallelism (EP), MoE only

`--enable-expert-parallel` is a **boolean** (no `--expert-parallel-size`); experts
are partitioned across the whole world size:

```text
EP = tensor_parallel_size × data_parallel_size    (EP = TP only when DP=1)
```

So on a fixed node you don't tune EP — you tune the TP/DP split, which only changes
the *attention* side:

| Layout (8 GPUs, all EP=8) | Attention | Best when |
| --- | --- | --- |
| `TP=1 DP=8` | 8 replicas, comm-free | **default** — one request's KV fits 1 GPU |
| `TP=2 DP=4` | 4 replicas | need ~2× per-replica KV (long ctx) |
| `TP=4 DP=2` | 2 replicas | ~4× per-replica KV, or weights too big for TP≤2 |
| `TP=8 DP=1` | 1 replica | trillion-scale weights / one huge KV pool |

Down the table = more per-replica KV/weight room, fewer replicas, higher all-reduce
cost; pick the **topmost row that fits**.

**Dataflow (DP-attention + EP-MoE):** the DP and EP groups are the **same GPUs**.
Attention is DP-local (no cross-rank comm); each MoE layer does a dispatch+combine
**all-to-all** to route tokens to the rank owning their expert. So comm is all-to-all
*only at MoE layers* (vs TP's per-layer all-reduce) — keep it **intra-node (NVLink)**.
Data-dependent routing → uneven load; vLLM runs dummy passes on idle ranks, so spread
load evenly.

**Enable for any MoE** (detect via `-A10B`/`-A3B`/`-A22B` handle, `num_experts` /
`n_routed_experts` in `config.json`); **not for dense**; no-op at `TP=DP=1`.
Cross-check `recipes.vllm.ai` for the validated layout, then adapt to your GPU count
via the fit math.

## Layer 2 — concurrency (`parallelism` / `--max-num-seqs`)

- **`parallelism`** = requests the client keeps in flight *per benchmark*.
- **`--max-num-seqs`** = sequences one replica decodes at once.

```text
serving_capacity = max-num-seqs × DP × num_instances
max-num-seqs     = ceil(parallelism / (DP × num_instances))   # keep matched
```

(TP/PP don't add capacity; replicas = DP, × `num_instances` for HAProxy — see
`multi-node.md`.) `parallelism` above capacity just queues in vLLM (and risks
`request_timeout`).

**`parallelism` ceiling = the smaller of:**

1. **total requests** = `dataset_size × repeats` (`n_samples` for simple-evals/tau2,
   `num_repeats` for nemo-skills) — can't have more in flight than exist;
2. **preemption-free capacity at the task's context** (KV-bound; below).

| Run | Set `parallelism` to |
| --- | --- |
| `total_requests ≤ capacity` (small) | `total_requests` (round up for uneven DP routing) → one wave |
| `total_requests ≫ capacity` (large) | the **preemption-free** capacity at the task's context (often *below* nominal) |

**Sizing `--max-num-seqs` vs KV** — capped by `context × concurrent seqs`; high
`max_new_tokens` shrinks the batch. Read vLLM startup `# GPU blocks` /
`Maximum concurrency for <max-model-len> tokens` (full-length floor — you fit more at
shorter context). Canary: `Preempted N` → lower; KV usage ≪100% with no preemption →
raise. **Relaxed by:** low-precision weights; **KV-cache quantization** — checkpoint
`kv_cache_scheme` **or serve-time `--kv-cache-dtype fp8`** (`fp8_e4m3`/`fp8_e5m2`) in
`deployment.command`, ~halving KV → ~2× concurrency/context (verify support; small
accuracy effect); and **hybrid/linear-attention** (near-constant KV).

## Balanced sizing — bigger is NOT always faster (esp. long context)

Past the KV-fit point throughput doesn't just plateau, it **regresses** — worst for
long-context / long-output:

1. **Preemption thrash** — over-admitted seqs get preempted; recomputing a ~120K
   prefill is huge wasted work, so a modest preemption-free concurrency finishes *sooner*.
2. **Prefill/decode contention** — many long prefills split `--max-num-batched-tokens`
   and starve decode.
3. **Timeout cascade** — too many in-flight → p99 > `request_timeout` → `max_retries`
   resubmissions pile on more load.

Sustainable concurrency is **context-dependent** — a `parallelism` good for GPQA
(short) thrashes AA-LCR (~120K). **Rule:** target ~**70–80% of the preemption-free
KV-fit concurrency at the task's working context × DP**; give long-context/long-output
tasks a **lower per-task override**; canary-tune up only while throughput↑,
preemption≈0, p99 < `request_timeout`; **err low** for long context (too-small mildly
underutilizes; too-large is *multiples* slower).

## Suites — set `parallelism` per task, not per run

Suite tasks hit **different bottlenecks** against one deployment; use a top-level
default for short model-bound tasks and override the outliers:

| Bottleneck | AA tasks | Cap by |
| --- | --- | --- |
| Model / GPU KV (short) | `gpqa_diamond_aa_v3`, `ns_ifbench` | top-level default (preemption-free KV-fit) |
| Long-context KV (~120K) | `ns_aa_lcr` | **low** override — prefill thrash; MLA ≫ GQA |
| Judge / user-sim rate limit | `ns_hle_aa`, `ns_aa_lcr`, `tau2_bench_telecom` | judge endpoint 429s, **not** the model |
| Sandbox execution | `ns_scicode` | sandbox slots |

- Judge/sandbox tasks bottleneck **before** the model — over-parallelizing yields
  429s/retries, not speed; cap to the endpoint, tune by *its* errors.
- `--max-num-seqs = ceil(max parallelism across tasks / DP)` (deployment must serve the
  busiest task) even if long-context tasks run lower.
- Canary each class (model / judge / sandbox) separately. Endpoint/context-dependent
  tasks (`ns_aa_lcr`, `tau2_bench_telecom`) ship `parallelism: ???` to force a choice.

## Worked examples (8×B200)

- **Dense 9B NVFP4** (~5–6 GB) → **TP=1/DP=8, no EP**. GPQA `n_samples=1` = 198 reqs
  (request-bound) → `parallelism=256`, `max-num-seqs=32`. `n_samples=8` = 1584
  (capacity-bound) → start 512; tune up only while preemption≈0 (~82K reasoning output
  → knee may be <1024).
- **Dense ~70B BF16, 8×H100/80GB** (~140 GB) → won't fit 1 GPU → **TP=2/DP=4, no EP**.
- **Large MoE ~235B-A22B** → EP on; layout `DP=8 + EP` (or `TP=8 + EP` if one replica
  needs the full node for KV).
- **Trillion-scale MoE (Kimi-class ~1T, MLA) — bit-width flips the split:** FP8
  (~1040 GB) is weight-bound → forced **TP=8/DP=1/EP**; 4-bit INT4/NVFP4 (~520–572 GB)
  frees room → **TP=1/DP=8/EP**. INT4 ≈ NVFP4 → same layout (don't let `moonshotai/…`
  vs `nvidia/…-NVFP4` mislead) — same reason a 4-bit Kimi needing TP=8 on 8×H200/640GB
  switches to TP=1/DP=8 on 8×B200.
