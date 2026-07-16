---
name: benchmark-model-kernels
description: >-
  Inspect Hugging Face decoder layers on meta tensors and plan or run per-rank
  BF16, FP8, and NVFP4 GEMM or fused-MoE microbenchmarks with the bundled
  scripts and a local FlashInfer checkout. Use when choosing a model, GPU, TP,
  EP, or M/token-concurrency sweep; deriving common fused QKV and gate/up
  shapes without loading checkpoint weights; or using FlashInfer benchmark
  utilities. Do not use for end-to-end server throughput or request latency.
---

# Benchmark Model Kernels

Plan a single-GPU microbenchmark for the per-rank shapes of an intended
deployment. The scripts never load checkpoint weights, launch distributed
workers, or measure collectives, serving throughput, or request latency.

## Interview, one decision per message

Ask exactly one unresolved decision per message with two or three concrete
options, state the default, and wait for the answer. Recommend an option only
when it is substantively better. Skip decisions already answered. Follow this
order:

1. **Model** — a local directory, a `config.json`, or a Hub ID (equivalent; no
   default). The script builds the model on meta tensors from configuration
   only. Do not enable `--trust-remote-code` without explicit approval; pin
   `--revision <commit>` when it is needed.
2. **TP and EP** — accept both directly, or derive them from the intended
   deployment's GPU model and count and confirm. Defaults: `TP=1`, `EP=1`.
   Derive parallelism from the intended deployment, never from the GPU that
   runs the microbenchmark. The script validates every sharding rule
   (divisibility, GQA replication, expert partitioning) and errors loudly.
3. **M sweep** — balanced default (recommended): `1 8 64 512`; decode-focused:
   `1 4 16 32`; throughput-focused: `64 256 1024 4096`. M is roughly the
   tokens scheduled per step (decode: active sequences), not endpoint
   concurrency.
4. **Shape preview** — always run this before anything else; no FlashInfer or
   GPU needed:

   ```bash
   python .agents/skills/benchmark-model-kernels/scripts/benchmark_model.py <model> \
     --tp <tp> --ep <ep> --ms <m1> <m2> ... --print_only
   ```

   Review the printed shapes, MoE tuple, and derived routing with the user.
   `# unsupported:` lines mean the list is partial and the script exits
   nonzero — handle those via **Manual supplements** below.
5. **FlashInfer checkout and GPU** — the full benchmark needs a FlashInfer
   *source checkout* containing `benchmarks/flashinfer_benchmark.py`; the
   installed wheel alone is not enough. Prefer a clean checkout matching the
   installed `flashinfer` version; ask before cloning or installing anything.
   On the benchmark machine, check `nvidia-smi` and package versions, and
   verify CUPTI timing with a tiny `bench_gpu_time(..., enable_cupti=True)`
   probe — a warning that falls back to CUDA events is a failure (the
   `cupti-python`/`nvidia-cuda-cupti` packages must match PyTorch's CUDA
   major). Ask for a GPU index only when several are visible. Pick a fresh
   workdir and state it.
6. **Full benchmark**:

   ```bash
   CUDA_VISIBLE_DEVICES=<gpu-index> \
   python .agents/skills/benchmark-model-kernels/scripts/benchmark_model.py <model> \
     --tp <tp> --ep <ep> --ms <m1> <m2> ... \
     --flashinfer_repo <flashinfer-repo> --workdir <workdir>
   ```

   Run a short plumbing check first (append
   `--dry_run_iters 1 --num_iters 3 --no_autotune`, throwaway workdir),
   especially after changing FlashInfer versions or shape logic; never present
   it as a performance result. Then run with defaults. The first fused-MoE
   build or autotune can take several minutes; its cache is reused.

## Rules the scripts own

Do not restate, re-derive, or override these — the code enforces them:

- `benchmark_model.py` derives `--nks`, `--nk_names`, and all `--moe_*`
  arguments including the routing method; overrides are rejected.
- Physical padding follows one rule: pad a dimension if and only if vLLM pads
  it for that case; otherwise keep the exact shape and let it fail the same
  way vLLM would. Row labels stay logical — report both shapes when padding
  changes a dimension.
- BF16 and FP8 MoE cases run before NVFP4 because a CUDA fault poisons the
  remaining cases in the same driver process.
- A failed case writes FlashInfer's error message (or a pointer to
  `driver.log`) into its `combined_results.csv` cell, and the command exits
  nonzero after the table is written. Never present a partial table as a
  successful benchmark; read `driver.log` before rerunning anything.

## Known backend and driver limits

- `mm_fp4` TensorRT-LLM needs `N % 128 == 0` (shuffled weight layout). Drivers
  with the conditional-shuffle fix drop only that backend (its cell reports no
  result row); stock 0.6.x drivers fail *all* backends for that shape with an
  empty assertion.
- `mm_fp8` (trtllm_low_latency) needs `K % 128 == 0`.
- Gated NVFP4 CUTLASS MoE with `2F % 128 != 0` per rank fails — vLLM raises
  instead of padding — often as a `CUDA error: misaligned address`. Prefer EP
  over TP for the experts to keep the per-rank width legal.
- Do not benchmark a padded shape to dodge these limits and call it
  vLLM-equivalent; report the limit instead.

## Manual supplements

For each `# unsupported:` layout from the preview: inspect that module's
forward path and the intended runtime's TP/EP sharding, derive the per-rank
shape (never guess sharding from a weight shape alone), and benchmark only the
missing shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> \
python .agents/skills/benchmark-model-kernels/scripts/benchmark_via_builtin.py \
  --flashinfer_repo <flashinfer-repo> --ms <m1> <m2> ... \
  --nks <n>,<k> --nk_names <name> --workdir <workdir>
```

For a manual MoE supplement, also derive and pass the routing arguments
(`--moe_routing_method` plus the DeepSeek group/scaling/bias flags when
applicable) — the runner otherwise defaults to `topk`, a different routing
kernel. Label these rows as manual supplements; this is not a substitute for
running `benchmark_model.py` first. Shell-quote user-supplied paths.

## Report

Report the command, GPU, versions, TP/EP, M values, shapes (logical and
physical where they differ), warnings, and the artifacts: `testlist.txt`,
`driver.log` (full driver output), `builtin_results.csv` (milliseconds,
success-only), and `combined_results.csv` (microseconds; GEMM rows grouped
under `<name>: <NxK>` labels, MoE rows flat). `*_with_quant` rows add a
separately measured activation-quantization time, except NVFP4 MoE with
quantization, which is a single fused measurement. These are kernel times
only — never describe them as end-to-end latency or throughput; they omit
weights, layer frequency, communication, KV cache, and scheduling.
