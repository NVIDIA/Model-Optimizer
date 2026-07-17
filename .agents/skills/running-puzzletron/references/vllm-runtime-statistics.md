# vLLM Runtime Statistics

## Inspect Runtime and Cost Code

Read the current implementations before changing formulas:

- `modelopt/torch/puzzletron/export/vllm.py`
- `modelopt/torch/puzzletron/utils/vllm_adapter.py`
- `modelopt/torch/puzzletron/subblock_stats/runtime_vllm.py`
- `modelopt/torch/puzzletron/subblock_stats/calc_subblock_params_and_memory.py`

Count total and active parameters separately. Include embeddings, projectors,
norms, routers, experts and top-k activation, tied weights, MTP, KV/SSM caches,
activations, and server overhead. State topology and workload assumptions.

## Exact Production Construction

Instantiate every production candidate at exact physical dimensions. Proxy caps
for expert count, routed/shared width, top-k, sequence length, or batch size are
legal only behind an explicit smoke-only flag. Put the flag and caps in cache
identity and never aggregate proxy measurements into production profiles.

Build temporary configs through the model descriptor. Derive the exact hybrid
pattern and mutually exclusive attention, Mamba/SSM, MoE, or dense-FFN
placement. Do not emit dense-FFN candidates/report rows for models without dense
FFNs. Keep a minimal valid scaffold only when homogeneous candidates cannot be
instantiated alone.

## Paired Marginal Estimation

Use paired `N` and `2N` layouts to cancel fixed overhead, defaulting to `N=4`
for production after a smaller numerical gate. Measure the identical candidate
and workload at both endpoints. Prefer candidate-only homogeneous layouts;
otherwise keep the same descriptor-defined scaffold in both so it cancels.

Changing `N`, warmups, iterations, scaffold, hybrid pattern, workload, or
estimator schema changes cache identity. Recover and validate the fixed overhead
and repeated controls; do not trust close rankings when control spread is large.

## One-GPU Worker Isolation

For each independent worker:

- bind exactly one explicit CUDA device;
- assign each vLLM subprocess a unique rendezvous port;
- isolate Triton, TorchInductor, TileLang, vLLM, FlashInfer, CUDA, temporary,
  PyTorch-kernel, and XDG caches;
- share only immutable model/data caches;
- stagger local starts when compilation/filesystem pressure is high;
- write immutable phase results and transactional shard completion markers.

Keep backend overrides scoped to this experiment. Do not globally disable a
fused path to make one benchmark pass. Verify `pip check`, vLLM and MoE native
extension imports, and one real-model fused-kernel gate before a large sweep.

## Cache and Aggregation

Cache identity covers the finalized model config, exact candidate, workload,
topology, estimator, environment overrides, GPU type, and software revisions.
Temporary paths and ports are non-semantic. If construction/timing code changes
without a schema bump, quarantine the cache manually.

Resume only compatible missing endpoints. Publish phase JSON transactionally,
then shard markers. One aggregator validates all shard identities, indices,
counts, finiteness, and candidate coverage before publishing canonical
`subblock_stats.json`. Exclude structural no-op controls from candidate counts
and MIP costs. A no-op or scaffold must not create a fake FFN row for a model
without FFNs.

## Anomalous Marginals

Never clamp negative or non-finite marginals. Identify and remeasure only the
exact original `N`/`2N` endpoints on quiet, pinned hardware while retaining
unaffected compatible caches. A higher-iteration diagnostic has a different
identity and cannot silently replace production data.

Non-positive total cost, attention/control failures, or widespread anomalies
fail the gate. An explicit `ignore_negatives` policy may retain a characterized
phase-only noise-dominated value with warning/provenance, but that value cannot
support a constraint on the affected phase. Record endpoints, remeasurement,
and acceptance rationale in the report.

## Completion Gate

Require the exact active-candidate count at every width/workload, all expected
layout phases and shard markers, matching identities, finite metrics, valid
fixed-overhead controls, and parameter/cache/memory accounting. Then publish the
canonical summary, stage manifest, and regenerated cumulative report. Partial
caches are progress, not stage completion.

