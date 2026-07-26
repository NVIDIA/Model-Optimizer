# AIPerf Best-Selection Design

## Goal

Benchmark every post-MIP model with one fixed vLLM topology and one configured concurrency
sweep, then support two deterministic ways to retain the fastest models.

## Configuration Contract

The serving node owns one topology and one ordered concurrency list:

```yaml
serving:
  type: aiperf
  config:
    concurrency: [1, 2, 4, 8, 16, 32]
    topology:
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
      data_parallel_size: 8
      enable_expert_parallel: true
```

The downstream selector remains a `top_k` filter and gains one optional field:

```yaml
fastest:
  type: filter
  input: serving
  mode: top_k
  metric: serving.output_token_throughput
  direction: maximize
  top_k: 4
  best_selection_mode: individual_best
```

`best_selection_mode` accepts:

- `individual_best`: reduce each model's concurrency sweep to its own best finite metric,
  then retain the global top K models.
- `best_per_concurrency`: rank models separately at every concurrency, retain top K at each
  point, then union and deduplicate the selected models.

Omitting `best_selection_mode` preserves the existing scalar `top_k` behavior. The new field
is valid only for `mode: top_k`.

The recommended flow defaults to `individual_best`. While configuring an AIPerf node, setup
v2 explicitly asks:

- `Serving concurrency sweep (comma-separated; one value is allowed):`
- `How should the best models be selected?`, with `individual_best` and
  `best_per_concurrency` choices.

It also asks for one serving topology, which is reused across the entire sweep. The wizard
writes the concurrency list into the AIPerf node and writes `best_selection_mode` into its
downstream `fastest` filter; the selection setting is not forwarded to the AIPerf CLI.
For a custom flow, the mode question is asked when configuring a filter that selects metrics
from an AIPerf node.

## Metric Resolution and Selection

The AIPerf node continues recording metrics using its existing keys:

```text
concurrency_1.output_token_throughput
concurrency_2.output_token_throughput
concurrency_4.output_token_throughput
```

For a filter metric such as `serving.output_token_throughput`, the selector resolves the
`serving` observation and matches only keys of the form
`concurrency_<positive integer>.output_token_throughput`.

For `individual_best`, `maximize` uses the maximum value and `minimize` uses the minimum.
Normal `top_k` tie-breaking remains deterministic by metric value and revision ID.

For `best_per_concurrency`, each concurrency uses the same direction and top K. The final
candidate order is deterministic: models are ordered by their best rank at any concurrency,
then by revision ID. A candidate missing any concurrency present in the completed sweep is
excluded rather than being compared on a partial sweep. The union may contain as many as
`top_k * number_of_concurrencies` models.

Filter-produced scores record the reduced metric for `individual_best` and the best
one-based rank for `best_per_concurrency`, so reports can explain why a model was retained.

## Setup and Resource Accounting

The v2 setup wizard parses and validates positive, unique concurrency integers while
preserving their entered order. Invalid or duplicate values are rejected and reprompted.
It validates the selection-mode answer at the same boundary. The serving topology is asked
once and reused for every concurrency and model.

The recommended post-MIP flow uses `serving.output_token_throughput`, not request throughput,
because output TPS is comparable across a fixed ISL/OSL workload and directly represents
system token capacity.

Candidate-limit estimation must account for `best_per_concurrency`. Its upper bound is the
smaller of the input candidate count and `top_k * len(concurrency)`. This prevents downstream
KD/evaluation resources from being under-provisioned when the per-concurrency union contains
more than top K models.

## Validation and Errors

Configuration validation rejects:

- unknown `best_selection_mode` values;
- `best_selection_mode` on non-`top_k` filters;
- a sweep-aware selector whose metric has no owner, such as an unqualified
  `output_token_throughput`;
- empty, non-positive, boolean, or duplicate concurrency values.

A model with missing or non-finite metrics is excluded with an explicit reason. Existing
AIPerf execution remains strict, so a failed concurrency normally prevents that model's
serving observation from being published at all.

## Test Corrections

Two stale tests are corrected alongside the feature:

- The mesh `PP=2, DP-replicate=2, DP-shard=4` uses 16 GPUs, spans two eight-GPU nodes, and
  is not an exclusive four-node allocation.
- `sort` is a one-GPU stage in the current execution model and is removed from the list of
  CPU-only stages. The test explicitly checks its GPU allocation instead.

## Test Coverage

Tests cover:

- configuration validation and metric-reference discovery for both selection modes;
- global top K using each model's individual best concurrency;
- top K per concurrency with deterministic union and deduplication;
- minimize direction, ties, missing points, and non-finite values;
- recommended-flow rendering and the setup v2 questions for concurrency-list and selection
  mode;
- candidate-limit estimation for the per-concurrency union;
- both corrected execution/mesh expectations;
- the existing Nano full orchestration dry-run.
