# vLLM Parameter-Inventory Lock Design

## Problem

A clean sharded vLLM-statistics stage launches eight independent one-GPU
workers. Each worker currently computes and publishes the same parameter
inventory for each embedding width. Although publication uses atomic file
replacement, workers can still overwrite one another's logical states.

The observed failure occurs after one worker publishes its complete inventory
and then re-reads the shared path while another worker has published an
intermediate inventory. The first worker consequently receives an incomplete
row set and fails static-stat assembly with a missing subblock key.

The reported parent layer `-1` is intentional. Runtime candidates loaded from
`subblock_library.json` are homogeneous and deduplicated across physical
layers, so they use `-1` to mean “no specific parent layer.” The completed
inventories contain all expected `-1` keys.

## Design

Serialize parameter-inventory ownership per embedding width with Puzzletron's
existing `distributed_eval.storage.file_lock`.

`_calculate_parameter_inventory_for_width` will acquire:

```text
artifacts/subblock_stats/parameter_inventory/.width-NNNN.lock
```

before reading or writing that width's cache. The cache completeness check must
run inside the lock. Therefore:

1. The first shard acquires the lock, builds the inventory, and publishes it.
2. Waiting shards acquire the lock one at a time.
3. Each waiting shard re-reads the now-complete cache and returns it without
   rebuilding.

Widths retain separate locks, so different widths may still be prepared in
parallel when `parameter_workers` is greater than one.

No model topology, candidate identity, `-1` semantics, runtime sharding, or
generated configuration changes.

## Implementation Boundary

Keep the existing inventory calculation as a private unlocked implementation.
The existing `_calculate_parameter_inventory_for_width` entry point becomes a
small locking wrapper so every caller receives the concurrency guarantee.

The lock is process-safe on the Linux/Slurm shared filesystem and is
automatically released if a worker exits.

## Verification

Add a deterministic concurrency regression test:

- create one synthetic `-1` FFN candidate;
- start two callers for the same width simultaneously;
- slow the mocked parameter calculation enough to expose overlap;
- assert both callers receive a complete inventory;
- assert the expensive parameter calculation runs exactly once.

Run the focused sparse-runtime and orchestration tests, then resume the same
campaign through the orchestrator. A successful rerun must show one inventory
builder per width and cache reuse by the other seven shards before runtime
benchmark progress begins.
