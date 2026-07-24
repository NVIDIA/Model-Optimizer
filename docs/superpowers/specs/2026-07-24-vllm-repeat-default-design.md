# vLLM Repeat-Count Default Design

## Goal

Change Puzzletron's global paired runtime-estimator repeat count from `10` to
`4`. Apply the new default consistently to direct configurations and
setup-wizard bundles while preserving explicit per-run overrides.

## Configuration Contract

Add `repeat_block_n_times: 4` to the base Puzzletron vLLM runtime-statistics
configuration. The setup bundle renderer already deep-merges this base
configuration, so generated smoke and production bundles inherit the value
without model-name conditionals.

Change the defensive fallback used by block and subblock runtime estimation
from `10` to `4`. Configurations that explicitly set
`repeat_block_n_times` remain unchanged.

The estimator continues to measure paired `N` and `2N` exact physical layouts.
Only the default value of `N` changes; cache identity already includes the
effective repeat count.

## Regression Coverage

Add focused tests proving:

- wizard-generated experiments contain `repeat_block_n_times: 4`;
- block runtime estimation uses `4` when the setting is absent;
- subblock runtime estimation uses `4` when the setting is absent; and
- an explicit repeat count still overrides the default.

## Current Nano Campaign

Regenerate both Nano bundles from canonical `answers.yaml`. Verify that smoke
and production YAMLs contain `repeat_block_n_times: 4`, validate both bundles,
and compile both dry-run plans without submitting jobs.

Remove invalidated vLLM-statistics outputs from the Nano production result
tree:

- `runtime_cache/`;
- `artifacts/vllm_stats/`;
- `artifacts/subblock_stats/parameter_inventory/`;
- `subblock_stats.json`, if present; and
- `manifests/vllm_stats.json`, if present.

Preserve all non-vLLM stage artifacts, orchestration history, and the failed
vLLM log. Confirm no active Nano Slurm job before deletion. The user will
relaunch the orchestrator manually.
