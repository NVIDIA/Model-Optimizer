# Dynamic Hugging Face Subset Picker Design

## Goal

Replace the free-form Nemotron-VLM subset prompt with a dynamic checkbox that
works for any dataset hosted on the Hugging Face Hub. Each choice shows the
subset name, row count, and original-file size. Multiple selected subsets are
sampled proportionally to their row counts.

The picker must not download dataset rows or media merely to render its choices.
It must preserve the existing bounded materialization controls and keep setup
usable without PyTorch.

## Source of Truth

Subset discovery uses Hugging Face's `datasets.get_dataset_config_names` for the
selected dataset and revision. The Hugging Face Dataset Viewer `/size` response
supplies per-configuration row counts and original-file byte sizes. `HfApi`
resolves the repository, immutable revision, authentication, and file metadata.

For `nvidia/Nemotron-VLM-Dataset-v2`, the current Dataset Viewer exposes 46
configurations. The dataset card's composition table has 51 source rows; that
documentation table is not the subset-discovery authority.

The merged catalog is ordered as returned by Hugging Face and contains:

- dataset repository ID and immutable revision;
- subset/configuration name;
- row count;
- original-file bytes;
- hosted-media availability when an adapter requires local media;
- selectable/disabled state and an actionable disabled reason.

No checked-in list of Nemotron subset names or sizes is used as a discovery
fallback.

## User Experience

After the user selects a Hugging Face dataset, setup fetches its catalog before
asking bounded acquisition questions. The checkbox labels use a compact form:

```text
[x] sparsetables — 100,000 rows — 14.36 GB
[x] plotqa_cot — 16,256 rows — 0.50 GB
[ ] activity_net_1 — 10,021 rows — 191.49 GB — external media required
```

Unavailable choices stay visible but disabled. The Nemotron-VLM adapter
preselects its existing `sparsetables`, `plotqa_cot`, and `wiki_en` defaults
when those configurations remain available. Other Hub datasets preselect their
declared default configuration, or the first selectable configuration if no
default is declared.

At least one selectable subset is required. A selected subset without a
reliable row count is rejected because proportional sampling would otherwise
be undefined.

The subsequent row-count, seed, shard-bound, layout, and sequence-length
questions remain unchanged.

## Generic Boundary

Catalog discovery is dataset-agnostic. Dataset-specific adapters provide only
capability checks:

- The Nemotron-VLM adapter requires media tar shards hosted in the repository.
  Configurations whose media must be obtained externally are visible but
  disabled.
- A generic Hub dataset has no Nemotron-specific path assumptions. Its
  configurations are selectable when Hugging Face exposes valid configuration
  and size metadata.

The selected configurations, immutable metadata identity, row counts, and
normalized proportional weights are persisted in canonical setup state and the
rendered data configuration. Downstream dataset construction consumes this
canonical selection instead of rediscovering or reordering configurations.

## Proportional Sampling

For selected configurations with row counts `r_i`, the canonical weight is:

```text
w_i = r_i / sum(r_j)
```

Bounded materialization converts those weights into deterministic integer
quotas using largest-remainder apportionment:

1. compute each exact quota as `num_samples * w_i`;
2. take the floor of each quota;
3. distribute remaining samples by descending fractional remainder, with
   Hugging Face catalog order as the deterministic tie-breaker.

If a subset yields fewer valid rows than its quota, the materializer
redistributes the deficit proportionally across selected subsets that still
have rows. It fails with per-subset diagnostics if the combined sources cannot
meet the requested total.

## Caching and Failure Handling

Catalog metadata is cached in wizard state by repository ID and immutable
revision. Resuming the same campaign reuses that exact catalog without a
network call.

On first use, failures are explicit:

- missing or gated repository: report authentication/access guidance;
- Hugging Face cannot enumerate configurations: report the upstream error;
- missing size metadata: show the configuration disabled with the reason;
- all configurations disabled: stop before asking acquisition questions;
- stale selected configuration after revision change: require reselection.

There is no static catalog fallback because it could silently select stale or
renamed configurations.

## Components

1. A dependency-light catalog module owns Hub source normalization, immutable
   revision resolution, Hugging Face configuration discovery, Dataset Viewer
   size parsing, capability annotation, formatting, and cache serialization.
2. The v2 wizard turns catalog entries into checkbox choices, validates the
   selection, and persists the selected metadata and weights.
3. Acquisition specifications carry ordered subset weights in their identity.
4. Materializers use deterministic proportional quotas and preserve the
   existing bounded-download guarantees.
5. Bundle rendering emits the selected ordered configurations and weights so
   smoke and production bundles are reproducible.

The setup requirements add Hugging Face `datasets` because
`get_dataset_config_names` is the required subset-discovery function. PyTorch
remains unnecessary.

## Testing

Hermetic tests inject a fake Hugging Face catalog provider at the network
boundary while exercising real catalog merging, checkbox behavior, state
persistence, and quota calculation.

Coverage includes:

- 46 dynamically returned configurations are all displayed;
- labels contain independently specified rows and sizes;
- hosted and external-media configurations are enabled/disabled correctly;
- defaults are selected without hard-coded catalog truncation;
- Back and resume preserve the exact catalog and selection;
- empty, disabled, stale, and size-less selections are rejected;
- proportional quotas sum exactly to the requested bound;
- deterministic tie-breaking and short-source redistribution;
- generic non-Nemotron Hub repositories use the same discovery path;
- existing single-subset and fixed local dataset behavior remains unchanged.

