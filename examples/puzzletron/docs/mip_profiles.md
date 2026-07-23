# MIP runs

Puzzletron separates MIP configuration into independent runs, variants within a
run, and concrete solves. A concrete solve is one run × variant × matrix row ×
objective. Results keep all four identities, so unrelated experiments never get
mixed together.

## Complete example

```yaml
mip:
  defaults:
    objectives: metrics.cosine_embedding_loss_hidden_states
    solver:
      backend: auto
      num_solutions: 1000
      min_hamming_distance: 2
      max_seconds_per_solution: 60
    homogeneous:
      enabled: false
      keep: all
      rank_by: objective

  workloads:
    serving-8k:
      isl: 8192
      osl: 1024
      batch_size: 4
      concurrency: 4

  runs:
    memory-050:
      objectives:
        - metrics.cosine_embedding_loss_hidden_states
        - {metric: metrics.lm_loss, direction: minimize}

      # Every variant retains this primary constraint.
      constraints:
        memory:
          at:
            serving-8k: 50%

      search_space:
        embedding: all
        depth: all

      variants:
        primary-only: {}

        params-too:
          constraints:
            params: 50%

        expert-bands:
          homogeneous:
            enabled: true
            keep: 100
            rank_by:
              constraint_closeness:
                weights:
                  memory: 2
                  experts: 1
          matrix:
            constraints.experts:
              - {range: [128, 144]}
              - {range: [144, 160]}
              - {min: 160}
```

All constraints in a concrete solve are combined with AND. Matrix fields form a
Cartesian product and create separately named concrete solves. An objective list
does the same, but each objective is optimized in an independent solver run.

## Runs, variants, and matrices

Use a new entry under `mip.runs` when two searches are conceptually independent.
Use variants when searches share a primary target but add different restrictions.
Run-level `constraints`, `objectives`, `search_space`, `solver`, and `homogeneous`
settings are inherited by every variant; variant settings override or extend them.
Set an inherited run to `false` in a derived config to disable it.

The explicit `matrix` is for intentional loops. Supported paths are `embedding`,
`depth`, `constraints.*`, `solver.*`, and `homogeneous.*`.

```yaml
variants:
  sweep:
    matrix:
      embedding: [4096, 3840, 3584]
      depth: [0, 2, 4]
      solver.min_hamming_distance: [1, 4]
```

Search-space lists choose options within one solve. Matrix lists create multiple
solves. This distinction keeps generated result identities predictable.

## Objectives and solver controls

An objective is a metric string, which minimizes by default, or a mapping with an
explicit direction. Put `objectives` under `mip.defaults` to share it, or override
it on a run or variant:

```yaml
objectives:
  - metrics.cosine_embedding_loss_hidden_states
  - {metric: metrics.lm_loss, direction: minimize}
```

Every objective gets its own solution pool. `num_solutions` is the requested pool
size. After each solution, Puzzletron adds a diversity constraint based on final
per-layer block configurations. `min_hamming_distance` is the minimum number of
layers whose final configuration must differ from every earlier solution.

```yaml
solver:
  backend: cuopt       # auto, pulp/cbc, or cuopt
  num_solutions: 2000
  min_hamming_distance: 3
  max_seconds_per_solution: 120
```

The solver may return fewer solutions when the feasible diverse pool is exhausted
or a time limit is reached.

## Constraints

A scalar is a directional bound: a maximum for costs and a minimum for benefits.

```yaml
params: 75%                   # at most 75% of the teacher
params: 22.5B                 # at most 22.5 billion parameters
experts: {range: [128, 144]}
throughput:
  at:
    serving-8k: 5000          # at least 5000 tokens/s
```

Use `min`, `max`, `eq`, or `range` for explicit bounds. Percentages are relative
to the full-width, depth-zero teacher, measured at the same workload when the
metric is workload-dependent. Friendly names are `params`, `active_params`,
`memory`, `runtime`, `prefill_runtime`, `throughput`, `kv_heads`, and `experts`.
A raw `stats.*` metric can be used when no friendly name exists.

Memory, runtime, prefill runtime, and throughput must select a named workload:

```yaml
constraints:
  memory:
    at:
      serving-8k: {min: 48%, max: 50%}
```

## Search space

```yaml
search_space:
  depth: {range: [0, 5]}
  embedding: [2688, 2560, 2432]
  axes_default: teacher
  axes:
    n_routed_experts: all
    moe_intermediate_size: [4096, 3072]
```

`all` selects every measured option. A scalar or list selects exact options, and
`{range: [low, high]}` selects an inclusive measured range. With
`axes_default: teacher`, omitted pruning axes stay at the teacher value.

Depth can also select typed subblock prefixes:

```yaml
search_space:
  depth:
    attention: 4
    moe: [1, 2]
```

Typed depth requires subblock granularity. It filters the existing global depth
trajectory by type; it does not run a separate importance ranking.

## Homogeneous search

Homogeneous search enumerates candidates that use one consistent choice for each
applicable pruning axis across layers. It is separate from the heterogeneous MIP
pool and is configured explicitly:

```yaml
homogeneous:
  enabled: true
  keep: 100            # positive integer or all
  rank_by: objective
```

`rank_by: objective` orders feasible homogeneous candidates using the current MIP
objective. To favor candidates near constraint boundaries instead, use:

```yaml
rank_by:
  constraint_closeness:
    weights:
      memory: 2
      params: 1
```

For multiple constraints, closeness is the weighted worst normalized distance to
the requested boundary (or interval midpoint). Objective value is the tie-breaker.
This keeps the policy deterministic without introducing another objective model.

MIP and homogeneous outputs may contain thousands of candidates. Candidate
deduplication and filtering belong to the configurable post-MIP pipeline; see
`post_mip_pipeline.md`.

Each completed invocation writes `mip/active_profiles.json`, the authoritative
snapshot of concrete profile IDs and their execution identity. Old profile directories
remain available as history, but post-MIP source selection only reads candidates from
this active snapshot. Changing constraints, objectives, search dimensions, solver-pool
settings, or materialization mode invalidates the corresponding cached scenario.
The identity also snapshots upstream score, statistics, library, teacher-control,
and stage-manifest artifacts, so regenerating MIP inputs invalidates old solutions.
