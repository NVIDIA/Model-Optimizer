# MIP Profiles

Puzzletron MIP profiles describe resource constraints, the workloads used to
measure them, and optional restrictions on the architecture search space. A
profile may produce the normal heterogeneous solution and a separate ranked
set of homogeneous solutions.

## Complete example

```yaml
mip:
  workloads:
    isl-heavy:
      isl: 8192
      osl: 128
      batch_size: 4
    osl-heavy:
      isl: 1024
      osl: 8192
      batch_size: 4
    serving:
      isl: 8192
      osl: 1024
      concurrency: 4

  profiles:
    memory-under-two-workloads:
      num_homogeneous_solutions: 3

      constraints:
        params: "75%"
        active_params:
          min: "70%"
          max: "90%"
        memory:
          at:
            isl-heavy: "80%"
            osl-heavy:
              max: "70%"
        runtime:
          at:
            serving:
              max: "75%"

      search_space:
        depth: {range: [0, 5]}
        embedding: [2688, 2560, 2432]
        axes_default: all
        axes:
          num_key_value_heads: null
          n_routed_experts: [128, 112, 96, 80, 64]
```

All constraints in a profile are combined with AND. Each entry under `at` is
also an independent constraint that the same architecture must satisfy.

## Constraint values

A scalar is a directional bound. It is a maximum for costs such as parameters,
memory, and runtime, and a minimum for benefits such as throughput.

```yaml
params: "75%"       # at most 75% of the teacher
params: 22.5B       # at most 22.5 billion parameters
throughput: 5000    # at least 5000 tokens/s
```

Use an object for intervals or equality:

```yaml
active_params: {min: "70%", max: "90%"}
params: {eq: 22.5B}
```

Use `values` to expand one named profile into multiple derived profiles. Lists
on multiple constraints form a Cartesian product.

```yaml
params: {values: ["70%", "75%", 22.5B]}
```

Percentages are measured against the original full-width, depth-zero teacher.
For a workload-dependent constraint, the denominator is the teacher measured
at that same named workload. Numbers are absolute values; supported unit
suffixes include `K`, `M`, `B`, and `T` for parameters, `MiB` and `GiB` for
memory, and `ms` and `s` for time.

Friendly constraint names include `params`, `active_params`, `memory`,
`runtime`, `prefill_runtime`, `throughput`, and `kv_heads`. A raw `stats.*`
metric may be used when no friendly name exists.

## Named workloads

Workloads are declared once under `mip.workloads` and referenced by name with
`at`. The statistics artifact must contain an exact matching measurement.
Missing workload names or measurements are configuration errors.

```yaml
constraints:
  memory:
    at:
      isl-heavy: "80%"
      osl-heavy: "70%"
```

The same mechanism applies to `memory`, `runtime`, `prefill_runtime`, and
`throughput`. Internally, each measurement remains distinct, for example
`stats.memory_mib@isl-heavy` and `stats.memory_mib@osl-heavy`.

## Search-space restrictions

Search-space lists restrict candidates within one solve; unlike constraint
`values`, they do not create profiles.

```yaml
search_space:
  depth: [0, 1, 4]
  embedding: [2688, 2560]
  axes_default: teacher
  axes:
    n_routed_experts: all
```

For depth, embedding, and pruning axes:

- `all` selects all measured options.
- `null` or `teacher` selects only the teacher value.
- A scalar selects one value.
- A list selects discrete values.
- `{range: [low, high]}` selects measured values in an inclusive range.
- An omitted axis inherits `axes_default`, which defaults to `all`.

Using `axes_default: teacher` and setting one axis to `all` creates a restricted
per-axis MIP experiment. Nano models without latent MoE projections simply omit
that axis.

### Typed sublayer depth selections

The existing scalar, list, and range forms for `depth` select prefixes from the
global iterative depth ranking. The following profile considers exactly two and
three total removals:

```yaml
search_space:
  depth:
    total: [2, 3]
```

`total` is an explicit spelling of the existing global-prefix behavior. It cannot
be combined with sublayer-type keys.

When depth importance uses `granularity: subblock`, a profile may instead select
counts by sublayer type:

```yaml
search_space:
  depth:
    attention: 4
    moe: [1, 2]
```

Lists on multiple type keys form a Cartesian product. This example creates two
depth scenarios: four attention removals plus one MoE removal, and four attention
removals plus two MoE removals. Omitted types contribute zero removals.

Typed selections reuse the single global iterative depth ranking; they do not run
a type-specific reranking. For each requested type, Puzzletron filters the global
ranking, takes the requested prefix, then combines the selected removals while
preserving their original global order. The scenario manifest records both the
typed counts and exact forced removals.

A typed count must be available in the collected global trajectory. Unknown types,
counts beyond the collected entries of that type, mixed `total` and typed keys, and
typed selectors with block-granularity depth importance are configuration errors.

## Homogeneous solutions

`num_homogeneous_solutions` controls a separate search in which each applicable
layer uses the same selected value for an axis:

- `0` disables homogeneous search and is the default.
- A positive integer retains the top-k feasible homogeneous solutions.
- `-1` retains every feasible homogeneous solution.

Homogeneous solutions use the same constraints and replacement-loss objective
as the heterogeneous MIP. Their ranking and output remain separate; when model
realization is enabled, every retained solution is materialized.
