# Puzzletron Wizard Candidate Counts

## Goal

After pruning axes and values are selected, show exact candidate counts in the
wizard's vLLM-runtime and replace-one-scoring granularity choices. The counts
must remain configuration-only so the setup environment does not require
PyTorch, converted checkpoints, or model weights.

## Counting Semantics

The two stages count different things:

- vLLM reports unique active configurations after deduplicating identical
  configurations across layers. The wizard displays this candidate count, not
  the paired `N`/`2N` benchmark-launch count.
- Replace-one scoring evaluates every non-teacher candidate at every applicable
  layer. The wizard displays solutions per embedding width and the total across
  all selected widths.

Teacher configurations are included in vLLM counts because they are measured.
They are excluded from replace-one counts because replacing a teacher with
itself is not a scored solution.

## Model Representation

Add a small, dependency-free counting representation to
`puzzletron_setup.profiles`:

- A subblock family names the axes that combine inside one subblock:
  `attention`, `gdn`, `ffn`, `moe`, or `mamba`.
- A block family is an ordered set of active subblock families.
- The inspected model resolves each physical layer to one block family.

Supported layouts are:

- Dense Qwen 3.5/3.6:
  `full_attention = attention + ffn`,
  `linear_attention = gdn + ffn`.
- MoE Qwen 3.5/3.6:
  `full_attention = attention + moe`,
  `linear_attention = gdn + moe`.
- Nemotron 3:
  `* = attention`, `M = mamba`, `E = moe`, `- = ffn`, following the exact
  hybrid pattern.

The subblock-family domain size is the Cartesian product of its enabled,
selected axis values. An absent or disabled axis is fixed at the teacher and
contributes a factor of one. Hidden width is a scenario dimension, not a
subblock-family axis.

## Formulas

For each distinct subblock family `s`, let `D(s)` be its domain size. For each
block family `b`, let `S(b)` be its active subblocks and `L(b)` its layer count.

- vLLM subblock candidates:
  `sum(D(s))` over distinct active subblock families.
- vLLM block candidates:
  `sum(product(D(s) for s in S(b)))` over distinct block families.
- Replace-one subblock solutions per width:
  `sum(L(b) * sum(D(s) - 1 for s in S(b)))`.
- Replace-one block solutions per width:
  `sum(L(b) * (product(D(s) for s in S(b)) - 1))`.
- Replace-one total:
  `per_width * number_of_selected_hidden_widths`.

Values are deduplicated before counting. Invalid or missing layer topology is a
setup error rather than a reason to print an approximate number.

## Wizard Presentation

The existing select prompts retain their values and defaults. Only their choice
titles and descriptions change.

Example:

```text
vLLM measurement granularity:
  Sublayer — 14 unique configurations
  Whole block — 24 unique configurations

Replace-one scoring granularity:
  Subblock — 168 solutions/width, 336 total across 2 widths
  Whole block — 312 solutions/width, 624 total across 2 widths
```

If only one width is selected, the title shows the single total without a
redundant multiplication phrase.

## Validation

Dependency-light unit tests will cover:

- the dense hybrid Qwen example above;
- MoE Qwen, proving the shared MoE domain is deduplicated for vLLM but repeated
  per layer for scoring;
- Nemotron's mutually exclusive hybrid pattern;
- choice titles in both normal and detailed wizard modes;
- teacher-only and duplicate selections.

The generated YAML values remain unchanged; this feature only improves the
information shown before the user selects a granularity.
