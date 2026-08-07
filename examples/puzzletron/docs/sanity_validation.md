# Sorting, width ranking, and slicing sanity

Puzzletron uses separate sanity checks to answer two different questions:

1. Did sorting and slicing preserve the intended model operation?
2. Did the importance ranking produce a better reduced candidate than its
   controls?

The first question is about implementation correctness. The second is about
ranking quality. Keeping them separate helps diagnose a failed campaign and
lets an acceptance plan decide how strict ranking quality must be.

## Sorting and slicing mental model

Width pruning makes selected model dimensions smaller, such as hidden channels,
attention heads, expert widths, or state-space dimensions. Puzzletron first
scores the relative importance of the units in each supported dimension. It
then permutes every coupled tensor into the same importance order so that a
prefix of the sorted dimension represents a reduced-width candidate. At full
width, sorting is only a permutation and must not change model behavior.

A **dynamic slice** runs a reduced candidate from the resident full-size sorted
teacher. Runtime hooks or views select only the requested prefix without first
writing a smaller checkpoint. A **physical slice**, also called physical
materialization, actually removes the excluded tensor rows, columns, heads, or
groups and updates the model config. The physical model is the exportable
ground truth for the dynamic path.

For example, reducing hidden width from 4096 to 3072 does not mean truncating
one weight matrix. Puzzletron must select the same 3072 channel identities
across every coupled embedding, residual, normalization, and projection
dimension. Sorting first moves the chosen identities into a common prefix;
slicing then applies that prefix consistently throughout the model.

## Compared views

The width sanity stages compare four views of one target width:

| View | Meaning |
|---|---|
| Activation-sorted | Keep the prefix that the importance scores rank highest. This is the candidate whose ranking quality is being tested. |
| Original or unsorted | Keep the same-sized prefix in the teacher's original order. Some older paths call this the random control, although the original prefix is deterministic. |
| Reverse-sorted | Keep the prefix ranked least important. This is a negative control for the importance ordering. |
| Physical | Rewrite the config and tensors as a smaller standalone model using the same target geometry as the activation-sorted candidate. |

## Measured values

Each comparison uses the same axis, layer, target width, model revision, and
input samples. Depending on the diagnostic configuration, the measured values
can include:

- replacement loss or language-model loss;
- hidden-state cosine distance, mean squared error, or mean absolute error;
- output-distribution KL divergence; and
- token top-k accuracy or consistency.

Loss, distance, error, and divergence metrics are lower-is-better. Accuracy and
consistency metrics are higher-is-better. The artifact records the metric,
direction, compared methods, observed degradation or difference, and allowed
tolerance so a finding can be traced to measured evidence.

## Width-ranking misses

For the same axis, layer, target width, samples, and metric, width sanity asks
whether the activation-sorted candidate is at least as good as the original
and reverse controls within the configured comparison tolerance. A ranking
miss occurs when the activation-sorted value is worse than a control by more
than that tolerance.

For example, if lower loss is better and the activation-sorted loss is `2.10`
while the original control is `2.00` with tolerance `0.01`, the degradation is
`0.10` and the check reports a ranking miss. This means the importance ordering
did not demonstrate the expected quality benefit for that case. It does not by
itself prove that sorting or slicing produced an invalid model.

## Equivalence failures

Equivalence checks compare two routes that are intended to represent the same
model operation:

- Sort sanity compares the full-width teacher with full-width sorted and
  reverse-sorted teachers. These permutations must preserve behavior within
  descriptor-owned tolerances.
- Slicing sanity compares a dynamic slice with a physically materialized model
  for the same axis, layers, target value, batch, and model revision. It checks
  loss, output shape, and output tensor differences using explicit absolute and
  relative tolerances.

An equivalence failure means those supposedly identical routes disagree beyond
their declared tolerance. Downstream measurements from the dynamic candidate
can no longer be assumed to describe the exportable physical checkpoint, so
the corresponding sanity stage fails unconditionally.

## Stage completion and qualification

Puzzletron currently records correctness failures and ranking-quality findings
through the same sanity-stage result, but they have different default effects:

| Result | Current stage behavior |
|---|---|
| Sort or slicing equivalence failure | Always fails the stage and is reported as a correctness error. |
| Width-ranking miss | Records a quality warning. By default the stage may complete successfully with `passed: false`. |
| Width-ranking miss with strict warning policy | Fails the stage when `sanity.fail_on_warnings` is enabled. |

Enable strict warning handling with:

```yaml
sanity:
  fail_on_warnings: true
```

This setting changes stage-completion policy; it does not reclassify a ranking
miss as an implementation-correctness error.

Campaign or release qualification is a separate policy decision. The current
code does not emit a distinct `qualification_blocked` state. Before treating a
campaign as accepted, define which ranking controls must pass, the metrics and
directions, sample counts, tolerances, covered axes and target widths, and the
aggregation rule. A stage-level quality warning may therefore remain
non-correctness evidence while still blocking scientific or release
acceptance.

For the broader pipeline design, see the
[semantic validation gates](v2_architecture.md#semantic-validation-gates).
