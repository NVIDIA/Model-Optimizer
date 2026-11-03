# Assess and report

Apply only after the parent skill's run-validation and comparability checks.
Use the task recipe's canonical metric and minimum run count.

## Retry mean, standard deviation, and SE

For each side, let `accuracy_r` be the full benchmark accuracy from fresh,
independent retry/run `r`, with `n` valid runs:

```text
mean = avg(accuracy_r)
s = sample_stddev(accuracy_r)         # divisor n - 1
SE = s / sqrt(n)
```

Use retry-level accuracy for both the mean and SD—not pooled question outcomes.
The SE denominator is the number of valid independent retries, not questions or
questions × retries. Cached replay and infrastructure recovery of the same
observation do not add independent retries. Retain every valid planned run.

Report each mean and SE in percent for percentage metrics. An approximate 95%
score interval is `mean ± 2*SE`; label it approximate. For small independent-run
samples, prefer `mean ± t(0.975, n-1)*SE`. Never substitute SD for SE. These
intervals describe generation/run variability on the fixed benchmark, not
uncertainty about generalization to new questions. Single-run tasks require a
metric-appropriate alternative analysis; do not invent retry variance.

## Confidence interval on the delta

Report an interval on the difference itself—not a test based on overlapping
individual score intervals. For independent baseline and candidate retry scores,
use a two-sided Welch interval:

```text
D = mean_B - mean_C                  # higher-is-better accuracy; positive is worse
vB = sB² / nB; vC = sC² / nC
SE_D = sqrt(vB + vC)
df = (vB + vC)² / (vB²/(nB-1) + vC²/(nC-1))
[L, U] = D ± t(0.975, df) * SE_D     # two-sided 95% Welch CI
```

Label the report `Baseline - Candidate`. For lower-is-better metrics, reverse
the delta and interval signs so positive still means degradation. Express
percentage-score differences in percentage points, not relative percent.

Calculate with a validated statistical library outside chat. For example,
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
`ttest_ind(baseline, candidate, equal_var=False).confidence_interval(0.95)`
returns the baseline-minus-candidate interval. Welch supports unequal counts and
variances; it requires at least two independent observations per side and
compatible run-level means, SDs, and counts. Do not use a fixed 2*SE multiplier
as an exact Welch interval at small degrees of freedom.

Missing counts/variance, jointly zero variance, or dependent observations do not
establish zero uncertainty. Mark the Welch interval unavailable and use a
reviewed alternative or report inconclusive. For genuinely paired observations,
use a paired-difference analysis; matched benchmark settings alone do not
establish statistical pairing. Preserve clustering; do not mix run-level SD
with question-level counts. Flag small-sample skew/boundary effects that make
the interval unreliable.

## Assessment

Compare the unrounded degradation CI with the user's nonnegative maximum
allowed loss `M`, in matching units:

- `U < M`: evidence the loss is within the target.
- `L > M`: evidence the loss exceeds the target.
- Otherwise, including equality: target unresolved.

Separately, `L > 0` indicates a significant degradation at the stated CI level.
A significant drop can still meet the target; failure to detect a drop does not
prove acceptability. Without an explicit target, feasibility is inconclusive.
Use judgment for borderline results and explain it without relabeling an
unresolved statistical result as established. Missing or unreliable uncertainty
cannot support a confidence-based acceptance claim.

Choose sample counts and analysis checkpoints before running; do not sample
until a favorable result appears. Propose bounded extra runs when uncertainty
prevents a decision. Results selected across many candidates/tasks or repeated
looks are exploratory unless confirmed independently or covered by an
appropriate sequential/multiple-comparison method.

## Report Format

Include:

- Baseline and candidate identifiers.
- Per-task metric path, baseline and candidate scores with SEs, degradation,
  confidence intervals and method, sample/run counts, and an Assessment column.
  Mark unavailable uncertainty explicitly; keep the assessment concise, not a
  statistics tutorial.
- Per-task external reference score, source URL, known protocol differences,
  percentage-point difference, and sanity status (`verified`, `failed`, or
  `externally unverified`).
- Comparability status for prompt/template, generation settings, sample counts,
  reasoning handling, judge/simulator setup, and score field.
- Per-task timeout and output-limit counts/rates for both runs, explicit
  denominators, telemetry coverage, effective limits, and unresolved effects on
  the score. Unknown accounting or unresolved infrastructure effects prevent an
  `acceptable` quantization-feasibility verdict.
- Comparability verdict: comparable, not comparable, or inconclusive.
- Quantization feasibility verdict: acceptable, not acceptable, or inconclusive.
  Never report `acceptable` when external baseline sanity failed. An externally
  unverified baseline does not block `acceptable`; apply the candidate-delta
  gate and report the missing external corroboration.
