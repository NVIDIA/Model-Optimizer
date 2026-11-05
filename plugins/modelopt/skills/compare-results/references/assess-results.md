# Assess results

Apply only after validating both evaluation runs and confirming comparability
using the [compare-results checklist](../SKILL.md#comparability-checklist).
Use the task recipe's canonical metric and minimum run count.

## Retry mean, standard deviation, and SE

For each side, let `accuracy_r` be the full benchmark accuracy from fresh,
independent retry/run `r`, with `R` valid runs:

```text
mean = avg(accuracy_r)
s = sample_stddev(accuracy_r)         # divisor R - 1
SE = s / sqrt(R)
```

Use retry-level accuracy for both the mean and SD—not pooled question outcomes.
The SE denominator is the number of valid independent retries, not questions or
questions × retries. Cached replay and infrastructure recovery of the same
observation do not add independent retries. Retain every valid planned run.

Report each mean and SE in percent for percentage metrics. Use the t-based 95%
score interval `mean ± t(0.975, R-1)*SE`, with at least two valid independent
runs. Never substitute SD for SE. These intervals describe generation/run
variability on the fixed benchmark, not uncertainty about generalization to new
questions. Single-run tasks require a metric-appropriate alternative analysis;
do not invent retry variance.

## Confidence interval on the delta

Report an interval on the difference itself—not a test based on overlapping
individual score intervals. For independent baseline and candidate retry scores,
use a two-sided Welch interval:

```text
D = mean_B - mean_C                  # higher-is-better accuracy; positive is worse
vB = sB² / RB; vC = sC² / RC
SE_D = sqrt(vB + vC)
df = (vB + vC)² / (vB²/(RB-1) + vC²/(RC-1))
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
compatible run-level means, SDs, and counts (`RB` and `RC`). Use the t quantile
at the calculated degrees of freedom, not a fixed multiplier.

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

Report results using the [report format](../SKILL.md#report-format).
