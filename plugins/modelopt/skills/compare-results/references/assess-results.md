# Assess results

Start by validating both evaluation runs and confirming comparability
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

Use retry-level accuracy for both the mean and SD, and the number of valid
independent retries for `R`. Count each observation once across cached replay
and infrastructure recovery. Retain every valid planned run.

Report each mean and SE in percent for percentage metrics. Use the t-based 95%
score interval `mean ± t(0.975, R-1)*SE`, with at least two valid independent
runs. Use `SE = s / sqrt(R)` for the interval width. These intervals describe
generation/run variability on the fixed benchmark. Assess generalization to new
questions separately. For single-run tasks, use a metric-appropriate alternative
analysis.

## Confidence interval on the delta

Assess the difference using its own confidence interval. For independent
baseline and candidate retry scores, use a two-sided Welch interval:

```text
L = mean_C - mean_B                  # signed loss; negative is worse accuracy
vB = sB² / RB; vC = sC² / RC
SE_L = sqrt(vB + vC)
df = (vB + vC)² / (vB²/(RB-1) + vC²/(RC-1))
[CI_low, CI_high] = L ± t(0.975, df) * SE_L  # two-sided 95% Welch CI
```

Label the report `L = Candidate - Baseline` for accuracy: `L = -1 pp` means
the candidate is 1 percentage point worse. For lower-is-better metrics, reverse
the delta and interval signs so negative still means worse. Express
percentage-score differences in percentage points.

Calculate with a validated statistical library outside chat. For example,
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
`ttest_ind(candidate, baseline, equal_var=False).confidence_interval(0.95)`
returns the candidate-minus-baseline interval. Welch supports unequal counts and
variances; it requires at least two independent observations per side and
compatible run-level means, SDs, and counts (`RB` and `RC`). Use the t quantile
at the calculated degrees of freedom.

For missing counts/variance, jointly zero variance, or dependent observations,
mark the Welch interval unavailable and use a reviewed alternative or report
inconclusive. Establish pairing from the observation design and use a
paired-difference analysis for paired observations. Preserve clustering and
match the SD and count to the same observation unit. Flag small-sample
skew/boundary effects that make the interval unreliable.

## Assessment

Compare the unrounded signed-loss CI with the user's minimum acceptable signed
loss `M ≤ 0`, in matching units. For example, `M = -1 pp` allows a 1 pp drop:

- **Within margin:** `M < CI_low`. The entire CI meets the target.
- **Substantial regression:** `CI_high < M`. The entire CI exceeds the allowed drop.
- **Possible regression:** `CI_low ≤ M ≤ CI_high < 0`. Worse than baseline,
  but unresolved against the allowed drop.
- **Inconclusive:** `CI_low ≤ M` and `0 ≤ CI_high`. Both an unacceptable drop
  and no degradation remain plausible.

The last two cases, including equality at `M`, leave the target unresolved.
Separately, `CI_high < 0` indicates a significant degradation at the stated CI
level. Assess target compliance separately from significance against zero.
With an unspecified target, report feasibility as inconclusive. Use judgment
for borderline results, explain it, and retain the statistical status alongside
the decision. Support confidence-based acceptance with reliable uncertainty.

Choose sample counts, analysis checkpoints, and stopping rules before running.
Propose bounded extra runs when uncertainty prevents a decision. Label results
selected across many candidates/tasks or repeated looks as exploratory; use
independent confirmation or an appropriate sequential/multiple-comparison
method for confirmatory claims.

Report results using the [report format](../SKILL.md#report-format).
