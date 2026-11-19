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
runs. If insufficient samples are available, report uncertainty unavailable.

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
the candidate is 1 percentage point worse.

Calculate with a validated statistical library. For example,
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
`ttest_ind(candidate, baseline, equal_var=False).confidence_interval(0.95)`
returns the candidate-minus-baseline interval.

Report uncertainty unavailable in degenerate cases.

## Assessment

Compare the signed-loss CI with the user's minimum acceptable signed
loss `M ≤ 0`, in matching units. For example, `M = -1 pp` allows a 1 pp drop:

- **Within margin:** `M < CI_low`. The entire CI meets the target.
- **Substantial regression:** `CI_high < M`. The entire CI exceeds the allowed drop.
- **Possible regression:** `CI_low ≤ M ≤ CI_high < 0`. Worse than baseline,
  but unresolved against the allowed drop.
- **Inconclusive:** `CI_low ≤ M` and `0 ≤ CI_high`. Both an unacceptable drop
  and no degradation remain plausible.

The last two cases, including equality at `M`, leave the target unresolved.
Use judgment for borderline results, explain it, and retain the statistical
status alongside the decision. Support confidence-based acceptance with
reliable uncertainty.
