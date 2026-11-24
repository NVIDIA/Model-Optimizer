Reproducibility notes
=====================

We use evaluation data from the `KTO tutorial <https://github.com/NVIDIA/Model-Optimizer/blob/13ad0ef401ec5e4480abd1b8eb236ccdf50ef802/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/kto/README.md#experiments-and-results>`__
for the comparisons and budget analyses in the :doc:`evaluation article <../index>`.
See its
`reproducibility instructions <https://github.com/NVIDIA/Model-Optimizer/blob/13ad0ef401ec5e4480abd1b8eb236ccdf50ef802/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/kto/reproducibility/README.md>`__
for the checkpoints, subtask split, and evaluation settings.

Saved metrics
-------------

We save the means, observed standard deviations, confidence intervals, and p-values as CSV
summaries:

- **SciCode: Candidate accuracy at fixed response budgets** — data used in the
  :ref:`Choose between question-level and repetition-level analysis <evaluation-choose-questions-and-repetitions>`
  example.
  CSV file: :download:`budget-allocation-accuracy.csv <budget-allocation-accuracy.csv>`.
- **SciCode: accuracy differences at fixed response budgets** — data used in the
  same comparison.
  CSV file: :download:`budget-allocation-differences.csv <budget-allocation-differences.csv>`.
- **SciCode: cap-hit rates** — data used in the
  :ref:`Collect as much evaluation data as you need <evaluation-collect-data>` example.
  CSV file: :download:`scicode-cap-data-volume.csv <scicode-cap-data-volume.csv>`.

.. _evaluation-statistical-methods:

Statistical methods
-------------------

- **SciCode accuracy:** to estimate the mean accuracy difference, first average each model’s
  correctness across repetitions within each subtask, then compute **Candidate − Baseline** for
  that subtask. Average those differences across subtasks.
  For the confidence interval and p-values, group all scored subtasks and their repetitions by
  parent problem. For the 95% interval, resample paired parent problems with replacement and
  take the 2.5th and 97.5th percentiles of the resulting mean differences.
  For the one-sided p-values, swap model labels for whole parent problems and count permuted mean
  differences at least as extreme as the observed difference in the tested direction.

  When question-level differences are independent and a t approximation is suitable, a paired
  t-based confidence interval and a one-sided paired t-test are alternatives.

- **Non-inferiority:** test whether Candidate's accuracy loss is smaller than an acceptable margin.
  For differences in pp, the t statistic is ``(mean_difference + margin) / standard_error``, using
  the upper tail of the t distribution. For SciCode, estimate the standard error across parent
  problems with degrees of freedom equal to the number of parent problems minus one.

- **Cap-hit rates:** calculate two-sided 95% paired-bootstrap
  confidence intervals by sampling with replacement, using whole parent problems for SciCode
  cap hits. Keep all repetitions and both models together within each sampled group. The
  interval endpoints are the 2.5th and 97.5th percentiles.
  If neither model has an observed failure, bootstrapping gives a [0, 0] interval. This does not
  establish that failures cannot occur or that the models have equal failure rates.

  For one-sided p-values, use a :ref:`paired permutation test <evaluation-pvalue-methods>`.
  Even when failures are rare, swapping the models' results for each question or parent problem
  shows how unusual the observed difference would be if the models behaved the same.
  Computing p-values with the bootstrap would be more complicated.

- **Verbosity:** to compare mean completion lengths, average each model’s response lengths across
  repetitions within each question, then subtract Baseline’s average from Candidate’s. Use these
  paired differences to calculate a two-sided t-based 95% confidence interval and run one-sided
  paired t-tests for the stated direction. Degrees of freedom equal the number of questions minus
  one.
  Compute each model's median completion length across all its responses. The verbosity tests
  compare mean lengths.

The SciCode accuracy and cap-hit tables use 50,000 bootstrap resamples and 50,000 random swap
patterns. The p-value is ``(extreme swaps + 1) / (total swaps + 1)``, counting swaps that give a
difference at least as extreme as the observed one in the tested direction. The SciCode cap-hit
plot uses the same bootstrap and exact permutation p-values.

.. _evaluation-fixed-response-budgets:

Observed variation at fixed response budgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a budget of B responses per model and R repetitions, each score averages B/R questions with
R repetitions each. We split the saved question order into consecutive groups of B/R questions
and the repetition order into blocks of R. Each response appears once per allocation; both models
use matching groups. This gives eight group scores for each allocation, without random resampling.
Each score uses 280 responses per model.

We shuffle the sorted IDs of the 294 eligible SciCode subtasks with seed ``20261124``. The
fixed-budget analyses use the first 280 subtasks in that order to allow exact grouping
across 1, 2, 4, and 8 repetitions. The SciCode cap-hit plot uses prefixes of the same
order. The dotted reference in the SciCode accuracy plot uses all 294 subtasks with
eight repetitions.

We report the mean and sample SD across the group scores. Accuracy means are percentages.
Candidate minus Baseline differences and all SDs are in percentage points. The means stay
fixed because every allocation partitions the same saved response pool. Groups can share
question or repetition IDs, so their observed spread is descriptive.

.. _evaluation-pvalue-methods:

Question-level and repetition-level p-values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the SciCode benchmark, a response scores 1 if it hits the completion cap and 0 otherwise.
Both methods test whether Candidate has a lower cap-hit rate, using identical responses from
294 subtasks at each repetition count.

**Question-level:** average each model's cap-hit outcomes across repetitions within each subtask,
then subtract Baseline's average from Candidate's. Sum these paired differences within each parent
problem. To calculate the exact `paired permutation p-value
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`__ for
the SciCode cap-hit plot, swap the models' results for each whole parent problem or leave them
unchanged, keeping all subtasks and repetitions together. The p-value is the fraction of possible
swaps giving a mean difference across subtasks equal to or lower than the observed difference,
including ties.

Cap hits are rare, so most parent-level differences are zero. The permutation test avoids relying
on a t-distribution approximation for these sparse differences. It assumes that the models' results
for each parent problem are interchangeable under the null hypothesis.

The outcomes are binary, but a simple binomial test on pooled responses would ignore question
pairing and repetitions within each question. With one response per model and question and
independent question pairs, an
`exact McNemar test
<https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html>`__
is another paired option.

For a cap-hit comparison, take the tail in the stated direction.

**Repetition-level:** for each model and repetition, divide the number of cap hits by 294. Apply a
`one-sided Welch t-test
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`__
to the models' per-repetition average scores (cap-hit rates), testing whether Candidate's mean
is lower than Baseline's.
The calculation uses each model's sample variance across repetitions and Welch's degrees of
freedom. It treats runs as independent and allows different variances for the two models.
At least two repetitions per model are needed to estimate those variances; with very few
repetitions, the estimates may be unstable.

Reproducibility limits
----------------------

The checkpoints and paired measurements are stored outside the public ModelOpt repository.
To keep the repository lightweight, we include summary results here rather than question-level
data, generated responses, or a full experiment environment.
