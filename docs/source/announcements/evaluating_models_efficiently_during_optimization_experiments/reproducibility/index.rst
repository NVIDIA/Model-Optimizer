Reproducibility notes
=====================

We use evaluation data from the `KTO tutorial <https://github.com/NVIDIA/Model-Optimizer/blob/13ad0ef401ec5e4480abd1b8eb236ccdf50ef802/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/kto/README.md#experiments-and-results>`__
for the examples in the :doc:`evaluation article <../index>`.
See its
`reproducibility instructions <https://github.com/NVIDIA/Model-Optimizer/blob/13ad0ef401ec5e4480abd1b8eb236ccdf50ef802/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/kto/reproducibility/README.md>`__
for the checkpoints, subtask split, and evaluation settings.

Saved metrics
-------------

We save the means, observed standard deviations, confidence intervals, and p-values as CSV
summaries:

- **SciCode: candidate accuracy at fixed response budgets** — data used in the
  :ref:`Choose between question-level and repetition-level analysis <evaluation-choose-questions-and-repetitions>`
  example.
  CSV file: :download:`budget-allocation-accuracy.csv <budget-allocation-accuracy.csv>`.
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
  For the one-sided `paired permutation p-values
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`__,
  swap model labels for whole parent problems and count permuted mean differences at least as
  extreme as the observed difference in the tested direction.

  When question-level differences are independent and a t approximation is suitable, a paired
  t-based confidence interval and a one-sided paired t-test are alternatives.

- **Non-inferiority:** test whether Candidate's accuracy loss is smaller than an acceptable margin.
  For differences in pp, the t statistic is ``(mean_difference + margin) / standard_error``, using
  the upper tail of the t distribution. For SciCode, estimate the standard error across parent
  problems with degrees of freedom equal to the number of parent problems minus one.

- **Cap-hit rates:** for each model, divide the number of responses that hit the completion cap
  by the total number of responses across all subtasks and repetitions.

  Calculate a two-sided 95% paired-bootstrap confidence interval for the rate difference by
  sampling whole SciCode parent problems with replacement. Keep all repetitions and both models
  together within each sampled group. The interval endpoints are the 2.5th and 97.5th percentiles.
  If neither model has an observed failure, bootstrapping gives a [0, 0] interval. This does not
  establish that failures cannot occur or that the models have equal failure rates.

  .. _evaluation-scicode-cap-hit-p-values:

  **SciCode cap-hit p-values**

  For the one-sided `paired permutation p-values
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`__,
  swap model labels for whole parent problems and count permuted cap-hit rate differences at least
  as extreme as the observed difference in the tested direction. For an exact p-value, divide
  this count by the total number of possible swap patterns.

  Cap hits are rare, so most parent-level differences are zero. The permutation test avoids relying
  on a t-distribution approximation for these sparse differences. It assumes that the models' results
  for each parent problem are interchangeable under the null hypothesis.

  The outcomes are binary, but a simple binomial test on pooled responses would ignore question
  pairing and repetitions within each question. With one response per model and question and
  independent question pairs, an
  `exact McNemar test
  <https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html>`__
  is another paired option.

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
SDs are in percentage points. The means stay
fixed because every allocation partitions the same saved response pool. Groups can share
question or repetition IDs, so their observed spread is descriptive.

Reproducibility limits
----------------------

The checkpoints and paired measurements are stored outside the public ModelOpt repository.
To keep the repository lightweight, we include summary results here rather than question-level
data, generated responses, or a full experiment environment.
