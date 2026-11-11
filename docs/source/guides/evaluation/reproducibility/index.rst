Reproducibility notes
=====================

These notes describe how the checkpoints were prepared and evaluated for the
:doc:`evaluation guidelines <../evaluation_guidelines>`.

Baseline and Candidate checkpoints
----------------------------------

Baseline is NVFP4 Qwen3.6-35B-A3B after PTQ and QAD. The `PTQ/QAD
tutorial <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md>`__
describes that recipe.

Candidate was trained directly from Baseline using KTO in NeMo RL, with an unchanged copy of
Baseline as the reference model. We identified 55 SciCode subtasks on which Baseline had shown
reasoning loops: 44 supplied training examples and 11 were reserved for evaluation. Training used
model-generated responses: 104 correct and 104 containing reasoning loops. After 13 optimizer steps
(KTO beta 0.05, learning rate 1e-6) with simulated NVFP4 quantization, Candidate was exported to
NVFP4.

The evaluated checkpoints are available in the `checkpoint collection
<https://modelopt.nvidia.com/hub/?collection=20eb6348-d437-4842-a67a-73d325238a27>`__
(NVIDIA internal access required).

Evaluation setup
----------------

Baseline and Candidate answered the same benchmark questions with matched generation seeds.

We evaluate on 294 of 338 SciCode subtasks, excluding the 44 used for KTO recovery training. KTO
used model-generated responses, not reference answers.

+------------------------------+------------------------------+-------------+---------------------+
| Benchmark                    | Questions or subtasks        | Repetitions | Responses per model |
+==============================+==============================+=============+=====================+
| MMLU-Pro                     | 1,000 questions sampled      | 20          | 20,000              |
|                              | across all 14 subjects       |             |                     |
+------------------------------+------------------------------+-------------+---------------------+
| SciCode                      | 294 subtasks                 | 20          | 5,880               |
+------------------------------+------------------------------+-------------+---------------------+
| SciCode focused subset       | 11 challenging, loop-prone   | 10          | 110                 |
|                              | held-out subtasks            |             |                     |
+------------------------------+------------------------------+-------------+---------------------+
| AA-Omniscience public v1     | 600 questions                | 10          | 6,000               |
+------------------------------+------------------------------+-------------+---------------------+

For benchmark evaluation, both models generated responses with temperature ``0.6``, top-p ``0.95``,
a ``131072``-token completion cap, and a ``262144``-token context window.

For AA-Omniscience, the Codex assistant graded each response using the reference answer, without
being told whether Baseline or Candidate produced it. The reported grading model was GPT-6.1 Sol
with xhigh reasoning effort. Deliberate abstention means choosing not to answer; empty answers and
uncertain guesses do not count.

Saved metrics
-------------

We save the accuracy differences, cap-hit rates, confidence intervals, and p-values shown in the
guide’s four plots as CSV summaries:

- **SciCode: question-level versus repetition-level analysis** — data used in the guide section
  :ref:`Prioritize question coverage over repetitions <evaluation-prioritize-question-coverage>`. CSV file: :download:`scicode-cap-analysis-unit.csv <scicode-cap-analysis-unit.csv>`.
- **MMLU-Pro: fixed response budget** — data used in the guide section :ref:`Prioritize question coverage over repetitions <evaluation-prioritize-question-coverage>`. CSV
  file: :download:`allocation-accuracy-gap.csv <allocation-accuracy-gap.csv>`.
- **SciCode: cap-hit rates** — data used in the guide section :ref:`Collect as much evaluation data as you need <evaluation-collect-data>`. CSV file:
  :download:`scicode-cap-data-volume.csv <scicode-cap-data-volume.csv>`.
- **MMLU-Pro: accuracy regression** — data used in the guide section :ref:`Collect as much evaluation data as you need <evaluation-collect-data>`. CSV
  file: :download:`mmlu-accuracy-data-volume.csv <mmlu-accuracy-data-volume.csv>`.

Statistical methods
-------------------

For accuracy, cap-hit and missing-final-answer rates, and mean completion length, we calculate the
difference by subtracting Baseline's value from Candidate's. Accuracy and rate changes are in
percentage points; length differences are in tokens, including reasoning tokens.

- **Accuracy and mean completion length:** average each model's repetitions within each question,
  then subtract Baseline's average from Candidate's for that question. Across these paired question
  differences, use a two-sided t-based 95% confidence interval and one-sided paired t-tests for the
  stated direction. Degrees of freedom equal the number of questions minus one. Median lengths
  describe all individual responses; the length tests concern means.
- **Non-inferiority:** test whether Candidate's accuracy loss is smaller than an acceptable margin.
  For differences in pp, the t statistic is ``(mean_difference + margin) / standard_error``, using
  the upper tail and the same degrees of freedom. The guide examines margins of 2, 1, 0.5, and 0.1
  pp.
- **Failure, abstention, and incorrect-answer rates:** calculate paired-bootstrap 95% confidence
  intervals by sampling questions with replacement, keeping each question's repetitions and both
  models together. MMLU-Pro and SciCode use 100,000 resamples; AA uses 50,000, preserving 100
  questions per domain. The interval endpoints are the 2.5th and 97.5th percentiles.

The question-level tests treat questions and SciCode subtasks as independent. Related subtasks from
the same parent problem may need to be analyzed together.

Each test uses p < 0.05. We do not make that threshold stricter because other metrics were also
tested. Reverse-direction p-values provide context only.

.. _evaluation-pvalue-methods:

Question-level and repetition-level p-values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the SciCode benchmark, a response scores 1 if it hits the completion cap and 0 otherwise.
Both methods test whether Candidate has a lower cap-hit rate, using identical responses from
294 subtasks at each repetition count.

**Question-level:** average each model's cap-hit outcomes across repetitions within each subtask,
then subtract Baseline's average from Candidate's. This gives 294 paired differences. To calculate
the exact `paired permutation p-value
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`__,
swap the models' results within each subtask or leave them unchanged, keeping all repetitions
together. The p-value is the fraction of possible swaps giving a mean difference equal to or lower
than the observed difference, including ties.

Cap hits are rare, so most subtask differences are zero. The permutation test avoids relying on a
t-distribution approximation for these sparse differences. It assumes that the models' results
within each subtask are interchangeable under the null hypothesis.

The outcomes are binary, but a simple binomial test on pooled responses would ignore question
pairing and repetitions within each question. With one response per model and question, an
`exact McNemar test
<https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html>`__
is another paired option.

We use the same paired permutation method for the guide's cap-hit, missing-final-answer, abstention,
and incorrect-answer p-values, taking the tail in the stated direction.

**Repetition-level:** for each model and repetition, divide the number of cap hits by 294. Apply a
`one-sided Welch t-test
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`__
to the two lists of repetition rates, testing whether Candidate's mean is lower than Baseline's.
The calculation uses each model's sample variance across repetitions and Welch's degrees of
freedom. It treats runs as independent and allows different variances for the two models.
At least two repetitions per model are needed to estimate those variances; with very few
repetitions, the estimates may be unstable.

Reproducibility limits
----------------------

The exact evaluated QAD and KTO checkpoints are not publicly available. To keep the ModelOpt
repository lightweight, we include summary results rather than question-level data, generated
responses, or a full experiment environment, consistent with the `Qwen3.6 PTQ/QAD
tutorial <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md>`__.
