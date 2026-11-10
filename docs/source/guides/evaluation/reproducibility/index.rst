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
- **P-values for cap-hit, missing-final-answer, abstention, and incorrect-answer rates:** use an
  exact paired permutation test. For each question, swap the two models' outcomes or leave them
  unchanged, keeping all repetitions together. The one-sided p-value is the fraction of possible
  swaps giving a difference at least as extreme in the tested direction, including ties.

The question-level tests treat questions and SciCode subtasks as independent. Related subtasks from
the same parent problem may need to be analyzed together.

Each test uses p < 0.05. We do not make that threshold stricter because other metrics were also
tested. Reverse-direction p-values provide context only.

Reproducibility limits
----------------------

The exact evaluated QAD and KTO checkpoints are not publicly available. To keep the ModelOpt
repository lightweight, we include summary results rather than question-level data, generated
responses, or a full experiment environment, consistent with the `Qwen3.6 PTQ/QAD
tutorial <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md>`__.
