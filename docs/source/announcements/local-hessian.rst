:orphan:

Local Hessian: Better NVFP4 Weight Scales from Layer Inputs
############################################################

:Author: Model Optimizer Team
:Date: September 9, 2026
:Tags: local-hessian, quantization, nvfp4, calibration, modelopt

NVFP4 represents each group of 16 weights with FP4 values and an FP8 block
scale [1]_. Choosing that scale is consequential: a large scale preserves a
block's outlier but spaces the representable values farther apart, while a
smaller scale resolves typical values better but clips the outlier. Max scaling
always chooses the first tradeoff. Weight mean squared error (MSE) improves it
by measuring the weights, but still treats every weight error as equally
important.

**Local Hessian**, also called **FP4 minimum-output-squared-error (MOSE)
scaling**, chooses the scale that best preserves what a linear layer computes.
It uses calibration inputs to score each candidate scale by its contribution to
layer-output error. The method does not update the full-precision weights: the
NVFP4 format, kernels, export path, and deployment runtime do not change.

The algorithm, without the algebra
**********************************

For each linear layer, Local Hessian does the following:

#. Run representative calibration data through the model and collect the input
   statistics for every 16-value input-channel block.
#. For each weight block and output channel, try the 126 positive, finite FP8
   E4M3 block-scale candidates supported by NVFP4.
#. Quantize and dequantize the block with each candidate.
#. Estimate how much that block's weight error changes the layer output on the
   calibration inputs.
#. Keep the candidate with the smallest estimated output error.

Searching every block scale jointly would be combinatorial. Local Hessian makes
the practical approximation that each block can be optimized independently.
This turns the search into many small problems that can run in parallel. The
full derivation, including what the approximation drops, is in the `appendix`_.

Why layer inputs change the answer
==================================

Consider two weight errors of the same size. If one multiplies an input feature
that is often large, it can perturb the layer output much more than the other.
Weight MSE assigns them the same cost; Local Hessian does not. For an input
block :math:`X_k` and weight error :math:`\Delta w`, it scores

.. math::

   \mathcal{J}_k(s) = \Delta w(s)^{\top} H_k \Delta w(s),
   \qquad
   H_k = \frac{1}{N}X_k^{\top}X_k.

The :math:`16\times16` matrix :math:`H_k` captures both the energy of each
input coordinate and correlations within the block. The scale :math:`s` affects
the score through :math:`\Delta w(s)=\mathcal{Q}_s(w)-w`. This is the same
quadratic form as the isolated block's output-reconstruction error. It is
sometimes called a local Hessian because :math:`X_k^{\top}X_k` is proportional
to the Hessian of squared reconstruction loss for that linear layer.

Using Local Hessian in Model Optimizer
**************************************

Use the Local Hessian NVFP4 configuration and provide a forward loop over
representative inputs:

.. code-block:: python

   import modelopt.torch.quantization as mtq

   model = mtq.quantize(
       model,
       mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
       forward_loop,
   )

The default Local Hessian block size is 16, matching NVFP4, and the calibration
forward loop is required. The implementation first max-calibrates the model,
collects per-block input second moments with weight fake quantization disabled,
and then runs a fused scale sweep where supported. Unsupported weight layouts
retain max-calibrated scales or fall back to weight MSE, with a warning where
applicable. At present, the per-block input statistics are not synchronized
across distributed ranks, so use the algorithm as a single-rank calibration
method.

Result 1: Scale selection matters on Qwen3.5-9B
************************************************

Table 1 isolates weight-scale selection on Qwen3.5-9B under NVFP4 W4A4: FP4
weights and FP4 activations. MMLU, HellaSwag, and WinoGrande are zero-shot;
GSM8K is five-shot with strict-match accuracy; WikiText reports word
perplexity. Accuracy values are percentage points. "Average drop" is the mean
of :math:`\max(\text{BF16 score}-\text{quantized score}, 0)` over the four
accuracy tasks, so lower is better. GPTQ here uses max-based scales [2]_.

All three tables in this article report Model Optimizer team measurements.
Their supporting checkpoints, run logs, evaluator outputs, and workbooks are
team-owned artifacts; a public artifact bundle is not currently available.
Derived summary values use the workbooks' full-precision scores; task scores in
the tables are rounded for display.

**Table 1. Qwen3.5-9B, NVFP4 W4A4 scale-setting comparison. Bold marks the
best quantized result in each column.**

.. list-table::
   :header-rows: 1

   * - Method
     - MMLU
     - HellaSwag
     - WinoGrande
     - GSM8K
     - Average drop
     - WikiText PPL
   * - BF16 reference
     - 78.69
     - 78.04
     - 73.40
     - 87.64
     - 0.00
     - 9.20
   * - Max scale
     - 75.81
     - 76.33
     - 70.64
     - 74.60
     - 5.10
     - 10.08
   * - Weight MSE scale
     - 76.49
     - 76.61
     - **72.45**
     - 76.72
     - 3.87
     - 9.98
   * - Four-over-six scale
     - 75.32
     - **76.62**
     - 70.40
     - 76.42
     - 4.75
     - 10.02
   * - Local Hessian scale
     - **76.81**
     - 76.50
     - 71.19
     - **80.89**
     - **3.10**
     - **9.90**
   * - GPTQ with max scale
     - 75.77
     - 76.51
     - 70.17
     - 75.97
     - 4.84
     - 10.02

Among the four scale-only methods, Local Hessian reduces average accuracy drop
from 5.10 to 3.10 percentage points relative to max scaling and has the lowest
perplexity. It also produces a smaller average drop and lower perplexity than
the recorded GPTQ baseline, without applying a full-precision weight update.
The result is not a claim that one method wins every cell: for example, weight
MSE is higher on HellaSwag and WinoGrande.

The retained team workbook identifies this model only as Qwen3.5-9B; it does
not record a model repository or checkpoint revision. It defines the reported
metrics as zero-shot MMLU accuracy, zero-shot HellaSwag normalized accuracy,
zero-shot WinoGrande accuracy, five-shot GSM8K strict-match accuracy, and
WikiText word perplexity. The same workbook contains one value per
configuration, but does not retain the calibration dataset or size, benchmark
dataset revisions, evaluator and version, hardware, repeat count, or
uncertainty. Tables 1 and 2 therefore support comparisons among these recorded
configurations, not claims of statistical significance or exact reproduction.

Result 2: Local Hessian composes with GPTQ
******************************************

Local Hessian changes scales; GPTQ uses approximate second-order information to
update quantized weights and compensate error [2]_. Because they act on
different parts of the quantization problem, they can be combined. Table 2 uses
the same Qwen3.5-9B W4A4 tasks and metric definitions as Table 1.

**Table 2. Qwen3.5-9B, NVFP4 W4A4 GPTQ composition.**

.. list-table::
   :header-rows: 1

   * - Method
     - MMLU
     - HellaSwag
     - WinoGrande
     - GSM8K
     - Average drop
     - WikiText PPL
   * - GPTQ with max scale
     - 75.77
     - 76.51
     - 70.17
     - 75.97
     - 4.84
     - 10.02
   * - GPTQ + weight MSE scale
     - 75.94
     - **76.59**
     - **71.51**
     - **82.18**
     - **2.89**
     - 9.95
   * - GPTQ + Local Hessian scale
     - **76.98**
     - **76.59**
     - 70.96
     - 81.50
     - 2.94
     - **9.91**

Replacing max scales with Local Hessian scales in the GPTQ pipeline improves
all four recorded task accuracies, lowers average drop from 4.84 to 2.94
percentage points, and lowers perplexity from 10.02 to 9.91. GPTQ plus weight
MSE has a slightly smaller average drop in this experiment; GPTQ plus Local
Hessian has the best MMLU and perplexity. This is exactly why scale selection
should remain a composable choice rather than being tied to one weight-update
algorithm.

The Qwen3.8-27B delivery candidate
**********************************

We then used Local Hessian while building a Qwen3.8-27B W4A4 delivery
candidate. The controlled ablation uses the same mixed-precision assignment for
both checkpoints: the dense model's MLP projections and language-model head use
either NVFP4 W4A4 or FP8, while all attention projections use FP8. Across the
layers common to both checkpoints, only the NVFP4 weight-scale rule changes
from max to Local Hessian. The max checkpoint omits the multi-token prediction
module, which accounts for the raw checkpoint-size difference; the retained
workbook normalizes both configurations to 6.31 effective bits for the
ablation. Every evaluation uses an FP8 E4M3 key-value cache. The BF16 baseline
and candidates derive from ``Qwen/Qwen3.8-27B`` revision
``1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0``. The retained records do not
specify the calibration dataset, sample count, sequence length, or batch size.

**Table 3. Qwen3.8-27B candidate scores. Values after ± are reported standard
errors.**

.. list-table::
   :header-rows: 1

   * - Scale rule
     - GPQA Diamond
     - Terminal-Bench 2.1
     - AA-LCR
     - MMMU-Pro
     - SciCode
     - IFBench
   * - BF16
     - 88.92 ± 0.36
     - 75.56
     - 72.63 ± 0.64
     - 75.14
     - 47.93
     - 80.07 ± 0.29
   * - Max
     - 87.94 ± 0.22
     - 70.79
     - **78.56 ± 0.72**
     - 74.39
     - 47.04
     - 75.89
   * - Local Hessian
     - **88.01 ± 0.32**
     - **74.02**
     - 73.38 ± 0.57
     - **74.86**
     - **48.41 ± 0.43**
     - **78.93 ± 0.59**

The team's retained evaluation workbook defines the aggregate metric as a
signed mean degradation against BF16:
:math:`\operatorname{mean}(\min(\text{candidate}-\text{BF16},0))` over the
available scores other than SciCode. Local Hessian moves it from -2.14 to -0.78
percentage points, recovering 1.36 points of mean degradation. Five of its six
scores are no more than 1.5 points below BF16, compared with four of six for
max scaling. The one regression relative to the max checkpoint is AA-LCR,
where max is also above the BF16 reference.

The evaluation records define GPQA Diamond as pass@1 symbolic-correct accuracy
over 198 questions, averaged across 16 responses per question; AA-LCR as
pass@1 judge-correct accuracy over 100 questions, also averaged across 16
responses; MMMU-Pro as pass@1 symbolic-correct accuracy over 1,730 entries;
SciCode as pass@1 subtask accuracy over 80 problems and 338 subtasks; IFBench
as pass@1 prompt-loose accuracy over 300 prompts, averaged across five
responses; and Terminal-Bench 2.1 as evaluator-final pass@1 over 89 tasks with
eight trials per task. MMMU-Pro uses one response per entry. SciCode uses one
evaluation run for BF16 and max and the mean of eight completed reruns for
Local Hessian; incomplete reruns are excluded.

For GPQA Diamond, AA-LCR, and IFBench, reported standard errors are the
evaluators' across-response standard errors, expressed in percentage points.
The Local Hessian SciCode standard error is the sample standard deviation of
the eight run-level scores divided by :math:`\sqrt{8}`. Blank standard-error
cells mean no uncertainty estimate was supplied; they are not zeros. These
repeats measure sampling within one checkpoint configuration, not independent
quantization or checkpoint builds.

The same workbook reports mean completion-token change relative to BF16,
excluding Terminal-Bench because the max-scale run has no comparable token
mean. That value is +46.1% for max scaling and +0.6% for Local Hessian.

These results are checkpoint- and evaluation-specific. Within this controlled
scale ablation, Local Hessian materially tightened the candidate's quality
profile; the results do not imply that it will dominate max scaling on every
task or model.

No new deployment contract
**************************

Local Hessian is an offline calibration rule. After calibration, each block
still contains ordinary NVFP4 values and an ordinary FP8 E4M3 scale. Inference
does not evaluate a Hessian, replay calibration data, or use a new operator.
An exported Local Hessian checkpoint therefore needs no Local-Hessian-specific
runtime support.

This separation is useful beyond GPTQ. A pipeline can change the scale rule
without changing its quantization format, mixed-precision assignment, export
schema, or serving stack. Scale search is also an active research area:
ScaleSearch searches nearby representable block scales using weight error
[3]_, ScaleSweep includes MSE and diagonally activation-weighted objectives
[4]_, and SOAR jointly optimizes global and block scales under reconstruction
error [5]_. Local Hessian's specific choice is the full :math:`16\times16`
within-block input second-moment matrix and an independent search for every
block and output channel.

Next steps
**********

Mixture-of-experts (MoE) models are the next important test. Routed experts see
different tokens, so their input statistics can differ sharply, and some
experts may see too few calibration samples. We plan to measure how calibration
coverage, routing balance, and scale sharing affect Local Hessian across MoE
architectures. Synchronizing the per-block statistics for distributed
calibration is another necessary step.

We also want broader evidence across model families and evaluations, including
repeat-based uncertainty, and systematic tests of combinations with GPTQ and
other algorithms. Activation-quantization-aware MOSE, which adds an explicit
activation-error coupling term, is a separate extension; the method and results
in this article are Local Hessian without that coupling.

Closing perspective
*******************

Building day-zero quantized checkpoints repeatedly teaches the same lesson: the
best algorithm can change with the model and sometimes with the evaluation.
Local Hessian is not a universal replacement for every calibration method. It
is a strong, deployment-compatible addition to the Model Optimizer toolbox.

That toolbox matters more than any single winner. Model Optimizer makes it
practical to combine algorithms, evaluate the resulting checkpoints, and
iterate while keeping the deployment target fixed. Better FP4 scale selection
is one more useful lever in that loop.

.. _appendix:

Appendix: Deriving the Local Hessian objective
**********************************************

Start with one output channel and one input row. Let :math:`w_0\in\mathbb{R}^d`
be the original weight vector, :math:`w_q=w_0+\Delta w` its quantized value,
and :math:`x\in\mathbb{R}^{1\times d}` the layer input. The original and
quantized scalar outputs are

.. math::

   y_0=xw_0,
   \qquad
   y_q=xw_q.

Their squared difference is exactly

.. math::

   \lVert y_q-y_0\rVert_2^2
   = \lVert x\Delta w\rVert_2^2
   = \Delta w^{\top}x^{\top}x\Delta w.

For :math:`N` calibration tokens, stack their inputs in
:math:`X\in\mathbb{R}^{N\times d}`. The average reconstruction loss becomes

.. math::

   \frac{1}{N}\lVert X\Delta w\rVert_2^2
   = \Delta w^{\top}\left(\frac{1}{N}X^{\top}X\right)\Delta w.

NVFP4 partitions the input dimension into blocks of :math:`b=16`. For block
:math:`k`, define :math:`X_k\in\mathbb{R}^{N\times b}` and

.. math::

   H_k=\frac{1}{N}X_k^{\top}X_k.

For output channel :math:`j`, a candidate scale :math:`s` produces

.. math::

   \Delta w_{j,k}(s)=\mathcal{Q}_s(w_{0,j,k})-w_{0,j,k}.

Local Hessian selects

.. math::

   s_{j,k}^{\star}
   = \underset{s\in\mathcal{S}_{\mathrm{FP8}}}{\arg\min}\;
     \Delta w_{j,k}(s)^{\top}H_k\Delta w_{j,k}(s).

The implementation anchors the tensor scale at
:math:`s_{\mathrm{tensor}}=\operatorname{amax}(W_0)/6` and evaluates the 126
valid positive, finite FP8 E4M3 block-scale multipliers. The normalization by
:math:`N` does not affect the selected scale because it multiplies every
candidate score by the same positive constant.

For an entire layer, the exact error contains cross terms between different
input blocks:

.. math::

   \left\lVert\sum_k X_k\Delta w_{j,k}\right\rVert_2^2.

Optimizing all block scales jointly would have a combinatorial search space.
Local Hessian drops the cross-block terms and minimizes each diagonal block's
contribution independently. It retains all correlations *within* each
16-coordinate block, unlike a diagonal importance-weighting approximation.
This is the approximation that makes the searches independent and parallel.

.. _local-hessian-references:

References
**********

.. [1] E. Alvarez, O. Almog, E. Chung, S. Layton, D. Stosic, R. Krashinsky,
   and K. Aubrey. `Introducing NVFP4 for Efficient and Accurate Low-Precision
   Inference <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>`_.
   NVIDIA Technical Blog, 2025.
.. [2] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. `GPTQ: Accurate
   Post-Training Quantization for Generative Pre-trained Transformers
   <https://arxiv.org/abs/2210.17323>`_. ICLR, 2023.
.. [3] T. Gupta et al. `Search Your Block Floating Point Scales!
   <https://arxiv.org/abs/2605.12464>`_. MLSys, 2026.
.. [4] L. Lin and X. Wan. `ScaleSweep: Accurate NVFP4 Post-Training
   Quantization of LLMs via Block Scale Initialization
   <https://arxiv.org/abs/2606.07618>`_. arXiv:2606.07618, 2026.
.. [5] C. Bao, X. Yan, Z. Li, G. Qin, G. Yu, and Y. Zhang. `SOAR: Scale
   Optimization for Accurate Reconstruction in NVFP4 Quantization
   <https://arxiv.org/abs/2605.12245>`_. arXiv:2605.12245, 2026.
