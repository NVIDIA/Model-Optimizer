:orphan:

Improving NVFP4 Accuracy with Local Hessian Weight Scales
#########################################################

:Author: Model Optimizer Team
:Date: September 9, 2026
:Tags: local-hessian, quantization, nvfp4, calibration, modelopt

.. role:: local-hessian-result(strong)
.. role:: table-header-note

In this blog, we share about Model Optimizer 'Local-Hessian', an algorithm for NVFP4 per-block scale selection
to minimize the output error. We used this algorithm to create a low loss checkpoint
`nvidia/Qwen3.8-27B-NVFP4 <https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4>`_ which can leverage NVFP4 tensor cores for performant inference on Blackwell GPUs.
Here is a comparison of accuracy results we observed for 'Local Hessian' algorithm compared to the default max algorithm:

.. image:: assets/qwen3-27b-w4a4-scale-rule-accuracy.png
   :alt: Qwen3.8-27B scores by NVFP4 weight-scale rule, BF16 baseline in gray
   :width: 100%

**Figure 1. Qwen3.8-27B NVFP4 accuracy comparison between the default NVFP4 algorithm (max) and 'Local-Hessian'.**

Background: NVFP4 Scale Selection
*********************************

NVFP4 represents each group of 16 weights with FP4 values and an FP8 block
scale [1]_. This block scale is used to scale the per-block values so to NVFP4 E2M1 range (-6.0, 6.0).
The default way is to set the block scale based on the per-block maximum value (max scaling) [1]_.

As shown, originally in 'Four-Over-Six' paper [3]_, this block scale can be selected based on other 
critieria like per-block error. While 'Four-Over-Six' selects the per-block scale from  2 candidates while 
`Model-Optimizer Mean Square Error (MSE) <https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html#modelopt.torch.quantization.model_calib.mse_calibrate>`_ algorithm sets this based on exhuastive sweep over all positive and non-zero FP8 
scales (126 values).

Both of these approaches for scale selection only considers weight tensor level error which we find does not correlate 
well with downstream accuracy evaluation results.

How Local Hessian Works
***********************

NVFP4 **Local-Hessian** chooses the scale to minimize the output error of NVFP4 matrix multiplication operation.
Specifically, we use this idea to select the per-block scale for NVFP4 weights only, 
which does not required any new deployment kernels or changes from the default NVFP4 deployment - we are just computing the 
per-block scales in a different way from the existing max based per-block scale selection.


Here is how it works. Consider a linear layer

.. math::
   :label: lh-linear

   Y = WX,
   \qquad
   W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}},
   \qquad
   X\in\mathbb{R}^{C_{\mathrm{in}}\times N},

for :math:`N` calibration tokens. We quantize the weights,

.. math::
   :label: lh-quant

   W_q=\mathcal{Q}(W,s)=W+\Delta(W,s),
   \qquad
   \mathcal{Q}(W,s)=\operatorname{Cast}(W/s)\cdot s,

where :math:`s` is the quantization scale, :math:`\operatorname{Cast}`
rounds to the low-precision datatype such as NVFP4, and :math:`\Delta`
is the resulting quantization error. 

Let :math:`Y_q=W_qX` be the output after quantizing the weights.

We consider one output channel at a time. Write its weights as the row
:math:`w` and its quantization error as :math:`\Delta(w,s)`. Its
output mean squared error is

.. math::
   :label: lh-output-error

   E(s) = \lVert wX-w_qX\rVert_2^2
     = \lVert \Delta(w,s)\,X\rVert_2^2
     = \Delta(w,s)\,(XX^{\top})\,\Delta(w,s)^{\top}.

Here :math:`XX^{\top}\in\mathbb{R}^{C_{\mathrm{in}}\times C_{\mathrm{in}}}`
is the Hessian (second order derivative) of :math:`E(s)` with respect to
:math:`\Delta(w,s)`, and it carries the output error minimization
objective into the scale decision.

For NVFP4, the weight scale :math:`s` for one output channel is a vector
of dimension :math:`C_{\mathrm{in}}/16`, one entry per block. With
:math:`M` candidate values per entry, minimizing :math:`E(s)` jointly
means searching :math:`M^{C_{\mathrm{in}}/16}` combinations -- not
tractable.

Per-Block Output Error Objective
================================

We simplify the objective in :eq:`lh-output-error` with one key
observation: each block's scale can be chosen in isolation, against the
output error that block alone contributes.
For block :math:`k`, with scale :math:`s_k`, weight error
:math:`\Delta(w_k,s_k)`, and inputs
:math:`X_k\in\mathbb{R}^{16\times N}`,

.. math::
   :label: lh-block-error

   E_k(s_k) = \Delta(w_k,s_k)\,(X_kX_k^{\top})\,\Delta(w_k,s_k)^{\top},

where :math:`X_kX_k^{\top}` is only :math:`16\times16`. That is the
local Hessian, and it turns the search into many small independent
problems instead of one large one.

For each per-block scale, we sweep over all possible 126 FP8 values, just like we do for MSE algorithm.
See the `Model Optimizer Local-Hessian code
<https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html#modelopt.torch.quantization.model_calib.layerwise_calibrate>`_
for more details.

Results
***********

Scale Selection Accuracy
========================

In Table 1 we compares Local-Hessian Vs other scale selection algorithms dor weights on Qwen 3.5 9B.

Local Hessian gives the overall best accuracy among the NVFP4
weight-scale selection methods, cutting the average drop from 5.10 to 3.10
points against the default max rule. We get that from nothing but a
smarter way of computing the weight scale -- which says something about
micro-block formats like NVFP4: **the scale carries a lot of
information, and it pays to set it diligently.**

.. list-table::
   :header-rows: 1

   * - Weight scale selection method
     - MMLU
     - HellaSwag
     - WinoGrande
     - GSM8K
     - Average drop :table-header-note:`(lower is better)`
     - WikiText PPL :table-header-note:`(lower is better)`
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
   * - MSE scale
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
     - :local-hessian-result:`3.10`
     - :local-hessian-result:`9.90`

.. rst-class:: table-note

All layers except the final output layer (``lm_head``) use NVFP4
weight and activation quantization (W4A4).


Local-Hessian + GPTQ Accuracy
=============================

Local Hessian changes scales; GPTQ [2]_ changes weight rounding to
minimize per-layer output error. The two are orthogonal, so they
compose: Local Hessian rounds to nearest (RTN) by default, and GPTQ can
replace that rounding step once the scales are set. In Table 2, we show
that Local-Hessian scales improve GPTQ as well.

Two things stand out:

#. Local-Hessian scale selection alone (3.10 average drop) beats GPTQ with
   max scales (4.84), with no weight update at all.
#. Composing the two improves further still, from 3.10 to 2.94.

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
   * - GPTQ + Local Hessian scale
     - **76.98**
     - **76.59**
     - 70.96
     - 81.50
     - :local-hessian-result:`2.94`
     - :local-hessian-result:`9.91`

Just Better Scales, No Runtime Cost
***********************************

Local Hessian and the other ModelOpt scale-selection algorithms for
NVFP4 weight scales are free. Weight scales are computed only once, at
checkpoint creation, and that same scale is reused on every
deployment. Selecting scales this way improves accuracy without
incurring any deployment throughput penalty.


Using Local Hessian
*******************

See the `local_hessian_calibrate API
<https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html#modelopt.torch.quantization.model_calib.local_hessian_calibrate>`_
for the calibration entry point.

To use it in your own configuration, set the ``algorithm`` field:

.. code-block:: python

   import modelopt.torch.quantization as mtq

   config = {
       "quant_cfg": [...],  # quantizer configuration
       "algorithm": {"method": "local_hessian", "fp8_scale_sweep": True},
   }

   model = mtq.quantize(model, config, forward_loop)

See :ref:`quant-cfg` for how to write the ``quant_cfg`` field.

To reproduce the published Qwen3.8-27B checkpoint end to end:

.. code-block:: bash

   python examples/hf_ptq/hf_ptq.py \
       --pyt_ckpt_path Qwen/Qwen3.8-27B \
       --recipe modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_local_hessian-fp8_attn-kv_fp8_cast.yaml \
       --dataset nemotron-post-training-v3 \
       --calib_size 512 \
       --calib_seq 2048 \
       --batch_size 1 \
       --export_path <export_dir>

Batch size 1 keeps padding tokens from polluting the output error
statistics used by Local Hessian and GPTQ.


Next steps
**********

- **Adapt Local Hessian for sparse MoEs.** Many experts in a sparse MoE
  see very little calibration data. Local-Hessian workflow needs to be adapted to that
  low-data regime.

Bitter Lesson: Best Method - It Depends!
****************************************************************

We, the Model Optimizer team, have quantized over 80 models, all available in
the `Inference Optimized Checkpoints (with Model Optimizer)
<https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer>`_
collection.

Building checkpoints for various models and evaluating them teaches this
bitter lesson: the best algorithm can change with the model, the
calibration dataset, and sometimes the evaluation.

Local Hessian gives the best accuracy recovery overall in our internal
evaluations on small dense models. However, the winning
algorithm/combination might change depending on the model. On
Qwen3.5-9B, Local Hessian + GPTQ came out ahead; on Qwen3.8-27B, plain
RTN rounding beat GPTQ on top of the same scales. Scale selection is the
reliable part; what you pair it with is not.

What matters more than any single algorithm is a place to build
quantized checkpoints, evaluate them (see our agentic skills for
frontier evaluation), and iterate quickly. Model Optimizer is that
one-stop shop for inference optimization, and Local Hessian is one more
useful lever in the loop.

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
.. [3] J. Cook, J. Guo, G. Xiao, Y. Lin, K. Wyss, M. Nazemi, A. Mishra,
   C. del Mundo, T. Blankevoort, and S. Han. `Four Over Six: More Accurate
   NVFP4 Quantization with Adaptive Block Scaling
   <https://arxiv.org/abs/2512.02010>`_. arXiv:2512.02010, 2025.