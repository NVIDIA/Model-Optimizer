:orphan:

AutoQuantize: Automatic Sensitivity-Guided Mixed-Precision Quantization Under a Cost Budget
###########################################################################################

:Authors: Asma Beevi K T, Wei Ming, Frida Hou, Juhi Mittal, Jenny Chen, Ajinkya Rasane, Meng Xin
:Date: July 15, 2026
:Tags: autoquantize, quantization, mixed-precision, modelopt

Why do we need AutoQuantize?
****************************

LLMs carry a lot of redundancy, but not uniformly: a few layers — attention projections, the final layers of the network — are disproportionately sensitive to quantization, while most others (like MoE experts) are quite forgiving. Keeping just those few sensitive layers at higher precision (FP8 or BF16) while quantizing the rest to FP4 preserves accuracy with nearly all of FP4's memory savings and speedups. The hard part is finding *which* layers to keep — traditionally a slow pile of per-model ablation experiments.

**AutoQuantize**, part of NVIDIA's `Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ library, automates this search: given a cost budget, it scores every layer's quantization sensitivity with a fast gradient-based heuristic and solves for a Pareto-optimal mixed-precision assignment — no per-model ablation studies required.

How AutoQuantize works
**********************

AutoQuantize is a neural architecture search (NAS) inspired method that works in three steps: score how sensitive each operation is to quantization, model the performance cost of each available format, and solve for the best layer-wise assignment under the cost budget with a knapsack-style optimization. The sensitivity score is a second-order Taylor approximation in the spirit of Optimal Brain Surgeon [1]_, as introduced in LLM-MQ [2]_.

Where LLM-MQ handles only weight quantization, AutoQuantize works at the operator level — including joint weight-and-activation quantization for GEMMs — and respects real deployment constraints such as operator fusion.

AutoQuantize gradient: A fast, yet accurate sensitivity scoring
===============================================================

The sensitivity score we want is simple to state: how much the model loss changes when a layer is quantized in isolation. Measuring that directly — quantize one layer at a time, re-evaluate the whole model — requires a full model evaluation per layer per candidate format, as we'll quantify later (Table 1). We need a cheaper estimate.

Two observations give us a shortcut. First, for a trained model, a Taylor expansion of the loss around a layer's output shows the loss change from a quantization perturbation is governed by the Hessian — the local curvature. Second, for cross-entropy loss, that Hessian is well approximated by the Fisher information matrix, whose diagonal is just the squared gradient — free from an ordinary backward pass. Together they turn sensitivity into a gradient-squared-weighted output error, no Hessian required.

Concretely, let :math:`Y_i` be the BF16 output of operator :math:`i`, :math:`Y_i^{Q_{i,f}}` its output under quantization format :math:`f`, :math:`g_i = \nabla_{Y_i}\mathcal{L}` the gradient at that output, and :math:`H_i` the local Hessian:

.. math::

   \mathcal{L}\!\left(Y_i^{Q_{i,f}}\right) = \mathcal{L}\!\left(Y_i\right) - g_i^{\top}\!\left(Y_i - Y_i^{Q_{i,f}}\right) + \tfrac{1}{2}\left(Y_i - Y_i^{Q_{i,f}}\right)^{\!\top} H_i \left(Y_i - Y_i^{Q_{i,f}}\right)

The first-order term vanishes in expectation for a trained model, leaving:

.. math::

   \Delta\mathcal{L}\!\left(Y_i^{Q_{i,f}}\right) = \mathcal{L}\!\left(Y_i^{Q_{i,f}}\right) - \mathcal{L}\!\left(Y_i\right) \approx \tfrac{1}{2}\left(Y_i - Y_i^{Q_{i,f}}\right)^{\!\top} H_i \left(Y_i - Y_i^{Q_{i,f}}\right)

Keeping only the Hessian diagonal and estimating it with the diagonal Fisher (squared gradients) gives the sensitivity score:

.. math::

   S(\mathrm{Op}_i, Q_{i,f}) = \Delta\mathcal{L}\!\left(Y_i^{Q_{i,f}}\right) \propto \sum_{k=1}^{d} \left(g_{i,k}\right)^2 \left(Y_{i,k} - Y_{i,k}^{Q_{i,f}}\right)^2

where :math:`d` is the feature dimension of the layer output.

The intuition: quantization perturbs the model, and the loss impact of that perturbation is the output error weighted by squared gradients. The error can be measured at the operation's immediate output or further downstream (e.g. the block output); for linear layers we use the linear-layer output. This output-side formulation is also what separates AutoQuantize from LLM-MQ, which measures error at each weight and therefore can't handle joint weight-and-activation quantization or the coupled decisions deployment-aware search needs.

Both ingredients are cheap: the output error :math:`Y_{i,k} - Y_{i,k}^{Q_{i,f}}` comes from replaying the operator's captured input through simulated quantization for each candidate format, and the gradient :math:`g_{i,k}` from a single backward pass.

Performance cost
================

ModelOpt uses *effective bits* after quantization as the cost. Effective bits is directly proportional to the compressed model weight size, which is a useful target in practice: large-batch inference is often bound by loading weights from memory (even for sparse MoEs), so weight compression pays off there too. That said, effective bits is a fast proxy, not truly hardware-aware — using measured hardware latency as the cost is a natural next step.

Putting it together
===================

Following the effective-bits objective above, the cost of assigning format :math:`f` to operator :math:`i` is its compressed weight size:

.. math::

   C(\mathrm{Op}_i, Q_{i,f}) = N_{\mathrm{params}}(\mathrm{Op}_i) \times \mathrm{bits}(Q_{i,f}),

where :math:`N_{\mathrm{params}}(\mathrm{Op}_i)` is the operator's parameter count and :math:`\mathrm{bits}(Q_{i,f})` the effective bits per weight of format :math:`f` (including scale-factor overhead). AutoQuantize then solves the constrained optimization

.. math::

   \min_{\{f\}} \sum_i S(\mathrm{Op}_i, Q_{i,f}) \quad \text{s.t.} \quad \sum_i C(\mathrm{Op}_i, Q_{i,f}) \leq B,

where :math:`Q_{i,f}` is the chosen format for operator :math:`i` and :math:`B` is the total weight-size budget — e.g. an average of 4.8 effective bits across the model. Structurally this is a multiple-choice knapsack; ModelOpt solves it with a linear-programming solver essentially instantaneously. Sweeping :math:`B` traces out an accuracy-vs-compression frontier.

Deployment-restriction-aware search
***********************************

A mixed-precision assignment is only useful if the runtime can execute it. So AutoQuantize performs a deployment-aware search — runtime coupling constraints are folded into the search rather than patched up afterwards, meaning the searched model is deployable out of the box in vLLM, SGLang, TensorRT-LLM, and similar inference runtimes. Any restriction of the form "this group of operators takes one joint format decision" becomes a merged knapsack item with aggregated sensitivity and cost.

**1) Joint quantization for fused linear layers.** Inference runtimes often fuse linear operators, which imposes a shared quantization format across the fused group. This constraint is applied within each layer: that layer's Q, K, and V projections are fused and must share one format, so the fused QKV projection becomes a single decision variable. The naive score would just sum the three per-projection sensitivities — but that treats their Hessians as independent, when the three outputs actually interact through the attention operation. Instead, AutoQuantize quantizes all three projections jointly with format :math:`f` and measures the sensitivity at the attention output, so the metric naturally captures how the projections' quantization errors combine through attention:

.. math::

   S(\mathrm{Op}_{\mathrm{qkv}}, Q_{\mathrm{qkv},f}) \propto \sum_{k=1}^{d} \left(g_{\mathrm{attn},k}\right)^2 \left(Y_{\mathrm{attn},k} - Y_{\mathrm{attn},k}^{Q_{\mathrm{qkv},f}}\right)^2,

where :math:`Y_{\mathrm{attn}}` is the attention output and :math:`g_{\mathrm{attn}}` its gradient. The cost, by contrast, has no interaction and is simply additive:

.. math::

   C(\mathrm{Op}_{\mathrm{qkv}}, Q_{\mathrm{qkv},f}) = C(\mathrm{Op}_{\mathrm{q}}, Q_{\mathrm{qkv},f}) + C(\mathrm{Op}_{\mathrm{k}}, Q_{\mathrm{qkv},f}) + C(\mathrm{Op}_{\mathrm{v}}, Q_{\mathrm{qkv},f}).

**2) MoE layer constraints.** vLLM and TensorRT-LLM quantized MoE APIs require all sparse experts in a constrained MoE group to share one quantization format. This restriction is also applied within each MoE layer: only sparse experts inside the same layer are coupled. In Nemotron 3 Super, each sparse expert contains ``up_proj`` and ``down_proj``, so these sparse-expert projections must be assigned jointly. We formulate the sparse-expert set as one operator-level decision,

.. math::

   \mathrm{Op}_{\mathrm{moe}} = \bigcup_{e \in \mathcal{E}} \{\mathrm{Op}_{e,\mathrm{up\_proj}}, \mathrm{Op}_{e,\mathrm{down\_proj}}\},

measure sensitivity at the MoE block output so the metric captures the combined contribution from all sparse experts,

.. math::

   S(\mathrm{Op}_{\mathrm{moe}}, Q_{\mathrm{moe},f}) \propto \sum_{k=1}^{d} \left(g_{\mathrm{moe},k}\right)^2 \left(Y_{\mathrm{moe},k} - Y_{\mathrm{moe},k}^{Q_{\mathrm{moe},f}}\right)^2,

and define the deployment cost as the sum over sparse experts,

.. math::

   C(\mathrm{Op}_{\mathrm{moe}}, Q_{\mathrm{moe},f}) = \sum_{e \in \mathcal{E}} \left(C(\mathrm{Op}_{e,\mathrm{up\_proj}}, Q_{\mathrm{moe},f}) + C(\mathrm{Op}_{e,\mathrm{down\_proj}}, Q_{\mathrm{moe},f})\right).

Other linear layers in the MoE block — latent projections and shared experts — are not part of this coupling and can be assigned formats independently.

Results
*******

.. image:: assets/autoquantize-qwen3-mmlu-effective-bits.png
   :alt: MMLU accuracy versus effective bits under AutoQuantize for Qwen3 1.7B, 4B, 8B, and 14B
   :width: 100%

**MMLU accuracy vs. effective bits under AutoQuantize, Qwen3 1.7B/4B/8B/14B.** Each point is one solve of the constrained search at that bit budget, followed by an MMLU evaluation. Solid: {NVFP4, FP8, BF16} menu; dashed: {NVFP4, BF16}.

The figure shows the accuracy-vs-compression frontier AutoQuantize traces on Qwen3 models: sweep the bit budget :math:`B`, solve at each point, evaluate on MMLU. Accuracy rises with the budget and tapers — once the sensitive layers are protected, extra bits buy little.

Two things to notice. First, the rise is essentially monotonic, which means the sensitivity score is ranking layers correctly — a noisy proxy would give a jagged frontier. Second, adding formats to the mix helps: at every budget, {NVFP4, FP8, BF16} sits at or above {NVFP4, BF16}. A sensitive layer doesn't need to back off all the way to BF16 — FP8 gives a good accuracy-vs-performance middle ground, protecting moderately sensitive layers at a fraction of the cost.

AutoQuantize gradient is fast!
==============================

**Speed.** The direct way to measure sensitivity — quantize one layer at a time and measure a downstream evaluation such as loss, accuracy — runs the whole model per measurement. The AutoQuantize gradient scores all layer × format combinations in one forward + backward sweep. On Qwen3.6-35B-A3B, with everything else identical, that's a ~51× difference (Table 1).

**Table 1. Scoring cost: AutoQuantize gradient vs. AutoQuantize KL-divergence (lower is better).**

.. list-table::
   :header-rows: 1

   * - Scoring method
     - Scoring complexity
     - Time taken for sensitivity estimation
     - Relative time
     - Peak GPU memory
   * - AutoQuantize gradient
     - :math:`O(N_{\mathrm{layers}}) \times O(N_{\mathrm{formats}})`
     - ~16 minutes
     - 1×
     - 29 GB
   * - AutoQuantize KL-divergence
     - :math:`O(N_{\mathrm{layers}}^2) \times O(N_{\mathrm{formats}})`
     - ~14 hours
     - ~51× slower
     - 23 GB

*Measured on 4× NVIDIA RTX 6000 Ada GPUs with 128 samples at sequence length 512. Times cover sensitivity scoring only — not the end-to-end AutoQuantize run, which also includes calibration time for each format.*

**Memory.** A backward pass is not inherently memory-heavy. With activation checkpointing, activations are recomputed on demand instead of retained for backward, trading additional compute for a smaller footprint. AutoQuantize also performs a scoring pass rather than a training step, so it needs no optimizer state or persistent weight-gradient buffers. Consequently, peak memory remains close to forward-only execution: 29 GB versus 23 GB for KL-divergence in Table 1.

How to use ModelOpt AutoQuantize
********************************

AutoQuantize is a one-call API in Model Optimizer — pass the model, a bit budget, the format menu to search over, and a calibration data loader:

.. code-block:: python

   import modelopt.torch.quantization as mtq

   model, search_state = mtq.auto_quantize(
       model,
       constraints={"effective_bits": 4.8},
       quantization_formats=[mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
       data_loader=calib_loader,
       forward_step=lambda model, batch: model(**batch),
       loss_func=lambda output, batch: output.loss,
       num_calib_steps=512,
       num_score_steps=128,
   )

The returned model carries the searched per-layer format assignment and is ready for export. For an end-to-end example on Hugging Face models — including export to a deployable checkpoint — see the `AutoQuantize section of the ModelOpt llm_ptq README <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq#autoquantize>`_. AutoQuantize also works on Megatron Core models — see the `AutoQuantize mixed-precision search example in Megatron-LM <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt#-auto-quantize-mixed-precision-search>`_.

Next steps
**********

We are working on improving AutoQuantize in the following ways:

#. **Hardware-aware cost.** Effective bits is a fast proxy for deployment cost. Relying instead on hardware-measured costs — such as per-operator latency on the target GPU and inference runtime — would let the solver optimize for what actually matters: end-to-end inference speed.
#. **Combinatorial effects of quantization.** AutoQuantize currently scores each layer quantized in isolation, but quantization errors interact — the loss impact of quantizing two layers together is not always the sum of their individual scores. Capturing these combinatorial effects in the sensitivity estimate is the next step toward tighter accuracy at the same budget.

Conclusion
**********

AutoQuantize turns mixed-precision quantization from trial and error into a principled search: gradient-based sensitivity scoring in a single sweep, a knapsack solve under your cost budget, and deployment constraints built in so the searched model runs directly in vLLM, SGLang, and TensorRT-LLM. Sweep the bit budget to find your model's accuracy-vs-compression sweet spot.

.. _references:

References
**********

.. [1] B. Hassibi and D. G. Stork. `Second Order Derivatives for Network Pruning: Optimal Brain Surgeon <https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_. *NeurIPS*, 1992.
.. [2] S. Li, X. Ning, K. Hong, T. Liu, L. Wang, X. Li, K. Zhong, G. Dai, H. Yang, and Y. Wang. `LLM-MQ: Mixed-Precision Quantization for Efficient LLM Deployment <https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/5c805adc-b555-499f-9882-5ca35ce674b5.pdf>`_. *NeurIPS Workshop on Efficient Natural Language and Speech Processing (ENLSP)*, 2023.
