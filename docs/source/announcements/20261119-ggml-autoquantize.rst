:orphan:

Sub-4-Bit AutoQuantize with GGML Formats: Matching a QAD Release Without Training
#################################################################################

:Author: Model Optimizer Team
:Date: November 19, 2026
:Tags: autoquantize, quantization, ggml, mixed-precision, gptq

GGML's IQ formats store an MoE expert weight in 1.75 to 2.56 bits, and the experts hold most of an
MoE's bytes. That makes them the place to spend a tight budget. A uniform assignment gives every
expert the same format, but some layers need more bits than others. AutoQuantize picks a format per
layer under a size budget, but below 4 bits its default search did no better than uniform IQ2. We
changed how it scores and calibrates candidates, and published the result as two AutoQuantize
recipes that run with one ``hf_ptq.py`` command.

On `NVIDIA Nemotron 3.5 Bolt 74B-A9B
<https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16>`__, post-training quantization
with the recipe matches the `24GB release
<https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB>`__ at the same size. That
release was trained with Quantization-Aware Distillation (QAD); the recipe needs no training.

Highlights
**********

* **QAD-release accuracy from PTQ alone.** At the release's 2.0625 bits per expert weight (22.4 GB of
  language-model weights), the recipe's held-out KL to BF16 is 0.076, against the release's 0.071:
  within one standard error. It scores AA-LCR 60.1, GPQA 79.0 and SciCode 41.9, against the
  release's 59.3, 76.0 and 40.2.
* **26% lower KL than uniform IQ2_XXS** at the same size, and 7.8 ± 1.4 points more on AA-LCR.
* **Scoring in a low-bit context matters most.** The default search scores each layer with the
  others in BF16. At the same budget its model has 94% higher KL and is 14 points lower on AA-LCR,
  5 on GPQA and 6 on SciCode.
* **GPTQ matters next.** Rounding the same format assignment with imatrix alone, without GPTQ,
  raises KL 17% and costs 3.8 ± 1.5 points on AA-LCR.
* **Lowest KL on Qwen3.6-35B-A3B too, by a smaller margin.** At 2.75 effective bits, the recipe's
  held-out KL is 0.100, against 0.105 for the default search and about 0.106 for uniform IQ2. These
  margins are within one standard error.
* **One command, two runs.** The same ``hf_ptq.py --recipe`` command searches, then finalizes and
  exports. Recipes are available for Nemotron 3.5 Bolt and Qwen3.6-35B-A3B.

Why Sub-4-Bit Needs a Different Search
**************************************

Each candidate format costs a fixed number of bits per weight:

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 10 10 10

   * - Format
     - IQ1_M
     - IQ2_XXS
     - IQ2_XS
     - IQ2_S
     - NVFP4
     - FP8
     - BF16
   * - Bits per weight
     - 1.75
     - 2.0625
     - 2.3125
     - 2.5625
     - 4.5
     - 8
     - 16

NVFP4 and FP8 are weight-only here (W4A16 and W8A16), like the IQ formats, because the budget
counts weight bits.

AutoQuantize scores how much each candidate format hurts each layer, then solves for the
lowest-scoring assignment under the budget (see :doc:`20260824-autoquantize`).
By default a layer is scored with every other layer in BF16. At NVFP4 and FP8 budgets that works,
because one layer's error stays small next to the others'. At 2 bits it does not. On Bolt, each
routed-expert group's IQ1_M error looks small on its own, so the default search put every routed
expert at IQ1_M and spent the bytes on the shared experts, at 14-16 bits. The routed experts'
errors compounded, and the model came out worse than uniform IQ2_XXS at the same size: held-out
KL 0.146 against 0.103.

What the Recipes Change
***********************

Three changes, all set in the recipe YAML:

1. **Score in a low-bit context.** With ``auto_quantize_method: kl_div`` and
   ``method_options: {context_quant_level: k}``, each candidate is scored while the other searched
   layers sit at the format ``k`` steps above their lowest-bit candidate (``0`` is the lowest). The
   score then includes how a layer's error compounds
   with the others', so the scores add up closer to the model that ships. Bolt uses
   ``context_quant_level: 1`` (IQ2_XXS, the second-lowest candidate). Qwen3.6 uses ``lowest``, also
   IQ2_XXS. With a context, AutoQuantize picks formats with a linear program, as the gradient method
   does.
2. **Calibrate with GPTQ, twice.** During the search, each IQ candidate is calibrated with GPTQ and
   imatrix on BF16 inputs (*local* GPTQ), so a layer's score does not depend on the formats picked
   for the layers before it. Once the formats are fixed, ``finalize_algorithm`` reruns GPTQ and
   imatrix sequentially, each layer on its quantized predecessors.
3. **Search the experts, fix the rest.** The experts hold about 90% of Bolt's linear weights. The
   recipe searches only the routed and shared experts and keeps every other layer at the release's
   format: W4A16 NVFP4 for the Mamba ``in_proj`` and ``out_proj``, and Q8_0 for attention, the MoE
   latent projections and ``lm_head``.

The search part of the Bolt recipe:

.. code-block:: yaml

   auto_quantize:
     constraints:
       effective_bits: 2.0625
     module_search_spaces:
       - module_name_patterns: ["*mixer.experts*", "*mixer.shared_experts.*"]
         candidate_formats:     # IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S with local GPTQ + imatrix,
           ...                  # then W4A16 NVFP4 and W8A16 FP8
         allow_no_quant: true   # BF16 is a candidate too
       ...                      # the other layers: one fixed format each
     auto_quantize_method: kl_div
     method_options: {context_quant_level: 1}
     finalize_algorithm:
       - method: gptq
         layerwise: {enable: true}
         ...
       - method: imatrix

Results on Nemotron 3.5 Bolt
****************************

Nemotron 3.5 Bolt is a 74B-parameter hybrid Mamba-Transformer MoE with 9B active parameters. The
24GB release keeps its experts at IQ2_XXS, the Mamba ``in_proj`` and ``out_proj`` at W4A16 NVFP4,
and attention, the MoE latent projections and ``lm_head`` at Q8_0, and was then trained with QAD.

The search scored candidates on 128 of 512 calibration documents (up to 512 tokens each) from
Nemotron post-training data. Held-out KL is measured against BF16 on 128 other documents (62.2k
tokens). The 2.25-, 2.5- and 2.75-bit rows reuse the 2.0625-bit search's scores, solved at their
budgets. Every row except BF16 keeps the release's formats outside the experts.

.. list-table::
   :header-rows: 1
   :widths: 34 11 11 11 11 11 11

   * - Checkpoint
     - Expert bits
     - LM weights
     - Held-out KL
     - AA-LCR
     - GPQA
     - SciCode
   * - BF16
     - 16
     - ~148 GB
     - –
     - 64.6
     - 80.6
     - 44.4
   * - 24GB release (IQ2_XXS experts, QAD)
     - 2.0625
     - 22.4 GB
     - 0.071
     - 59.3
     - 76.0
     - 40.2
   * - Uniform IQ2_XXS experts, PTQ (GPTQ)
     - 2.0625
     - 22.4 GB
     - 0.103
     - 52.3
     - 79.3
     - 41.1
   * - **AutoQuantize recipe**
     - 2.0625
     - 22.4 GB
     - **0.076**
     - **60.1**
     - **79.0**
     - **41.9**
   * - AutoQuantize recipe
     - 2.25
     - 24.0 GB
     - 0.060
     - 59.1
     - 81.0
     - 41.6
   * - AutoQuantize recipe
     - 2.5
     - 26.1 GB
     - 0.045
     - 60.7
     - 80.2
     - 41.9
   * - AutoQuantize recipe
     - 2.75
     - 28.2 GB
     - 0.038
     - 62.1
     - 80.5
     - 42.6

Held-out KL has a standard error of 0.002-0.008. AA-LCR is pass@1 over 16 seeds with a
65,536-token generation budget, GPQA Diamond is averaged over 32 repeats and SciCode subtask accuracy
over 8. Between two rows, paired by question, differences resolve to about ±1.5 points on AA-LCR,
±0.7 on GPQA and ±1 on SciCode. Our harness scores BF16 and the release lower than the model card
does on AA-LCR and GPQA, so compare rows within this table only. Every quantized row was evaluated
on an exact BF16 dequantization of its quantized weights, so the table measures accuracy, not
speed.

At the release's size, PTQ with uniform IQ2_XXS experts loses 7 AA-LCR points to the release.
The recipe recovers them with the same bytes and is 3 points higher on GPQA. Most of the
remaining AA-LCR gap to BF16 is runaway reasoning, not wrong answers: a generation that hits the
token budget scores as wrong, and those are repetition loops. Among finished generations, every
model answers about 70% correctly, BF16 included, but at 2.25 bits (24.0 GB) the recipe's model hits
the budget in 15.9% of generations against BF16's 7.6%. A 512-token KL objective does not see that, so
adding bytes moves AA-LCR less than it moves KL.

These numbers come from a script with the recipe's search and finalize settings. The script loads
the checkpoint with its own weight renaming and calls ``mtq.auto_quantize``. One setting differs
from the recipe: the script's scoring context held the Mamba ``conv1d`` at FP8, while the recipe
keeps it in BF16. Both finalize it in BF16.

What Each Change Buys
=====================

Each row changes one part of the recipe and keeps the rest. Differences are paired against the
recipe at 22.4 GB; a negative eval delta means lower accuracy than the recipe.

.. list-table::
   :header-rows: 1
   :widths: 44 14 14 14 14

   * - Change
     - Held-out KL
     - AA-LCR
     - GPQA
     - SciCode
   * - Default search: other layers in BF16 while scoring, threshold search :sup:`a`
     - +94%
     - −13.8 ± 2.0
     - −4.8 ± 0.9
     - −5.8 ± 1.8
   * - Other experts in BF16 while scoring, linear program
     - +65%
     - −10.6 ± 1.8
     - −5.6 ± 1.2
     - −2.6 ± 1.0
   * - Non-expert layers at IQ2_XXS too while scoring
     - +23%
     - −5.6 ± 1.4
     - −1.1 ± 0.8
     - −0.2 ± 1.0
   * - Non-expert layers in BF16 while scoring
     - +1.8%
     - −2.9 ± 1.6
     - −1.0 ± 0.7
     - −0.4 ± 1.0
   * - Same assignment, rounded with imatrix only (no GPTQ)
     - +17%
     - −3.8 ± 1.5
     - −0.9 ± 0.7
     - −1.1 ± 0.9
   * - Same assignment, rounded with plain max scaling
     - +32%
     - −2.0 ± 1.7
     - −1.2 ± 0.6
     - −0.0 ± 1.1
   * - Search every layer, not only the experts
     - +6%
     - −0.4 ± 1.6
     - −1.1 ± 0.7
     - −0.4 ± 0.8

:sup:`a` The threshold search left budget unused, so this model is 22.2 GB.

* **Context.** The best scoring context is the model that ships: 2-bit experts with the other
  layers at their final formats. Erring toward BF16 for the non-expert layers costs little; scoring
  them at 2 bits, below their final formats, costs 23% in KL and 5.6 AA-LCR points. Scoring with
  the other experts in BF16 is the costliest choice, even with the linear program.
* **Calibration.** Without GPTQ, the same assignment loses about as much KL as going from 2.0625 to
  2.25 bits (22.4 to 24.0 GB) gains.
* **Scope.** Searching the non-expert layers too lost at every size we tried, 6-17% in KL and 1-2
  GPQA points. A layer type with a well-tested format is better fixed than searched.

Results on Qwen3.6-35B-A3B
**************************

The `Qwen3.6-35B-A3B recipe
<https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt_recipes/models/Qwen/Qwen3.6-35B-A3B/auto_quantize>`__
searches the routed experts over IQ2_XXS, IQ2_XS, IQ2_S, W4A16 NVFP4, W8A16 FP8 and BF16 at 2.75
effective bits. Unlike Bolt's, its budget counts every linear except embeddings, ``lm_head`` and
the vision tower, and those other linears stay BF16. Held-out KL against BF16, over 128 documents
(63.5k tokens; standard error about 0.01):

.. list-table::
   :header-rows: 1
   :widths: 75 25

   * - Search at 2.75 bits
     - Held-out KL
   * - **This recipe**
     - **0.100**
   * - Uniform IQ2 + sequential GPTQ (interpolated between IQ2_XXS and IQ2_XS)
     - ~0.106
   * - Default gradient search, W4A4/W8A8 NVFP4/FP8 candidates
     - 0.105
   * - KL scored in context, W4A4/W8A8 NVFP4/FP8 candidates
     - 0.101

The recipe measures lowest, but on this smaller MoE its margins are within one standard error. The
Bolt results above are where the method's gain is clear.

Try It
******

Run the command once to search, then run it again to finalize and export. Both runs share the
``--auto_quantize_checkpoint`` file:

.. code-block:: bash

   python examples/hf_ptq/hf_ptq.py \
       --pyt_ckpt_path nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 \
       --trust_remote_code \
       --recipe models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/auto_quantize/iq1_iq2_gptq_nvfp4_fp8_experts_at_2p0625bits \
       --auto_quantize_checkpoint <search.pth> \
       --export_path <output-path>

For Qwen3.6-35B-A3B, use ``--pyt_ckpt_path Qwen/Qwen3.6-35B-A3B`` and
``--recipe models/Qwen/Qwen3.6-35B-A3B/auto_quantize/iq2_gptq_nvfp4_fp8_experts_at_2p75bits``.

* **Other budgets.** Change ``constraints.effective_bits`` in a copy of the recipe. The Bolt
  table's 2.25- to 2.75-bit rows are the same search solved at those budgets.
* **One GPU or several.** ``hf_ptq.py`` holds Bolt on one GPU: on a GB200, a search with two expert
  candidates peaked at 184 GB and the finalize at 193 GB. To score the candidate formats in
  parallel, run one format shard per GPU and merge them, as described in `Scoring and finalizing
  AutoQuantize in separate runs
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq#scoring-and-finalizing-autoquantize-in-separate-runs>`__.
* **Bolt loads through its own code**, hence ``--trust_remote_code``; it was tested with
  transformers 5.18.
* **Serving.** The export is a Hugging Face checkpoint with packed GGML blocks. For serving GGML
  checkpoints on vLLM, see `vLLM with GGML recipes
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq#vllm-with-ggml-recipes-iq-q8_0>`__.

Each recipe's README lists its candidates and results: `Nemotron 3.5 Bolt
<https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt_recipes/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/auto_quantize>`__
and `Qwen3.6-35B-A3B
<https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt_recipes/models/Qwen/Qwen3.6-35B-A3B/auto_quantize>`__.
For writing your own, the `sub-4-bit AutoQuantize guide
<https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq#sub-4-bit-autoquantize-ggml-formats>`__
covers the same settings.
