:orphan:

DSpark vs Domino: Same DFlash Backbone, Different Correction Heads
##################################################################

:Author: Model Optimizer Team
:Date: July 13, 2026
:Tags: speculative-decoding, dflash, dspark, domino, architecture

DSpark (DeepSpec) and Domino both build on block-parallel DFlash draft generation but diverge in their token-level correction heads. DSpark's default head is a stateless first-order Markov transition; Domino's is a GRU that conditions on the draft prefix. Both must unroll sequentially at inference, so the tradeoff is per-step cost against how much prefix context the correction can use. During teacher-forced training, DSpark's Markov transition can also be parallelized over positions.
See the DSpark and Domino papers in :ref:`dspark-domino-references` for the original method descriptions.

Highlights
**********

* Both systems share the DFlash block-parallel backbone, so their parallel draft throughput starts from a similar foundation.
* In ModelOpt, DSpark defaults to ``markov_head_type="vanilla"``: stateless ``W1`` and ``W2`` embedding lookups with no hidden state to thread through.
* Domino uses ``nn.GRU`` and carries recurrent state across draft positions.
* Both correction heads are sequential at inference because ``x_{k-1}`` must be sampled before step ``k``.

Shared Foundation: DFlash Block-Parallel Backbone
*************************************************

Both systems use DFlash: a draft backbone that runs a single causal attention forward pass over all draft positions in parallel, producing per-position hidden states and base draft logits. This is the expensive step; the correction head adds token-level adjustment on top of those outputs.

Where They Diverge: The Correction Head
***************************************

DSpark uses a first-order Markov transition. For each draft position ``k``:

.. code-block:: text

   e_{k-1} = W1[x_{k-1}]
   bias_k  = W2 * e_{k-1}
   p_k     = softmax(U_k + bias_k)
   x_k     ~ p_k

The correction at position ``k`` depends only on ``x_{k-1}``; no RNN hidden state threads across steps. The dominant work is a table lookup and projection rather than a recurrent rollout.

Domino uses a GRU correction head. A recurrent hidden state accumulates information about the draft prefix and is concatenated at readout:

.. code-block:: text

   gru_h_k = GRU(input_k, gru_h_{k-1})
   p_k     = softmax(U_k + W * [h_k; gru_h_k])
   x_k     ~ p_k

These descriptions compare the underlying architectures. ModelOpt's Domino support is currently training-only, so it does not apply the correction head in serving.

Correction Head Comparison
**************************

.. list-table::
   :header-rows: 1

   * - System
     - Per-step compute
     - State carried
   * - DSpark ``markov_head_type="vanilla"``
     - ``W1[x_{k-1}]`` plus transition projection
     - None
   * - Domino GRU
     - Full GRU cell over a high-dimensional input
     - Recurrent hidden state

Both heads must unroll left-to-right at inference. The practical distinction is qualitative: the vanilla Markov head uses only the prior sampled token, while the GRU carries a prefix-dependent recurrent state.

Takeaways
*********

#. DFlash draft generation is shared; the correction head is the main differentiator.
#. Both default correction heads are sequential at inference; their tradeoff is local transition structure versus prefix-dependent state.
#. ModelOpt exposes the DSpark variants through ``markov_head_type``: ``vanilla`` (the default), ``gated``, and ``rnn``. The ``rnn`` option is the closest analogue to Domino's GRU.
#. Architectural comparisons do not establish a universal quality or throughput ranking; evaluate the chosen head on the target model and serving configuration.

.. _dspark-domino-references:

References
**********

* Xin Cheng et al., `DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation <https://arxiv.org/abs/2607.05147>`_,
  arXiv:2607.05147, 2026.
* Jianuo Huang et al., `Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding <https://arxiv.org/abs/2605.29707>`_,
  arXiv:2605.29707, 2026.

Resources
*********

* `DeepSpec / DSpark repo <https://github.com/deepseek-ai/DeepSpec>`_
* `DeepSeek-V4-Pro-DSpark checkpoint <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark>`_
* `Domino repo <https://github.com/jianuo-huang/Domino>`_
* `Domino checkpoint: Qwen3-8B-Domino-b16 <https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16>`_
* `ModelOpt PR #1710 <https://github.com/NVIDIA/Model-Optimizer/pull/1710>`_
