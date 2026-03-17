====================
Speculative Decoding
====================

ModelOpt's Speculative Decoding module (:mod:`modelopt.torch.speculative <modelopt.torch.speculative>`)
enables your model to generate multiple tokens in each generate step, reducing inference latency.

ModelOpt implements the **EAGLE3** algorithm, which attaches a lightweight autoregressive draft
module to a frozen base model. The draft module operates at the feature level—predicting future
hidden states rather than tokens directly—to achieve high acceptance rates at low compute cost.

Medusa is also supported for models that require it; see :ref:`Medusa <speculative-medusa>` below.


Where to Start
--------------

- **New to speculative decoding?** Start with the :doc:`Workflow <_eagle_workflow>` for a
  step-by-step guide from model conversion to deployment.
- **Tuning training or architecture?** See the :doc:`Configuration Reference <_eagle_config_reference>`
  for all available options.
- **Want better acceptance rates?** See :doc:`Best Practices <_eagle_best_practices>` for data
  synthesis, draft model configuration, and vocabulary compression.
- **Looking for the Python API?** See the :doc:`Module Guide <_speculative_module_guide>`.


.. toctree::
    :maxdepth: 1
    :caption: EAGLE

    ./_eagle_workflow.rst
    ./_eagle_config_reference.rst
    ./_eagle_best_practices.rst

.. toctree::
    :maxdepth: 1
    :caption: Reference

    ./_speculative_module_guide.rst


.. _speculative-concepts:

Speculative Decoding Concepts
==============================

Speculative decoding
--------------------

The standard way of generating text from a language model is with autoregressive decoding: one token
is generated each step and appended to the input context for the next token generation. This means
to generate *K* tokens it will take *K* serial runs of the model. Inference from large autoregressive
models like Transformers can be slow and expensive. Therefore, various *speculative decoding* algorithms
have been proposed to accelerate text generation, especially in latency critical applications.

Typically, a short draft of length *K* is generated using a faster model, called the *draft model*.
Then, a larger and more powerful model, called the *target model*, verifies the draft in a single
forward pass. A sampling scheme decides which draft tokens to accept, recovering the output
distribution of the target model in the process.

EAGLE3 algorithm
----------------

EAGLE3 attaches a lightweight autoregressive decoder (the draft module) to a frozen base model.
Unlike token-level autoregression, the draft module operates at the *feature level*: it predicts
future hidden states, which are then projected to token logits. Autoregression over hidden states
is an easier task than over tokens, so the draft module achieves high prediction accuracy with low
compute cost.

Compared to earlier EAGLE versions, EAGLE3 uses auxiliary hidden states from **multiple intermediate
layers** of the base model as additional input to the draft decoder, not just the final layer output.
This richer signal enables the draft module to more accurately predict the base model's next-layer
representations, resulting in higher token acceptance rates and greater inference speedup.

.. _speculative-medusa:

Medusa algorithm
----------------

Medusa is an alternative speculative decoding algorithm that adds *K* independent prediction heads
on top of the base model, each predicting a future token position in parallel. Unlike EAGLE3, Medusa
does not use a separate autoregressive draft decoder.

Medusa is supported alongside EAGLE3 in ModelOpt. To use it, pass ``"medusa"`` as the algorithm name
to :meth:`mtsp.convert() <modelopt.torch.speculative.speculative_decoding.convert>` with a
:class:`MedusaConfig <modelopt.torch.speculative.config.MedusaConfig>` dict:

.. code-block:: python

    mtsp.convert(model, [("medusa", {"medusa_num_heads": 2, "medusa_num_layers": 1})])

The training and checkpointing workflow is identical to EAGLE3. See the
:doc:`Module Guide <_speculative_module_guide>` for the full API reference.
