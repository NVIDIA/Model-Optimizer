====================
Speculative Decoding
====================

Introduction
============

ModelOpt's Speculative Decoding module (:mod:`modelopt.torch.speculative <modelopt.torch.speculative>`)
enables your model to generate multiple tokens in each generate step. This can be useful for reducing the
latency of your model and speeds up inference.

Below are the speculative decoding algorithms supported by ModelOpt:
- Medusa
- EAGLE


Follow the steps described below to obtain a model with Medusa or EAGLE speculative decoding using ModelOpt's
Speculative Decoding module :mod:`modelopt.torch.speculative`:

#.  **Convert your model via** :meth:`mtsp.convert <modelopt.torch.speculative.speculative_decoding.convert>`:
    Add Medusa heads or EAGLE module to your model.
#.  **Fine-tune Medusa heads or EAGLE module**: Fine-tune the Medusa heads or EAGLE module.
    The base model is recommended to be frozen.
#.  **Checkpoint and re-load**: Save the model via :meth:`mto.save <modelopt.torch.opt.conversion.save>` and
    restore via :meth:`mto.restore <modelopt.torch.opt.conversion.restore>`

.. _speculative_conversion:

Convert
=======

You can convert your model to a speculative decoding model using :meth:`mtsp.convert()
<modelopt.torch.speculative.speculative_decoding.convert>`.

Example usage:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.speculative as mtsp

    # User-defined model
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if mode == "medusa":
        # Configure and convert to medusa
        config = {
            "medusa_num_heads": 2,
            "medusa_num_layers": 1,
            }
    elif mode == "eagle":
        config = {
            "eagle_num_layers": 1
            }
    mtsp.convert(model, [(mode, config)])


Fine-tune speculative decoding model and store/restore the model
----------------------------------------------------------------

After converting to a speculative decoding model, you need to fine-tune the decoding module:

.. code-block:: python

    import os
    from transformers import Trainer
    import modelopt.torch.opt as mto

    mto.enable_huggingface_checkpointing()

    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    trainer._move_model_to_device(model, trainer.args.device)

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model("<path to the output directory>")


To restore the saved speculative decoding model:

.. code-block:: python

    model = AutoModelForCausalLM.from_pretrained("<path to the output directory>")

.. _speculative-concepts:

Speculative Decoding Concepts
=============================

Below, we will provide an overview of ModelOpt's speculative decoding feature as well as its basic
concepts and terminology.

Speculative decoding
--------------------
The standard way of generating text from a language model is with autoregressive decoding: one token
is generated each step and appended to the input context for the next token generation. This means
to generate *K* tokens it will take *K* serial runs of the model. Inference from large autoregressive
models like Transformers can be slow and expensive. Therefore, various *speculative decoding* algorithms
have been proposed to accelerate text generation, especially in latency critical applications.

Typically, a short draft of length *K* is generated using a faster, auto-regressive model, called draft
model. This can be attained with either a parallel model or by calling the draft model *K* times.
Then, a larger and more powerful model, called target model, is used to score the draft. Last, a sampling
scheme is used to decide which draft to accept by the target model, recovering the distribution of the
target model in the process.

Medusa algorithm
----------------

There are many ways to achieve speculative decoding. A popular approach is Medusa where instead of
using an additional draft model, it introduces a few additional decoding heads to predict multiple
future tokens simultaneously. During generation, these heads each produce multiple likely words for
the corresponding position. These options are then combined and processed using a tree-based attention
mechanism. Finally, a typical acceptance scheme is employed to pick the longest plausible prefix from
the candidates for further decoding. Since the draft model is the target model itself, this guarantees
the output distribution is the same as that of the target model.

EAGLE algorithm
---------------

Unlike Medusa that predicts future tokens based on the base model hidden states, EAGLE predicts
future hidden states through a lightweight autoregressive decoder, which is then used to
predict the future tokens. Since autoregression at the feature (hidden states) level is simpler
than at the token level, EAGLE can predict future tokens more accurately than Medusa. Therefore, it
achieves higher speedup.
