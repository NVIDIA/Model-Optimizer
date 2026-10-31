========================================
Loading Hugging Face models across ranks
========================================

:func:`load_pretrained <modelopt.torch.utils.plugins.model_load_utils.load_pretrained>` loads a
Hugging Face model for ModelOpt to optimize and export, whole on every rank or distributed. It loads
by a :class:`DistributionPlan <modelopt.torch.utils.distributed.DistributionPlan>`. Every rank takes
rank 0's plan: the one rank 0 passes as ``distribution``, or, when it passes none, the one
:func:`resolve_distribution_plan <modelopt.torch.utils.plugins.model_load_utils.resolve_distribution_plan>`
decides from ``MODELOPT_FSDP_SIZE``, ``MODELOPT_EP_DEGREE`` and rank 0's GPU memory. The plan is
recorded on the model, where calibration and export read it.

.. code-block:: python

    # torchrun --nproc_per_node 8 quantize.py
    from modelopt.torch.utils.distributed import distributed_run
    from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

    with distributed_run():
        model = load_pretrained("meta-llama/Llama-3.1-8B")  # a whole copy on every rank's GPU
        # With MODELOPT_FSDP_SIZE=world: FSDP2-sharded over the 8 ranks instead.

Routes
======

* A plan with ``dense="fsdp"`` takes the FSDP2 load (``parallel_load_and_prepare_fsdp2``): a causal
  LM, or a text-only VLM it supports (Qwen3.5), is FSDP2-sharded over the process group, each rank
  reading its own layers.
* Any other plan gives every rank the whole model, through ``auto_class.from_pretrained``
  (``auto_class`` defaults to ``AutoModelForCausalLM``).

No load shards the routed experts yet. A plan the load decides keeps the expert-parallel degree at
1, so an MoE loads without expert parallelism; an EP degree ``MODELOPT_EP_DEGREE`` asks for, or a
plan with ``ep_size > 1``, is refused.

A refusal on any rank raises on every rank, before the load, and says how to load the model whole:
``MODELOPT_FSDP_SIZE=1`` for a plan the load decided, or
``distribution=DistributionPlan(<ranks>, dense="replicate")``.

The plan is recorded once, so ``fsdp2_wrap`` refuses a model ``load_pretrained`` returned: to shard
a model yourself, load it with ``from_pretrained``.

Whole-model loads
=================

* On a distributed run, each rank loads the model onto its own GPU when it has one, no
  ``device_map`` is passed and ``init_quantized_weights`` does not place it: the GPU its process is
  bound to once CUDA is set up, else the one its local rank numbers. ``device_map=None``,
  ``from_pretrained``'s own default, loads it on the CPU for the caller to place. A rank without a
  GPU of its own warns and loads where ``from_pretrained`` places it.
* The options only the FSDP2 load takes are refused, except at the FSDP2 load's own defaults
  (``cpu_offload=False``, ``mp_policy=None``, ``broadcast_chunk_size=8``) or
  ``cpu_offload=None``, which ask nothing of a whole-model load, so a caller can forward them
  whatever the plan. ``broadcast_chunk_size=None`` asks for all of a source's layers at once.

Sharded loads
=============

* The FSDP2 load builds the model class and places the model itself: a text-only VLM through its
  image-text class, anything else through ``AutoModelForCausalLM``. Leave ``auto_class`` out.
* Besides its own options (``cpu_offload``, ``mp_policy``, ``broadcast_chunk_size``) it takes
  ``trust_remote_code``, ``attn_implementation`` and ``hf_config``, and refuses any other keyword
  rather than ignore it. An option passed as ``None``, ``from_pretrained``'s own default, asks
  nothing and is not refused.
* It reads the checkpoint's tensors as stored and keeps their dtype, so it refuses a ``dtype`` and
  a quantized checkpoint. It cannot run the model type's loading hook, so it refuses a checkpoint
  the hook would build modules for, such as a Nemotron-H MTP head.
* A single process, or a group of one, reads no knob: it takes the FSDP2 load only from a plan
  passed as ``distribution``, and sets a group of one up for it, which needs
  ``torchrun --nproc_per_node=1``.

The automatic layout
====================

With neither a plan nor ``MODELOPT_FSDP_SIZE``, rank 0's GPU memory decides whether the dense part
is sharded. The model is weighed in the checkpoint's own dtype (its config's, else bfloat16), not
one the load asks for: hand a load that widens it, ``dtype=torch.float32`` say, its plan.

A call that places the model (``device_map``, ``quantization_config``, a ``dtype``) loads only
where the model fits one GPU; where it does not, the refusal says the layout was decided rather
than asked for. Pass ``distribution=`` or set ``MODELOPT_FSDP_SIZE=1`` to keep the load whole. An
option at ``from_pretrained``'s own default, ``device_map=None`` included, places nothing and does
not keep the load whole.

Configs
=======

``hf_config`` is the config ``resolve_distribution_plan`` returned with the plan handed in as
``distribution``: the load builds from it as given, so without that plan it is refused. A
``config`` passed for the load, or a path to one, is the one the plan is decided from, and a sharded
load takes it as its ``hf_config``.

Hub options
===========

A Hub model ID is made local first, so the export can read the source checkpoint from disk; the
``revision``, ``token``, ``cache_dir`` and ``local_files_only`` options go to that fetch. The config
read and a whole-model load also take all but ``revision``: an ``auto_map`` entry can name custom
code in another Hub repo, which Transformers fetches with them. The FSDP2 load builds the model
from the config with ``from_config``, which takes none of them, so such code comes with the
environment's Hub settings.

Launches without a process group
================================

A launch whose processes set no process group up (a launcher that only exports ``WORLD_SIZE``, as
``hf_ptq.py`` without ``--use_fsdp2``) has each process load as a single process does: the whole
model, with no plan to agree on and no ``device_map`` default. A plan for several ranks, a sharded
load, and a knob asking to shard need the group, so they are refused there: set it up first, with
``distributed_run()``.

Inside ``init_quantized_weights``
=================================

``init_quantized_weights`` (see :doc:`_compress_quantized_models`) quantizes and compresses the
model in the ``from_pretrained`` it patches, which the FSDP2 load never calls. Inside it, a plan
the load decides gives every rank the whole model, with nothing to measure, and the load leaves
the device map to ``init_quantized_weights``, which computes it and refuses one it is passed. A
knob or a plan that asks to shard is refused there.
