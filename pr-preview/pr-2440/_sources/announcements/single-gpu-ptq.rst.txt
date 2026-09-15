:orphan:

Quantizing a 1.5 TB Model on a Single GPU
#########################################

:Author: Model Optimizer Team
:Date: September 15, 2026
:Tags: quantization, nvfp4, layerwise, moe, single-gpu, modelopt

Post-training quantization needs calibration, and calibration needs forward passes over real
data. Traditionally that has meant holding the whole model in accelerator memory — so the
hardware floor for *quantizing* a model has been roughly the floor for *serving* it. For a
large mixture-of-experts checkpoint, that puts an otherwise routine PTQ run behind a
multi-node allocation.

`Model Optimizer <https://github.com/NVIDIA/Model-Optimizer>`_ now calibrates and exports one
decoder layer at a time. The memory a calibration run needs is set by the largest layer, not
by the model, and the floor drops from *one model* to *one layer*.

As an existence proof: ``moonshotai/Kimi-K3`` — a 1.5 TB checkpoint with 896 experts across
93 layers — quantized to NVFP4 on a **single B200**.

The memory floor for calibration
********************************

The constraint worth naming is not that models are large. It is that calibration has been
*all-or-nothing*: the forward pass wants every layer resident, even though it only ever reads
one layer at a time. Two consequences follow, and both cost hardware rather than accuracy.

First, the whole checkpoint has to fit somewhere fast. Second, the classic PTQ shape —
calibrate the whole model, then export the whole model — traverses it twice, so a completed
calibration still owes a second full pass before there is a checkpoint on disk.

Neither is inherent to the math. Calibration statistics for layer *i* depend on the
activations entering layer *i*, which the previous layer already produced. If those
activations are carried forward explicitly, layers can be visited strictly one at a time, and
the resident set never has to exceed one of them.

One layer at a time
*******************

Three pieces make that concrete.

**Weights spill to disk.** An ``accelerate`` device map with explicit GPU and CPU budgets
keeps the bulk of the checkpoint on disk or in host RAM and materializes only what the
current step touches.

**Calibration walks layer by layer.** ``layerwise`` calibration runs the forward pass one
decoder layer at a time, caching the activations at each boundary so the next layer has its
input without replaying the ones before it.

**Each layer is exported the moment it is finished.** With ``layerwise.export_dir`` set, a
layer is quantized and written to its own checkpoint shard as soon as calibration is done
with it — and then released.

That third piece is the one worth remembering, because it collapses two problems into one
artifact:

.. note::

   **The shards are the resume artifact.** There is no separate full-precision scratch copy
   accumulating beside the run, and no second whole-model export pass owed at the end. A
   shard on disk *is* the record that its layer is done.

Running it
**********

Setting the config field is the entire switch — there is no CLI flag:

.. code-block:: yaml

   quantize:
     algorithm:
       method: max
       layerwise:
         enable: true
         calib_mutates_weights: false                 # amax-only fast path
         export_dir: /tmp/modelopt_layerwise_export   # presence is the switch;
                                                      # value is replaced with --export_path
         # checkpoint_dir omitted -> derived as <export_path>.layerwise_resume

Then the run itself:

.. code-block:: bash

   python examples/hf_ptq/hf_ptq.py \
       --pyt_ckpt_path  <bf16_ckpt> \
       --recipe         models/moonshotai/Kimi-K3/ptq/nvfp4_experts-kv_none_layerwise_export \
       --export_path    <out> \
       --qformat nvfp4 --trust_remote_code --attn_implementation eager \
       --offload_folder <scratch> --max_gpu_memory_gb 140 --max_cpu_memory_gb 1700 \
       --calib_size 256 --batch_size 8 --skip_generate

What you need on hand: one GPU, a GPU and CPU memory budget you choose, and fast scratch
sized for the checkpoint you are writing.

A few configurations are refused *before* calibration starts, rather than producing a quietly
different checkpoint — worth checking against your model before you spend a session:

* AWQ and SVDQuant, which need whole-model pre-quant-scale steps
* Models with tied weights (``tie_word_embeddings``); use ``export_hf_checkpoint()`` instead
* Multi-process jobs such as FSDP2, where every rank would write the same shards

Per-layer export also leaves the in-memory model in export form, so ``hf_ptq.py`` sets
``--skip_generate`` for you.

Interrupt it
************

Because a committed shard means a finished layer, resume needs no special invocation: rerun
the identical command. Finished layers are skipped, and calibration picks up at the boundary
it last committed.

.. code-block:: text

   Checkpoint: resuming layerwise calibration from layer 13/93

The Kimi-K3 checkpoint was produced this way across three separate four-hour GPU sessions.
Resume is exact rather than approximate: a run killed with ``SIGKILL`` after 25 of 48 layers
and then resumed produced a checkpoint identical, tensor for tensor, to the uninterrupted run.

Results
*******

.. warning::

   **Draft:** the Kimi-K3 row is pending reconfirmation against a full run on the current
   exporter. Figures marked ``TODO(reconfirm)`` must be replaced from that run before publish.

.. list-table::
   :header-rows: 1

   * - Model
     - Layers
     - GPU budget
     - Wall clock
     - Peak GPU
     - Output
   * - Kimi-K3 (1.5 TB)
     - 93
     - 140 GB
     - ``TODO(reconfirm)``
     - ``TODO(reconfirm)``
     - 93 layer shards + tail + index
   * - DeepSeek-R1 671B (642 GB)
     - 61
     - 80 GB
     - 40 min 12 s
     - 88.9 GB
     - 403 GB, 40 shards
   * - Nemotron-3-Ultra 550B (~1.1 TB)
     - 108
     - 80 GB
     - 47 min 16 s
     - 76.7 GB
     - 365 GB, 34 shards

Every Kimi-K3 expert projection — ``TODO(reconfirm)``, or 92 × 896 × 3 — carries a calibrated
``input_scale``, and vLLM selects the FlashInfer TRT-LLM NVFP4 MoE kernel rather than the
emulation fallback.

Resume state stays bounded, because only the committed boundary's activations are kept: 332 KB
beside 22 GB of shards on a 35B model, 396 KB beside 19 GB on a 30B.

The correctness claim behind these runs is narrow and worth stating precisely: **per-layer
export produces the same checkpoint as whole-model export.** Across four models, the two paths
were compared tensor for tensor and config for config — 123,513 tensors on a 35B MoE, 74,163
on a 30B, 0 mismatched — with every ``weight_map`` entry resolving to the shard that actually
holds it. Under vLLM, checkpoints exported both ways produce identical greedy generations.

What you're trading
*******************

Each of these is a consequence of the design rather than a defect, so the useful question is
whether the trade fits your constraints.

**Time.** The layer walk is sequential by construction: roughly 40–47 minutes for models in
the 550–671B range on one GPU. You are trading wall clock for hardware, which is the point,
but it is a real cost on a large model.

**Calibration algorithms are restricted — for now.** ``calib_mutates_weights: false``, the
flag that makes resume cheap, is currently whitelisted to amax-only methods: max, MSE, and
local Hessian. Weight-mutating calibration (GPTQ, AWQ, SmoothQuant) and AutoQuantize are
refused today. That restriction is conservative rather than fundamental — under per-layer
export the mutated weights land in the layer's shard before the layer is released — so GPTQ in
particular is expected to need little or no change. Gradual enablement is on the way. Formats
today are FP8 and NVFP4.

**It solves calibration memory, not serving memory.** The checkpoint this produces still has
to be served, and whether your hardware can serve it is a separate question this workflow does
not answer.

**It scales in depth, not width.** Peak memory here is a lower bound, not a knob: the GPU
still has to hold one decoder layer, plus its activations, at once. A model with many modest
layers is easy; a model with one enormous layer is the boundary this design cannot move. That
is also how to predict whether your model fits before spending a session finding out — divide,
don't guess.

Resources
*********

* `Single-GPU disk-offload PTQ (PR #2008) <https://github.com/NVIDIA/Model-Optimizer/pull/2008>`_
* `Per-layer shard export (PR #2136) <https://github.com/NVIDIA/Model-Optimizer/pull/2136>`_
* `Multimodal and MTP support for layerwise export (PR #2303) <https://github.com/NVIDIA/Model-Optimizer/pull/2303>`_
* `Kimi-K3 on layerwise fused export (PR #2218) <https://github.com/NVIDIA/Model-Optimizer/pull/2218>`_
