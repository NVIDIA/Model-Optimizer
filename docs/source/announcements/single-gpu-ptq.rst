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
one layer at a time. The classic PTQ shape compounds it — calibrate the whole model, then
export the whole model — so a finished calibration still owes a second full traversal before
any checkpoint exists on disk.

Neither is inherent to the math. Layer *i*'s calibration statistics depend only on the
activations entering layer *i*, which layer *i-1* already produced.

One layer at a time
*******************

The mechanism is a loop interchange. Conventional calibration puts data on the outside and
depth on the inside, so every layer has to be resident for every batch:

.. code-block:: python

   for batch in calib_data:           # outer: data
       h = embed(batch)
       for layer in model.layers:     # inner: depth
           h = layer(h)               # all 93 layers live, the whole time

Layerwise calibration swaps the two loops — depth outside, data inside:

.. code-block:: python

   acts = [embed(batch) for batch in calib_data]    # activations at the boundary
   for layer in model.layers:                       # outer: depth
       for i, h in enumerate(acts):                 # inner: data
           acts[i] = layer(h)                       # one layer live at a time
       calibrate(layer); quantize(layer); export(layer); release(layer)

The interchange is what buys everything else. After the swap a layer is *finished* the moment
its inner loop ends — every batch it will ever see has already been through it — so the four
calls on that last line are well defined, and a shard written there is a truthful record that
its layer is done.

The price is the boundary. Instead of one activation tensor in flight per batch, the whole
calibration set's activations are held between layers. That is a real cost, but it is bounded
by the calibration set rather than by the model.

Two pieces turn that into a run.

**Weights spill to disk.** An ``accelerate`` device map with explicit GPU and CPU budgets
keeps the bulk of the checkpoint on disk or in host RAM and materializes only what the
current step touches.

**Each layer is exported the moment it is finished.** With ``layerwise.export_dir`` set, a
layer is quantized and written to its own checkpoint shard as soon as calibration is done
with it — and then released. That is the piece worth remembering, because it collapses two
problems into one artifact:

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

   **Draft:** these figures are pending reconfirmation against a full run on the current
   exporter. Everything marked ``TODO(reconfirm)`` must be replaced from that run before
   publish.

.. list-table::
   :header-rows: 1

   * - Kimi-K3
     -
   * - Layers
     - 93
   * - GPU budget (``--max_gpu_memory_gb``)
     - 140 GB
   * - CPU budget (``--max_cpu_memory_gb``)
     - 1700 GB
   * - Wall clock
     - ``TODO(reconfirm)``
   * - Peak GPU
     - ``TODO(reconfirm)``
   * - Peak RSS
     - ``TODO(reconfirm)``
   * - Output
     - 93 layer shards + tail + index

**Those budgets are weight-placement budgets, not caps**, and it is worth knowing that before
you size them. ``--max_gpu_memory_gb`` and ``--max_cpu_memory_gb`` feed ``accelerate``'s device
map: they decide how much of the *checkpoint* is assigned to each device, and everything the
run allocates on top of the weights falls outside them. So expect peak GPU to land somewhat
above the GPU budget — activations, calibration buffers and the CUDA context are not counted
against it — and expect a much larger transient spike in host RSS while shards are read and
dispatched, settling to a far lower steady state once the offload folder is populated. Size
both with headroom rather than to the exact capacity of the machine.

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

**Calibration algorithms are restricted — for now.** The workflow supports max, MSE and local
Hessian calibration today, in FP8 and NVFP4. Weight-mutating calibration (GPTQ, AWQ,
SmoothQuant) and AutoQuantize are refused. That restriction is conservative rather than
fundamental: under per-layer export a mutated weight is already written into the layer's shard
before the layer is released, so the machinery is in the right shape for it. A PR extending the
workflow to GPTQ and friends is on the way.

**It solves calibration memory, not serving memory.** The checkpoint this produces still has
to be served, and whether your hardware can serve it is a separate question this workflow does
not answer.

**It scales in depth, not width.** Peak memory here is a lower bound, not a knob: the GPU
still has to hold one decoder layer, plus its activations, at once. A model with many modest
layers is easy; a model with one enormous layer is the boundary this design cannot move. That
is also how to predict whether your model fits before spending a session finding out — divide,
don't guess.
