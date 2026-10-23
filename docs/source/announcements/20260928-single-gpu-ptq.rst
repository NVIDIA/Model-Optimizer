:orphan:

Quantizing a 4.9 TB Qwen3.8 Model on a Single GPU
#################################################

:Author: Model Optimizer Team
:Date: September 28, 2026
:Tags: quantization, nvfp4, layerwise, moe, single-gpu, modelopt

Post-training quantization needs calibration, and calibration needs forward passes over real
data. Every batch runs through every layer, so the whole model either sits in accelerator
memory or gets streamed back in from CPU and disk for each batch. The first option makes the
hardware you need to *quantize* a model roughly the hardware you need to *serve* it. The second
moves the full weight set once per batch. For a large mixture-of-experts checkpoint, an
otherwise routine PTQ run ends up waiting on a multi-node allocation.

`Model Optimizer <https://github.com/NVIDIA/Model-Optimizer>`_ now calibrates and exports one
decoder layer at a time. The largest layer sets the memory a calibration run needs. The floor
drops from *one model* to *one layer*.

We took ``Qwen/Qwen3.8-2.4T-A95B``, a 2.4 T parameter model whose BF16 checkpoint weighs
**4.892 TB** and carries 512 routed experts in each of its 92 layers, and quantized it with
NVIDIA's published production recipe on a **single GB300**. It took **3 h 04 m**. GPU memory
peaked at **141.7 GB** of the card's 283 GB.

The memory floor for calibration
********************************

Calibration has been *all-or-nothing*: every batch needs every layer, even though it only ever
reads one layer at a time. The usual PTQ flow adds a second cost. You calibrate the whole model,
then export the whole model, so a finished calibration still owes another full traversal before
a single checkpoint byte reaches disk.

The math requires neither. Layer *i*'s calibration statistics depend only on the activations
entering layer *i*, and layer *i-1* has already produced those.

One layer at a time
*******************

Mechanically, this is a loop interchange. Conventional calibration puts data on the outside
and depth on the inside:

.. code-block:: python

   for batch in calib_data:           # outer: data
       h = embed(batch)
       for layer in model.layers:     # inner: depth
           h = layer(h)               # all 92 layers, once per batch

Layerwise calibration swaps them, with depth outside and data inside:

.. code-block:: python

   acts = [embed(batch) for batch in calib_data]    # activations at the boundary
   for layer in model.layers:                       # outer: depth
       for i, h in enumerate(acts):                 # inner: data
           acts[i] = layer(h)                       # one layer live at a time
       calibrate(layer); quantize(layer); export(layer); release(layer)

Everything else follows from that swap. When a layer's inner loop ends, the layer is finished.
Every batch it will ever see has already gone through it, so the four calls on that last line
are well defined, and the layer can be written out and forgotten right there.

You pay at the boundary. A conventional run keeps one activation tensor in flight per batch.
Layerwise keeps the activations for the entire calibration set between layers. That cost is
real. It grows with the calibration set and the model's hidden size, and the number of layers
never enters into it.

Two more pieces make it a working run.

**Weights spill to disk.** An ``accelerate`` device map with explicit GPU and CPU budgets
leaves most of the checkpoint on disk or in host RAM. Only what the current step touches gets
materialized.

**Each layer is exported as soon as it's finished.** With ``layerwise.export_dir`` set, a layer
is quantized, written to its own checkpoint shard, and released the moment calibration is done
with it. Remember this part, because it lets one artifact handle two jobs:

.. note::

   **The shards are the checkpoint, and most of the resume state.** No full-precision scratch
   copy piles up beside the run, and nobody owes a second whole-model export pass at the end.
   A small manifest in ``<export_path>.layerwise_resume`` commits each shard and holds the
   activations entering the next layer, so keep that directory alongside the output.

Running it
**********

Setting the config field turns it on. There's no CLI flag:

.. code-block:: yaml

   quantize:
     algorithm:
       method: mse                                    # max / mse / local Hessian all work
       fp8_scale_sweep: true
       layerwise:
         enable: true
         get_qdq_activations_from_prev_layer: false   # next layer sees FP inputs, as in a
                                                      # whole-model pass; use true for local Hessian
         calib_mutates_weights: false                 # mse and the sweep only touch _amax
         export_dir: /tmp/modelopt_layerwise_export   # presence is the switch;
                                                      # value is replaced with --export_path
         # checkpoint_dir omitted -> derived as <export_path>.layerwise_resume

That block is the **only** difference between this run and the published whole-model recipe,
so the single-GPU variant is a file you create: copy
``modelopt_recipes/models/Qwen/Qwen3.8-2.4T-A95B/ptq/nvfp4_experts_mse-fp8_self_attn-fp8_linear_attn-kv_fp8_cast.yaml``
and swap in the block above.
``modelopt_recipes/general/ptq/nvfp4_experts_only-kv_fp8_layerwise_export.yaml`` has the same
block in a complete recipe if you want something to diff against. Everything under
``quant_cfg`` stays as published:

.. list-table::
   :header-rows: 1

   * - Scope
     - Format
   * - Routed experts (92 × 512)
     - NVFP4, MSE-searched static weight scales, dynamic input scales
   * - Self-attention ``q/k/v/o_proj`` (23 layers)
     - FP8 W8A8
   * - Linear-attention / gated-delta path, including ``conv1d`` (69 layers)
     - FP8 W8A8
   * - KV cache
     - FP8, cast mode (constant amax, no KV calibration)
   * - MTP block, router, shared experts, ``lm_head``, embeddings
     - BF16, unquantized

Calibration used **512 samples** of
`Nemotron-Post-Training-Dataset-v2 <https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2>`_
at sequence length 512, **batch size 8**.

.. note::

   **Pick the batch size on purpose.** For layerwise runs, ``hf_ptq.py`` defaults to
   ``--batch_size 1``, but only if you leave the flag at its auto-probe default. An explicit
   value passes straight through. Going to 8 cuts wall clock roughly in half. It does move the
   numbers a little. Weights, ``weight_scale`` and ``weight_scale_2`` come out bit-identical,
   but activation ``input_scale`` values shift slightly, because batching pads each sequence to
   the longest one in the batch and those pad positions feed the activation amax.

Then launch the run:

.. code-block:: bash

   python examples/hf_ptq/hf_ptq.py \
       --pyt_ckpt_path  <Qwen3.8-2.4T-A95B> \
       --recipe         <single-gpu variant of the recipe above> \
       --export_path    <out> \
       --qformat nvfp4 --attn_implementation eager \
       --offload_folder <scratch> --max_gpu_memory_gb 120 --max_cpu_memory_gb 600 \
       --dataset nemotron-post-training-dataset-v2 \
       --calib_size 512 --calib_seq 512 --batch_size 8

You'll need one GPU, GPU and CPU memory budgets of your choosing, and fast scratch space big
enough for the checkpoint you're writing.

.. note::

   **This is the production recipe, calibrated from a BF16 source.** ``Qwen/Qwen3.8-2.4T-A95B``
   ships as plain BF16 with no ``quantization_config``. Nothing in the path dequantizes, so the
   source format carries no accuracy caveat. The quantization config is the one behind the
   published ``nvidia/Qwen3.8-2.4T-A95B-NVFP4`` checkpoint. We picked the calibration
   hyperparameters (sample count, sequence length, batch size) for this demonstration. We don't
   claim they reproduce the published checkpoint's numerics.

Some configurations are refused outright, so you never get a checkpoint that's quietly different
from what you asked for. Check your model against this list before you commit a session to it:

* AWQ and SVDQuant, which need whole-model pre-quant-scale steps. These can surface at the first
  layer's export rather than up front, once the calibrator has registered them.
* Models with tied weights (``tie_word_embeddings``); use ``export_hf_checkpoint()`` instead
* Multi-process jobs such as FSDP2, where every rank would write the same shards

``examples/hf_ptq/hf_ptq.py`` refuses a few more up front, including AutoQuantize recipes,
speculative-decoding models and ``--qformat int8_smoothquant``, a format per-layer export can't
write.

Per-layer export also leaves the in-memory model in export form, so ``hf_ptq.py`` sets
``--skip_generate`` for you.

Interrupt it
************

Each layer's shard is written first, and then the manifest in ``<export_path>.layerwise_resume``
commits it. As long as both directories survive, resuming takes nothing special. Rerun the exact
same command. Committed layers get skipped, and calibration picks up at the last boundary it
committed. A layer whose shard landed but never got committed is calibrated and exported again,
as long as an earlier layer was committed. If the run died before its first commit, the rerun
stops rather than overwrite those shards, and you delete the export directory to start over.

.. code-block:: text

   Checkpoint: resuming layerwise calibration from layer 42/92

An earlier Qwen3.8 run on this workflow survived a CUDA OOM, a deliberate stop, a session
teardown, and a machine rebuild that wiped site-packages mid-run, finishing across **four
process lifetimes**. Every restart skipped the finished layers. Nothing was recalibrated or
re-exported. Resume is exact, too. We killed a run right after layer 2 of 8, resumed it, and got
a checkpoint identical, tensor for tensor, to the uninterrupted run.

The resume directory stays small. In flight it holds the per-layer output shapes plus the last
committed boundary's activations, which scale with the calibration set, and each commit prunes
the set before it. Once this run finished, it came to **748 KB beside 1.4 TB of shards**.

Results
*******

.. list-table::
   :header-rows: 1

   * - ``Qwen/Qwen3.8-2.4T-A95B``
     -
   * - Hardware
     - 1 × GB300 (283 GB)
   * - Source checkpoint
     - 4.892 TB BF16, 213 shards
   * - Wall clock
     - 3 h 04 m (11,026 s), ≈ 120 s/layer
   * - Peak GPU
     - 141.7 GB against a 120 GB budget (mean utilization 33.8 %)
   * - Peak host RSS
     - 604.8 GB against a 600 GB budget
   * - Output
     - 1.444 TB (3.4 × smaller): 92 layer shards + tail + index. All 141,312 expert
       projections (92 × 512 × 3) carry their own ``input_scale`` and ``weight_scale``, so vLLM
       picks the FlashInfer TRT-LLM NVFP4 MoE kernel over the emulation fallback.

**The budgets only govern where weights go.** Both flags size ``accelerate``'s device map, so
activations, calibration buffers and the CUDA context sit on top of them. That's why GPU peaked
21.7 GB over budget, and host RSS, after plateauing near 600 GB as weights paged in from disk,
peaked 4.8 GB over. Leave headroom on both instead of sizing to the machine's exact capacity.

GPU memory rises as each layer's weights materialize and falls as they're released, with no
drift across all 92 layers. That flat envelope is why depth has no effect on peak memory, so
watch the first few layers of any new model: a floor that keeps rising means something is being
retained when it should be released.

The correctness claim is narrow: **per-layer export produces the same checkpoint as whole-model
export.** Across four models, the two paths matched tensor for tensor and config for config
(123,513 tensors on a 35B MoE, 74,163 on a 30B, 0 mismatches), every ``weight_map`` entry
resolved to the shard holding its tensor, and vLLM produced identical greedy generations from
both. A reduced model with this architecture matched too before the full run: 7,933 tensors, 0
mismatched, identical ``hf_quant_config.json`` and ``config.json``.

What you're trading
*******************

All of these come straight from the design. Whether they work for you depends on your
constraints.

**Time.** The layer walk is sequential by construction. Models in the 550–671B range take
roughly 40–47 minutes on one GPU, and the 2.4 T run is in *Results*. You're paying for smaller
hardware with wall clock, which is the whole idea, but on a large model you'll feel it. Batch
size is the main lever inside a run (see the note under *Running it*).

**Calibration algorithms are limited, for now.** Max, MSE and local Hessian, in FP8 and NVFP4,
only touch ``_amax``, so they run with ``calib_mutates_weights: false`` and keep resume state
small. GPTQ rewrites weights and runs layerwise with the default ``true``, which snapshots each
layer's full weights for resume. A mutated weight is written into its shard before the layer is
released, so per-layer export is shaped for GPTQ, but we haven't validated the pair yet, and a
PR doing that is on the way. The refused configurations are listed under *Running it*.

**Serving is a separate problem.** The checkpoint this produces still has to be served
somewhere, and this workflow can't tell you whether your hardware is up to that.

**Layer size sets the floor.** Peak memory here is a lower bound you can't tune away. The GPU
still has to hold one decoder layer plus its activations at once. A model built from many modest
layers is easy. If one layer is enormous, you've hit the limit of this design. The same
arithmetic tells you whether your model fits before you burn a session finding out: take the
largest decoder layer, add its calibration activations, and compare the total against your card.
