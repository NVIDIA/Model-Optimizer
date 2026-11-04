.. _local-quantization-distillation:

Local quantization distillation (Bypass)
=======================================

Bypass repairs an existing ModelOpt PTQ model by independently matching each
selected unit's output to the original teacher. One teacher forward supplies
inputs and targets; each student unit replays the teacher input and immediately
backpropagates its own mean squared error. All selected units train in that same
pass, without caching a corpus of hidden states or backpropagating through the
whole student.

The default unit is a complete quantized MLP/MoE, including routing and expert
combination. Normalization outside the selected module is outside its loss.
Explicit, disjoint module paths can select attention, recurrent mixers or whole
decoder blocks. Attention masks, document resets and recurrent state belong to
the replay contract. The objective includes every valid token, including prompts;
it does not use label masking or a logit-distillation loss.

Starting from PTQ
-----------------

Load the original teacher and its ModelOpt-quantized student, then call:

.. code-block:: python

    import modelopt.torch.distill as mtd
    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

    result = mtd.local_distill(
        student,
        teacher,
        packed_train_batches,
        {"updates": 512, "global_batch_size": 8, "micro_batch_size": 1},
        validation_dataloader=packed_validation_batches,
        adapter=AutoModelLocalDistillationAdapter(mesh_context),
    )
    repaired = result.student

The result uses the final update by default. Best-validation selection is an
explicit option. Mixed formats remain local to their existing quantizers: gate/up
may use IQ2_S while down uses NVFP4, for example. Enabled activation quantizers
execute during replay and retain their original calibration by default, including
W4A4 models.

Native scales and modes
-----------------------

Scale-only is the default. It freezes element codes, signs and lookup catalogs;
positive FP32 log-scale parameters project to legal native scale fields on every
forward. Rounding uses a straight-through gradient. Finalization stores those
native fields and removes the temporary training parameters. There is no added
inference branch. Weight-only and joint/code-changing modes are explicit options
with narrower coverage.

Native ModelOpt/AutoModel save and restore preserve the repaired fields.
Deployment export can apply its usual fusion, resmoothing and requantization;
it does not promise identical element codes after those transformations.
Measure teacher KLD again on the converted artifact before reporting serving
quality.

.. list-table:: Current native-field coverage
   :header-rows: 1
   :widths: 35 25 20 20

   * - Weight family
     - Frozen-code scales
     - Weight-only
     - Joint / re-quantization
   * - Static integer, including static affine
     - Supported
     - Supported
     - Supported
   * - Dynamic tensorwise integer
     - Learned static native threshold
     - Original dynamic rule
     - Learned static threshold
   * - Fake GGML IQ1/IQ2/Q8, FP8, NVFP4, MXFP4/6/8, MXINT8
     - Supported native fields
     - Original quantizer with weight STE
     - Not implemented
   * - Compressed native QTensor
     - Supported native fields
     - Not implemented
     - Not implemented

Read-only preflight reports unsupported quantizers before training setup. Active
LSQ, unknown backends, rotate-back weight transforms and floating affine weight
fields require another adapter. Sequential casts retain their original downstream
forward; grouped weights use their physical members. Dynamic activation rules
remain dynamic. Optional activation tuning currently supports calibrated static
integer thresholds without TP; other activation formats remain frozen.

Optional external backends register a repair factory with
``register_repair_backend`` and load their native tensor provider before restore.
Their codecs, catalogs, calibration algorithms and deployment policies remain
owned by the provider; they are not bundled into ModelOpt.

Execution and checkpointing
---------------------------

The AutoModel adapter uses realized EP, CP, TP and FSDP2 meshes rather than
assuming that each rank represents independent data. Packed-document masks and
position resets are preserved. Physical non-interleaved PP uses a forward-only
teacher schedule and stage-local student replay; empty stages still participate.
Interleaved PP is refused. The tutorial CLI and diagnostics currently use PP=1.

Checkpoints commit every rank's new state before retiring the previous generation.
Resume restores optimizer state, scheduler position, RNG and the consumed unique
data prefix. Native DCP publication/restore requires a matching model skeleton and
topology. Deployment export retains the target exporter's format/layout limits;
a native checkpoint round trip does not establish GGUF or HF conversion for a
new architecture. Packed real-weight TP repartitioning and TP activation tuning
are not supported.

Tutorial and validation
-----------------------

The `GLM Flash GPTQ IQ2_S tutorial
<https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/local_quantization_distillation>`_
prepares unique, conversation-disjoint Nemotron Post-Training Dataset v2 packs,
produces GPTQ PTQ, repairs it, and compares held-out PPL and full-vocabulary
KL(teacher || candidate). Its explicit defaults are 512 updates, eight global
32K packs, Adagrad, scale LR 0.01, 5% warmup/cosine decay and saves every 128
updates. These are starting values, not universal optimums.

Short native GPU checks cover actual format gradients and reloads, EP/FSDP/TP/CP
reference updates, packed GLM attention and decoder boundaries, Mamba W4A4 and
physical PP checkpoint/resume. They use tiny random models. Full-model quality,
throughput and memory claims require separate measurement. Repair-call process
GPU-hours and Slurm allocation GPU-hours are reported separately.

The tutorial's ``MODEL_COVERAGE.md`` records the top 15 distinct open-weight
Artificial Analysis checkpoints, exact native architecture matches and retained
layer prefixes. Ten are supported by the validated AutoModel checkout. The
offline native matrix also covers all 18 identified official generative Qwen
3.5/3.6/3.8 checkpoints, including dense/MoE variants, base models, and the
Flash-Next registry alias. Unsupported architectures are excluded explicitly;
these random-model checks establish execution and serialization, not quality.

Native AutoModel expert exposure is reported as routed-assignment counts, with
unobserved expert IDs listed explicitly. These counts use valid training tokens
and carry their observed update range through resume. They describe exposure,
not a quality guarantee for individual experts.

Initializer and repaired-model selection use held-out full-vocabulary
KL(teacher || candidate). Surrogate-corpus PPL is secondary; an overflowing PPL
is recorded as null without discarding finite cross entropy or KLD.
