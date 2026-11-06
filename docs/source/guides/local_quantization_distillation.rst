.. _local-quantization-distillation:

Local quantization distillation (Bypass)
=======================================

Bypass repairs an existing ModelOpt PTQ model by independently matching each
selected unit's output to the original teacher. One teacher forward supplies
inputs and targets; each student unit replays the teacher input and immediately
backpropagates its own valid-token mean squared error. Units train in the same
pass without storing a corpus of hidden states or backpropagating through
neighboring units.

The Transformers adapter selects quantized text MLP/MoE units by default. Exact,
disjoint module paths or semantic roles can select attention, recurrent mixers
or whole decoders. Normalization outside a selected module is outside its loss.
Prompts contribute alongside all other valid tokens; repair does not optimize
language-model logits.
An unselected output head is bypassed during repair while the normal root
lifecycle remains intact. Selecting the head retains all-token targets;
calibration keeps the real head forward.

Starting from PTQ
-----------------

Keep the original teacher and the independently quantized student on matching
architectures and placement. Then call:

.. code-block:: python

    import modelopt.torch.distill as mtd
    from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
    from examples.local_quantization_distillation.evaluate import evaluate_pair

    adapter = TransformersLocalDistillationAdapter()
    before = evaluate_pair(teacher, student, diagnostic_batches, adapter=adapter)
    result = mtd.local_distill(
        student, teacher, packed_train_batches,
        {"updates": 512, "sequence_length": 32768,
         "global_batch_size": 8, "micro_batch_size": 1},
        validation_dataloader=packed_validation_batches,
        adapter=adapter,
    )
    after = evaluate_pair(teacher, result.student, diagnostic_batches, adapter=adapter)

Use re-iterable calibration and diagnostic data and unique training batches.
Keep calibration, repair, MSE validation and whole-model diagnostic conversations
disjoint. The example's ``prepare_data`` tool writes deterministic document IDs,
reset positions, shifted targets and a provenance manifest without repeating
source token intervals. ``PackedBatchSequence`` slices over unique data lanes
without epoch wrap or sampler padding.
Supported Qwen hybrid and Nemotron H families can coalesce up to eight
independent documents within the original token budget, preserving MRoPE,
labels and valid-token counts. Calibration, custom masks, unknown metadata and
context-coupled activation statistics keep serial document forwards.

Scale-only repair freezes codes, signs and catalogs and updates legal native
scale fields through FP32 log-scale masters. Enabled activation quantizers retain
their calibrated forward. Finalization writes back the fields and removes
temporary training state. Mixed formats keep their individual quantizers;
unsupported format/layout/mode combinations fail preflight.

Diagnostics and evidence
------------------------

Use held-out full-vocabulary KL(teacher || candidate) to select PTQ initializers
and assess whole-model fidelity. PPL on a surrogate corpus is secondary. Local
repair and its validation loader still use hidden-output MSE; the default
retained state is the final update. Best-per-unit MSE selection is explicit.

``evaluate_pair`` requires PP=1 and full-context batches. It captures actual
head inputs and computes exact full-vocabulary statistics with defaults
``token_chunk=128`` and ``vocabulary_chunk=8192``. ``projection_modes`` reports
``chunked_linear`` for ordinary vocabulary-sliced Linear heads or
``chunked_head`` for real custom/quantized/TP head forwards on token chunks.
These paths avoid full-sequence vocabulary tensors while retaining final hidden
states; logits-only models without an exposed head report ``full_logits``.
Muse/Mistral post-projection transforms remain family-owned. Heads with
context-coupled activation statistics require a family projector and are refused
by the generic chunked path.

KL includes every valid input position; NLL/PPL use already shifted
within-document targets. All ranks need matching outer batch schedules; the
adapter coordinates exhaustion/errors and rejects unequal schedules. An
overflowing PPL is null without discarding finite KLD.

The November 6, 2026 compatibility matrix completed 58 reduced random-model
cases on pinned Transformers 5.23/Torch 2.14 with IQ2_XXS or NVFP4 W4A4. It
establishes short numerical/ownership compatibility, not pretrained quality.
Muse Spark NVFP4 uses an explicit 2.5% relative scalar bound, with 1.20% observed
variation; other scalar and all vector relative-L2 bounds remain 0.3%.

Distributed execution and persistence
-------------------------------------

The caller owns process launch and matching teacher/student placement. Native
HF TP/EP/FSDP uses ``TransformersDeviceMesh``: dense ``(pp, fsdp, tp)`` and expert
``(pp, efsdp, ep)`` views contain the same ranks and satisfy
``efsdp * ep == fsdp * tp``. EP overlaps dense ranks; it does not create another
independent data axis. Calibrate PTQ before applying the supported native plans.

Compatible half-precision full-causal SDPA can use scoped native CP. Packed,
recurrent, sparse/compressed, custom-mask and other incompatible paths use
``replicated_full_context``. This fallback retains full forward/replay work on
each CP rank and masks loss ownership; it saves neither activation memory nor
forward compute. All 58 release-matrix cases used this fallback. Inspect
``execution.context_parallel_modes`` before making efficiency claims.

``output_dir`` publishes a topology-native Torch DCP model checkpoint. All ranks
restore with ``adapter.restore(matching_ptq_skeleton, directory)``. Exact repair
resume uses ``checkpoint_dir`` and ``resume_from`` to restore optimizer,
scheduler, RNG and consumed data position. These artifacts are distinct from
HF/GGUF deployment export, whose fusion/resmoothing may change codes; assess the
converted artifact again with teacher KLD.

Tutorial and benchmark
----------------------

The `Transformers API guide
<https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/local_quantization_distillation>`_
contains GPTQ initialization, packed-data preparation, diagnostics, native meshes
and checkpoint examples. The broad family baseline is pinned Transformers
5.23.0.dev0 at ``084b1fd1b9983dca2ce12441aec4478371b9fe8f``. Differentiable
V4.1, Kimi K3.1 and Step 5 text extensions and the MiMo V2.6 bridge retain their
architecture-specific math and native precision boundaries. Floating source
import validates keys/layouts, and native reload archives original packed-format
metadata rather than reactivating the source quantizer. MiMo preserves FP32
routing bias under native BF16 construction. Raw Kimi MXFP4, MiMo MXFP4/FP8
and Step5 FP8 tensors remain refused; Step5 also needs its source activation
rounding. Bounded source-layout tests do not establish pretrained quality. Text support excludes vision/audio and
draft execution.

``benchmark_transformers`` measures random Qwen3 or Moonlight prefixes in an
existing allocation. It writes physical layouts, execution modes and timing
metadata without downloading pretrained weights or writing model checkpoints.
Prefix speed/memory and tiny-model tests do not establish full-model throughput,
realistic peak memory or downstream quality. Four-rank regression tests require
an explicit distributed launch and are not exercised by ordinary single-GPU
pytest collection.
