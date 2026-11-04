# Review map: local quantization distillation

The comparison base is `origin/main` at `7d442467504cdae155812baf15242628d96971b5`.
Include untracked files when reviewing this working tree. Nothing has been committed
or published. The AutoModel dependency has no working-tree modifications.

The feature has four dependency-ordered review areas. These are review slices,
not separate user APIs or separate copies of the training loop.

| Area | Main entry points | Why it changes |
|---|---|---|
| Native quantization prerequisites | `quantization/model_calib.py`, `mode.py`, `model_quant.py`, `utils/core_utils.py`, `plugins/{automodel,huggingface}.py` | Native expert calibration, distributed statistics and FSDP-owned weight access. Preserve activation calibration and expert routing. |
| Native format fields and persistence | `quantization/{repair,repair_formats}.py`, `qtensor/`, `ggml/scale_fields.py`, `utils/{native_weight,weight_source}.py`, `export/`, `backends/nvfp4_gemm.py` | Train legal format fields while keeping frozen codes and native checkpoint semantics. Optional external formats register their own provider. |
| Generic local distillation | `distill/{config,local_distillation,local_units,local_checkpoint,layerwise_distillation_model}.py` | Shared streaming teacher capture/replay, independent unit optimizers, masking, validation and transactional resume. No AutoModel imports in the generic runner. |
| AutoModel and tutorial | `distill/plugins/automodel/`, this example directory and the public RST guide | Optional model-family selection, packed inputs, EP/CP/TP/FSDP2/PP ownership, native DCP and reproducible end-to-end recipes. |

## AutoModel integration

The public import remains
`modelopt.torch.distill.plugins.automodel.AutoModelLocalDistillationAdapter`.
Everything specific to AutoModel execution lives in its plugin package:

- `adapter.py`: interface and orchestration.
- `families/`: registered family-owned specs, packing, transport and replay-state hooks.
- `runners.py`: shared native fast replay, replaceable by a family descriptor.
- `packing.py`: common repair/diagnostic packing and context-parallel preparation.
- `runtime.py` and `parallel.py`: objective, gradient and native parameter ownership.
- `pipeline.py`: physical-stage scheduling and replay.
- `checkpoint.py`: native distributed state and publishing.

The quantization plugin is separate because PTQ also needs native expert projection
ownership and calibration, before any Bypass runner exists. General format and
FSDP fixes stay in their existing owners rather than being hidden in model adaptors.

## Selection and evidence

Held-out full-vocabulary KL(teacher || candidate) selects the PTQ initializer and
retained repaired checkpoint. PPL/CE on this surrogate dataset is secondary.
The repair objective remains local hidden-output MSE on all valid packed tokens.
Native ModelOpt saves preserve the repaired format fields; deployment conversion
can perform its existing fusion/resmoothing and should be checked again with KL.

`VALIDATION.md` separates tiny native/distributed correctness checks from pretrained
model measurements. Random-prefix tests establish compatibility, not model quality.
The completed MiMo experiment used a frozen source snapshot. Its pretrained-model
results are recorded separately in `RESULTS.md`.

## Publishing boundaries

Keep operational run manifests, source snapshots, cluster batch scripts and secrets
and internal development notes outside the package. Review shared-core/format
prerequisites separately before the higher-level runner. Do not describe the entire
working tree as a tutorial-only change.

## Performance and optional providers

The generic runner owns no model-name switch. AutoModel families own temporary
routing state and can replace unit execution inside the existing FSDP ownership
window. The default native MoE path batches homogeneous expert fields and uses
one-launch native IQ1_M/IQ2_S/IQ2_XS level decoding plus differentiable block scales.
Other quantizers retain their native execution. Performance evidence must distinguish
full-width component replay from end-to-end model training.

External codecs are registered, not embedded: `register_repair_backend` chooses
native repair state and TP capability, and native tensor subclasses register their
checkpoint identity. Built-in examples and tests use the formats supplied by
ModelOpt; optional providers are imported explicitly by their callers.

## Adapter-owned native writeback

Each weight repair state implements `finalize_(quantizer, weight)`; activation
states implement `finalize_(quantizer)`. The policy resolves current native
weight bindings, delegates writeback, and removes temporary state. Format states
own legal field projection, weight-coordinate folding and native payload pinning.
External providers inherit or override this contract without public type dispatch.
Both grouped and individual IQ quantizers share native IQ1_M/IQ2_S/IQ2_XS level
decoding during native replay; CPU, missing-Triton and other-codec fallbacks remain.
