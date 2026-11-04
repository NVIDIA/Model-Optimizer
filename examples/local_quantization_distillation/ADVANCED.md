# Local quantization distillation: advanced configuration

[Return to the tutorial](README.md).

## Existing PTQ models and mixed formats

The core entry point does not run or replace PTQ:

```python
import modelopt.torch.distill as mtd
from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

result = mtd.local_distill(
    student, teacher, packed_train_batches,
    {"updates": 512, "sequence_length": 32768, "global_batch_size": 8,
     "micro_batch_size": 1, "checkpoint_dir": resume_directory},
    validation_dataloader=packed_validation_batches,
    adapter=AutoModelLocalDistillationAdapter(existing_mesh_context),
)
```

`student` is an existing ModelOpt PTQ model/checkpoint, including a mixture of
formats. Each enabled quantizer uses its own native field adapter; the repair
does not force an entire unit into one format. For example, change only the
down-projection rule in the YAML to use IQ2_S gate/up and NVFP4 down:

```yaml
- quantizer_name: '*mlp.experts.down_projs_weight_quantizers.*'
  cfg:
    num_bits: [2, 1]
    block_sizes: {-1: 16, type: dynamic, scale_bits: [4, 3]}
```

For W4A4, configure the corresponding enabled **input** quantizers with the
chosen activation format during PTQ. Repair executes these activation quantizers
in the actual student unit, preserving their calibration by default. Disabling
them would optimize a different model. `tune_activations: true` is a separate
opt-in with an explicit activation learning rate, subject to native adapter
capabilities.

Scale-only freezes element codes, signs and lookup tables. FP32 log-scale masters
project to each format's legal native scale representation in the forward pass,
using STE through rounding. Finalization stores legal native fields and removes
repair modules. IQ local multipliers remain valid format values; fixed catalog
entries are not optimized. Native ModelOpt save/restore preserves codes. GGUF/HF
deployment export still depends on that exporter's format/layout support.

Full HF deployment export retains its existing fusion and resmoothing steps.
These can fold pre-quant scales into neighboring weights and quantize those
weights again. A direct per-weight export reuses a valid native payload, but a
complete deployment conversion does not promise identical codes or predictions.
The native ModelOpt/AutoModel checkpoint is the fidelity-preserving repair
artifact. Recheck KLD on the converted artifact before reporting deployment
quality or choosing it for serving.

Optional `mode: weight_only` or `joint`, `requantize: true`, activation tuning and
best-state selection are explicit settings. Unsupported combinations fail during
preflight before training mutates the model. They are not silently converted to
another format. The acceptance matrix below describes the checked combinations;
other format, layout and topology combinations retain their native restrictions.

The current native-field mode coverage is:

| Weight representation | Scale-only (frozen codes) | Weight-only | Joint / scale re-quantization |
|---|---|---|---|
| Static integer fake quantization, including static affine bias | Yes | Yes | Yes |
| Dynamic tensorwise integer weights | Yes; learned thresholds are stored as native static weight fields | Original dynamic rule | Yes; learned thresholds are stored |
| Fake GGML IQ, FP8, NVFP4 and MX formats in the validated matrix | Yes | Original native quantizer with weight STE | Adapter still required |
| Compressed native QTensor weights | Yes | Adapter still required | Adapter still required |

Activation tuning currently accepts calibrated static integer thresholds with an
explicit LR and without TP. Other activation formats execute their original
forward with frozen calibration. Weight scale repair does not make activation
codes constant. For scale projection, the reported LR applies to FP32 log-scale
masters; each forward uses the format's representable values.

## Boundaries and distributed execution

Default discovery picks quantized outer `mlp`/`moe` modules and skips fully
unquantized units. Set `units` to exact, disjoint paths for attention, GDN, Mamba,
whole decoder blocks, or architecture-specific boundaries. A custom adapter can
provide output selection and recurrent-state ownership. Shared trainable fields
must have one owner; overlapping units are rejected.

Native family adapters resolve semantic `unit_types` through AutoModel's own
decoder discovery, including `ffn` on DeepSeek V4.1, `linear_attn` for Qwen GDN,
and `mixer` for Nemotron Mamba. For example, `unit_types: [gdn]` selects native
Qwen GDN blocks, and `unit_types: [decoder]` selects complete decoder blocks.
Native GDN's fused convolution honors weight and static input quantization;
its output quantization needs an unfused adapter. Dynamic convolution activation
thresholds with CP are currently refused.

The AutoModel adapter reuses its native mesh, dispatch, CP sharder and FSDP owner
unsharding. It supports physical non-interleaved PP via the existing teacher
`AutoPipeline`:

```python
adapter = AutoModelLocalDistillationAdapter(
    mesh_context, teacher_pipeline=teacher_pipeline,
)
result = mtd.local_distill(
    student_pipeline.info.model_parts[0], teacher_pipeline.info.model_parts[0],
    packed_train_batches, config, adapter=adapter,
)
```

PP finishes the current forward-only teacher microbatch, then replays its
stage-local records with immediate backward, as in Puzzletron. Empty embedding/head stages still participate in execution and
checkpoint transactions. Virtual/interleaved schedules are refused. The simple
GPTQ/diagnostic CLI above uses PP=1. Native topology-preserving DCP checkpoints
are not topology-changing deployment exports.

## Focused validation

The development checks use tiny native models, not a full GLM quality run:

- Native integer, IQ2_S, NVFP4 W4A4, MXFP4/6/8 and MXINT8: nonzero scale gradients
  and exact native checkpoint round trips.
- Native expert-parallel mixed-format gradients/updates against an unsharded
  reference; FSDP2, TP2 and packed CP2 repair/reference comparisons.
- Native PP2 with uneven ownership and an embedding-only stage; interrupted
  all-rank checkpoint resume and native DCP restore into a fresh skeleton.
- Unique packed-data cursor and paired full-vocabulary diagnostics/masking.

See the development run ledger for measured short-probe results. Full-model
quality, throughput and memory measurements remain separate experiments.

[Native model coverage](MODEL_COVERAGE.md) lists the ranked-model and Qwen
compatibility probes, native family boundaries, and implementation limits.

### Reviewing this feature

[REVIEW_MAP.md](REVIEW_MAP.md) covers the complete implementation against main,
including shared quantization prerequisites and the optional AutoModel integration.
Training reports include per-native-expert routed-assignment counts and explicitly
list unobserved experts. Counts exclude padding and validation, reduce over unique
data/context lanes before EP gather, and are accumulated in resume state. Their
first/last observed update indices identify partial coverage after resuming an older
checkpoint that did not record these counters. Routed counts describe exposure;
they do not guarantee each expert improved.

A secondary PPL that exceeds the floating-point range is reported as `null`;
finite mean cross entropy and KLD remain available and KLD selection continues.

## Native family execution

`distill/plugins/automodel/families/` follows the per-family descriptor pattern
used by ModelOpt model specs and Puzzletron AnyModel. A small registry resolves
the native implementation; each family owns semantic boundaries, packed inputs,
CP setup and optional per-forward replay state. Adding a family does not require
editing a central model-name switch. The registry is backend-specific: HF model
specs remain architecture data shared by other ModelOpt subsystems.

Each descriptor can select `unit_runner`. The default native fast runner preserves
the original model's router, shared/latent experts, activation/clipping and dispatch,
while batching compatible IQ expert scale work. IQ1_M/IQ2_S/IQ2_XS immutable levels
use a native GPU decoder; heterogeneous representations retain their original
quantizers. CPU and unavailable-kernel paths retain native reference behavior.
No decoded weight cache persists between repair units. Correctness precedes speed:
family-specific runners must match native outputs, scale gradients and routing.

DeepSeek-V4 hash/vision routing has temporary token metadata outside the MoE's
ordinary arguments. Its descriptor snapshots that state before the teacher gate
consumes it and restores the corresponding state around student replay. Kimi's
latent projections and SiTU remain native; they are not replaced by generic SwiGLU.

Repair teacher forwards set `logits_to_keep=1` when the explicit forward signature
supports it. All decoder hidden tokens still train; only unused terminal logits
shrink. PP terminal metadata follows that shape and is restored afterward.
Standalone PPL/KLD diagnostics retain full token logits.

## Optional external format plugins

ModelOpt does not import optional format providers automatically. Import the
provider before quantization or checkpoint restore. Providers may register a
quantization backend/GPTQ helper through the existing APIs and a native scale
repair factory through `register_repair_backend`. Native tensor subclasses register
for pinned-payload restoration; names must be unique. TP ownership must be declared
and implemented explicitly. No external provider is required for the GLM IQ2_S
tutorial or the built-in format compatibility suite.
