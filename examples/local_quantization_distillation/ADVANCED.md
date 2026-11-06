# Advanced Transformers repair configuration

[Return to the API guide](README.md).

## Native fields and modes

The core `mtd.local_distill` API repairs an existing ModelOpt PTQ student; PTQ
remains a separate step. Each quantizer keeps its own format, so a unit may mix
integer, IQ, FP8, NVFP4 and MX weight representations. Calibration and native
code/scale block geometry remain part of the model contract.

| Weight representation | Scale-only | Weight-only | Joint / requantization |
|---|---|---|---|
| Static integer fake quantization, including affine bias | Native fields | Supported | Supported |
| Dynamic tensorwise integer | Learned static weight threshold | Original dynamic rule | Learned threshold |
| Fake GGML IQ, FP8, NVFP4 and MX formats | Supported native fields | Original quantizer with weight STE | Format adapter required |
| Compressed native QTensor | Supported native fields | Format adapter required | Format adapter required |

Scale-only freezes codes, signs and catalogs. FP32 log-scale masters project to
legal native fields in each forward with straight-through gradients through
rounding. Finalization writes back those fields and removes temporary training
state. Deployment fusion/resmoothing may change codes; reassess the converted
artifact with held-out KLD.

Enabled activation quantizers execute during replay with their original
calibration. `tune_activations=True` additionally requires an explicit
`activation_lr` and supported calibrated static integer thresholds; TP/EP have
further capability guards. Unsupported combinations fail preflight rather than
silently changing format or recalibrating shards.

Grouped reconstruction retains at most 256 KiB on a group's first repair state,
for immutable metadata or small packed payloads that fit. Payload/placement
changes invalidate reuse. Trainable reconstructions, expanded weights and
gathered parameters are never cached. [Component measurements](RESULTS.md)
preserve exact values/gradients; they do not establish a full repair speedup.

## Native distributed placement

The caller launches processes and constructs the meshes. For four ranks, this
example overlaps dense TP2/FSDP2 with expert EP4:

```python
from torch.distributed.device_mesh import init_device_mesh
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.utils import TransformersDeviceMesh
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter

dense = init_device_mesh("cuda", (1, 2, 2), mesh_dim_names=("pp", "fsdp", "tp"))
expert = init_device_mesh("cuda", (1, 1, 4), mesh_dim_names=("pp", "efsdp", "ep"))
adapter = TransformersLocalDistillationAdapter(TransformersDeviceMesh(dense, expert))
for model in (teacher, student):  # PTQ already calibrated identically on all ranks
    model.config.distributed_config = DistributedConfig(tp_size=2, fsdp_size=2, ep_size=4)
    adapter.parallelize(model)
dp_size = dense["fsdp"].size()
dp_rank = dense["fsdp"].get_local_rank()
```

Dense and expert meshes contain the same ranks and satisfy
`efsdp * ep == fsdp * tp`; EP is a multiple of TP. Native HF experts are sharded
over the expert axis rather than internally tensor-sharded. EP therefore does
not multiply the number of unique samples. Feed the same batch to TP/EP peers;
`fsdp` identifies unique data lanes. Frozen base weights need no DDP reducer;
the repair runtime reduces trainable fields according to their physical owners.

For combined CP/EP, use dense `(pp, cp, fsdp, tp)` and expert
`(pp, cp, efsdp, ep)` roots with matching ordered CP peer groups. Dense
statistics/gradients span `cp * fsdp`; expert replicas span `cp * efsdp`.
Read `execution.context_parallel_modes`: `native_sdpa` uses chronological
sequence shards, while `replicated_full_context` retains full inputs/compute on
every CP rank and masks loss ownership. The latter provides no forward-compute
or activation-memory savings.

Torch `DeviceMesh` is also supported for compatible DP/CP/TP/PP arrangements,
including data-axis aliases `fsdp` and HSDP `dp_replicate`/`dp_shard`. The newer
HF EP interface requires `TransformersDeviceMesh`. Native `DistributedConfig`
rejects FSDP+PP and dispatch-EP+PP. Whole-model diagnostics require PP=1 and
matching outer dataloader schedules on all ranks; the adapter rejects unequal
exhaustion before entering model collectives.

Physical TP partitions must preserve complete native code/scale blocks,
activation statistics and transforms. Output quantization that needs unsupported
post-reduction execution is refused. For a custom feature-sharded repair unit,
the family can override `objective_tp_mesh`; ordinary column-sharded Linear
units already use a global feature denominator and count tokens once.

## Model publication and exact resume

`output_dir` publishes a topology-native Torch DCP model checkpoint. Every rank
participates; immutable files have integrity metadata and an atomic commit
manifest. Restore using `adapter.restore(model, directory)` into the matching
converted graph, modes, aliases and physical topology. Committed directories
may be relocated. This is distinct from a consolidated HF/GGUF deployment export.

Use `checkpoint_dir`, `checkpoint_interval` and `resume_from` for interrupted
repair. Resume restores per-unit optimizer state, scheduler position, RNG and
the consumed unique data prefix. Recreate the same teacher, PTQ initializer,
data order and topology before resuming:

```python
repair = mtd.LocalDistillationConfig(
    updates=512, global_batch_size=8, micro_batch_size=1,
    checkpoint_dir="/path/to/resume", checkpoint_interval=128,
    resume_from="/path/to/resume",
)
```

`selection="final"` retains the last update. The explicit
`best_validation_per_unit` option uses local validation MSE; whole-model KLD
selection remains the caller's responsibility.

## Custom families and optional formats

For a custom HF text root, pass a family descriptor explicitly:

```python
from modelopt.torch.distill.plugins.transformers import TransformersFamilyAdapter

family = TransformersFamilyAdapter(
    name="custom_text", decoder_paths=("model.text_decoder.layers",),
)
adapter = TransformersLocalDistillationAdapter(family_adapter=family)
```

Override `unit_attributes`, `document_batches` and `snapshot_unit_state` for
semantic boundaries, isolation and temporary routing state. `batch_packed_documents`
is a family opt-in to conservative text-document coalescing, not permission to
batch arbitrary masks or metadata. `output_head` identifies the actual task head;
`transform_logits` preserves post-projection math such as Muse/Mistral transforms.
Context-coupled head activation statistics require a specialized diagnostic path and are
explicitly refused by the generic chunked evaluator. Model forward math stays
in the model. `functional_children`
describes child weights read directly by their parent; native FSDP lifecycle
wrapping keeps gather/backward/reshard inside the owning calls.
`additional_fsdp_children` can separate called projections from an FP32 parent.
Preserve native per-module dtypes: blanket BF16 conversion or FSDP mixed
precision can round required FP32 recurrence/router/compression parameters.

Optional format providers explicitly register a factory with
`register_repair_backend` and their native checkpoint tensor identity. Import
the provider before PTQ or restore. TP partition support must be declared and
implemented by that provider. No external provider is needed for built-in formats.
