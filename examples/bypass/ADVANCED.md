# Bypass selection, native placement, and extension

## Callable trimmed models

A callable original-weight prefix is a valid teacher. For an `nn.Sequential` of MLPs, trim before PTQ so the prefix owns normal ModelOpt conversion metadata:

```python
import copy
import modelopt.torch.quantization as mtq
import modelopt.torch.distill as mtd

teacher_prefix = teacher[:k]
student_prefix = mtq.quantize(copy.deepcopy(teacher_prefix), quant_cfg, forward_loop)
mtd.bypass(student_prefix, teacher_prefix, train_batches,
           {"modules": [str(i) for i in range(k)]})
```

A bare `nn.ModuleList` has no forward. Keep the callable HF root, original embeddings, position/mask construction, norms, and head. For a simple Llama-style prefix, retain `teacher.model.layers[:k]` inside the original root and align `config.num_hidden_layers` before PTQ. Preserve all relevant layer-pattern and source-layer configuration for hybrid/shared-attention models. Selecting `layer_indices=list(range(k))` on the original model avoids reconstructing those dependencies. Selection chooses repair boundaries; it does not promise to remove all other forward computation.

## Different formats in one MoE

Formats are properties of the existing quantizers. This Qwen3-Next-style PTQ fragment assigns the shared expert IQ2_S, fused routed gate/up IQ1_M, and routed down NVFP4:

```python
quant_cfg = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*mlp.shared_expert.*weight_quantizer",
         "cfg": {"num_bits": "iq2_s", "backend": "ggml"}},
        {"quantizer_name": "*mlp.experts.gate_up_proj_weight_quantizers*",
         "cfg": {"num_bits": "iq1_m", "backend": "ggml"}},
        {"quantizer_name": "*mlp.experts.down_proj_weight_quantizers*",
         "cfg": {"num_bits": (2, 1),
                 "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}},
    ],
    "algorithm": "max",
}
student = mtq.quantize(copy.deepcopy(teacher), quant_cfg, forward_loop)
mtd.bypass(student, teacher, train_batches)
```

Use the actual quantizer paths for your architecture. The NVFP4 down projection sends gradients to the IQ gate/up scales while its weights, codes, and quantization parameters stay fixed. A frozen projection uses a differentiable reference when its inference kernel lacks an input VJP. Different experts may also use different IQ formats. Grouped packed execution requires compatible banks; heterogeneous banks retain the native/reference computation.

## Caller-owned native parallelism

The caller launches workers, loads/places models, and supplies the actual Torch `DeviceMesh` or stable-5.23 `TransformersDeviceMesh`. Bypass does not interpret display-only topology settings or create allocations. Prepare undistributed native weights with `TransformersBypassAdapter.parallelize` before Bypass, or adopt exactly the same already prepared mesh. Preparing twice does not install duplicate hooks.

For TP=2, FSDP=2, EP=4 on four workers:

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.distributed.utils import TransformersDeviceMesh
from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
from examples.bypass.prepare_data import PackedBatchSequence

# Launch this program with torchrun --nproc-per-node=4.
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
mesh = TransformersDeviceMesh(
    init_device_mesh("cuda", (1, 1, 2, 2),
                     mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")),
    init_device_mesh("cuda", (1, 1, 1, 4),
                     mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")),
)
adapter = TransformersBypassAdapter(mesh)
# teacher and student are caller-loaded models; this recipe does not load weights.
adapter.parallelize(student)
adapter.parallelize(teacher)
context = adapter.context
train_batches = PackedBatchSequence(
    "packed-cascade", "train", global_batch_size=8, device="cuda",
    dp_size=context.dp_size, dp_rank=context.data_mesh.get_local_rank(),
)
mtd.bypass(student, teacher, train_batches,
           {"checkpoint_dir": "owner-checkpoints"}, device_mesh=mesh)
dist.destroy_process_group()
```

EP overlaps TP/FSDP: `efsdp * ep == fsdp * tp`. EP is not an independent GPU-count multiplier. Unique data lanes are `dp_replicate * fsdp`; TP, CP, and EP owners share the corresponding samples. The example has two unique data lanes, each consuming four of the eight global sequences per update. Batch sizes must divide the unique data lanes exactly; no sampler padding or repetition is added. Empty expert owners still participate in matching collectives.

The native repair path accepts the default `DistributedConfig.ep_implementation="auto"` or explicit `"all_to_all"` and forwards it to the installed dispatcher. Other transports are refused before placement: an unused expert owner must still participate in backward. Prepare independent teacher and student models before applying native placement.

Native CP is enabled only for declared, faithful causal SDPA paths when Torch exposes the required scoped API. Other cases run and report `replicated_full_context`. Packed documents and positional semantics remain intact. Logs and the private `_modelopt_bypass_execution` diagnostics identify actual CP mode, data lanes, local batch, and local projection call counts. PP greater than one is refused until a model-owned transport is validated.

The native validation matrix uses Torch 2.14/CUDA 13.4; portable core/math and CPU distributed checks also run with supported Torch 2.11. Check the exact tested topology in [MODEL_COVERAGE.md](MODEL_COVERAGE.md).

## Exact resume and finalized state

```python
config = {"checkpoint_dir": "owner-checkpoints", "checkpoint_interval": 128}
mtd.bypass(student, teacher, train_batches, config, device_mesh=mesh)
# Recreate the same original PTQ skeleton, teacher, data lanes, and placement.
mtd.bypass(student, teacher, train_batches,
           {**config, "resume_from": "owner-checkpoints"}, device_mesh=mesh)
```

Owner checkpoints commit all rank shards before advancing `latest`. They contain native scale masters, name-keyed Lion moments, RNG, numerical/data/topology identity, and the exact data cursor. Partial generations are refused. A replayable `Sequence` is content-checked; a stateful loader must provide `state_dict`, `load_state_dict`, and JSON-compatible `data_identity`. Numerical recipe, teacher/PTQ/data identity, and topology must match. Output paths may move.

Use a separate checkpoint directory for each active run. A failed transaction removes its own pending writes; a directory left by a killed process can be removed after that process exits. Loading uses only committed generations.

After Bypass returns, an unsharded student uses ordinary `mto.save`/`mto.restore` and supported ModelOpt export. Native owner checkpoints resume on the same original model skeleton and mesh, including from the final update. This example does not introduce a second distributed deployment/export format or claim consolidated HF export from a native EP owner shard.

## Add a model or IQ format

Ordinary text decoders usually need one `ModelSpec.bypass_spec` in their model package's `specs.py`: declare decoder paths, feed-forward attribute names, and document independence. Match the existing `BypassSpec` fields. Register metadata without importing optional modeling code. An unusual mixer or side input can provide a model-owned `ModuleBypassAdapter` subclass through `adapter_path`; the Nemotron-H adapter is an example. Declare task-owned post-head diagnostic math through `logits_transform_path`, a callable accepting `(model, logits)`. It receives the complete vocabulary per token chunk by default. Only elementwise transforms, such as a scale or tanh softcap, may set `logits_transform_slices=True`; a transform that reduces across vocabulary entries must keep the default. Keep exceptional math beside the model, with independent numerical and source-loading oracles.

An IQ codec owns its existing `GGMLFormat.payload_scales` and `replace_payload_scales` global-scale accessors. Its `scale_spec` adds paired local-field readers/writers, legal bounds, and multiplier/bias/divisor geometry; the format also declares integer-level decoding and block sizes. Register that descriptor in the existing GGML registry. The common scale policy then supplies FP32 masters, native rounding/projection, ownership, checkpointing, and finalization. Preserve code/additive bits and fixed signs. Extend the independent native-byte/gradient/roundtrip tests. Codec availability enables the correct reference; a packed kernel needs its own decoder, forward/input/scale VJP checks, and measured dispatch qualification.

## Kernel choices and measurements

All five IQ1/IQ2 layouts have native CUDA level decoding, fused scale reconstruction, packed dense forward/input/scale VJPs, and grouped expert tiles. The reference remains available for unsupported layouts, dtypes, devices, and heterogeneous banks. Immutable grouped pointer metadata is bounded. Expanded replay weights and scales projected for each forward are not persistently cached.

Packed payloads and FP32 scale masters for all selected boundaries remain on their preparation devices throughout training; their totals add across boundaries, including with offload or FSDP. Finalization runs one policy per boundary in sequence, retaining CPU rollback snapshots for that boundary until publication succeeds. Larger or custom MoE units need proportional host and device capacity; selectors or trimmed callable prefixes bound the selected scope.

Automatic packed dense dispatch is limited to measured GB300 BF16 IQ1_M/IQ2_XXS 2048×4096 matrices with at most eight tokens; grouped dispatch further requires a compatible four-expert bank. The default 32K path uses fused scale reconstruction and standard GEMM, which is faster in that regime. Fused valid-token MSE saves the original narrow activations instead of full FP32 differences. Measurements qualify individual operations, not full-model training throughput. See the validation evidence in the PR stack for time, peak memory, and first-call costs.

Measured on GB300, Torch 2.14/CUDA 13.4, with native VJP checks:

| Operation | Reference | Selected path | Memory observation |
| --- | ---: | ---: | --- |
| IQ1_M/IQ2_XXS dense 2048×4096, 1–8 tokens, forward/backward | 1.48–1.58 ms | 0.94–1.06 ms | 31–36 MiB less peak allocation |
| Four compatible experts at that width, 8 total tokens, balanced/skewed/empty routing | 5.21–5.40 ms | 2.91–3.03 ms | 162–169 MiB less peak allocation |
| Valid-token MSE forward/backward, 32K×4096 BF16 | 2.58 ms | 0.285 ms | 1.75 GiB less peak allocation |
| Native one-layer 4096-wide Llama diagnostic probe, 32K context and 32K vocabulary | 0.482 s | 0.655 s chunked | 4.95 to 2.59 GiB peak allocation |

The kernel rows use warm CUDA-event medians. First fused-MSE invocation took 0.584 s including compilation; CUDA extension builds are separate setup costs. The diagnostic row is a single invocation of each path on the same randomly initialized task and favors bounded memory. It is not a released-model throughput result. At 32K tokens, the tested packed dense projections took 26.8–136.9 ms versus 1.46–1.75 ms for materialized GEMM, so automatic dispatch keeps the materialized path there.
