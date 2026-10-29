# Automatic AutoQuant execution (experimental)

Supply an ordinary AutoQuant recipe and execution limits. ModelOpt resolves the
recipe's algorithms and grouping, generates independent format calibration/scoring
workers, and joins them with one coordinator. Users do not define stages.

This preview supports gradient weight search with max, MSE, imatrix and GGML GPTQ.
Layerwise max, imatrix and GGML GPTQ use source-activation search estimates followed
by final calibration of the selected configuration. BF16 creates no calibration
worker. Layerwise MSE, KV search, disk-offloaded/meta weights and multi-GPU worker
computation are rejected. Existing serial APIs and manual workers keep their defaults.

## Generate and inspect a launch

Run the generator in an environment with ModelOpt installed, from the repository
root. The resulting YAML uses the existing launcher; generation submits no jobs.

```bash
python tools/launcher/autoquant.py --config request.yaml --output generated.yaml
```

Example `request.yaml` for a small model (these limits are illustrative, not a
capacity recommendation for Qwen3.6-35B-A3B):

```yaml
recipe: general/auto_quantize/nvfp4_fp8_at_5p4bits
model: /shared/source-model
dataset: /shared/calibration.jsonl
work_dir: /shared/autoquant-artifacts
export_path: /shared/selected-model
calib_size: 512
calib_seq: 512
batch_size: 1
slurm_config:
  _factory_: slurm_factory
  time: '00:30:00'
resources:
  max_concurrent_formats: 2
  max_total_gpus: 2
  allocated_gpus_per_worker: 1
  gpu_bytes: 8589934592
  host_bytes: 34359738368
  input_bytes: 1048576
  candidate_bytes: 1073741824
  checkpoint_bytes: 4294967296
  metadata_bytes: 1048576
  export_bytes: 2147483648
  artifact_storage_bytes: 17179869184
  retries: 1
```

Submit the generated YAML through your configured ModelOpt launcher. Source,
input and output paths must be available inside its container mounts. Each worker
requests one node, one rank and `allocated_gpus_per_worker` GPUs; its Slurm step
binds one GPU for computation. Count whole-node reservations in this limit: a
four-GPU exclusive node requires `allocated_gpus_per_worker: 4`, so two concurrent
workers need `max_total_gpus: 8`. Generation rejects an insufficient total budget
or conflicting Slurm GPU requests. Host memory is requested through Slurm.
The PyTorch allocator ceiling excludes allocations made by external CUDA libraries.
Verify actual allocation sizes on clusters that schedule whole nodes.
When Slurm reports a larger allocation than declared, workers refuse before
creating workspace metadata or loading weights.

## Numerical and recovery contract

Each worker loads the same source and snapshots the same calibration/scoring
inputs. It calibrates its assigned format and the fixed reference configuration,
then reuses the existing gradient scorer with only candidate replay restricted.
Configurable reference groups remain unquantized. The coordinator merges raw
contributions before distributed reduction and monotonic score adjustment; the
existing solver selects one complete configuration.

Artifacts bind source tensors, inputs, callbacks, implementation, numerical
settings, candidate identities and resolved group mappings. Packed tensors are
preserved exactly. Incomplete, conflicting or incompatible results are refused.
Completed format artifacts are immutable and reused on retry; at most three
retries are allowed. A failed format does not prevent independent remaining
formats from running, but the coordinator refuses incomplete coverage.

Each work directory belongs to one search/input/resource contract. Budgets and
export destinations may vary while completed statistics remain reusable. Process
leases bound active workers across repeated submissions and exclude overlapping
coordinators before model loading. Redundant Slurm submissions can still allocate
GPUs before runtime refusal, so allocation caps apply to each generated launch.
All retained recipe and assignment metadata shares the explicit `metadata_bytes`
ceiling; conflicting metadata is refused without replacement.

`artifact_storage_bytes` bounds pipeline-owned regular-file payloads: candidate
artifacts and their publication headroom, two shared checkpoint generations,
metadata, and every retained export reservation. Each export reserves
`export_bytes` before model loading; changed budgets can reuse statistics but
require room for another retained output. Reservations persist after failure and
are reused on retry. This conservative accounting does not automatically release
a reservation when someone deletes its output. Source checkpoints, installed
packages, launcher logs and external dataset/kernel caches are outside this bound.

The existing exporter uses bounded shard and metadata writers for this opt-in path.
The byte limit covers cumulative writes, including copied source files and
conservative safetensors header allowances. `metadata_bytes` is reserved inside
each export for its manifest. A private stage is published only after export and
payload hashing complete; a retry verifies the assignment and all file digests
before reusing an existing output. Failed stages are removed, and interrupted
stages are reclaimed under the destination lease. Unmanaged exports are refused.

The coordinator owns the shared checkpoint lock and publishes by atomic generation
replacement. Per-artifact byte limits apply during reads and writes. Interrupted
managed generations are reclaimed under the destination lock. A frozen assignment
records each group's complete algorithm and settings; changing only the target bit
budget reuses validated calibration and sensitivity statistics.

### Layerwise recipes

For supported layerwise recipes, `source_activation_v1` captures original-source
layer inputs and applies each candidate's original algorithm and settings. This is
an explicit search approximation: it does not include the eventual selected
quantized prefix. GPTQ remains GPTQ; it is never silently replaced by imatrix.
Reference and per-group scoring still use the existing gradient scorer.

After the coordinator freezes the complete assignment, it releases the search
model and reloads original weights. Non-layerwise groups reuse their accepted
states; layerwise groups receive final calibration through the native sequential
traversal, preserving their algorithm chains and activation policy. Incompatible
traversal settings are refused. Enabled quantizers must lie within frozen decoder
groups; each group must belong to one decoder layer.

Layerwise requests additionally require explicit byte limits:

```yaml
resources:
  # Include the other limits from the request above.
  activation_bytes: 1073741824
  final_checkpoint_bytes: 4294967296
```

The activation limit covers simultaneously cached adjacent layers and must fit
both host and GPU limits. The final checkpoint reservation counts toward
`artifact_storage_bytes`; increase that total accordingly. The pipeline owns final
checkpoint/export paths, so recipe-level layerwise `checkpoint_dir` and `export_dir`
must be unset. Source weights must be contiguous and unshared on one device.

Completed final-calibration layers resume only with matching assignment, source,
inputs and implementation. Their exact packed payloads are retained. The one owned
final checkpoint is reclaimed only after checking the exported assignment and file
hashes. Interrupted reclamation can resume; incomplete exports do not authorize
deletion. Held-out KL is validation, not selection or reranking.

## Current validation limits

CPU/example checks and tiny-model GPU tests establish scoped correctness.
Representative-layer measurements do not establish full-model quality or a
repeatable concurrency speedup. The bounded exporter
currently supports a single-process transformers model without offload, FSDP,
layerwise export or a ModelOpt-state sidecar. Limits describe managed pipeline
writes, not a filesystem quota protecting against external writers. Use a dedicated
study area and retain the existing experiments and recovery checkpoints.
