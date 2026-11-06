# Source and review map

[Return to the API guide](README.md). The review base is `origin/main` at
`8c614286acd653670fb7ee6e9b4fac4da62a1626`; historical measurements retain their
own runtime/source revisions.

| Owner | Entry points | Review focus |
|---|---|---|
| Repair API | `modelopt/torch/distill/config.py`, `local_distillation.py`, `local_units.py` | Once-resolved unit/quantizer preflight plan, valid-token objectives and independent unit optimizers |
| Shared runtime | `modelopt/torch/distill/local_parallel.py`, `local_runtime.py`, `local_checkpoint.py` | Typed RealizedMeshContext, physical gradient/statistic ownership, PP scheduling and exact resume |
| Native fields | `modelopt/torch/quantization/repair.py`, `repair_formats.py`, `repair_partition.py` | Legal native scale projection, bounded immutable constants, partition geometry and final writeback |
| Native kernels | `modelopt/torch/kernels/quantization/repair.py` | IQ decoding, grouped scale application, gradients and large-index addressing |
| HF adapter | `modelopt/torch/distill/plugins/transformers/adapter.py`, `families/`, `packing.py`, `output_head.py` | Semantic units, conservative document coalescing, output-head contracts and per-forward side state |
| HF distributed | `parallel.py`, `context_parallel.py`, `functional_fsdp.py`, `checkpoint.py` in that plugin | Native HF plans, CP execution modes, callable FSDP owners and topology-native DCP |
| Text extensions | `modelopt/torch/models/{deepseek_v41,kimi_k31,step5,mimo_v2}/` | Published architecture math, native precision, checkpoint reconstruction and cache semantics |
| Example tools | `prepare_data.py`, `evaluate.py`, `benchmark_*.py` | Unique splits, chunked full-vocabulary KLD, projection modes and explicit benchmark topology |

Start with native format/partition invariants, then the shared local runner,
then HF placement and family behavior. Model-specific math belongs in the model;
family descriptors describe ownership and replay rather than replacing forward
kernels. Optional formats register their repair/checkpoint contracts explicitly.

Training uses hidden-output MSE; whole-model initializer/result assessment uses
held-out teacher KLD. Inspect exact token counts and `context_parallel_modes`
before interpreting a metric. Native checkpoint restore preserves topology and
fields; deployment conversion has separate fusion/layout restrictions.

[Validation](VALIDATION.md) lists regression coverage and executed checks;
[Results](RESULTS.md) distinguishes current compatibility from older timings.
Keep operational snapshots, cluster scripts, full run ledgers and credentials
outside the distributable example.
