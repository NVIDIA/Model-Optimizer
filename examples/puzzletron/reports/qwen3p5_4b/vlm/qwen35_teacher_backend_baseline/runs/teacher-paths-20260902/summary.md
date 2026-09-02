# Qwen 3.5 4B teacher paths: 2026-09-02 record

Evidence status: `teacher_only`. This record contains one completed evaluation
per path on an identical ordered selection of 344 rows. It contains no student
result and no repeated measurement.

## Results

| Evaluation path | RealWorldQA (64) | MMMU (120) | MVBench (160) |
| --- | ---: | ---: | ---: |
| General vLLM adapter | 65.6250% | 35.8330% | 20.0000% |
| Qwen-specific Transformers adapter | 57.8125% | 35.8330% | 66.8750% |

The Qwen-specific adapter was also run on the same 64 RealWorldQA rows with an
explicit empty system message. That distinct prompt condition scored 57.8125%.
It is an ablation, not a second repetition of the default condition.

The two paths differ in evaluator revision, runtime adapter, prompt
serialization, and video timestamp handling. The numbers establish
path-specific teacher references; they do not measure an inference-engine-only
effect. A fixed `test:0` serialization probe produced 59 tokens for the general
path, 70 for the default Qwen-specific path, and 64 for its empty-system
ablation. The rendered prompts and token sequences were distinct.

## Scope and limitations

- The ordered evaluation contract is
  `qwen35-vlm-rwqa64-mmmu120-mvbench160-frozen-v1`.
- Each condition has one retained execution. Sampling was disabled, but one
  execution is not repetition evidence.
- Correct-count numerators, uncertainty estimates, exact model and execution
  revisions, the general path's manifest-byte hash, accelerator topology, and
  per-condition GPU-hours were not retained as first-class evidence.
- Raw evaluator exports remain external and unpublished; their checksums were
  not retained in the canonical handoff.
- The empty-system ablation is RealWorldQA-only and is not available as a
  checked-in versioned profile.

See [result_record.json](result_record.json) for provenance,
[metrics.csv](metrics.csv) for structured scores, and the campaign
[README](../../README.md) for the reproduction entrypoint.
