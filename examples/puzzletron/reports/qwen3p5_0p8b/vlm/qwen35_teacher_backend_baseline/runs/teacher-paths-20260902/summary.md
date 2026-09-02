# Qwen 3.5 0.8B teacher paths: 2026-09-02 record

Evidence status: `teacher_only`. This record contains one completed evaluation
per path on an identical ordered selection of 344 rows. It contains no student
result and no repeated measurement. The exact upstream model revision was not
retained, so the run is not reproducible from the published record.

## Results

| Evaluation path | RealWorldQA (64) | MMMU (120) | MVBench (160) |
| --- | ---: | ---: | ---: |
| General vLLM adapter | 51.5625% | 25.0000% | 48.1250% |
| Qwen-specific Transformers adapter | 40.6250% | 25.0000% | 50.6250% |

The Qwen-specific adapter was also run on the same 64 RealWorldQA rows with an
explicit empty system message. That distinct prompt condition scored 48.4375%.
It is an ablation, not a second repetition of the default condition.

The two paths differ in evaluator revision, runtime adapter, prompt
serialization, and video timestamp handling. The numbers establish
path-specific teacher references; they do not measure an inference-engine-only
effect. A fixed `test:0` serialization probe produced 59 tokens for the general
path, 70 for the default Qwen-specific path, and 64 for its empty-system
ablation. The rendered prompts and token sequences were distinct.

## Scope and limitations

- The ordered evaluation contract is
  `qwen35-vlm-short-v1-exact-rows-v1`.
- Each condition has one retained execution. Sampling was disabled, but one
  execution is not repetition evidence.
- Correct-count numerators, uncertainty estimates, exact model and execution
  revisions, the general path's manifest-byte hash, accelerator topology, and
  per-condition GPU-hours were not retained as first-class evidence.
- Raw evaluator exports remain external and unpublished. Only the default
  Qwen-specific export has a retained checksum in this record.
- The empty-system ablation is RealWorldQA-only and is not available as a
  checked-in versioned profile.

See [result_record.json](result_record.json) for provenance,
[metrics.csv](metrics.csv) for structured scores, and the campaign
[README](../../README.md) for the reproduction entrypoint.
