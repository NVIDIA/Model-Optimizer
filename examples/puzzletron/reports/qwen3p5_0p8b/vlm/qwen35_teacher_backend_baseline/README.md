# Qwen 3.5 0.8B VLM teacher evaluation paths

This campaign retains teacher-only results for two evaluation paths on
the same ordered 344 rows: the general `lmms-eval` vLLM adapter and the
Qwen-specific `qwen3_5` Transformers adapter. The paths use different evaluator
revisions and construct different model inputs, so the results do not isolate
the effect of the inference engine.

## Retained run

- [2026-09-02 teacher paths](runs/teacher-paths-20260902/summary.md): RealWorldQA
  64 rows, MMMU 120 rows, and MVBench 160 rows, plus a separate 64-row
  RealWorldQA prompt ablation for the Qwen-specific path.

The run leaf contains the numerical table, structured provenance, limitations,
and sanitized reproduction commands. Dataset and model files are downloaded or
mounted at run time; they are not stored in this repository.
