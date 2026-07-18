# Qwen3.6 FP8 granularity report workspace

This directory is the local, self-contained report workspace for the Qwen3.6-35B-A3B and
Qwen3.6-27B ModelOpt study. The remote experiment root is recorded in
`study_manifest.json`.

Regenerate the report after copying remote `results.json` artifacts beneath `results/`:

```bash
python workspaces/qwen36_fp8_granularity_study/scripts/render_report.py
```

The renderer deliberately shows a pending state instead of inventing values when no valid
result artifacts are present. It accepts nested result directories, validates the study schema
and BF16-reference cohort, and retains the source path for every artifact. Successful artifacts
must also pass complete role coverage and cross-candidate quantizer-owner-set checks. They populate
within-scope rankings, tokenwise metrics, equal-row bootstrap intervals, quantizer-MSE family
and highest-layer summaries, logical FP8 weight cost, and phase wall times.

The evaluation is a deterministic 32-row slice from the same staged CNN/DailyMail training split
used for calibration. It contains 31 unique text payloads because staged rows 1026 and 1029 have
identical content hashes; deduplicating that repeated text leaves every reported rank unchanged.
It is a controlled numerical screen, not a claim of broad task or corpus generalization. The W8A8
candidates are format-policy bundles: temporal scale policy is different between per-tensor and
dynamic block formats, and MXFP8 changes both block size and scale encoding. The W8A16 controls
remove activation quantization but do not eliminate the MXFP8 block-size/scale-encoding confound.

Run the renderer regression tests with:

```bash
python workspaces/qwen36_fp8_granularity_study/scripts/test_render_report.py
```
