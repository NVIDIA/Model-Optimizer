# Qwen 3.5 0.8B VLM KD learning curve

Status: successful measurement run, superseded selection policy.

Six student checkpoints were evaluated on the same frozen 344-row manifest (64
RealWorldQA rows, 120 MMMU validation rows, and 160 MVBench rows) before KD and
after 64, 128, and 256 cumulative KD steps. The FFN-3328 control was
additionally evaluated at 512 steps. Teacher evaluation was performed once and
reused at every comparison milestone. The semantic evaluation contract is
`qwen35-vlm-rwqa64-mmmu120-mvbench160-frozen-v1`; the runtime profile is
`qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v1`.
The exact 344-row selection is bundled as `row_manifest.json`; its byte hash
and the runtime's semantic row-selection identity are recorded separately in
`result_record.json`.

## Teacher and 256-step students

| Model | Cohort | RealWorldQA | MMMU | MVBench reported | MVBench fixed-160 |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher | Reference | 0.5625 | 0.29167 | 41.875% | N/A |
| `architecture_2a343aec3100a1b4` | Retained-95, hidden 960, depth 0 | 0.578125 | 0.325 | 37.5% | 37.5% |
| `architecture_f01974e675dc335b` | Retained-95, hidden 960, depth 1 | 0.546875 | 0.30833 | 35.625% | 35.625% |
| `architecture_abbb430e18472a62` | Retained-90 exploratory | 0.515625 | 0.275 | 30.267857% | 30.0% |
| `architecture_60ef4164a42c59b3` | Retained-85 exploratory | 0.46875 | 0.24167 | 13.839286% | 13.75% |
| `architecture_9f4b0f4244aaca1c` | Exact FFN-3328 control | 0.671875 | 0.35 | 43.125% | 43.125% |
| `architecture_05357c1774af922e` | Exact FFN-3072 control | 0.625 | 0.28333 | 40.0% | 40.0% |

At 256 steps, FFN-3328 is the only student above the teacher on all three
evaluator-reported metrics. Its separately gated 512-step scores are 0.65625
RealWorldQA, 0.325 MMMU, and 42.5% MVBench macro, so this fixed 344-row
evaluation does not support extending it past 256 steps.

The evaluator-reported MVBench macro excludes empty generations from affected
leaf denominators. `mvbench_audit.csv` records all 20 leaves for each retained
post-KD evaluation and recomputes empty generations as incorrect over the fixed
160 selected rows. At 256 steps, the retained-90 student has 7 empty responses
and the retained-85 student has 2; the other four students have none. Retained
raw per-leaf samples were not available for the teacher or pre-KD evaluations,
so those entries remain reported aggregates and have no fixed-160 value.

## KD exposure

All trajectories start from their own identity-bound pre-KD checkpoint and
preserve optimizer state across milestones. Global batch size is 4. Cumulative
examples are 256, 512, and 1,024 at 64, 128, and 256 steps. The 512-step
extension has 2,048 cumulative examples. The runtime counter reports 352,028,
704,056, 1,408,112, and 2,816,224 effective tokens at those milestones.

Actual cumulative KD GPU-hours at 256 steps range from 0.183 to 0.351 per
candidate. Detailed per-candidate exposure is in `result_record.json`.
Effective-token accounting follows the training objective's runtime counter and
can exceed a simple examples-times-sequence-length estimate.

## Executed selection protocol

The historical run used a bespoke prototype:

1. Generate up to eight MIP solutions per retained band with minimum Hamming
   distance 2.
2. Deduplicate exact architectures and keep 24 by MIP objective.
3. Screen those candidates with 64 image-text LM-loss samples.
4. Retain eight by LM loss with a minimum of two from each retained-95,
   retained-90, and retained-85 band.
5. Materialize and observe serving performance.
6. Retain four by a weighted mean rank using LM-loss weight 1.0 and throughput
   weight 0.25, with at least one candidate from each band.

This policy was campaign-specific and was not justified as a Puzzletron
default. The throughput term did not change the final four because the band
quota dominated. The integrated recipe in this change removes these heuristics,
so a reproduction with the corrected recipe can select different students.

## Reproduction and limitations

Use the integrated campaign configuration for the corrected, supported
selection flow. Exact replay of the historical selection requires archived
runtime source evidence; it is not represented by the corrected repository
recipe.

The comparison is one deterministic evaluation of the fixed 344-row selection,
without repeated sampling or confidence intervals. The separate eight-task
all-rows scope (`qwen35-vlm-judge-free8-all-rows-v1`) was not run. Attention and
GDN were fixed at teacher geometry, and this result is not evidence for a
full-axis campaign. The frozen row manifest is included. External runtime
manifests and checkpoints are not included; their content hashes provide the
immutable source-evidence join.

See the [structured record](result_record.json), [tidy metrics](metrics.csv),
[frozen row manifest](row_manifest.json), and [MVBench denominator
audit](mvbench_audit.csv). Campaign-level reproduction guidance is in the
[campaign README](../../README.md).
