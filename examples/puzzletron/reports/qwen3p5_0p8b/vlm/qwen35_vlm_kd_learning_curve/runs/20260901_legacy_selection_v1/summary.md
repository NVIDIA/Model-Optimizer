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

## Models

| Label | Checkpoint | Cohort |
| --- | --- | --- |
| A | `architecture_2a343aec3100a1b4` | Retained-95, hidden 960, depth 0 |
| B | `architecture_f01974e675dc335b` | Retained-95, hidden 960, depth 1 |
| C | `architecture_abbb430e18472a62` | Retained-90 exploratory |
| D | `architecture_60ef4164a42c59b3` | Retained-85 exploratory |
| E | `architecture_9f4b0f4244aaca1c` | Exact FFN-3328 control |
| F | `architecture_05357c1774af922e` | Exact FFN-3072 control |

## Quality measurements

| Model | KD steps | RealWorldQA | MMMU | MVBench reported | MVBench fixed-160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher | N/A | 0.5625 | 0.29167 | 41.875% | N/A |
| A | Pre-KD | 0.4375 | 0.225 | 25.625% | N/A |
| A | 64 | 0.59375 | 0.29167 | 40.625% | 40.625% |
| A | 128 | 0.59375 | 0.325 | 41.25% | 41.25% |
| A | 256 | 0.578125 | 0.325 | 37.5% | 37.5% |
| B | Pre-KD | 0.296875 | 0.25 | 24.375% | N/A |
| B | 64 | 0.625 | 0.30833 | 35.970238% | 31.875% |
| B | 128 | 0.59375 | 0.34167 | 34.065476% | 32.5% |
| B | 256 | 0.546875 | 0.30833 | 35.625% | 35.625% |
| C | Pre-KD | 0.125 | 0.26667 | 13.75% | N/A |
| C | 64 | 0.5 | 0.21667 | 28.47619% | 18.125% |
| C | 128 | 0.484375 | 0.30833 | 32.642857% | 31.25% |
| C | 256 | 0.515625 | 0.275 | 30.267857% | 30.0% |
| D | Pre-KD | 0.109375 | 0.25 | 10.625% | N/A |
| D | 64 | 0.484375 | 0.34167 | 10.375% | 10.0% |
| D | 128 | 0.453125 | 0.25 | 16.339286% | 16.25% |
| D | 256 | 0.46875 | 0.24167 | 13.839286% | 13.75% |
| E | Pre-KD | 0.5625 | 0.28333 | 32.5% | N/A |
| E | 64 | 0.625 | 0.325 | 39.375% | 39.375% |
| E | 128 | 0.640625 | 0.325 | 44.375% | 44.375% |
| E | 256 | 0.671875 | 0.35 | 43.125% | 43.125% |
| E | 512 | 0.65625 | 0.325 | 42.5% | 42.5% |
| F | Pre-KD | 0.5625 | 0.24167 | 28.75% | N/A |
| F | 64 | 0.546875 | 0.29167 | 42.5% | 42.5% |
| F | 128 | 0.609375 | 0.28333 | 39.375% | 39.375% |
| F | 256 | 0.625 | 0.28333 | 40.0% | 40.0% |

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

Actual cumulative KD GPU-hours were recorded per checkpoint:

| Model | 64 steps | 128 steps | 256 steps | 512 steps |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0873 | 0.1752 | 0.2930 | N/A |
| B | 0.0798 | 0.1159 | 0.1829 | N/A |
| C | 0.0871 | 0.1748 | 0.2925 | N/A |
| D | 0.0828 | 0.1184 | 0.1853 | N/A |
| E | 0.0897 | 0.2318 | 0.3516 | 0.5383 |
| F | 0.0900 | 0.2320 | 0.3509 | N/A |

The effective-token values are the training objective's runtime counters.

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

This policy was campaign-specific and is not part of Puzzletron's standard
candidate selection. The integrated recipe does not reproduce this historical
shortlist policy.

## Reproduction and limitations

Use the integrated campaign configuration for the corrected selection flow.
Exact replay of the historical selection requires archived runtime source
evidence; it is not represented by the corrected repository recipe.

The comparison is one deterministic evaluation of the fixed 344-row selection,
without repeated sampling or confidence intervals. The separate eight-task
all-rows scope (`qwen35-vlm-judge-free8-all-rows-v1`) was not run. Attention and
GDN remained at teacher geometry. The frozen row manifest is included. External
runtime manifests and checkpoints are not included; their content hashes are
recorded in `result_record.json`.

See the [structured record](result_record.json), [tidy metrics](metrics.csv),
[frozen row manifest](row_manifest.json), and [MVBench denominator
audit](mvbench_audit.csv). Campaign-level reproduction guidance is in the
[campaign README](../../README.md).
