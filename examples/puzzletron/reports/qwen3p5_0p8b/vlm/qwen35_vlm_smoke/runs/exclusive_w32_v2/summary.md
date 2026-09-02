# Historical pre-KD FFN-2048 comparison

Evidence status: `preliminary`

This run measured the Qwen 3.5 0.8B teacher and a pre-distillation FFN-2048
student. It is retained as historical evidence from an earlier campaign policy
that measured serving before KD. The maintained campaign now measures only its
selected final post-KD student.

## Quality

The evaluation covered the first 100 RealWorldQA rows and the first 100 MMMU
validation rows. Each task was repeated twice with deterministic generation.

| Model | Checkpoint stage | RealWorldQA | MMMU |
| --- | --- | ---: | ---: |
| Teacher | Teacher | 60% | 35% |
| FFN-2048 student | Pre-KD | 19% | 24% |

## Serving

Each cell used 32 warmup requests on one exclusive eight-H100 node. Teacher and
student placement was swapped so both models ran on every GPU. The reported
difference is the average of eight same-GPU comparisons. The comparisons came
from one node allocation, and the models used different synthetic request
streams.

| Requests | Images | Concurrency | Teacher tok/s | Student tok/s | Student difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 1 | 93.78 | 94.98 | +1.28% |
| 64 | 1 | 4 | 348.18 | 353.27 | +1.48% |
| 64 | 6 | 1 | 69.11 | 69.91 | +1.17% |
| 64 | 6 | 4 | 199.79 | 204.52 | +2.42% |
| 64 | 12 | 1 | 50.22 | 50.55 | +0.67% |
| 64 | 12 | 4 | 134.95 | 135.04 | +0.07% |
| 256 | 6 | 1 | 66.61 | 67.87 | +1.94% |
| 256 | 6 | 4 | 194.18 | 199.05 | +2.53% |

## Scope and limitations

- The student uses FFN intermediate width 2048 instead of the teacher width
  3584 in every language-model layer. Parameter counts were not retained.
- The quality scores cover fixed 100-row prefixes, not the complete benchmarks.
- The two deterministic repetitions reused the same rows.
- All serving measurements came from one node allocation; they are not
  independent experiment repetitions.
- Teacher and student serving measurements used different synthetic request
  streams.
- The 256-request measurements cover only the six-image workload.
- Serving recorded checkpoint paths but not content hashes. Later quality-run
  hashes cannot prove that the checkpoint bytes were unchanged between runs.

The structured evidence is in [result_record.json](result_record.json), summary
metrics are in [metrics.csv](metrics.csv), and the 128 serving observations are
in [observations.csv](observations.csv). [recipe.json](recipe.json) records the
serving setup. The exact launcher for this historical study was not retained.
