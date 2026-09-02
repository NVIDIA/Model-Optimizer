# Qwen 3.5 4B VLM FFN-width KD-search run 2026-09-01-r2

Evidence status: `preliminary`.

This run completed the three-candidate screening path through 64-step TP2 KD
and the `qwen35-vlm-rwqa100-mmmu100-prefix100-repeat2-v1` evaluation contract.
The [structured record](result_record.json) is the source of truth for exact
provenance, identities, and hashes. The [metric table](metrics.csv) contains
the curated numerical observations.

## Results

| Checkpoint | Parameters | Reduction | Post-KD loss | Teacher top-1 agreement | RealWorldQA | MMMU |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 5,295,564,288 | baseline | - | - | 0.70 | 0.42 |
| FFN 7168 | 4,792,247,808 | 9.5045% | 0.275756 | 0.935247 | 0.74 | 0.33 |
| FFN 6144 | 4,540,589,568 | 14.2567% | 0.335212 | 0.928762 | 0.56 | 0.37 |
| FFN 5120 | 4,288,931,328 | 19.0090% | 0.238116 | 0.921556 | 0.71 | 0.29 |

The 9.5%-pruned student is the provisional first example candidate because it
has the strongest teacher-token agreement and the smallest combined absolute
downstream delta under this pinned screen. The 19.0% student remains a useful
higher-compression point, but its MMMU regression is larger.

The two downstream repetitions used the same first 100 rows per task with
deterministic decoding. Their identical metrics establish execution
reproducibility, not independent sampling evidence.

The heterogeneous and homogeneous MIP origins deduplicated to these same three
physical uniform-width students. This run therefore contains no
homogeneous-versus-heterogeneous quality comparison.

## Serving screen

Output-token throughput at concurrency four used eight requests, 100 requested
input tokens, 80 output tokens, and 1280 by 720 synthetic images.

| Student | 1 image | 4 images | 8 images |
|---|---:|---:|---:|
| FFN 7168 | 217.93 tok/s | 240.26 tok/s | 199.19 tok/s |
| FFN 6144 | 224.55 tok/s | 253.52 tok/s | 217.28 tok/s |
| FFN 5120 | 237.27 tok/s | 247.89 tok/s | 206.25 tok/s |

These are bounded screening measurements, not a production throughput claim.
The teacher was not measured under the same serving contract.

## Runtime

| Work | Wall time | Allocation |
|---|---:|---:|
| Three concurrent 64-step TP2 KD students | 7m35s | 6 GPUs |
| KD controller and report completion | 10m08s | 6-GPU worker plus CPU controller |
| Post-KD 32-sample loss screen | 4m05s | 3 GPUs |
| Post-KD bounded downstream evaluation | 14m09s | 3 GPUs |
| All retained attempts, diagnostics, and validation | - | 8.054 GPU-hours |

Each KD student used two tensor-parallel ranks, local and global batch size one,
a frozen vision tower, activation checkpointing, 64-token KD chunks, and the
uncapped 64-row multimodal dataset. Peak memory was 63.90, 63.19, and 62.48 GiB
per rank for the 9.5%, 14.3%, and 19.0% students.

## Limitations

- The configured 256-step finalist continuation was not run. It was an authored
  exploratory budget and has not been established as necessary or sufficient
  for convergence.
- RealWorldQA and MMMU cover only two task families and use the first 100 rows,
  not a randomized or coverage-preserving sample.
- Deterministic repetitions reuse the same rows and do not reduce sampling
  uncertainty. Per-item outputs and a frozen source-ID manifest were not
  retained in this repository package.
- The exact execution Git revision and accelerator model were not recorded in
  the runtime evidence. The checked-in configuration is a validated successor
  containing the executed campaign behavior; no reproduction revision is claimed.
- Single-GPU full-objective KD did not fit the observed 80 GiB memory envelope.
  TP2 completed without reducing batch size or image resolution.
- The initial thinking-enabled downstream profile and incomplete or failed KD
  attempts are superseded evidence and are excluded from the reported scores.
- External checkpoints and runtime artifacts are not published with this
  repository. The structured record retains opaque identities and hashes for
  the canonical summaries that support these results.

Reproduction instructions are in the [campaign guide](../../README.md).
