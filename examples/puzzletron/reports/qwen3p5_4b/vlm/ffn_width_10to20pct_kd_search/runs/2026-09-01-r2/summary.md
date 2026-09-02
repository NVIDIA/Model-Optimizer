# Qwen 3.5 4B VLM FFN-width KD-search run 2026-09-01-r2

Status: early comparison (`preliminary`). These results help choose what to test
next; they do not show that pruning preserves quality across VLM tasks.

This run created three smaller models, measured their serving speed, trained
each one for 64 KD steps on two GPUs, and evaluated them on RealWorldQA and
MMMU. The [structured record](result_record.json) contains exact configuration
and artifact details. The [metric table](metrics.csv) contains the recorded
measurements.

## Results

| Checkpoint | Parameters | Reduction | Post-KD loss | Teacher top-1 agreement | RealWorldQA | MMMU |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 5,295,564,288 | baseline | - | - | 0.70 | 0.42 |
| FFN 7168 | 4,792,247,808 | 9.5045% | 0.275756 | 0.935247 | 0.74 | 0.33 |
| FFN 6144 | 4,540,589,568 | 14.2567% | 0.335212 | 0.928762 | 0.56 | 0.37 |
| FFN 5120 | 4,288,931,328 | 19.0090% | 0.238116 | 0.921556 | 0.71 | 0.29 |

FFN 7168 is the first student to test further. It agreed most often with the
teacher's token choices and stayed closest to the teacher across these two
benchmarks. FFN 5120 removes more parameters, but its MMMU score fell further.

## How to read the scores

- We evaluated the first 100 examples from RealWorldQA and the first 100 from
  MMMU. The examples were not randomly sampled.
- We ran those same examples twice and got the same scores. This checks that the
  evaluator repeats its result; it does not test more examples.
- We did not save the exact example IDs, so a future run cannot prove that it
  used the identical set.
- The teacher scores came from an earlier evaluation. We did not run the planned
  final teacher and student comparison, and we do not have the exact teacher
  checkpoint fingerprint or runtime settings.

The search produced each of the three FFN widths in two different ways, but the
resulting model shapes were identical. There are three distinct students, not
six, and this run cannot compare the two search methods.

## Serving check

We sent eight synthetic requests, four at a time. Each request asked for 100
input tokens and 80 output tokens and used 1280 by 720 images.

| Student | 1 image | 4 images | 8 images |
|---|---:|---:|---:|
| FFN 7168 | 217.93 tok/s | 240.26 tok/s | 199.19 tok/s |
| FFN 6144 | 224.55 tok/s | 253.52 tok/s | 217.28 tok/s |
| FFN 5120 | 237.27 tok/s | 247.89 tok/s | 206.25 tok/s |

This is a small comparison of the three students, not a production performance
benchmark. We did not measure the teacher with the same request settings.

## Runtime

| Work | Wall time | Allocation |
|---|---:|---:|
| Three concurrent 64-step TP2 KD students | 7m35s | 6 GPUs |
| KD controller and report completion | 10m08s | 6-GPU worker plus CPU controller |
| Post-KD 32-sample loss screen | 4m05s | 3 GPUs |
| Post-KD RealWorldQA/MMMU check | 14m09s | 3 GPUs |
| All runs, including retries | - | 8.054 GPU-hours |

Each student used two GPUs, batch size one, and 64 training rows. We left the
vision tower unchanged and used activation checkpointing and 64-token KD chunks
to reduce memory use. Peak memory was 63.90, 63.19, and 62.48 GiB per GPU for
the 9.5%, 14.3%, and 19.0% students. We did not retain the total number of
examples or tokens processed or the optimizer history.

## Limitations

- The planned 256-step KD run was not run. It would start again from the selected
  model before KD; the 64-step results only choose which model to use. The 256
  steps were an experiment budget, not a proven training requirement.
- We did not record the exact source revision or GPU model used for this run.
  The checked-in recipe behaves the same in the tested paths, but it is newer
  than the code that launched the experiment.
- The full KD loss ran out of memory on one 80 GiB GPU. Splitting each student
  across two GPUs worked without reducing the batch size or image resolution.
- The reported scores exclude the first evaluation with thinking enabled and
  any incomplete or failed KD attempts.
- The trained checkpoints and raw run files are stored outside this repository.
  The structured record includes their identifiers and hashes.

Reproduction instructions are in the [campaign guide](../../README.md).
