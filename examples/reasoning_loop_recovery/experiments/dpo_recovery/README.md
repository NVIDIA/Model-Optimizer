# DPO recovery

Test whether direct preference optimization (DPO) reduces repetitive reasoning after iterative
recovery QAD on **Qwen3.6-35B-A3B**, without anti-repetition instructions or penalties at evaluation.

The starting policy is the **round-3 iterative recovery QAD checkpoint**, not the original
500-iteration QAD checkpoint used by the [prompt-7 experiment](../prompt7_recovery_qad/README.md).
Its cumulative recovery training used 2,181 examples. DPO keeps fake quantizers active for the
trainable policy and uses the same starting weights for its frozen reference policy. In this
implementation, quantization is temporarily disabled during the frozen-reference forward pass.
It does not use logits from the original BF16 teacher.

## Data, training, and evaluation

- [Data](data/README.md): 115 preferred/rejected pairs, with 104 for training and 11 for diagnostic
  pair validation. The 11 held-out SciCode subtasks are separate from pair validation.
- [Training](training/README.md): 13 DPO steps on four GB300 GPUs, objective, settings, and export.
- [Seven environment paths](training/paths/README.md): preserve all alternatives, including their
  failures, compatibility fixes, validation milestones, and remaining work.
- [Compatibility](training/compatibility/README.md): checkpoint layout, model construction,
  quantization state, attention, and memory fixes.
- [Evaluation](evaluation/README.md): 55 SciCode subtasks × 10 attempts with the original prompt.

The YAML files record experiment settings and source provenance; they are not standalone launcher
inputs. Dataset/checkpoint paths use placeholders for an external workspace. Generated responses,
checkpoints, container images, and experiment-local driver implementations are not included.
Source filenames and checksums are preserved in the [source manifest](training/compatibility/source_manifest.yaml).

## Results

The completed DPO evaluation below is for **path 4: NeMo RL source in the original QAD container**.
It does not establish that the other paths produce equivalent checkpoints. No final environment
choice is made here; all seven paths remain documented.

### Comparison with earlier QAD variants

Same **55 SciCode subtasks × 10 attempts per model**, without extra prompting or penalties.
The DPO row is the completed path-4 evaluation; the other rows are the earlier QAD comparisons.

| Model | Cap: train /440 | Cap: held-out /110 | Correct: train /440 | Correct: held-out /110 |
| --- | ---: | ---: | ---: | ---: |
| Initial QAD | 41 (9.32%) | 12 (10.91%) | 57 | 6 |
| Non-iterative QAD — 2,181 examples | 25 (5.68%) | 6 (5.45%) | 57 | 5 |
| Iterative QAD — round 3 | 17 (3.86%) | 0 (0%) | 53 | 4 |
| **Round 3 + DPO** | **0 (0%)** | **0 (0%)** | **65** | **5** |

Non-iterative and round-3 iterative QAD each used 2,181 cumulative recovery examples. DPO then
used **104 training preference pairs** over 13 steps. Zero observed cap hits still needs confirmation
with more fresh-seed attempts. The following tables give additional metrics for the direct
round-3-versus-DPO comparison.

### Training subtasks: 44 × 10 attempts

| Model | Cap hits | No final | Correct | Mean tokens |
| --- | ---: | ---: | ---: | ---: |
| Iterative QAD, before DPO | 17/440 (3.86%) | 17/440 | 53/440 | 17,090 |
| Iterative QAD + DPO, path 4 | 0/440 (0%) | 0/440 | 65/440 | 7,470 |

### Held-out subtasks: 11 × 10 attempts

| Model | Cap hits | No final | Correct | Mean tokens |
| --- | ---: | ---: | ---: | ---: |
| Iterative QAD, before DPO | 0/110 (0%) | 0/110 | 4/110 | 11,736 |
| Iterative QAD + DPO, path 4 | 0/110 (0%) | 0/110 | 5/110 | 8,667 |

See [evaluation/results.yaml](evaluation/results.yaml) for exact counts and token statistics.
Zero observed cap hits does not establish a zero underlying failure rate. The held-out baseline
already had zero cap hits; this comparison cannot demonstrate a reduction there. Accuracy changes
are descriptive, not an established improvement. These selected failure-prone subtasks do not
represent the full SciCode distribution, and related subtasks can share a parent problem.
