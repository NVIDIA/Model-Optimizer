# DoGE data-blend tuning for distillation

This experimental workflow uses DoGE-style gradient alignment to tune data-blend weights for
Megatron-Bridge distillation. Use it to run a short proxy distillation experiment, update candidate
source weights against a fixed target blend, and decide which fixed blend is worth validating with a
normal `distill.py` run.

## How DoGE works

[DoGE](https://arxiv.org/abs/2310.15393) is a bilevel data-weighting method. The inner loop trains
the student on source datasets; the outer loop updates the source blend weights using a target-data
signal.

At each DoGE step in this PoC:

1. Draw one batch from each candidate source and one batch from the target blend.
2. Compute one KD gradient per source batch and one KD gradient for the target batch.
3. Score each source by gradient alignment with the target gradient. This PoC uses cosine
   similarity.
4. Increase the relative weight of better-aligned sources and decrease the relative weight of
   weaker or opposed sources with a normalized exponentiated update.
5. Return the weighted source KD loss to Megatron-Bridge so the normal optimizer step updates the
   student.

The output is a weight trajectory in `doge_weights.jsonl`. Treat the learned weights as a hypothesis:
validate them with a standard fixed-blend distillation run and compare against fixed-blend baselines.

## Workflow

1. Prepare tokenized Megatron datasets for each candidate source.
2. Choose a target objective as Megatron `WEIGHT PATH` pairs, for example a held-out validation
   blend.
3. Run `examples/megatron_bridge/doge_distill.py` with initial source weights in `--data_paths` and
   the fixed target objective in `--target_data_paths`.
4. Inspect `doge_weights.jsonl` under `--output_dir`.
5. Run a normal `examples/megatron_bridge/distill.py` job with the selected fixed weights and the
   same target validation blend.

Example:

```bash
torchrun --nproc_per_node 8 examples/megatron_bridge/doge_distill.py \
    --tp_size 8 \
    --teacher_hf_path /path/to/teacher_hf \
    --student_hf_path /path/to/pruned_student_hf \
    --data_paths \
        0.05 /data/tokenized_wikitext_text_document \
        0.05 /data/tokenized_math_text_document \
        0.90 /data/tokenized_stem_text_document \
    --target_data_paths \
        0.50 /data/target_wikitext_text_document \
        0.25 /data/target_math_text_document \
        0.25 /data/target_stem_text_document \
    --seq_length 4096 \
    --mbs 1 \
    --gbs 1 \
    --train_iters 2448 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_iters 120 \
    --eval_interval 400 \
    --eval_iters 256 \
    --log_interval 80 \
    --doge_meta_lr 0.1 \
    --output_dir /path/to/doge_run
```

`doge_weights.jsonl` records the trajectory:

```json
{"iteration": 100, "alignment_scores": {"...": 0.12}, "blend_weights": {"...": 0.54}}
```

Use the learned weights as a hypothesis, not as a final result. The final comparison should come
from a standard `distill.py` run with fixed weights and the same validation setup as the baselines.

## Runtime compared with normal distillation

DoGE is slower per training step because each step computes gradient-alignment probes before the
normal weighted source update. With three source datasets, one target blend, Qwen3-8B teacher,
Qwen3-8B 0.7x pruned student, sequence length 4096, `gbs=mbs=1`, and eight H100 80GB GPUs, the
observed times were:

| Workflow | Approx. step time | Relative speed |
|---|---:|---:|
| Normal `distill.py` | 0.28-0.38 s/iter | 1.0x |
| `doge_distill.py` | 1.7 s/iter | about 5-6x slower |

This overhead is expected for the PoC. DoGE performs extra target/source gradient calculations to
update blend weights, while normal distillation only performs the training update.

## Qwen3-8B 0.7x PoC result

The initial PoC tuned a three-source blend for a Qwen3-8B teacher and Qwen3-8B 0.7x pruned
student. The target validation blend was fixed at 50% WikiText, 25% Nemotron-v2 math, and 25%
Nemotron-v2 stem. All runs below used 2448 iterations, `gbs=mbs=1`, `eval_iters=256`, sequence
length 4096, and eight H100 80GB GPUs.

The DoGE run started from 5% WikiText, 5% math, and 90% stem. It converged to approximately
99.877% WikiText, 0.123% math, and 0.00035% stem. Those learned weights were then evaluated by
running normal `distill.py` with the fixed learned blend.

The following trajectory was read from `doge_weights.jsonl`. Values are blend weights in percent.

| Iteration | WikiText | Nemotron-v2 math | Nemotron-v2 stem |
|---:|---:|---:|---:|
| 0 | 5.000 | 5.000 | 90.000 |
| 1 | 5.305 | 4.989 | 89.706 |
| 100 | 35.506 | 6.169 | 58.325 |
| 500 | 98.995 | 0.366 | 0.639 |
| 1000 | 99.983 | 0.009 | 0.007 |
| 1500 | 99.997 | 0.003 | 0.001 |
| 2000 | 99.992 | 0.008 | 0.000 |
| 2448 | 99.877 | 0.123 | 0.000 |

| Run | Training blend | Train CE | Train KD | Target CE | Target KD |
|---|---|---:|---:|---:|---:|
| 50/25/25 baseline, previous | 50% WikiText / 25% math / 25% stem | 1.794982 | 0.256528 | 1.806069 | 0.251977 |
| 50/25/25 baseline, rerun | 50% WikiText / 25% math / 25% stem | 1.753319 | 0.284975 | 1.764029 | 0.277251 |
| DoGE initial | 5% WikiText / 5% math / 90% stem | 1.273813 | 0.262383 | 1.821381 | 0.332609 |
| DoGE final learned | 99.877% WikiText / 0.123% math / ~0% stem | 2.614528 | 0.331276 | 1.817401 | 0.401323 |

Lower KD is better. In this PoC, the DoGE-learned final blend did not beat the fixed 50/25/25
baseline on the target validation blend. The baseline rerun was close to the previous run, while
the learned near-WikiText blend was clearly worse on target KD.

## Current PoC limitations

- Only `gbs == mbs` is supported.
- Alignment scoring currently uses a hardcoded Qwen3-8B final MLP projection parameter.
- Pipeline parallelism and broader model-family support are not validated.
- The learned weights collapsed toward WikiText in the Qwen3-8B 0.7x PoC.
- Future work: investigate why the learned weights collapsed.
