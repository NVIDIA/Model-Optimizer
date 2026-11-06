# Muse-Spark-Open-816B-A42B DFlash2

These two recipes train the DFlash2 drafter released as
[nvidia/Muse-Spark-Open-816B-A42B-DFlash2](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2)
for [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B).

| Stage | Recipe | Launcher example | Steps | Sequence | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1, from scratch | `dflash2.yaml` | `hf_streaming_dflash2.yaml` | 390,000 of 434,744 (8 epochs) | 4,096 | 1e-3, 4% warmup, linear |
| 2, long context | `dflash2_long_context.yaml` | `hf_streaming_dflash2_long_context.yaml` | 25,000 of 59,502 (6 epochs) | 12,288 | 3e-5, 1,500 warmup steps, linear |

The launcher examples are under `tools/launcher/examples/meta-models/Muse-Spark-Open-816B-A42B/`.
Stage 2 starts from the export of stage 1's checkpoint-390000, and the release is stage 2's
checkpoint-25000. Neither is the last checkpoint of its run, and the recipes keep only the last
three, so stop each run once its checkpoint is written, or copy the checkpoint out. Both stages
stream hidden states from the NVFP4 target,
[meta-models/Muse-Spark-Open-816B-A42B-NVFP4](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4),
served by vLLM.

## Training data

The drafter learns from the target's own outputs. The prompts come from
[nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2);
its responses are not used.

- The target answered each prompt at temperature 1, with up to 8,192 output tokens, at one of
  its five reasoning-effort levels: xhigh for 78% of the samples, and minimal, low, medium and
  high for about 200,000 each.
- Each response is split into its reasoning and its answer. Responses that reached the token
  limit without an answer, and a few malformed ones, were dropped, leaving 3.48 million samples.
- Each row is a `conversations` list: a system turn that sets the reasoning effort the way
  the model's chat template writes it, the user turn, and an assistant turn whose `content` is
  the answer and whose `reasoning` is the reasoning.

Stage 2 uses the samples whose conversation, rendered with the chat template, is longer than
4,096 tokens: 634,723 samples, 18% of the corpus.

Synthesizing the corpus took about 525 node-hours on 4-GPU GB300 nodes, one TP4 replica of the
target per node.

## Compute

Both stages ran on 22 nodes of 4 GB300 GPUs: 14 vLLM serve replicas of the target and 8
trainer nodes.

| Stage | Run time | GPU-hours |
| --- | --- | --- |
| 1 (390,000 steps) | 37.7 h | ~3,320 |
| 2 (25,000 steps) | 6.0 h | ~530 |

## Results

Acceptance length with vLLM, draft length 7, reasoning effort xhigh, 4,096 output tokens,
concurrency 32, on one GB300 node (TP4, EP4). At temperature 1 the draft samples with
`draft_sample_method: probabilistic`.

| Benchmark | Greedy | Temperature 1 |
| --- | --- | --- |
| MT-Bench | 4.01 | 3.66 |
| SPEED-Bench, qualitative | 4.08 | 3.68 |
| SPEED-Bench, throughput 32K (first 100 per category) | 3.59 | 3.26 |

Against stage 1's checkpoint-390000, stage 2 raised SPEED-Bench acceptance by 0.5-1.9% and
moved MT-Bench by -0.9% (greedy) and +0.4% (temperature 1).
