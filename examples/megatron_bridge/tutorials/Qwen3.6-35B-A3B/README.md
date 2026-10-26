# Qwen3.6-35B-A3B: W4A4 NVFP4 PTQ + QAD + vLLM Deployment

End-to-end W4A4 optimization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) demonstrating how to quantize both weights and activations: NVFP4 W4A4 post-training quantization → Quantization-Aware Distillation (QAD) to recover the accuracy it costs → evaluation benchmarking → vLLM deployment.

> [!NOTE]
> This tutorial complements the [Nemotron-3-Nano-30B-A3B tutorial](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md), which covers pruning + distillation + FP8 PTQ. Here the starting point is an unpruned model and the technique under test is **W4A4** — aggressive enough that PTQ alone leaves a measurable accuracy gap, which is exactly what QAD exists to close.

## Highlights

- **W4A4 PTQ is close to BF16**: a six-benchmark average of 69.9 vs 70.4, matching the published W4A16 checkpoint (69.8). Only IFBench (−2.6 pp) and MMMU-Pro (−1.2 pp) drop by more than run-to-run noise ([paired per-question deltas](#where-qad-actually-helps)).
- **QAD (500 iterations, ~735 GB200 GPU-hours) brings IFBench back to BF16** and closes about 40% of the MMMU-Pro gap (same paired comparison), for an average of 70.1.
- **W4A4 is faster than BF16, W4A16 is not**: up to 1.30× BF16 output throughput in vLLM and faster in 9 of 12 shapes, while weight-only NVFP4 is slower than BF16 in 10 of 12. The checkpoint shrinks from 67 to 22 GiB.
- **Caveats**: each step makes the model guess more instead of abstaining (AA-Omniscience), most of all QAD, and QAD lengthens SciCode outputs by 91% through non-terminating sub-steps, which `presence_penalty: 1.5` fixes at decode time.

## Contents

1. **[Results](#results)** — accuracy, knowledge recall and calibration, output length, and vLLM throughput
2. **Steps to Reproduce**
   1. **[Data Preparation](#1-data-preparation)** — tokenizing the SFT blend for distillation
   2. **[Quantization](#2-quantization)** — W4A4 NVFP4 PTQ with `examples/megatron_bridge/quantize.py`
   3. **[QAD](#3-quantization-aware-distillation-qad)** — recovering accuracy with `examples/megatron_bridge/distill.py`
   4. **[Export](#4-export)** — converting the quantized Megatron checkpoint to a deployable HF checkpoint
   5. **[Evaluation](#5-evaluation)** — benchmarking with NeMo Evaluator across MMMU-Pro, GPQA Diamond, SciCode, and more, plus AA-Omniscience for knowledge recall and calibration
   6. **[vLLM Inference Benchmarking](#6-vllm-inference-benchmarking)** — throughput comparison against BF16 on GB200
3. **[Potential further improvements](#potential-further-improvements)** — settings we did not test and their potential benefits

Experiments that did not become the recommended recipe — chat-format SFT data with assistant-only loss, self-distilled data, and the SciCode non-termination diagnosis — are in [ABLATIONS.md](ABLATIONS.md).

## Results

![W4A4 NVFP4 accuracy recovery during QAD](figures/qad_learning_curves.png)

<b>Main results</b> — all models evaluated with the same [setup](evaluation/README.md). Values are `mean ± sem` across repeats; intermediate QAD checkpoints are in the figure above.

| Model | MMMU-Pro | GPQA Diamond | SciCode (Subtask) | AA-LCR | IFBench | tau2-bench Telecom | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **BF16** (teacher) | 74.6 ± 0.2 | 84.7 | 39.9 ± 0.6 | 69.1 ± 0.8 | 60.0 ± 0.5 | 94.2 ± 1.0 | 70.4 |
| [W4A16 NVFP4 PTQ](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) (published)<sup>1</sup> | 73.8 ± 0.2 | 84.4 | 38.5 ± 0.5 | 66.3 ± 1.0 | 59.4 ± 0.5 | 96.4 ± 0.4 | 69.8 |
| **W4A4 NVFP4 PTQ** (QAD student) | 73.4 ± 0.2 | 84.7 | 39.1 ± 0.7 | 70.0 ± 1.1 | 57.9 ± 0.5 | 94.2 ± 1.2 | 69.9 |
| **↳ + QAD 500 iters** | **73.9 ± 0.2** | **84.2** | **40.2 ± 0.6** | **69.4 ± 1.5** | **59.6 ± 0.5** | **93.4 ± 0.4** | **70.1** |

><sup>1</sup> The W4A16 NVFP4 row is the published checkpoint re-evaluated here with the same evaluation harness.

### Where QAD actually helps

W4A4 quantization only measurably hurt **two** of the six benchmarks, so those are the only two where QAD has anything to recover:

| Benchmark | W4A4 PTQ vs BF16 | After 500 QAD iters | Outcome |
| --- | --- | --- | --- |
| **IFBench** | **−2.6 pp** (real loss) | **−0.3 pp** | ✅ recovered — back to BF16 |
| **MMMU-Pro** | **−1.2 pp** (real loss) | **−0.7 pp** | ⚠️ about 40% recovered, gap remains |
| GPQA Diamond, SciCode, AA-LCR, tau2-bench | no loss | no change | — nothing to recover |

"Real loss" means the gap is larger than the run-to-run noise, so it is not a measurement artifact. The four benchmarks in the last row land within ±1.2 pp of BF16 both before and after QAD, which is inside their noise — W4A4 is effectively lossless there.

<details>
<summary>Statistical detail (click to expand)</summary>

Comparisons are **paired per-question t-tests**: the same question answered by both models is compared directly, which cancels question-difficulty variance and is far more sensitive than comparing run averages. `p` is the probability of seeing a gap this large if the two models were actually equal; below 0.05 is conventionally "real".

| Benchmark | PTQ vs BF16 | QAD 500 vs PTQ | QAD 500 vs BF16 |
| --- | --- | --- | --- |
| MMMU-Pro | −1.23 (p=0.0002) | +0.51 (p=0.078) | −0.72 (p=0.023) |
| GPQA Diamond | +0.03 (p=0.96) | −0.41 (p=0.60) | −0.38 (p=0.58) |
| SciCode (Subtask) | −0.81 (p=0.40) | +1.15 (p=0.22) | +0.33 (p=0.72) |
| AA-LCR | +0.75 (p=0.73) | −0.62 (p=0.68) | +0.12 (p=0.95) |
| IFBench | −2.62 (p=0.017) | +2.33 (p=0.036) | −0.29 (p=0.79) |
| tau2-bench Telecom | −0.00 (p=1.00) | −0.88 (p=0.36) | −0.88 (p=0.44) |

</details>

See [Potential further improvements](#potential-further-improvements) for what would likely close the remaining MMMU-Pro gap.

### Knowledge recall and calibration

The six benchmarks above barely test recalling facts, which is what weight quantization tends to cost first. [AA-Omniscience](evaluation/omniscience.yaml) does: 600 questions across domains, each answered 10 times, and a judge grades every answer as correct, partially correct, incorrect or abstained ("I don't know"). Its index is the share correct minus the share incorrect, so a wrong guess costs as much as a right answer earns.

| Model | Correct | Incorrect | Abstained | Omniscience index | Hallucination rate |
| --- | --- | --- | --- | --- | --- |
| **BF16** (teacher) | 21.0 | 32.3 | 46.0 | −11.3 | 40.9% |
| W4A16 NVFP4 PTQ (published) | 20.9 | 36.3 | 42.2 | −15.4 | 45.8% |
| **W4A4 NVFP4 PTQ** | 20.2 | 39.3 | 39.8 | −19.1 | 49.2% |
| **↳ + QAD 500 iters** | 21.0 ± 0.3 | 48.5 ± 2.3 | 29.8 ± 2.4 | −27.5 | 61.4% |

>`pass@1[avg-of-10]`, judged by Gemini 3 Flash. Hallucination rate = incorrect / (100 − correct). The ~0.7% graded partially correct is not shown, so the three columns sum to ~99.3. `±` is the sem across the 10 repeats of one run (≤0.6 on the rows without one); the repeats share seeds across models, so it is not the run-to-run sem used elsewhere on this page.

- **Knowledge is kept.** Every model answers the same ~21% correctly; 4-bit weights, 4-bit activations and QAD all leave recall intact.
- **The model guesses more instead of abstaining.** Each step abstains less and answers wrong more often: the published W4A16 checkpoint already loses 4 index points, W4A4 PTQ 8, and **QAD 500 loses 16**, at an unchanged correct rate. None of the six benchmarks above can see this.
- **QAD does most of it early**: the index drops to −31.4 at iteration 50, then partly recovers (−25.1 at 300, −27.5 at 500), the same shape as SciCode's non-terminating sub-steps (see [ABLATIONS.md](ABLATIONS.md#data-format-pre-tokenized-blend-vs-chat-sft-data)).

If abstaining matters for your use case (assistants that should say "I don't know"), evaluate it explicitly: QAD as run here did not preserve it.

### Output length

Accuracy is only half the serving cost — a model that scores the same while emitting more tokens is slower end to end. Mean completion tokens per benchmark (`response_stats.avg_completion_tokens`, same runs as the results table):

| Benchmark | BF16 | W4A16 NVFP4 | W4A4 NVFP4 PTQ | + QAD 500 iters | runs/side |
| --- | --- | --- | --- | --- | --- |
| SciCode (Subtask) | 5,348 | +22.9% | +25.5% | **+90.9%** | 8 |
| IFBench | 12,177 | +7.0% | +9.3% | +17.4% | 8 |
| AA-LCR | 2,560 | +7.1% | +6.9% | +9.8% | 8 |
| MMMU-Pro | 9,382 | +3.0% | +6.2% | −3.8% | 8 |
| GPQA Diamond | 13,561 | +17.1% | +6.6% | −8.5% | 1 |

**4-bit weights lengthen SciCode outputs by ~23-25% on their own** — the published W4A16 checkpoint does it too, so it is not something QAD or W4A4 introduced. **QAD then pushes SciCode to +90.9%**, for an unchanged score (40.2 vs 39.9), while pulling GPQA and MMMU-Pro back toward BF16. The throughput table is measured at fixed output length, so it does not capture this.

The SciCode increase is not verbosity: **3.6% of sub-steps never terminate** (BF16: 0.7%). They hit the 131,072-token cap with no answer and use 45.7% of all completion tokens. **`presence_penalty: 1.5` fixes it at decode time**: capped sub-steps drop from 96 to 2 of 2,704, accuracy stays within noise, and SciCode runs 12-44% faster. [scicode.yaml](evaluation/scicode.yaml) has the line commented out so it reproduces the results table; uncomment it to apply the fix. Do not set it for every benchmark: on GPQA Diamond it cuts generation throughput 4.8× (5,109 to 1,067 tokens/s) for no accuracy change. Details are in [ABLATIONS.md](ABLATIONS.md#scicode-non-termination-diagnosis-and-decode-time-fix).

### vLLM Throughput (4× GB200, vLLM 0.28.0, TP=4 + EP)

| shape (ISL/OSL) | concurrency | BF16 | W4A16 NVFP4 | **W4A4 NVFP4** | W4A4 / BF16 |
| --- | --- | --- | --- | --- | --- |
| decode 128/2048 | 1 | 351 | 225 | 307 | 0.88× |
| decode 128/2048 | 32 | 6,544 | 5,602 | **7,320** | **1.12×** |
| decode 128/2048 | 128 | 17,636 | 15,222 | **20,076** | **1.14×** |
| chat 8000/1000 | 32 | 3,649 | 2,373 | **4,079** | **1.12×** |
| chat 8000/1000 | 128 | 5,571 | 6,138 | **6,370** | **1.14×** |
| prefill 32000/400 | 128 | 854 | 1,092 | **1,107** | **1.30×** |

Output tokens/s; 6 of the 12 measured shapes shown. **W4A4 beats BF16 in 9 of 12 and beats W4A16 in all 12.** Checkpoint size drops from **67 GiB → 22 GiB (3.1×)**.

The W4A16 column is why this tutorial targets W4A4 at all: **weight-only NVFP4 is *slower* than BF16** in 10 of 12 shapes. A BF16 activation forces vLLM onto the Marlin dequant-to-BF16 fallback, which never reaches the Blackwell FP4 tensor cores. Only quantizing activations too unlocks them.

---

## Steps to Reproduce

**Environment:** Results were produced with container `nvcr.io/nvidia/nemo:26.08`, ModelOpt 0.47.0 and `nemo-evaluator-launcher` 0.2.6 (with `nemo-evaluator` 0.2.8) on GB200 (aarch64). See the [Megatron-Bridge README](../../README.md) for environment setup (including ModelOpt mount path) and container usage. Deployment and evaluation of NVFP4 checkpoints require a Blackwell GPU.

### 1. Data Preparation

QAD currently supports **language-model distillation with text-only data**, so the blend is SFT text: [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data), all 8 splits, 17.3B tokens.

Tokenize with the [token-budgeted blend workflow](../../../dataset/MEGATRON_DATA_PREP.md#prepare-token-budgeted-data-blends) using [data_blend.yaml](data_blend.yaml) — set `output_dir` in that file first:

| Split | Tokens | Weight |
| --- | --- | --- |
| chat | 9.73B | 56.2 |
| math | 3.65B | 21.1 |
| science | 1.89B | 10.9 |
| instruction_following | 0.57B | 3.3 |
| conversational_agent | 0.57B | 3.3 |
| terminal_agent | 0.57B | 3.3 |
| swe | 0.31B | 1.8 |
| safety | 0.003B | 0.02 |

Weights are each split's share of the dataset's own token count — a single pass over the natural mixture, not a tuned blend. The 500-iteration schedule consumes **8.4B tokens, about half an epoch**, so nothing is resampled. Note the agent/SWE splits total under 9%, which is relevant to the agentic coverage discussed under [potential further improvements](#potential-further-improvements).

---

### 2. Quantization

W4A4 NVFP4 PTQ using the recipe at [`modelopt_recipes/model_type/qwen3_6_moe/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast_mcore.yaml`](../../../../modelopt_recipes/model_type/qwen3_6_moe/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast_mcore.yaml). See [examples/megatron_bridge/README.md](../../README.md) for full PTQ documentation.

PTQ takes **~9 min on 2 GB200 nodes (1.2 GPU-hours)** at EP=8.

**What gets quantized** (read from the exported checkpoint's `quantization_config`):

| Component | Precision | Tensors |
| --- | --- | --- |
| MoE routed experts (`down`/`gate`/`up_proj`) | **NVFP4 W4A4** (block 16) | 30,720 |
| Shared expert | **NVFP4 W4A4** | 120 |
| `lm_head` | **NVFP4 W4A4** | 1 |
| Linear attention (`in_proj_qkv`/`in_proj_z`/`out_proj`, 30 layers) | **FP8 W8A8** | 90 |
| Full attention (`q`/`k`/`v`/`o_proj`, 10 layers) | **FP8 W8A8** | 40 |
| KV cache | **FP8** (static) | — |
| MTP head, MoE router, `conv1d`, `in_proj_a`/`in_proj_b`, embeddings, vision tower | BF16 | — |

99.2% of quantized tensors are MoE experts — where the FP4 throughput win comes from. Attention stays at FP8: only 130 tensors, and numerically more sensitive.

<details>
<summary>W4A4 NVFP4 PTQ command (click to expand)</summary>

`--ep_size 8` needs 8 ranks, which is **2 GB200 nodes** (4 GPUs each) — launched with `srun`, one task per GPU. On 8-GPU nodes a single-node `torchrun --nproc_per_node 8` is equivalent.

```bash
# SBATCH --nodes=2 --ntasks-per-node=4 --gpus-per-node=4
srun ... python /opt/Model-Optimizer/examples/megatron_bridge/quantize.py \
    --hf_model_name_or_path Qwen/Qwen3.6-35B-A3B \
    --recipe model_type/qwen3_6_moe/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast_mcore \
    --tp_size 1 --ep_size 8 --pp_size 1 \
    --calib_dataset_name cnn_nemotron_v2_mix \
    --calib_num_samples 1024 \
    --calib_batch_size 1 \
    --seq_length 8192 \
    --export_megatron_path /path/to/qwen36_w4a4_megatron \
    --skip_generate
```

Set `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from `SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_LOCALID` inside the `srun` body, as in the [QAD command](#3-quantization-aware-distillation-qad) below.

> [!IMPORTANT]
> Set `--ep_size` to the expert-parallel size you will run QAD at. QAD loads this checkpoint directly and **EP must match** — the expert layout is baked into the distributed checkpoint. Re-exporting at a different EP later means re-running PTQ.

</details>

---

### 3. Quantization-Aware Distillation (QAD)

QAD fine-tunes the quantized student against the BF16 teacher, so the student learns weights that survive 4-bit rounding. See the [QAD section of the Megatron-Bridge README](../../README.md#quantization-aware-distillation-qad).

Minimum hardware: the student and teacher are both resident, plus an fp32 gradient buffer and optimizer state — roughly 124 GB/GPU of the 185 GiB on a GB200 at the settings below. We used **32 nodes × 4 GB200 (128 GPUs)**; 500 iterations took **5.7 hours wall-clock (~735 GB200 GPU-hours)**. Steady-state training is ~37 s/iter (~5.1 h of the total); the remaining ~35 min is per-job startup, loading the 67 GiB teacher and the student — the run was split across two 4-hour jobs, so that cost is paid twice.

<details>
<summary>QAD command (click to expand)</summary>

Launched with `srun`, one task per GPU, across 32 nodes (128 ranks). Set `RANK` / `WORLD_SIZE` /
`LOCAL_RANK` from `SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_LOCALID` in the `srun` body; `python -u`
keeps the multi-node logs unbuffered.

```bash
# SBATCH --nodes=32 --ntasks-per-node=4 --gpus-per-node=4
srun ... python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
    --teacher_hf_path Qwen/Qwen3.6-35B-A3B \
    --student_hf_path Qwen/Qwen3.6-35B-A3B \
    --student_megatron_path /path/to/qwen36_w4a4_megatron \
    --tp_size 1 --pp_size 1 --cp_size 1 --ep_size 8 \
    --data_paths "${DATA_BLEND}" \
    --data_path_to_cache /path/to/cache \
    --seq_length 32768 \
    --mbs 1 \
    --gbs 512 \
    --train_iters 500 \
    --lr 1e-5 --min_lr 1e-6 --lr_warmup_iters 50 \
    --logit_kl_top_k 4096 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --no_async_save \
    --eval_iters 0 \
    --save_interval 50 \
    --output_dir /path/to/qad_output
```

Non-default arguments:

- `--student_megatron_path` — the quantized checkpoint from Section 2; `--student_hf_path` still points at the BF16 model, which supplies the architecture.
- `--tp_size 1 --pp_size 1` — **required, not chosen** (see below). `--ep_size 8` must match the PTQ checkpoint. You may increase `--cp_size` to enable context parallelism for longer sequence lengths (`nemo:26.10` container onwards).
- `--seq_length 32768 --gbs 512` — 16.8M tokens/iteration, 1.7B per 100 iterations.
- `--lr 1e-5 --min_lr 1e-6` — an order of magnitude below typical distillation LRs: the job is to adapt weights to quantization, not to learn the task.
- `--logit_kl_top_k 4096` — restricts the KD loss to the teacher's top-4096 vocab entries. With a 248,320-token vocabulary the dense `[seq, vocab]` fp32 logits are **30.31 GiB per tensor** at 32K, which OOMs on its own.
- `--recompute_*` / `--no_async_save` / `--eval_iters 0` — all needed to fit. Async save spawns a worker needing its own CUDA context; the validation path computes full-vocab LM and MTP cross-entropy (top-k applies to training only), so eval OOMs at 32K even though training fits.

</details>

---

### 4. Export

Convert the quantized Megatron checkpoint to a deployable unified HuggingFace checkpoint (**~7 min on 1 node, 0.5 GB200 GPU-hours**). The unified exporter loads at TP=1, so use pipeline parallelism to shard across GPUs.

<details>
<summary>Export command (click to expand)</summary>

```bash
torchrun --nproc_per_node 4 /opt/Model-Optimizer/examples/megatron_bridge/export_quantized_megatron_to_hf.py \
    --hf_model_name_or_path Qwen/Qwen3.6-35B-A3B \
    --megatron_path /path/to/qad_output/checkpoints/iter_0000500 \
    --pp_size 4 \
    --export_unified_hf_path /path/to/qwen36_w4a4_qad_hf
```

</details>

The exported checkpoint is directly deployable with [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [SGLang](https://github.com/sgl-project/sglang). Sanity-check it with `generate_vllm.py --model <path>`.

---

### 5. Evaluation

[evaluation/](evaluation/README.md) holds everything needed to reproduce the [results](#results): one [NeMo Evaluator Launcher](https://github.com/NVIDIA-NeMo/evaluator) config per benchmark, the benchmark table with sampling settings and repeat counts, the judge and user-simulator endpoints that AA-LCR, AA-Omniscience and tau2-bench need, and the launch commands.

---

### 6. vLLM Inference Benchmarking

Throughput was measured with [AIPerf](https://github.com/ai-dynamo/aiperf) against a served vLLM endpoint on 4× GB200 — three ISL/OSL shapes × four concurrencies = the 12 shapes in the [results table](#vllm-throughput-4-gb200-vllm-0280-tp4--ep).

<details>
<summary>Serve + benchmark commands (click to expand)</summary>

The same `vllm serve` command works for BF16 and NVFP4; the quantized checkpoint carries its own format and FP8 KV-cache settings in `quantization_config`.

```bash
vllm serve <checkpoint_path> --served-model-name bench \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel \
    --max-model-len 262144 --reasoning-parser qwen3 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
    --max-num-batched-tokens 8192 --enable-chunked-prefill

# decode-bound, chat, and prefill-bound shapes at four concurrencies each
for shape in "128 2048 decode" "8000 1000 chat" "32000 400 prefill"; do
    set -- $shape; ISL=$1; OSL=$2; NAME=$3
    for C in 1 8 32 128; do
        aiperf profile -m bench --endpoint-type chat --streaming -u localhost:8000 \
            --synthetic-input-tokens-mean $ISL --output-tokens-mean $OSL \
            --concurrency $C --request-count $(( C * 5 )) \
            --tokenizer <checkpoint_path> \
            --extra-inputs ignore_eos:true --random-seed 42 \
            --artifact-dir bench/${NAME}_isl${ISL}_osl${OSL}/c${C}
    done
done
```

</details>

> [!TIP]
> Check the endpoint answers a known question correctly before benchmarking it. A checkpoint served with the wrong KV dtype or an unexpected kernel still produces tokens at a plausible rate — the throughput number looks fine and is meaningless. We gate every benchmark on a short prompt with a verifiable answer.

> [!TIP]
> To deploy the model with vLLM, refer to the [vLLM Quickstart documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart/).

---

## Potential further improvements

This is a single 500-iteration run at 32K on a text-only blend. Each of those three choices is an unexplored lever.

**1. Continue QAD longer.** The recovery had not saturated: IFBench's gain arrived *late* (−0.11 pp vs the student at iteration 300, +2.33 pp at 500) and MMMU-Pro was still climbing (+0.51 pp, p=0.078). Training also used only **8.4B of the blend's 17.3B tokens**, so it can roughly double before repeating a sample. At ~735 GPU-hours per 500 iterations this is the most direct experiment, and IFBench + MMMU-Pro alone (~55 GPU-hours) are enough to read the result.

**2. Longer sequence length (64K).** The model serves at 262K context, so 32K trains on a fraction of it; AA-LCR, the one long-context benchmark, showed no movement either way. At 32K the run already needs top-k KD plus full recompute, and memory is dominated by the resident student + teacher + fp32 gradient buffer rather than by sequence length — so 64K needs a different memory lever.

**3. A better data blend.** The clearest signal in the study: **MMMU-Pro is multimodal, the blend is text-only, and MMMU-Pro is the one deficit QAD did not close.** Since QAD supports language-model distillation with text-only data, recovering a multimodal benchmark that way asks the method to do something it is not set up for. Agentic coverage is also thin (agent/SWE splits under 9%, and tau2-bench showed the transient 8× `agent_error` spike), and weights here are simply proportional to token count — contrast the [Nemotron-3-Nano blend](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md#data-blend), which is deliberately designed against target benchmarks and [ablated](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/ABLATIONS.md#effect-of-data-blend-tool_calling). So the lever is *what* the blend contains: feeding the same conversations as chat data with assistant-only loss did not help, and isolating attention per document is still untested on its own (see the [data-format ablation](ABLATIONS.md#data-format-pre-tokenized-blend-vs-chat-sft-data)).

If you run only one, run **(1)**: no engineering risk, the trajectory says the gain is still arriving, and it produces the evidence to judge whether (2) and (3) are worth their cost.
