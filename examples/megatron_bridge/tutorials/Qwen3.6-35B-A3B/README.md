# Qwen3.6-35B-A3B: W4A4 NVFP4 PTQ + QAD + vLLM Deployment

End-to-end W4A4 optimization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) demonstrating how to push past weight-only quantization: NVFP4 W4A4 post-training quantization → Quantization-Aware Distillation (QAD) to recover the accuracy it costs → evaluation benchmarking → vLLM deployment. This document covers:

1. **[Data Preparation](#1-data-preparation)** — tokenizing the SFT blend for distillation
2. **[Quantization](#2-quantization)** — W4A4 NVFP4 PTQ with `examples/megatron_bridge/quantize.py`
3. **[QAD](#3-quantization-aware-distillation-qad)** — recovering accuracy with `examples/megatron_bridge/distill.py`
4. **[Export](#4-export)** — converting the quantized Megatron checkpoint to a deployable HF checkpoint
5. **[Evaluation](#5-evaluation)** — benchmarking with NeMo Evaluator across MMMU-Pro, GPQA Diamond, SciCode, and more
6. **[vLLM Inference Benchmarking](#6-vllm-inference-benchmarking)** — throughput comparison against BF16 on GB200

It closes with [potential further improvements](#potential-further-improvements) — the levers this run did not explore, and what the evidence says each might buy.

> [!NOTE]
> This tutorial complements the [Nemotron-3-Nano-30B-A3B tutorial](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md), which covers pruning + distillation + FP8 PTQ. Here the starting point is an unpruned model and the technique under test is **W4A4** — aggressive enough that PTQ alone leaves a measurable accuracy gap, which is exactly what QAD exists to close.

## Results

![W4A4 NVFP4 accuracy recovery during QAD](figures/qad_learning_curves.png)

<b>Main results</b> — all models evaluated with the same [setup](#5-evaluation). Values are `mean ± sem` across repeats; intermediate QAD checkpoints are in the figure above.

| Model | MMMU-Pro | GPQA Diamond | SciCode (Subtask) | AA-LCR | IFBench | tau2-bench Telecom | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **BF16** (teacher) | 74.6 ± 0.2 | 84.7 | 39.9 ± 0.6 | 69.1 ± 0.8 | 60.0 ± 0.5 | 94.2 ± 1.0 | 70.4 |
| **W4A4 NVFP4 PTQ** (QAD student) | 73.4 ± 0.2 | 84.7 | 39.1 ± 0.7 | 70.0 ± 1.1 | 57.9 ± 0.5 | 94.2 ± 1.2 | 69.9 |
| **↳ + QAD 500 iters** | **73.9 ± 0.2** | **84.2** | **40.2 ± 0.6** | **69.4 ± 1.5** | **59.6 ± 0.5** | **93.4 ± 0.4** | **70.1** |

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

### vLLM Throughput (4× GB200, vLLM 0.28.0, TP=4 + EP)

| shape (ISL/OSL) | concurrency | BF16 | W4A16 NVFP4 | **W4A4 NVFP4** | W4A4 / BF16 |
| --- | --- | --- | --- | --- | --- |
| decode 128/2048 | 1 | 351 | 225 | 307 | 0.88× |
| decode 128/2048 | 32 | 6,544 | 5,602 | **7,320** | **1.12×** |
| decode 128/2048 | 128 | 17,636 | 15,222 | **20,076** | **1.14×** |
| chat 8000/1000 | 32 | 3,649 | 2,373 | **4,079** | **1.12×** |
| chat 8000/1000 | 128 | 5,571 | 6,138 | **6,370** | **1.14×** |
| prefill 32000/400 | 128 | 854 | 1,092 | **1,107** | **1.30×** |

Output tokens/s; 6 of the 12 measured shapes shown. **W4A4 beats BF16 in 9 of 12 and beats W4A16 in all 12.** Checkpoint size drops from **67 GiB → 23 GiB (2.9×)**.

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

Weights are each split's share of the dataset's own token count — a single pass over the natural mixture, not a tuned blend. The 500-iteration schedule consumes **8.4B tokens, about half an epoch**, so nothing is resampled. Note the agent/SWE splits total under 9%, which is relevant to the tau2-bench behaviour in Section 3.

---

### 2. Quantization

W4A4 NVFP4 PTQ using the recipe at [`modelopt_recipes/models/Qwen/Qwen3.6-35B-A3B/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast.yaml`](../../../../modelopt_recipes/models/Qwen/Qwen3.6-35B-A3B/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast.yaml). See [examples/megatron_bridge/README.md](../../README.md) for full PTQ documentation.

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
    --recipe models/Qwen/Qwen3.6-35B-A3B/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast \
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

Minimum hardware: the student and teacher are both resident, plus an fp32 gradient buffer and optimizer state — roughly 124 GB/GPU of the 185 GiB on a GB200 at the settings below. We used **32 nodes × 4 GB200 (128 GPUs)**; 500 iterations took **5.7 hours wall-clock (~735 GB200 GPU-hours)** at ~37 s/iter.

<details>
<summary>QAD command (click to expand)</summary>

> NOTE: We use `python -u` for slurm multi-node runs.

```bash
python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
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
    --logit_kl_topk 4096 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --no_async_save \
    --eval_iters 0 \
    --save_interval 50 \
    --output_dir /path/to/qad_output
```

Non-default arguments:

- `--student_megatron_path` — the quantized checkpoint from Section 2; `--student_hf_path` still points at the BF16 model, which supplies the architecture.
- `--tp_size 1 --pp_size 1 --cp_size 1` — **required, not chosen** (see below). `--ep_size 8` must match the PTQ checkpoint.
- `--seq_length 32768 --gbs 512` — 16.8M tokens/iteration, 1.7B per 100 iterations.
- `--lr 1e-5 --min_lr 1e-6` — an order of magnitude below typical distillation LRs: the job is to adapt weights to quantization, not to learn the task.
- `--logit_kl_topk 4096` — restricts the KD loss to the teacher's top-4096 vocab entries. With a 248,320-token vocabulary the dense `[seq, vocab]` fp32 logits are **30.31 GiB per tensor** at 32K, which OOMs on its own.
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

One config per benchmark in [eval_configs/](eval_configs/), so each can be launched independently with its own walltime and sampling:

| Benchmark | Config | Temp | Runs | Walltime | Metric name |
| --- | --- | --- | --- | --- | --- |
| MMMU-Pro | [mmmu_pro.yaml](eval_configs/mmmu_pro.yaml) | 1.0 | 8 | 2:30 | `mmmu-pro_pass_at_1_symbolic_correct` |
| GPQA Diamond | [gpqa.yaml](eval_configs/gpqa.yaml) | 1.0 | 1 (avg-of-16) | 4:00 | `gpqa_pass_at_1_avg-of-16_symbolic_correct` |
| SciCode (Subtask) | [scicode.yaml](eval_configs/scicode.yaml) | **0.6** | 8 | 2:00 | `scicode_pass_at_1_subtask_accuracy` |
| AA-LCR | [aa_lcr.yaml](eval_configs/aa_lcr.yaml) | 1.0 | 8 | 1:00 | `aalcr_pass_at_1_judge_correct` |
| IFBench | [ifbench.yaml](eval_configs/ifbench.yaml) | 1.0 | 8 | 1:30 | `ifbench_pass_at_1_average_score` |
| tau2-bench Telecom | [tau2_telecom.yaml](eval_configs/tau2_telecom.yaml) | 1.0 | 3 | 1:30 | `tau2_bench_telecom_pass_at_1_pass_at_1` |

Sampling follows the [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) card (`T=1.0, top_p=0.95, max_new_tokens=131072`), except SciCode, which the card specifies at `T=0.6`. `parallelism` varies (32 default, 16 for AA-LCR's long contexts, 8 for SciCode's long generations) as a throughput knob only. The same configs serve BF16 and NVFP4 unchanged — vLLM reads the FP8 KV-cache setting from the checkpoint's own `quantization_config`.

> [!IMPORTANT]
> Every task sets `num_repeats: 1`; the repeat counts above come from **launching a config that many times**. N launches give N independent `pass@1` values, which is what `mean ± sem` and the paired tests need — `num_repeats: N` instead yields a single `pass@1[avg-of-N]` with no spread. GPQA is the deliberate exception.

> [!IMPORTANT]
> **Run enough repeats.** These benchmarks are noisy, and 3 was not enough to tell signal from noise in either direction. At 3 repeats, AA-LCR (only 100 questions) showed a 3 pp swing that looked real and **vanished completely** at 8; IFBench's genuine +2.3 pp gain was **invisible** at 3 and only appeared at 8. Three runs also *understates* how noisy a benchmark is, so the measured spread looks tighter than it is.

<details>
<summary>Evaluation launch steps (click to expand)</summary>

Set `execution.hostname`, `execution.account` and `deployment.checkpoint_path` in the config, or override with `-o <option>=<value>`.

```bash
pip install "nemo-evaluator-launcher[all]==0.2.6"

# The only environment variables these configs reference:
export HF_TOKEN=<your_huggingface_token>
export SLURM_JOB_DIR=<path_to_slurm_job_output_dir>

# One benchmark. To verify the pipeline first, add
# `-o ++evaluation.nemo_evaluator_config.config.params.limit_samples=8`
nemo-evaluator-launcher run --config eval_configs/mmmu_pro.yaml

# The 8 independent runs behind the results table: launch the same config 8 times and pool the
# per-run pass@1 values. Each launch returns its own invocation id; record them so you can select
# exactly those runs when pooling (each eval directory also holds a small canary run).
for i in $(seq 1 8); do
    nemo-evaluator-launcher run --config eval_configs/mmmu_pro.yaml
done

# All six benchmarks at their reported repeat counts
for cfg_runs in mmmu_pro:8 gpqa:1 scicode:8 aa_lcr:8 ifbench:8 tau2_telecom:3; do
    cfg=${cfg_runs%%:*}; runs=${cfg_runs##*:}
    for i in $(seq 1 "$runs"); do
        nemo-evaluator-launcher run --config "eval_configs/${cfg}.yaml"
    done
done
```

</details>

For more details on NeMo Evaluator, see the [GitHub repo](https://github.com/NVIDIA-NeMo/evaluator) and [documentation](https://docs.nvidia.com/nemo/evaluator/latest/).

---

### 6. vLLM Inference Benchmarking

Throughput was measured with [AIPerf](https://github.com/ai-dynamo/aiperf) against a served vLLM endpoint on 4× GB200 — three ISL/OSL shapes × four concurrencies = the 12 shapes in the [results table](#vllm-throughput-4-gb200-vllm-0280-tp4--ep).

<details>
<summary>Serve + benchmark commands (click to expand)</summary>

The same `vllm serve` command works for BF16 and NVFP4; the quantized checkpoint carries its own format and FP8 KV-cache settings in `quantization_config`.

```bash
vllm serve <checkpoint_path> --served-model-name bench \
    --host 0.0.0.0 --port 8000 \
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

**3. A better data blend.** The clearest signal in the study: **MMMU-Pro is multimodal, the blend is text-only, and MMMU-Pro is the one deficit QAD did not close.** Since QAD supports language-model distillation with text-only data, recovering a multimodal benchmark that way asks the method to do something it is not set up for. Agentic coverage is also thin (agent/SWE splits under 9%, and tau2-bench showed the transient 8× `agent_error` spike), and weights here are simply proportional to token count — contrast the [Nemotron-3-Nano blend](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md#data-blend), which is deliberately designed against target benchmarks and [ablated](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/ABLATIONS.md#effect-of-data-blend-tool_calling).

If you run only one, run **(1)**: no engineering risk, the trajectory says the gain is still arriving, and it produces the evidence to judge whether (2) and (3) are worth their cost.
