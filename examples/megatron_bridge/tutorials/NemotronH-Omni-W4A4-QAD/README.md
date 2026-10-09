# NemotronH Omni VL: W4A4 NVFP4 PTQ + QAD + Deployment

End-to-end W4A4 quantization of a NemotronH omni vision-language checkpoint (architecture `NemotronH_Omni_Reasoning_V3`), a Mamba-Transformer MoE language model with a vision tower: mixed NVFP4/FP8 post-training quantization → Quantization-Aware Distillation (QAD) → Hugging Face export → vLLM serving. This document covers:

1. **[Data Preparation](#1-data-preparation)**: the chat-format QAD blend
2. **[Quantization](#2-quantization)**: W4A4 PTQ with `examples/megatron_bridge/quantize.py`
3. **[QAD](#3-quantization-aware-distillation-qad)**: distillation from the BF16 teacher with `examples/megatron_bridge/distill.py`
4. **[Export](#4-export)**: the deployable Hugging Face checkpoint
5. **[Evaluation and Serving](#5-evaluation-and-serving)**: the vLLM settings behind the results

Only the language model is quantized; the vision tower, projector and MTP head stay BF16 and are copied from the source checkpoint.

## Results

All checkpoints were served with the same vLLM settings ([Section 5](#5-evaluation-and-serving)); quantized ones with an NVFP4 KV cache.

| Model | AA-LCR | SciCode (subtask) | GPQA Diamond pass@1 (maj@8) |
| --- | --- | --- | --- |
| **BF16** (teacher) | 63.31 | 46.48 | 89.52 (90.66) |
| **W4A4 PTQ** (QAD student) | 62.00 | 46.92 | 89.27 (90.66) |
| ↳ QAD 200 iters | 61.69 | 43.62 | 88.70 (89.65) |
| ↳ QAD 300 iters | 62.56 | 45.53 | 88.70 (88.13) |
| ↳ QAD 400 iters | 64.06 | 45.36 | 88.89 (90.40) |
| ↳ QAD 500 iters | 62.94 | 46.22 | 89.84 (90.66) |
| **↳ QAD 600 iters** | **63.75** | **46.35** | **89.77 (91.67)** |

Sample counts: AA-LCR 1,600 (100 questions × 16), SciCode 520, GPQA Diamond 1,584 (198 questions × 8).

The recipe is already close to lossless after PTQ: every PTQ score is within 1.4 points of BF16. The first few hundred QAD iterations dip before recovering, and QAD 600 lands within 0.5 points of BF16 on all three benchmarks. The exported W4A4 checkpoint is **74 GB**.

> [!NOTE]
> The numbers above come from runs whose code was functionally equivalent to the commands below but predates them: the Omni export mapping and chat-data distillation were local patches then, and the student was built without its MTP head (`--mtp_num_layers 0`). This code path was checked end to end with a short run (a 32-sample PTQ, 20 QAD iterations and the export); QAD keeps the MTP head unchanged either way (see [Section 3](#3-quantization-aware-distillation-qad)).

---

## Steps to Reproduce

**Environment:** a NeMo 26.08 container (`nvcr.io/nvidia/nemo:26.08`) on GB300 (4 GPUs per node, aarch64), with this repository mounted at `/opt/Model-Optimizer` and Megatron-Bridge `main` at [`d23c1f98`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/d23c1f981596c227b13d98c95b09afb865d80258) (and its pinned Megatron-LM) mounted at `/opt/Megatron-Bridge`. See the [Megatron-Bridge README](../../README.md) for the container setup. Serving NVFP4 checkpoints requires a Blackwell GPU.

Every command below runs with `srun`, one task per GPU, and sets `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from `SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_LOCALID` in the `srun` body. Pass `--trust_remote_code`, since the model ships its own modeling code.

### 1. Data Preparation

QAD distills on text chat data, read directly from jsonl by `distill.py --sft_hf_dataset`. [`build_qad_blend.py`](build_qad_blend.py) materializes a 36,800-record sample of public Nemotron post-training datasets, 400 records per weight point:

| Dataset / split | Weight | Records |
| --- | --- | --- |
| Nemotron-Pretraining-SFT-v1 / Nemotron-SFT-General | 20 | 8,000 |
| Nemotron-SFT-Math-v3 / train | 17 | 6,800 |
| Nemotron-Competitive-Programming-v1 / python_part00 | 15 | 6,000 |
| Nemotron-Math-v2 / high_part00 | 10 | 4,000 |
| Nemotron-Post-Training-Dataset-v1 / stem | 8 | 3,200 |
| Nemotron-Pretraining-SFT-v1 / Nemotron-SFT-Code | 5 | 2,000 |
| Nemotron-Pretraining-SFT-v1 / Nemotron-SFT-MATH | 5 | 2,000 |
| Nemotron-Competitive-Programming-v1 / cpp_part00 | 5 | 2,000 |
| Nemotron-Science-v1 / MCQ | 3 | 1,200 |
| Nemotron-Science-v1 / RQA | 2 | 800 |
| Nemotron-SFT-Instruction-Following-Chat-v2 / reasoning_off | 2 | 800 |

The script also lists `Nemotron-SFT-Instruction-Following-Chat-v2 / reasoning_on` and `Nemotron-Agentic-v1 / tool_calling`; in our run they contributed no records (tool-calling rows are not text-only chat), so the blend has no agentic data. After a seeded shuffle, 1,000 records are held out for validation and 35,800 are used for training.

```bash
python examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD/build_qad_blend.py \
    --output_dir /path/to/qad_blend
```

Records follow the `{"messages": [{"role": ..., "content": ...}, ...]}` schema and are rendered with the model's own chat template during QAD. Records longer than the 32K sequence length used below are truncated.

---

### 2. Quantization

The recipe is a fixed mixed-precision scheme; it will be published under `modelopt_recipes`, and its path goes to `--recipe` below. Counts are read from the exported checkpoint's `hf_quant_config.json`:

| Component | Precision | Tensors |
| --- | --- | --- |
| MoE routed experts (`up`/`down_proj`) | **NVFP4 W4A4**, block 16, MSE-selected static weight scales | 40,960 |
| MoE shared experts | **NVFP4 W4A4** | 80 |
| Mamba `in_proj` / `out_proj` | **FP8 W8A8** per-tensor | 80 |
| Attention `q`/`k`/`v`/`o_proj` (8 layers) | **FP8 W8A8** per-tensor | 32 |
| `lm_head` | **FP8 W8A8** | 1 |
| KV cache | **NVFP4** cast, constant amax 448 | — |
| Vision tower, projector, MTP head, routers, embeddings | BF16 | — |

PTQ takes **16 minutes on one 4-GPU GB300 node** with 512 calibration samples.

<details>
<summary>PTQ command (click to expand)</summary>

```bash
# SBATCH --nodes=1 --ntasks-per-node=4 --gpus-per-node=4
srun ... python -u /opt/Model-Optimizer/examples/megatron_bridge/quantize.py \
    --hf_model_name_or_path <nemotron_h_omni-checkpoint> \
    --trust_remote_code \
    --recipe /path/to/recipe.yaml \
    --tp_size 1 --ep_size 4 --pp_size 1 \
    --calib_dataset_name cnn_nemotron_v2_mix \
    --calib_num_samples 512 \
    --calib_batch_size 1 \
    --export_megatron_path /path/to/nemotron_h_omni_w4a4_megatron \
    --skip_generate
```

- `--tp_size 1` is **required**: static NVFP4 weight scales are calibrated per full weight, so the recipe needs TP=1. Expert parallelism spreads the experts instead.
- Calibration is text-only (`cnn_nemotron_v2_mix`): the language model is calibrated on its own, and the vision tower is not run.
- The MoE experts are built as grouped GEMMs (`TEGroupedMLP`), which is what the recipe's `mlp.experts.linear_fc*` rules match. The log reports `moe_grouped_gemm: true`; do not pass `--no_moe_grouped_gemm`.
- The MTP head is built and stays BF16 (the recipe disables `language_model.mtp.*`).

</details>

---

### 3. Quantization-Aware Distillation (QAD)

QAD trains the quantized student against the frozen BF16 teacher with a logit KL loss only (no cross-entropy). See the [QAD section of the Megatron-Bridge README](../../README.md#quantization-aware-distillation-qad).

We used **8 nodes × 4 GB300 (32 GPUs)** with context parallelism 4 and expert parallelism 32. 600 iterations took **7.7 hours** (about 44 s/iteration at steady state). The validation KD loss fell from 1.36e-2 at iteration 100 to 1.01e-2 at iteration 600.

<details>
<summary>QAD command (click to expand)</summary>

```bash
# SBATCH --nodes=8 --ntasks-per-node=4 --gpus-per-node=4
srun ... python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
    --teacher_hf_path <nemotron_h_omni-checkpoint> \
    --student_hf_path <nemotron_h_omni-checkpoint> \
    --student_megatron_path /path/to/nemotron_h_omni_w4a4_megatron \
    --trust_remote_code \
    --tp_size 1 --pp_size 1 --cp_size 4 --ep_size 32 \
    --sft --sft_hf_dataset /path/to/qad_blend/train.jsonl \
    --sft_hf_validation /path/to/qad_blend/validation.jsonl \
    --sft_loss_mode assistant \
    --seq_length 32768 --mbs 1 --gbs 64 \
    --train_iters 600 \
    --lr 2e-5 --min_lr 5e-6 --lr_warmup_iters 30 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --no_async_save \
    --save_interval 100 --checkpoint_keep_last -1 \
    --eval_iters 4 --eval_interval 100 \
    --log_interval 10 \
    --output_dir /path/to/qad_output
```

- `--student_megatron_path` is the PTQ checkpoint from Section 2; `--student_hf_path` still points at the BF16 model, which supplies the architecture. The PTQ checkpoint was written at EP=4 and loads at EP=32.
- `--sft --sft_hf_dataset ... --sft_loss_mode assistant` renders each conversation with the model's chat template and computes the loss on assistant tokens only.
- `--seq_length 32768 --gbs 64`: 600 iterations cover 38,400 records, about 1.07 epochs of the 35,800-record training split.
- `--recompute_*` and `--no_async_save` are needed to fit: the async-save worker needs its own CUDA context and runs out of memory next to the resident teacher.
- The MTP head is built but receives no loss under KD-only training (`mtp_1 loss` and `mtp_2 loss` log as 0), so it stays bit-identical to the BF16 source.
- Each saved iteration is about 1.6 TB; size `--save_interval` and `--checkpoint_keep_last` to your storage.

</details>

---

### 4. Export

Export the QAD checkpoint to a Hugging Face checkpoint on one 4-GPU node (pipeline parallel 4, about 26 minutes), then add the NVFP4 KV-cache scales:

```bash
# SBATCH --nodes=1 --ntasks-per-node=4 --gpus-per-node=4
srun ... python -u /opt/Model-Optimizer/examples/megatron_bridge/export_quantized_megatron_to_hf.py \
    --hf_model_name_or_path <nemotron_h_omni-checkpoint> \
    --trust_remote_code \
    --megatron_path /path/to/qad_output/checkpoints \
    --export_unified_hf_path /path/to/nemotron_h_omni_w4a4_hf \
    --pp_size 4

python examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD/add_nvfp4_kv_scales.py \
    /path/to/nemotron_h_omni_w4a4_hf
```

- The exporter writes the quantized language model under `language_model.`, the MTP head from the model in BF16, and copies `vision_model.`, `vision_projector.` and `mlp1.` from the source checkpoint.
- It exports the latest `iter_*` under `--megatron_path`. To export another iteration, point `--megatron_path` at a directory holding a link to that `iter_*` and a matching `latest_checkpointed_iteration.txt`.
- [`add_nvfp4_kv_scales.py`](add_nvfp4_kv_scales.py) is needed because the recipe's KV quantizers use a constant amax, so the export carries NVFP4 KV metadata but no `k_scale` / `v_scale` tensors. The script writes the scale that amax implies, 448 / (6 × 448) = 1/6, for each of the 8 attention layers.

---

### 5. Evaluation and Serving

All checkpoints were served with vLLM on one 4-GPU GB300 node, TP=4 with expert parallelism. Quantized checkpoints used these engine arguments; the BF16 baseline used `--kv-cache-dtype auto`.

| Setting | Value |
| --- | --- |
| `--kv-cache-dtype` | `nvfp4` |
| `--max-model-len` | 262144 |
| `--mamba-ssm-cache-dtype` | `float16` |
| `--mamba-conv-cache-dtype` | `float32` |
| Mamba cache rounding | stochastic rounding, 5 Philox rounds |
| Reasoning parser | `nemotron_v3` |
| Chat template | the checkpoint's `chat_template.jinja` |

Benchmarks: AA-LCR (mean reward × 100, 16 repeats), SciCode (subtask accuracy), and GPQA Diamond without tools (pass@1 averaged over 8 samples, plus majority@8).

## Potential further improvements

- **Long reasoning:** on a separate long-reasoning benchmark (APEX shortlist, no tools), PTQ stayed close to BF16 (87.8 vs 88.4) while QAD 600 scored 83.2 and produced about 18% fewer output tokens. One hypothesis is that distilling on 32K-truncated data shortens long reasoning; a longer `--seq_length` or longer-context data would test it.
- **Data:** the blend is mostly math and code (64% of records) and has no agentic or tool-calling data.
- **Multimodal data:** QAD here is text-only. Image-text distillation would exercise the language model on vision-conditioned inputs as well.
