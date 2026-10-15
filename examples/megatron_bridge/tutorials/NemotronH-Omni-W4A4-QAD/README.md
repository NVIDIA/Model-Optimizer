# NemotronH Omni VL: W4A4 NVFP4 PTQ + QAD + Deployment

End-to-end W4A4 quantization of a NemotronH omni vision-language checkpoint (architecture `NemotronH_Omni_Reasoning_V3`), a Mamba-Transformer MoE language model with a vision tower: mixed NVFP4/FP8 post-training quantization → Quantization-Aware Distillation (QAD) → Hugging Face export → vLLM serving. This document covers:

1. **[Data Preparation](#1-data-preparation)**: the chat-format QAD blend, from `data_blend.yaml`
2. **[Quantization](#2-quantization)**: W4A4 PTQ with `examples/megatron_bridge/quantize.py`
3. **[QAD](#3-quantization-aware-distillation-qad)**: distillation from the BF16 teacher with `examples/megatron_bridge/distill.py`
4. **[Export](#4-export)**: the deployable Hugging Face checkpoint
5. **[Evaluation and Serving](#5-evaluation-and-serving)**: the vLLM settings behind the results

Only the language model (including its MTP head) is quantized; the vision tower and projector stay BF16 and are copied from the source checkpoint.

## Results

All checkpoints were served with the same vLLM settings (see [Section 5](#5-evaluation-and-serving)); the quantized ones with an FP8 KV cache. BF16 is the mean of two runs.

| Benchmark | BF16 (teacher) | W4A4 PTQ | W4A4 QAD (600 iters) |
| --- | --- | --- | --- |
| AA-LCR | 68.63 | 68.62 | 67.94 |
| AA-Omniscience | 26.21 | 25.63 | 25.87 |
| AA-Omniscience (non-hallucination) | 70.03 | 71.34 | 69.65 |
| GPQA Diamond | 89.52 | 89.84 | 89.90 |
| HLE (no tools) | 31.79 | 30.49 | 30.72 |
| MMLU-ProX | 76.91 | 74.41 | 81.15 |
| SciCode (subtask) | 47.13 | 46.48 | 46.58 |
| SciCode (problem) | 15.10 | 13.65 | 15.38 |
| τ³-bench banking | 18.88 | 17.40 | 20.10 |

MMLU-ProX scores a response only if it ends with the answer phrase in the question's language. Most of the QAD gain comes from fewer responses missing that phrase (11.3% for PTQ, 5.4% after QAD); on the responses with a readable answer, accuracy rises from 83.9 to 85.8.

The exported W4A4 checkpoint is **70 GB**.

---

## Steps to Reproduce

**Environment:** a NeMo 26.08 container (`nvcr.io/nvidia/nemo:26.08`) on GB300 (4 GPUs per node, aarch64), with this repository mounted at `/opt/Model-Optimizer` and Megatron-Bridge `main` (and its pinned Megatron-LM) mounted at `/opt/Megatron-Bridge`. See the [Megatron-Bridge README](../../README.md) for the container setup. Serving NVFP4 checkpoints requires a Blackwell GPU.

PTQ and export launch with `torchrun`; QAD runs on 8 nodes with the `sbatch` script in [Section 3](#3-quantization-aware-distillation-qad).

### 1. Data Preparation

QAD trains on a weighted blend of 11 text-only chat sources, defined in [`data_blend.yaml`](data_blend.yaml) and computing the loss on assistant tokens only. Set `output_dir` in that file, then write the blend as chat JSONL with the [chat JSONL mode of `prepare_megatron_data_blend`](../../../dataset/MEGATRON_DATA_PREP.md#chat-jsonl-blends-for-assistant-only-loss):

```bash
python -m modelopt.torch.utils.plugins.prepare_megatron_data_blend --config data_blend.yaml
```

This writes `train.jsonl` (35,800 conversations) and `validation.jsonl` (1,000) to `output_dir`. Run it in the container: Megatron-Bridge's text-only chat filter selects the rows.

---

### 2. Quantization

The recipe is [`models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse`](../../../../modelopt_recipes/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse.yaml), the PTQ stage for NVIDIA Nemotron 3.5 Super VL and the same file `hf_ptq.py` uses: its rules list both the Hugging Face and the Megatron-Core module names where they differ.

| Component | Precision |
| --- | --- |
| MoE routed and shared experts | **NVFP4 W4A4**, block 16, MSE-selected static weight scales |
| Mamba `in_proj` / `out_proj` | **FP8 W8A8** per-tensor |
| Attention `q`/`k`/`v`/`o_proj` (Megatron-Core `linear_qkv` / `linear_proj`) | **FP8 W8A8** per-tensor |
| `lm_head` (Megatron-Core `output_layer`) | **FP8 W8A8** |
| KV cache | **FP8** cast at a fixed range (no `k_scale` / `v_scale` exported) |
| MTP block-1 routed and shared experts | **NVFP4 W4A4**, as above |
| MTP block-0 KV cache | **FP8**, calibrated |
| Vision tower, projector, the rest of the MTP head, latent MoE, routers, embeddings | BF16 |

PTQ runs on **one 4-GPU GB300 node** with 512 calibration samples.

<details>
<summary>PTQ command (click to expand)</summary>

```bash
torchrun --nproc_per_node 4 /opt/Model-Optimizer/examples/megatron_bridge/quantize.py \
    --hf_model_name_or_path <nemotron_h_omni-checkpoint> \
    --trust_remote_code \
    --recipe models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse \
    --tp_size 1 --ep_size 4 --pp_size 1 \
    --calib_dataset_name cnn_nemotron_v2_mix \
    --calib_num_samples 512 \
    --calib_batch_size 1 \
    --export_megatron_path /path/to/nemotron_h_omni_w4a4_megatron \
    --skip_generate
```

- `--tp_size 1` is **required**: with tensor parallelism, static NVFP4 weight scales are per-shard but the Megatron checkpoint saves them as replicated, so calibration raises `NotImplementedError` for TP > 1. Expert parallelism spreads the experts instead.
- Calibration is text-only (`cnn_nemotron_v2_mix`): the language model is calibrated on its own, and the vision tower is not run.
- The MTP head is quantized too, so the exported checkpoint has it in the serving format: its MoE block (`language_model.mtp.layers.0.mtp_model_layer.layers.1` in Megatron-Core) gets the same NVFP4 treatment as the backbone experts, and the rest of the head stays BF16.

</details>

---

### 3. Quantization-Aware Distillation (QAD)

QAD trains the quantized student against the frozen BF16 teacher with a logit KL loss only (no cross-entropy). See the [QAD section of the Megatron-Bridge README](../../README.md#quantization-aware-distillation-qad).

We used **8 nodes × 4 GB300 (32 GPUs)** with context parallelism 4 and expert parallelism 32.

<details>
<summary>QAD command (click to expand)</summary>

`srun` starts one task per GPU (32 in all), and each task maps its Slurm rank variables to the `RANK` / `WORLD_SIZE` / `LOCAL_RANK` that `torch.distributed` reads, with a shared `MASTER_ADDR` / `MASTER_PORT`. Add the container options your cluster needs (image, mounts as in the Environment note) to `srun`.

```bash
#!/bin/bash
#SBATCH --nodes=8 --ntasks-per-node=4 --gpus-per-node=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) MASTER_PORT=29500
srun bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
    --teacher_hf_path <nemotron_h_omni-checkpoint> \
    --student_hf_path <nemotron_h_omni-checkpoint> \
    --student_megatron_path /path/to/nemotron_h_omni_w4a4_megatron \
    --trust_remote_code \
    --tp_size 1 --pp_size 1 --cp_size 4 --ep_size 32 \
    --sft --sft_hf_dataset /path/to/qad_blend/train.jsonl \
    --sft_hf_validation /path/to/qad_blend/validation.jsonl \
    --sft_loss_mode assistant \
    --freeze_mtp \
    --seq_length 32768 --mbs 1 --gbs 64 \
    --train_iters 600 \
    --lr 2e-5 --min_lr 5e-6 --lr_warmup_iters 30 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --no_async_save \
    --save_interval 100 --checkpoint_keep_last -1 \
    --eval_iters 4 --eval_interval 100 \
    --log_interval 10 \
    --output_dir /path/to/qad_output'
```

- `--student_megatron_path` is the PTQ checkpoint from Section 2; `--student_hf_path` still points at the BF16 model, which supplies the architecture. The PTQ checkpoint was written at EP=4 and loads at EP=32.
- `--sft --sft_hf_dataset ... --sft_loss_mode assistant` renders each conversation with the model's chat template and computes the loss on assistant tokens only.
- `--seq_length 32768 --gbs 64`: 600 iterations cover 38,400 records, about 1.07 epochs of the 35,800-record training split.
- `--recompute_*` and `--no_async_save` are needed to fit: the async-save worker needs its own CUDA context and runs out of memory next to the resident teacher.
- `--freeze_mtp` keeps the MTP head out of training: its loss is skipped and its parameters are frozen, so the exported MTP head keeps its PTQ weights. Without it, the quantized MTP head's cross-entropy loss trains the head and, through the hidden states, embeddings and output layer it shares, the rest of the student too.
- Each saved iteration is about 1.6 TB; size `--save_interval` and `--checkpoint_keep_last` to your storage.

</details>

---

### 4. Export

Export the QAD checkpoint to a Hugging Face checkpoint on one 4-GPU node (pipeline parallel 4):

```bash
torchrun --nproc_per_node 4 /opt/Model-Optimizer/examples/megatron_bridge/export_quantized_megatron_to_hf.py \
    --hf_model_name_or_path <nemotron_h_omni-checkpoint> \
    --trust_remote_code \
    --megatron_path /path/to/qad_output/checkpoints \
    --export_unified_hf_path /path/to/nemotron_h_omni_w4a4_hf \
    --pp_size 4
```

- The exporter writes the quantized language model, including the MTP head, under `language_model.`, and copies `vision_model.`, `vision_projector.` and `mlp1.` from the source checkpoint.
- The MTP head exports through the standard language-model mapping; `--export_extra_modules` is for EAGLE and Medusa heads and is not needed here.
- The MTP head can be replaced after export by a separately trained MTP head in the same format.
- It exports the latest `iter_*` under `--megatron_path`. To export another iteration, point `--megatron_path` at a directory holding a link to that `iter_*` and a matching `latest_checkpointed_iteration.txt`.
- The backbone KV cache is cast at the fixed FP8 range, so the export has no backbone `k_scale` / `v_scale`; vLLM then uses a scale of 1.0, which is that range. Only the MTP attention carries a calibrated pair.

---

### 5. Evaluation and Serving

Serve with vLLM on one 4-GPU GB300 node, TP=4 with expert parallelism. Quantized checkpoints use these engine arguments; the BF16 baseline uses `--kv-cache-dtype auto`.

| Setting | Value |
| --- | --- |
| `--kv-cache-dtype` | `fp8` |
| `--max-model-len` | 262144 |
| `--mamba-ssm-cache-dtype` | `float16` |
| `--mamba-conv-cache-dtype` | `float32` |
| Mamba cache rounding | stochastic rounding, 5 Philox rounds |
| Reasoning parser | `nemotron_v3` |
| Chat template | the checkpoint's `chat_template.jinja` |
