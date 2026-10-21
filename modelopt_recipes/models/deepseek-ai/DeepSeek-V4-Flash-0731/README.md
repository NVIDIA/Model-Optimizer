# DeepSeek-V4-Flash IQ2_XXS with mergeable LoRA QAD

Use the standard Megatron-Bridge entry points to import the official HF checkpoint,
calibrate IQ2_XXS routed experts, train adapters, and export packed IQ2 or folded BF16.
Bridge decodes the original MXFP4 expert weights and FP8 weights into BF16 during import.
Other weights stay floating point, including native FP32 mHC controls.

## Inputs and topology

- Prepare a main-only HF metadata view of the official checkpoint: keep the original
  shards, remove `mtp.*` index entries, set `num_nextn_predict_layers=0`, and trim
  per-layer metadata to the main depth. Preserve the original checkpoint separately.
- Use TP1 / PP1 / CP1 / ETP1 and SequentialMLP (`--no_moe_grouped_gemm`). The recorded
  integration uses EP8 on two nodes with four GB300 GPUs each. V4 TP/PP splitting and
  MTP export are not supported by this initial implementation.
- Supply a calibration JSONL and Megatron pre-tokenized QAD training data. The recorded
  calibration uses a seven-source `nemotron-post-training-v3` mix, 512 windows of 4096
  tokens, and batch size 1. Pin dataset revisions and keep calibration, training and
  held-out documents disjoint.

Run the commands from the repository root on both nodes. Set `NODE_RANK` to 0 or 1
and `MASTER_ADDR` to the first node's address. Set the input/output variables
`HF_MAIN`, `CALIB_JSONL`, `TRAIN_PREFIX`, `PTQ_MEGATRON`, `QAD_OUTPUT`,
`TRAINED_ITERATION`, `PACKED_HF`, and `FOLDED_HF` to your paths.

```bash
LAUNCH=(torchrun --nnodes=2 --nproc_per_node=4 --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}")
```

## Calibrate and save the training checkpoint

```bash
"${LAUNCH[@]}" examples/megatron_bridge/quantize.py \
  --hf_model_name_or_path "$HF_MAIN" --trust_remote_code \
  --recipe models/deepseek-ai/DeepSeek-V4-Flash-0731/ptq/iq2_xxs_experts-gptq \
  --no_moe_grouped_gemm --tp_size 1 --pp_size 1 --ep_size 8 --cp_size 1 \
  --dsa_kernel_backend none \
  --calib_dataset_name "$CALIB_JSONL" --calib_num_samples 512 \
  --calib_batch_size 1 --seq_length 4096 \
  --lora_rank 8 --lora_alpha 16 \
  --lora_target_modules '*mlp.experts.local_experts.*.linear_fc1' \
    '*mlp.experts.local_experts.*.linear_fc2' \
  --skip_generate --export_megatron_path "$PTQ_MEGATRON"
```

Omit `--compress`: QAD needs floating-point training weights, ModelOpt quantizer state,
GPTQ payloads, and zero-initialized adapters. The recipe uses importance-weighted,
layerwise GPTQ with block size 256 and damping 0.3. It targets only routed-expert
gate/up and down projections.

## Train with distill.py

```bash
"${LAUNCH[@]}" examples/megatron_bridge/distill.py \
  --student_hf_path "$HF_MAIN" --teacher_hf_path "$HF_MAIN" --trust_remote_code \
  --student_megatron_path "$PTQ_MEGATRON" \
  --no_moe_grouped_gemm --tp_size 1 --pp_size 1 --ep_size 8 --cp_size 1 \
  --data_paths "$TRAIN_PREFIX" --seq_length 512 --mbs 1 --gbs 8 \
  --kd_loss_alpha 1.0 --moe_aux_loss_coeff 0 --moe_router_bias_update_rate 0 \
  --dsa_kernel_backend none --deterministic_mode --mhc_fused_backend triton \
  --recompute_granularity selective --recompute_modules mhc moe \
  --lr 0.001 --min_lr 0.001 --lr_warmup_iters 0 \
  --train_iters 4 --save_interval 2 --checkpoint_keep_last -1 \
  --eval_iters 0 --no_async_save --log_interval 1 --output_dir "$QAD_OUTPUT"
```

Only the routed-expert rank-8 / alpha-16 LoRA factors are trainable. The frozen teacher
runs online; leave `--logit_kl_top_k` and `--logit_kl_top_p` unset for full-vocabulary KL.
The four-step configuration is an integration smoke test, not a tuned training recipe.
Exact resume additionally requires the Bridge data-setup RNG and Core frozen-optimizer
checkpoint fixes described in the PR's dependency notes.

## Merge and export

Keep the unmerged training checkpoint for resume. Point `TRAINED_ITERATION` to the
saved iteration directory. The standard exporter restores and merges the adapters.

```bash
"${LAUNCH[@]}" examples/megatron_bridge/export_quantized_megatron_to_hf.py \
  --hf_model_name_or_path "$HF_MAIN" --trust_remote_code \
  --megatron_path "$TRAINED_ITERATION" --export_unified_hf_path "$PACKED_HF" \
  --pp_size 1 --ep_size 8 --no_moe_grouped_gemm
```

Run a separate export with `--vllm_fakequant_export` and
`--export_unified_hf_path "$FOLDED_HF"` to store the quantized expert values as BF16.
Packed export stores IQ2_XXS expert payloads; non-experts remain floating point in both
representations. Optimized packed-IQ2 serving is a separate runtime integration.

## Recorded validation

The v02 stack passed full-model GPTQ, exact eight-rank PTQ reload, four QAD updates,
fresh step-two resume, adapter merge, and both exports. All 33,024 packed expert tensors
decode exactly to folded BF16; all 34,223 exported tensors were compared. Fresh folded-HF
import reproduces the merged Bridge output probe on all eight ranks. The output probe
uses one 32-token input; held-out quality and production training are separate checks.
