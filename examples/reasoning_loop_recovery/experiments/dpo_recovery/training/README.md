# DPO training and NVFP4 export

**Procedure used for the reported results:** Path 4—NeMo RL installed in the original QAD
NeMo container. See [path 4](paths/04_rl_in_qad_container/README.md) for setup and compatibility
fixes. The other six paths are alternative environments under investigation.

The policy and frozen reference both start from the **round-3 iterative recovery QAD checkpoint**.
The reference is a frozen copy of that fake-quantized model, not the BF16 teacher used during QAD.
Keep the saved ModelOpt state, quantizer configuration, and scales; do not replace the input with
an exported NVFP4 deployment checkpoint or recalibrate it as a new PTQ model.

DPO minimizes `-log sigmoid(beta * ((log π(chosen|x) - log π(rejected|x)) -
(log πref(chosen|x) - log πref(rejected|x))))`, with beta 0.05. Use summed output-token log
probabilities, including the chosen EOS but no rejected EOS. Mask input tokens. There is no SFT
cross-entropy term or teacher-logit KL distillation loss. Quantization remains active while policy
weights update; the reference does not update.

## Settings and environments

[dpo.yaml](dpo.yaml) preserves the full release-path configuration, including the effective Megatron
optimizer rather than the inactive AutoModel optimizer fields. Paths use `${VARIABLE}` placeholders
and need substitution in an external workspace. The experiment driver merges this record with
NeMo RL's `examples/configs/dpo.yaml`, validates the data processor and framework provenance, then
calls `nemo_rl.algorithms.dpo.setup()` and `dpo_train()`. It uses NeMo RL's DPO implementation;
there is no custom DPO loss or replacement training loop.

Use 104 training pairs, 13 steps, global batch 8, micro batch 1, maximum sequence 32,768, constant
learning rate 1e-6, seed 20261024, TP=1 / EP=4 / PP=1 / CP=1, and one node with four GB300 GPUs.
Enable full activation recomputation and **128-token chunked linear log-probability computation**.
The release-path settings are `policy.megatron_cfg.use_linear_ce_fusion_loss: true` and
`linear_ce_fusion_chunk_size: 128`; earlier `use_fused_linear_logprobs` options were ignored there.
An unchunked full-vocabulary FP32 temporary caused an approximately 39.58-GiB allocation failure.

See [versions.yaml](versions.yaml) for image tags, actual package versions, and commit pins.
A matching Bridge version string does not imply matching source commits. The seven paths are
[documented separately](paths/README.md); none is selected as the final supported workflow.

## Checkpoint validation

Before optimization, verify architecture parity and that every requested language-model shard is
present. Audit policy and reference quantizers on each rank immediately after restoration; a
nonzero active quantizer count is required. Preserve the original checkpoint unchanged.

Save the policy and optimizer after step 13. Start a **separate process** with the same path-specific
runtime, reload the checkpoint, require `save_state.total_steps == 13`, and validate all 11 diagnostic
pairs with finite losses. `DPOSaveState` is a dataclass, not a dictionary. A marker that training
finished alone is insufficient evidence that checkpoint reload works.

Path 4's saved policy is under `step_13/policy/weights/iter_0000000`; the Megatron iteration directory
is zero even though NeMo RL's `training_info.json` records 13 completed training steps. Do not select
an older smoke checkpoint by looking only at `iter_*` filenames.

## Export and evaluate

Path 4 used the original QAD ModelOpt source and NeMo container for export after reload validation:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${MODEL_OPT_SOURCE}/examples/megatron_bridge/export_quantized_megatron_to_hf.py" \
  --hf_model_name_or_path "${TEACHER_HF_PATH}" \
  --megatron_path "${DPO_POLICY_CHECKPOINT}" \
  --pp_size 4 \
  --export_unified_hf_path "${EXPORT_DIR}"
```

These commands run inside an approved four-GPU allocation. Here `TEACHER_HF_PATH` supplies the HF
architecture/tokenizer; no teacher is used in DPO. Export uses PP=4 even though training uses EP=4.
Require safetensors and an NVFP4 `quantization_config` before serving. Export does not by itself
validate accuracy; follow the [SciCode protocol](../evaluation/README.md).
