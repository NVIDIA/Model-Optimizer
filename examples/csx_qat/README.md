# 3-bit CSX QAT for GPT-OSS-20B on tulu@8k

Recovers accuracy lost to 3-bit CSX post-training quantization by finetuning with
the quantizer in the forward loop, then exports the result in QLab deploy format.

The CSX scheme (rectangular `dH x dF` groups, per-PE codebooks, signed fp16
scales) and the trainable FakeQuantize come from QLab; Model-Optimizer supplies
the SFT pipeline and the MoE quantizer plumbing. **No Model-Optimizer source
changes are needed** — this directory is only the example wiring.

## Setup

```bash
pip install -U "nvidia-modelopt[hf]"
pip install -e /path/to/quantization_tool     # QLab, branch ak/csx-fakequant-qat
```

Two GPT-OSS-specific notes:

* It cannot use `flash_attention_2` or `sdpa` at all (learned attention sinks).
  Use `kernels-community/vllm-flash-attn3`, which needs `kernels>=0.15.2` —
  conflicting with the `<0.13` pin in `examples/gpt-oss/requirements.txt`.
  `eager` also works but OOMs at larger eval batches.
* Released `nvidia-modelopt` wheels may not ship the carrier recipe
  `general/ptq/mxfp4_mlp_weight_only`; it is a thin wrapper over a preset that
  *is* shipped, so copy it from `main` if `load_recipe` fails.

## Run

```bash
# 0. corpus: 8,000 train / 500 held-out from allenai/tulu-3-sft-mixture (~5.7M tokens)
python examples/csx_qat/prepare_tulu8k.py --out /scratch/tulu8k

# 1. PTQ: fit the artifacts the QAT run will train under
python examples/csx_qat/fit_csx.py --model openai/gpt-oss-20b --num-bits 3 \
    --out artifacts/gptoss20b_csx3bit.pt

# 2. the CEILING: high-precision SFT, no quantization
accelerate launch --config_file examples/gpt-oss/configs/zero3.yaml \
    examples/csx_qat/sft_csx.py \
    --config examples/csx_qat/configs/sft_gptoss20b_tulu8k.yaml \
    --dataset_name /scratch/tulu8k --csx_artifacts none \
    --output_dir out/ceiling

# 3. QAT
accelerate launch --config_file examples/gpt-oss/configs/zero3.yaml \
    examples/csx_qat/sft_csx.py \
    --config examples/csx_qat/configs/sft_gptoss20b_tulu8k.yaml \
    --dataset_name /scratch/tulu8k \
    --csx_artifacts artifacts/gptoss20b_csx3bit.pt \
    --output_dir out/csx3-qat

# 4. the FLOOR: PTQ applied to the *ceiling* model, and the deployed QAT model.
#    Consolidate each run's ZeRO shards first (see "Checkpoints" below).
python examples/csx_qat/eval_csx.py --model out/ceiling-consolidated \
    --dataset /scratch/tulu8k --label ceiling
python examples/csx_qat/eval_csx.py --model out/ceiling-consolidated \
    --dataset /scratch/tulu8k --csx-artifacts artifacts/ptq_on_ceiling.pt \
    --csx-bake --label floor
python examples/csx_qat/eval_csx.py --model out/csx3-qat-consolidated \
    --dataset /scratch/tulu8k --csx-artifacts artifacts/gptoss20b_csx3bit.pt \
    --csx-bake --label qat_deployed

# 5. export for deployment
python examples/csx_qat/export_qlab.py --model out/csx3-qat-consolidated \
    --artifacts artifacts/gptoss20b_csx3bit.pt --out out/csx3-deploy
```

Note the floor is PTQ fitted on the **finetuned** model, not the base model — fit
step 1 again against `out/ceiling-consolidated` to produce
`artifacts/ptq_on_ceiling.pt`. That distinction is not cosmetic; see below.

## Measured results

1 node, 8x H100-80GB, ZeRO-3, 250 steps (1 epoch, effective batch 32). Corpus
loss is token-weighted over the corpus's own 500-example held-out split
(346,085 label tokens). PTQ fit SQNR 12.5-13.4 dB.

| condition | loss | token acc |
|---|---|---|
| BF16 + SFT (ceiling) | 0.7917 | 0.7991 |
| CSX 3-bit PTQ on the SFT model (floor) | 0.8717 | 0.7825 |
| **CSX 3-bit QAT, deployed** | **0.8270** | **0.7926** |

QAT closes **55.9%** of the loss gap and 60.8% of the accuracy gap. For contrast,
4-bit on the same corpus closes 50.0% / 62.9% from a much smaller gap (0.0270 vs
0.0800), and 3-bit QAT (0.8270) still does not beat plain 4-bit PTQ (0.8187) —
3-bit's gap starts ~3x larger, so it recovers more but does not catch up.

Peak memory 43.8 GiB/rank at `CSX_EXPERT_CHUNK=4` (3-bit; 45.3 GiB at 4-bit).
`CSX_EXPERT_CHUNK` bounds the op's peak by chunking the expert dimension and is
bit-identical for any value.

## Parallelism

| | peak/rank | notes |
|---|---|---|
| DeepSpeed ZeRO-3 | 43.8 GiB | `CSX_EXPERT_CHUNK=4`. Use `examples/gpt-oss/configs/zero3.yaml` — it sets `deepspeed_moe_layer_cls_names`, without which ZeRO-3 traces per-expert parameter access (which varies with routing) and ranks fail with "Detected a disagreement on list length" |
| FSDP2 | 60.9 GiB | `configs/fsdp2.yaml`, needs `CSX_EXPERT_CHUNK=2`. Verified for 1 step, not a full equivalence run. Known issue: completes training, eval and save, then hangs at teardown on a stuck NCCL collective — kill it afterwards |

## Checkpoints

**Do not trust the trainer's own save.** On transformers <= 5.14,
`save_pretrained` for an MXFP4-native model silently collapses every layer's
expert weights into two prefix-less keys named `gate_up_proj$` / `down_proj$`,
dropping 19.1B of 20.9B parameters with no error. Reconstruct from the ZeRO
shards instead:

```bash
python out/csx3-qat/checkpoint-*/zero_to_fp32.py \
    out/csx3-qat/checkpoint-250 out/csx3-qat-consolidated --safe_serialization
```

Then copy `config.json` and the tokenizer files across, and sanity-check that all
24 layers' expert weights are present.

## Pitfalls

**A QAT checkpoint is only valid with its quantizer.** Straight-through training
moves weights to positions that are good *after* snapping to the codebook, not in
full precision. Measured at 3-bit: 1.0897 unquantized versus 0.8733 deployed —
i.e. the bare weights are worse than the PTQ floor they started from. Always
export before evaluating or shipping.

**Do not measure quantization damage on a base model.** On a model not adapted to
the eval distribution, cross-entropy and token accuracy move *opposite* to
damage: CSX 3-bit (12.4 dB SQNR) scored better than 4-bit (18.7 dB), which scored
better than BF16. Weight noise raises entropy on a format the base model was
never trained on. Take the PTQ reference on the finetuned model, and cross-check
with KL against the unquantized model plus top-1 agreement.

**Quote recovery with its denominator.** How much of the gap QAT closes depends
strongly on how quantization-sensitive the target distribution is: the same 4-bit
scheme closes 17.5% on one corpus and 50.0% on another. Report the ceiling, the
floor and the absolute recovery, not a bare percentage.
