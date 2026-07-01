# Quantize Deepseek models to FP4

This example will demonstrate the steps to quantize DeepSeek models to FP4 and export a unified checkpoint that can be deployed with TRT-LLM.

## Setup

Due to the model size, currently it requires 8xH200 or 16xH100 to quantize the FP8 model, we will use 8xH200 as example.

## Directory Layout

- `deepseek_v3/`: DeepSeek V3, R1, V3.1, and V3.2 FP4 quantization.
- `deepseek_v4/`: DeepSeek V4 routed-expert NVFP4 quantization.

## DeepSeek V3 FP4

### Convert the HF checkpoint for DeepSeek FP8 inference

```bash
# set up variables to run the example
export HF_FP8_CKPT={path_to_downloaded_hf_checkpoint}
export DS_CKPT={path_to_save_converted_checkpoint}
export FP4_QUANT_PATH={path_to_save_quantization_results}
export HF_FP4_PATH={path_to_save_the_final_FP4_checkpoint}
```

### DeepSeek V3 R1 V3.1

```bash
# download the FP8 checkpoint from Hugginface. This is an example of DeepSeek-R1
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir $HF_FP8_CKPT

# clone DeepSeek-V3 (base model of R1) Github repository for FP8 inference,
git clone https://github.com/deepseek-ai/DeepSeek-V3.git && cd DeepSeek-V3 && git checkout 9b4e978
```

### [Experimental] DeepSeek V3.2

```bash
# download the FP8 checkpoint from Hugginface.
huggingface-cli download deepseek-ai/DeepSeek-V3.2-Exp --local-dir $HF_FP8_CKPT

# clone DeepSeek-V3.2 Github repository for FP8 inference,
git clone https://github.com/deepseek-ai/DeepSeek-V3.2-Exp.git && cd DeepSeek-V3.2-Exp && git checkout 87e509a

# Install requirements
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git
pip install -r inference/requirements.txt
```

### Convert the Checkpoint

```bash
# convert the HF checkpoint to a specific format for Deepseek
python inference/convert.py --hf-ckpt-path $HF_FP8_CKPT --save-path $DS_CKPT --n-experts 256 --model-parallel 8
```

## Post-training quantization

### Run the calibration scripts

DeepSeek V3, R1, V3.1

```bash
torchrun --nproc-per-node 8 --master_port=12346 deepseek_v3/ptq.py --model_path $DS_CKPT --config DeepSeek-V3/inference/configs/config_671B.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH
```

DeepSeek V3.2

```bash
torchrun --nproc-per-node 8 --master_port=12346 deepseek_v3/ptq.py --model_path $DS_CKPT --config DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH
```

#### MoE expert calibration

By default, calibration uses the model's native top-k routing and then runs a
post-calibration sync that sets every expert's `input_quantizer.amax` (w1/w2/w3)
to the per-layer global peer max (all-reduced across EP ranks).
`weight_quantizer.amax` stays per-expert; any uncalibrated expert falls back to
a compute path over the dequantized FP8 weight. This mirrors the
`layer_sync_moe_local_experts_amax` flow that mtq runs automatically for
QuantSequentialMLP-derived MoEs.

To restore the original behavior — force every token through every expert
during calibration (slower, ~2x forwards, no post-calibration sync) — pass
`--calib_all_experts`:

```bash
torchrun --nproc-per-node 8 --master_port=12346 deepseek_v3/ptq.py --model_path $DS_CKPT --config DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH --calib_all_experts
```

A summary of every TensorQuantizer is written to `$FP4_QUANT_PATH/.quant_summary.txt`.

### Quantize the FP8 hf checkpoint to FP4

We provide a one-step-script which will:

- Quantize the weights to NVFP4
- Copy miscellaneous files to the quantized checkpoint

```bash
./deepseek_v3/quantize_fp8_to_nvfp4.sh --amax_path $FP4_QUANT_PATH --fp4_output_path $HF_FP4_PATH --fp8_hf_path $HF_FP8_CKPT --world_size 8
```

## DeepSeek V4 routed-expert NVFP4

DeepSeek V4 uses a mixed native checkpoint layout. The V4 recipe quantizes
only the routed experts to NVFP4 W4A4 and leaves attention projections, the
router gate, shared experts, embeddings, and the LM head in their original
formats.

### Prepare the MP checkpoint

Keep experts in MXFP4 when resharding with DeepSeek's own `convert.py`:

```bash
export DS_V4=/path/to/DeepSeek-V4-Pro
export MP=8
export MP_CKPT=/path/to/DeepSeek-V4-Pro-mp${MP}-mxfp4
export AMAX=/path/to/amax-nvfp4-experts
export HF_NVFP4_PATH=/path/to/DeepSeek-V4-Pro-nvfp4-experts

python ${DS_V4}/inference/convert.py \
    --hf-ckpt-path ${DS_V4} \
    --save-path ${MP_CKPT} \
    --n-experts 384 \
    --model-parallel ${MP}
```

### Calibrate routed experts

Single node:

```bash
torchrun --nproc-per-node ${MP} --master_port 12346 deepseek_v4/ptq.py \
    --model_path ${MP_CKPT} \
    --config ${DS_V4}/inference/config.json \
    --dsv4_inference_dir ${DS_V4}/inference \
    --output_path ${AMAX}
```

Two 4-GPU nodes for `MP=8`:

```bash
# node 0
torchrun --nnodes=2 --node_rank=0 --master_addr=<ip> --master_port=12346 \
    --nproc-per-node 4 deepseek_v4/ptq.py \
    --model_path ${MP_CKPT} \
    --config ${DS_V4}/inference/config.json \
    --dsv4_inference_dir ${DS_V4}/inference \
    --output_path ${AMAX}

# node 1
torchrun --nnodes=2 --node_rank=1 --master_addr=<ip> --master_port=12346 \
    --nproc-per-node 4 deepseek_v4/ptq.py \
    --model_path ${MP_CKPT} \
    --config ${DS_V4}/inference/config.json \
    --dsv4_inference_dir ${DS_V4}/inference \
    --output_path ${AMAX}
```

### Export back to HF shard layout

`deepseek_v4/quantize_to_nvfp4.py` operates on the original HF-style V4 checkpoint and
produces a new HF-style checkpoint with routed expert weights replaced by
NVFP4 tensors plus `weight_scale`, `weight_scale_2`, and `input_scale`.

```bash
python deepseek_v4/quantize_to_nvfp4.py \
    --amax_path ${AMAX} \
    --source_ckpt ${DS_V4} \
    --output_ckpt ${HF_NVFP4_PATH}
```

The output includes an updated `model.safetensors.index.json`, a `config.json`
with `quantization_config.moe_quant_algo = "NVFP4"`, and `hf_quant_config.json`
describing the mixed NVFP4 expert layers.

When the source routed experts are MXFP4 (as in the V4 release), add
`--cast_mxfp4_to_nvfp4` for a lossless weight conversion — recommended over the
default lossy dequant/re-quant path. See below.

#### Lossless MXFP4 → NVFP4 weight cast (`--cast_mxfp4_to_nvfp4`)

The routed experts in the source checkpoint are already MXFP4 (E2M1 nibbles +
a power-of-two E8M0 scale per 32-element block). Without the flag, the export
dequantizes them to BF16 and re-quantizes to NVFP4 using the calibrated
per-tensor weight amax, which re-derives the per-block scales from the data and
is therefore lossy. With `--cast_mxfp4_to_nvfp4`, the per-tensor `scale_2` is
pinned to `2^(k_max - 8)` and each per-block E4M3 scale to `2^(k_j - m)` straight
from the source E8M0 scales, so `per_block_scale * scale_2 = 2^k_j` and the NVFP4
nibbles equal the source MXFP4 nibbles bit-for-bit (for every block whose `k_j`
lands in E4M3's representable window; the rare out-of-range block falls back to a
data-derived scale). The flag only affects routed-expert **weights** — activation
`input_scale` still comes from `${AMAX}` calibration — and the run prints a
`[cast] lossless MXFP4->NVFP4 blocks: …` summary. This mirrors the GPTOSS cast in
[`examples/hf_ptq/cast_mxfp4_to_nvfp4.py`](../hf_ptq/cast_mxfp4_to_nvfp4.py); the
V4 twist is that w1/w3 share one `scale_2` (fused GEMM1), so `k_max` is taken over
both projections.
