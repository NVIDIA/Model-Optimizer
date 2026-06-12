# Quantize Deepseek models to FP4

This example will demonstrate the steps to quantize DeepSeek models to FP4 and export a unified checkpoint that can be deployed with TRT-LLM.

## Setup

Due to the model size, currently it requires 8xH200 or 16xH100 to quantize the FP8 model, we will use 8xH200 as example.

## Convert the HF checkpoint for deepseek FP8 inference

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
torchrun --nproc-per-node 8 --master_port=12346 ptq.py --model_path $DS_CKPT --config DeepSeek-V3/inference/configs/config_671B.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH
```

DeepSeek V3.2

```bash
torchrun --nproc-per-node 8 --master_port=12346 ptq.py --model_path $DS_CKPT --config DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH
```

### Quantize the FP8 hf checkpoint to FP4

We provide a one-step-script which will:

- Quantize the weights to NVFP4
- Copy miscellaneous files to the quantized checkpoint

```bash
./quantize_fp8_to_nvfp4.sh --amax_path $FP4_QUANT_PATH --fp4_output_path $HF_FP4_PATH --fp8_hf_path $HF_FP8_CKPT --world_size 8
```
