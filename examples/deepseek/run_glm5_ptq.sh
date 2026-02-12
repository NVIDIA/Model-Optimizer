#!/bin/bash
set -e

JOBID=${1:?Usage: $0 <job_id> [--skip-convert]}
SKIP_CONVERT=false
[[ "${2}" == "--skip-convert" ]] && SKIP_CONVERT=true

CONTAINER_IMAGE=$(readlink -f ~/fsw/containers/modelopt-v2.sqsh)
CONTAINER_MOUNTS=$(readlink -f ~/fsw):/fsw

HF_CKPT=/fsw/models/glm-5-bf16
DS_CKPT=/fsw/models/glm-5-ds
AMAX_PATH=/fsw/models/glm-5-nvfp4-amax
DS_V3_2_DIR=/fsw/Model-Optimizer/examples/deepseek/DeepSeek-V3.2-Exp
GLM5_CONFIG=${DS_V3_2_DIR}/inference/config_glm5.json

srun --overlap --jobid=${JOBID} --nodes=1 --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --export="ALL,HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN env var}" \
    bash -c '
set -e
pip install --no-deps -e /fsw/Model-Optimizer

cd /fsw/Model-Optimizer/examples/deepseek

# Step 1: Convert HF bf16 checkpoint to DeepSeek sharded format
if [ "'"${SKIP_CONVERT}"'" != "true" ]; then
    python '"${DS_V3_2_DIR}"'/inference/convert.py \
        --hf-ckpt-path '"${HF_CKPT}"' \
        --save-path '"${DS_CKPT}"' \
        --n-experts 256 \
        --model-parallel 8
else
    echo "Skipping conversion (--skip-convert)"
fi

# Step 2: Run PTQ calibration
torchrun --nproc-per-node 8 --master_port=12346 ptq.py \
    --model_path '"${DS_CKPT}"' \
    --config '"${GLM5_CONFIG}"' \
    --quant_cfg NVFP4_DEFAULT_CFG \
    --output_path '"${AMAX_PATH}"' \
    --trust_remote_code \
    --batch_size 8 \
    --calib_size 512
'
