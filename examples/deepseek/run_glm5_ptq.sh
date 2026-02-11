#!/bin/bash
set -e

JOBID=${1:?Usage: $0 <job_id>}

CONTAINER_IMAGE=$(readlink -f ~/fsw/containers/modelopt.sqsh)
CONTAINER_MOUNTS=$(readlink -f ~/fsw):/fsw

srun --overlap --jobid=${JOBID} --nodes=1 --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -c '
set -e
pip install --no-deps -e /fsw/Model-Optimizer
pip install git+https://github.com/huggingface/transformers

cd /fsw/Model-Optimizer/examples/deepseek

python ptq.py \
    --model_path /fsw/models/glm-5-bf16 \
    --model_type hf \
    --quant_cfg NVFP4_DEFAULT_CFG \
    --output_path /fsw/models/glm-5-nvfp4-amax \
    --trust_remote_code \
    --batch_size 8 \
    --calib_size 512
'
