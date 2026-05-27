#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source ${SCRIPT_DIR}/../service_utils.sh

###################################################################################################

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    TASK_ID=0
else
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
    TASK_ID=${SLURM_ARRAY_TASK_ID}
fi

if [ -z ${SLURM_ARRAY_TASK_COUNT} ]; then
    TASK_COUNT=1
else
    echo "SLURM_ARRAY_TASK_COUNT ${SLURM_ARRAY_TASK_COUNT}"
    TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
fi

trtllm-llmapi-launch python3 modules/Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py \
    --model ${HF_MODEL_CKPT} \
    --dp-rank ${TASK_ID} \
    --dp-world-size ${TASK_COUNT} \
    ${@}
