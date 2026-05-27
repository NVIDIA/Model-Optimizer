#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source ${SCRIPT_DIR}/../../service_utils.sh

###################################################################################################
# HF-based hidden state dumping for models not supported by TRT-LLM.
# Uses compute_hidden_states_hf.py with device_map="auto" (no TP/EP flags needed).
#
# Required environment:
#   HF_MODEL_CKPT   Path to the HF model checkpoint
#
# Args passed through to compute_hidden_states_hf.py:
#   --input-data, --output-dir, --max-seq-len, etc.
###################################################################################################

pip install datasets 2>/dev/null || true

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

python3 modules/Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py \
    --model ${HF_MODEL_CKPT} \
    --dp-rank ${TASK_ID} \
    --dp-world-size ${TASK_COUNT} \
    --trust_remote_code \
    ${@}
