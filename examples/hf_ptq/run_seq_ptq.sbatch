#!/usr/bin/env bash
#SBATCH --job-name=hfptq-seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
# Single-process ("pipelined") PTQ runner: ONE python process, model split across the node's GPUs by
# accelerate's device_map, calibration flowing through the stages one GPU at a time. This is the
# non-DP/EP counterpart to the distributed harness runner -- same model, same recipe, so the two
# paths' quantizer values can be compared directly.
#
# MODELOPT_VL_DEVICE_MAP=auto (balanced) is deliberate: 'sequential' fills GPU0 to
# gpu_mem_percentage before spilling, which leaves GPU0 with no room for calibration activations.
set -uo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}
export MODELOPT_SRC=${MODELOPT_SRC:?set MODELOPT_SRC to the modelopt repo root}
export PTQ_SCRIPT=${PTQ_SCRIPT:-${MODELOPT_SRC}/examples/hf_ptq/hf_ptq.py}
export MODEL=${MODEL:?set MODEL}
export RECIPE=${RECIPE:?set RECIPE}
export EXPORT_PATH=${EXPORT_PATH:?set EXPORT_PATH}
export CALIB_SIZE=${CALIB_SIZE:-64}
export CALIB_SEQ=${CALIB_SEQ:-4096}
export BATCH_SIZE=${BATCH_SIZE:-1}
export DATASET=${DATASET:-cnn_dailymail}
export TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
export SKIP_GENERATE=${SKIP_GENERATE:-1}
export MODELOPT_VL_DEVICE_MAP=${MODELOPT_VL_DEVICE_MAP:-auto}
export MODELOPT_USE_MAMBA_KERNELS=${MODELOPT_USE_MAMBA_KERNELS:-0}
export GPU_MAX_MEM_PERCENTAGE=${GPU_MAX_MEM_PERCENTAGE:-0.8}
export TF_PIP=${TF_PIP:-"transformers==5.13.0 einops accelerate datasets peft sentencepiece pydantic omegaconf ml_dtypes safetensors"}
export HF_HOME=${HF_HOME:?set HF_HOME}
export PYTHONUNBUFFERED=1

CONTAINER_MOUNTS=""
for _m in /lustre /scratch; do [ -d "${_m}" ] && CONTAINER_MOUNTS="${CONTAINER_MOUNTS:+${CONTAINER_MOUNTS},}${_m}:${_m}"; done

echo "hf_ptq SEQUENTIAL: MODEL=${MODEL} RECIPE=${RECIPE} vl_device_map=${MODELOPT_VL_DEVICE_MAP} calib=${CALIB_SIZE}x${CALIB_SEQ}"

srun --unbuffered \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${PWD}" \
  --no-container-mount-home --no-container-entrypoint \
  --container-env=HF_HOME,HF_TOKEN,MODELOPT_SRC,PTQ_SCRIPT,MODEL,RECIPE,EXPORT_PATH,CALIB_SIZE,CALIB_SEQ,BATCH_SIZE,DATASET,TRUST_REMOTE_CODE,SKIP_GENERATE,MODELOPT_VL_DEVICE_MAP,MODELOPT_USE_MAMBA_KERNELS,GPU_MAX_MEM_PERCENTAGE,TF_PIP,PYTHONUNBUFFERED \
  bash -lc '
    set -o pipefail
    export PYTHONPATH="${MODELOPT_SRC}:${MODELOPT_SRC}/examples/hf_ptq${PYTHONPATH:+:$PYTHONPATH}"
    if [ -n "${TF_PIP:-}" ]; then
      C=/tmp/pip-constraints.txt; : > "$C"
      for _p in torch torchvision; do
        _v="$(python -m pip show "${_p}" 2>/dev/null | awk "/^Version:/{print \$2}")"
        [ -n "${_v}" ] && echo "${_p}==${_v}" >> "$C"
      done
      python -m pip install --no-cache-dir --no-input --root-user-action=ignore \
        --constraint "$C" ${TF_PIP} 2>&1 | tail -3
    fi
    nvidia-smi --query-gpu=index,memory.total --format=csv,noheader
    case "${TRUST_REMOTE_CODE}" in 0|off|false|no) TRC="" ;; *) TRC="--trust_remote_code" ;; esac
    case "${SKIP_GENERATE}" in 0|off|false|no) SKG="" ;; *) SKG="--skip_generate" ;; esac
    python "${PTQ_SCRIPT}" \
      --pyt_ckpt_path "${MODEL}" \
      --recipe "${RECIPE}" \
      --batch_size "${BATCH_SIZE}" --calib_size "${CALIB_SIZE}" --calib_seq "${CALIB_SEQ}" \
      --dataset "${DATASET}" \
      --gpu_max_mem_percentage "${GPU_MAX_MEM_PERCENTAGE}" \
      --export_path "${EXPORT_PATH}" \
      ${TRC} ${SKG}
  '
