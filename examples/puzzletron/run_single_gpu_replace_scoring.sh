#!/usr/bin/env bash

set -Eeuo pipefail

: "${CONFIG_PATH:?set CONFIG_PATH}"
: "${SCENARIO_DIR:?set SCENARIO_DIR}"
: "${MODEL_TEACHER_DIR:?set MODEL_TEACHER_DIR}"
: "${TARGET_TEACHER_DIR:?set TARGET_TEACHER_DIR}"

ROOT=${ROOT:-$(pwd)}
PUZZLETRON_VENV=${PUZZLETRON_VENV:-"${ROOT}/.venv_new"}
TASK_ID=${SLURM_PROCID:-0}
LOCAL_ID=${SLURM_LOCALID:-0}
TASK_COUNT=${SLURM_NTASKS:-1}
JOB_ID=${SLURM_JOB_ID:-local}

cd "${ROOT}"
if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then
  source "${PUZZLETRON_SETUP_ENV}"
fi
source "${PUZZLETRON_VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SCENARIO_DIR=$(realpath "${SCENARIO_DIR}")
MODEL_TEACHER_DIR=$(realpath "${MODEL_TEACHER_DIR}")
TARGET_TEACHER_DIR=$(realpath "${TARGET_TEACHER_DIR}")
SOURCE_DIR=${SCENARIO_DIR}/ckpts/sorted_teacher
SOLUTIONS_PATH=${SCENARIO_DIR}/single_sequence_replacement_solutions.json
LIBRARY_PATH=${SCENARIO_DIR}/replacement_library.json
CAMPAIGN_DIR=${SCENARIO_DIR}/distributed_eval/campaign
OUTPUT_DIR=${SCENARIO_DIR}/distributed_eval/output
COMPATIBILITY_OUTPUT_DIR=${SCENARIO_DIR}/single_sequence_replacement_solutions--validation
LOG_DIR=${SCENARIO_DIR}/distributed_eval/logs
mkdir -p "${CAMPAIGN_DIR}" "${OUTPUT_DIR}" "${COMPATIBILITY_OUTPUT_DIR}" "${LOG_DIR}"

for required in \
  "${SOURCE_DIR}/config.json" \
  "${MODEL_TEACHER_DIR}/config.json" \
  "${TARGET_TEACHER_DIR}/config.json" \
  "${SOLUTIONS_PATH}" \
  "${LIBRARY_PATH}"; do
  [[ -f "${required}" ]] || { echo "missing required replacement-scoring input: ${required}" >&2; exit 3; }
done

WIDTH=$(python - "${SCENARIO_DIR}/scenario_manifest.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["hidden_width"]))
PY
)

export CAMPAIGN_DIR CONFIG_PATH SOLUTIONS_PATH OUTPUT_DIR COMPATIBILITY_OUTPUT_DIR
export WORLD_SIZE=1
export TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-7200}
export STALE_SECONDS=${STALE_SECONDS:-120}
export DISTRIBUTED_EVAL_OVERRIDES
DISTRIBUTED_EVAL_OVERRIDES=$(printf '%s\n' \
  "puzzle_dir=${SCENARIO_DIR}" \
  "model.force_hf=false" \
  "replacement_library_path=${LIBRARY_PATH}" \
  "scoring.replacement_library_path=${LIBRARY_PATH}" \
  "scoring.teacher_dir=${MODEL_TEACHER_DIR}" \
  "scoring.source_checkpoint_dir=${SOURCE_DIR}" \
  "scoring.target_teacher_dir=${TARGET_TEACHER_DIR}" \
  "scoring.output_dir=${COMPATIBILITY_OUTPUT_DIR}" \
  "scoring.automodel.force_hf=false" \
  "scoring.automodel.parallel.tp=1" \
  "scoring.automodel.parallel.cp=1" \
  "scoring.automodel.parallel.pp=1" \
  "scoring.automodel.parallel.ep=1" \
  "scoring.automodel.parallel.dp_shard=1" \
  "scoring.automodel.parallel.dp_replicate=1" \
  "scoring.distributed_eval.gpus_per_task=1" \
  "scoring.eval_samples=${SCORING_EVAL_SAMPLES:-8}" \
  "scoring.micro_batch_size=${SCORING_MICRO_BATCH_SIZE:-4}" \
  "scoring.block_size=${SCORING_BLOCK_SIZE:-1024}")

coordinator_pid=
if [[ "${TASK_ID}" == "0" ]]; then
  bash examples/puzzletron/distributed_eval/run_coordinator.sh \
    >"${LOG_DIR}/coordinator_${JOB_ID}.log" 2>&1 &
  coordinator_pid=$!
fi

waited=0
while [[ ! -f "${CAMPAIGN_DIR}/manifest.json" ]]; do
  sleep 1
  waited=$((waited + 1))
  [[ ${waited} -lt 300 ]] || { echo "RPC campaign initialization timed out" >&2; exit 4; }
done

export NNODES=1 NPROC_PER_NODE=1 NODE_RANK=0
export RDZV_ENDPOINT=127.0.0.1:$((35000 + LOCAL_ID))
export RDZV_ID="replace-${JOB_ID}-w${WIDTH}-task${TASK_ID}"
export WORKER_HOST=${WORKER_HOST:-$(hostname -f)}
export WORKER_PORT=$((5200 + LOCAL_ID))
export WORKER_GROUP_INDEX=${TASK_ID}
export WORKER_ID="${JOB_ID}-w${WIDTH}-task${TASK_ID}"
bash examples/puzzletron/distributed_eval/run_worker.sh \
  >"${LOG_DIR}/worker_${JOB_ID}_task${TASK_ID}.log" 2>&1 &
worker_pid=$!

if [[ "${TASK_ID}" == "0" ]]; then
  set +e
  wait "${coordinator_pid}"; coordinator_rc=$?
  python -m modelopt.torch.puzzletron.distributed_eval.cli drain \
    --campaign-dir "${CAMPAIGN_DIR}" --stale-seconds 180 \
    >>"${LOG_DIR}/coordinator_${JOB_ID}.log" 2>&1
  drain_rc=$?
  wait "${worker_pid}"; worker_rc=$?
  set -e
  python -m modelopt.torch.puzzletron.distributed_eval.cli rebuild-summary \
    --campaign-dir "${CAMPAIGN_DIR}"
  if [[ ${coordinator_rc} -ne 0 || ${drain_rc} -ne 0 || ${worker_rc} -ne 0 ]]; then
    echo "RPC failure: coordinator=${coordinator_rc} drain=${drain_rc} worker=${worker_rc}" >&2
    exit 5
  fi
else
  wait "${worker_pid}"
fi

if [[ "${TASK_ID}" == "0" ]]; then
  expected=$(python - "${SOLUTIONS_PATH}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
  actual=$(find "${COMPATIBILITY_OUTPUT_DIR}" -maxdepth 1 -type f -name 'solution_*.json' | wc -l)
  [[ "${actual}" == "${expected}" ]] || {
    echo "replacement result cardinality mismatch: ${actual}/${expected}" >&2
    exit 6
  }
  echo "replacement scoring complete: width=${WIDTH} results=${actual}/${expected} workers=${TASK_COUNT}"
fi
