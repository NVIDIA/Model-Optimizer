#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/shared/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/modelopt_qwen
CAMPAIGN_ROOT=${ROOT}/puzzle_runs/clean/acceptance/2026-07-06-cross-model-stage-matrix
PREFLIGHT=${CAMPAIGN_ROOT}/campaign/preflight.json
RUNNER=${ROOT}/examples/puzzletron/run_cross_model_stage.sh
INVENTORY=${ROOT}/examples/puzzletron/cross_model_campaign_inventory.py
IMAGE=/shared/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/gkarch/images/nvidia+nemo+26.04.01.sqsh

mapfile -t MODEL_ROWS < <(
  inventory_args=("${PREFLIGHT}")
  if [[ -n "${CAMPAIGN_START_MODEL:-}" ]]; then
    inventory_args+=(--start-model "${CAMPAIGN_START_MODEL}")
  fi
  python3 "${INVENTORY}" "${inventory_args[@]}"
)

for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_id nodes gpus_per_node nproc_per_node exclusive <<<"${row}"
  model_root=${CAMPAIGN_ROOT}/models/${model_id}
  teacher=${model_root}/ckpts/teacher
  probe_root=${model_root}/probes/activation
  manifest=${probe_root}/manifests/activation.json
  if python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and json.loads(path.read_text()).get("status") == "success" else 1)
PY
  then
    echo "[activation-probe] reuse ${model_id}"
    continue
  fi

  srun_args=(
    -p interactive -t 1:00:00 -A coreai_dlalgo_llm
    --container-image "${IMAGE}" --container-mounts /shared:/shared
    --container-workdir "${ROOT}" --mpi=pmix
    --nodes="${nodes}" --ntasks="${nodes}" --ntasks-per-node=1
    --gres="gpu:${gpus_per_node}"
  )
  [[ "${exclusive}" == 0 ]] || srun_args+=(--exclusive)
  export CAMPAIGN_NNODES=${nodes}
  export CAMPAIGN_NPROC_PER_NODE=${nproc_per_node}
  echo "[activation-probe] launch ${model_id}: nodes=${nodes} gpus/node=${gpus_per_node} exclusive=${exclusive}"
  srun "${srun_args[@]}" "${RUNNER}" "${model_id}" activation \
    --override "puzzle_dir=${probe_root}" \
    --override "teacher_dir=${teacher}" \
    --override "pruning.model_name_or_path=${teacher}" \
    --override "pruning.activations_log_dir=${probe_root}/pruning_scores" \
    --override pruning.eval_samples=2 \
    --override pruning.micro_batch_size=2 \
    --override data.calibration.num_samples=2 \
    --override data.calibration.micro_batch_size=2
done
