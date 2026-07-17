#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${PUZZLETRON_ROOT:-${SCRIPT_ROOT}}
: "${PUZZLETRON_CAMPAIGN_CONFIG:?export PUZZLETRON_CAMPAIGN_CONFIG with the campaign YAML path}"
CAMPAIGN=${PUZZLETRON_CAMPAIGN_CONFIG}
CAMPAIGN_ROOT=${PUZZLETRON_CAMPAIGN_ROOT:-${ROOT}/puzzle_runs/clean/acceptance/cross-model-stage-matrix}
PREFLIGHT=${CAMPAIGN_ROOT}/campaign/preflight.json
RUNNER=${ROOT}/examples/puzzletron/run_cross_model_stage.sh
RESUME=${ROOT}/examples/puzzletron/acceptance_resume.py
: "${PUZZLETRON_IMAGE:?export PUZZLETRON_IMAGE with the container image path}"
: "${PUZZLETRON_CONTAINER_MOUNTS:?export PUZZLETRON_CONTAINER_MOUNTS for the container}"
IMAGE=${PUZZLETRON_IMAGE}
PARTITION=interactive
ACCOUNT=coreai_dlalgo_llm
TIME_LIMIT=3:50:00

if [[ "${1:-}" != "--foreground" ]]; then
  requested=${1:-all}
  session="puzzletron-cross-model-${requested//_/-}"
  tmux new-session -d -s "${session}" "${0} --foreground ${requested}"
  echo "started tmux session ${session}"
  exit 0
fi
STAGE_REQUEST=${2:-all}

STAGES=(
  convert activation sort sort_equivalence activation_diagnostic bypass build_library
  scoring depth mip evaluation distillation aiperf
)
declare -A UPSTREAM=(
  [activation]=convert
  [sort]=activation
  [sort_equivalence]=sort
  [activation_diagnostic]=sort_equivalence
  [bypass]=activation_diagnostic
  [build_library]=bypass
  [scoring]=build_library
  [depth]=scoring
  [mip]=depth
  [evaluation]=mip
  [distillation]=evaluation
  [aiperf]=distillation
)
if [[ "${STAGE_REQUEST}" != all ]]; then
  STAGES=("${STAGE_REQUEST}")
fi

mapfile -t MODEL_ROWS < <(
  inventory_args=("${PREFLIGHT}")
  if [[ -n "${CAMPAIGN_START_MODEL:-}" ]]; then
    inventory_args+=(--start-model "${CAMPAIGN_START_MODEL}")
  fi
  python3 "${ROOT}/examples/puzzletron/cross_model_campaign_inventory.py" "${inventory_args[@]}"
)

for stage in "${STAGES[@]}"; do
  echo "[campaign] stage ${stage}: starting all ${#MODEL_ROWS[@]} models"
  for row in "${MODEL_ROWS[@]}"; do
    IFS=$'\t' read -r model_id nodes gpus_per_node nproc_per_node exclusive <<<"${row}"
    config=${CAMPAIGN_ROOT}/configs/${model_id}.yaml
    model_root=${CAMPAIGN_ROOT}/models/${model_id}
    identity=(
      --root "${model_root}" --config "${config}" --mode "${stage}"
      --require "manifests/${stage}.json"
    )
    upstream_stage=${UPSTREAM[${stage}]:-}
    if [[ -n "${upstream_stage}" ]]; then
      identity+=(
        --upstream-marker "${model_root}/manifests/completions/${upstream_stage}.json"
      )
    fi
    if python3 "${RESUME}" check "${identity[@]}"; then
      echo "[campaign] reuse ${model_id}/${stage}"
      continue
    fi

    srun_args=(
      -p "${PARTITION}" -t "${TIME_LIMIT}" -A "${ACCOUNT}"
      --container-image "${IMAGE}" --container-mounts "${PUZZLETRON_CONTAINER_MOUNTS}"
      --container-workdir "${ROOT}" --mpi=pmix
    )
    case "${stage}" in
      activation|sort_equivalence|activation_diagnostic|bypass|scoring|depth|evaluation|distillation)
        srun_args+=(
          --nodes="${nodes}" --ntasks="${nodes}" --ntasks-per-node=1
          --gres="gpu:${gpus_per_node}"
        )
        [[ "${exclusive}" == 0 ]] || srun_args+=(--exclusive)
        launch_exclusive=$([[ "${exclusive}" == 0 ]] && echo no || echo yes)
        export CAMPAIGN_NNODES=${nodes}
        export CAMPAIGN_NPROC_PER_NODE=${nproc_per_node}
        ;;
      aiperf)
        srun_args+=(--nodes=1 --ntasks=1 --gres=gpu:4)
        launch_exclusive=no
        export CAMPAIGN_NNODES=1
        export CAMPAIGN_NPROC_PER_NODE=4
        ;;
      *)
        srun_args+=(--nodes=1 --ntasks=1 --gres=gpu:1)
        launch_exclusive=no
        export CAMPAIGN_NNODES=1
        export CAMPAIGN_NPROC_PER_NODE=1
        ;;
    esac

    echo "[campaign] launch ${model_id}/${stage}: nodes=${CAMPAIGN_NNODES} gpus/node=${CAMPAIGN_NPROC_PER_NODE} exclusive=${launch_exclusive}"
    srun "${srun_args[@]}" "${RUNNER}" "${model_id}" "${stage}"
    python3 "${RESUME}" mark "${identity[@]}"
  done
  echo "[campaign] stage ${stage}: all models complete"
done
