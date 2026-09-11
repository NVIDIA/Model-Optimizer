#!/bin/bash
# One entry point for the PineDrift-820B DFlash2 training harness on PDX.
#
#   ~/pinedrift_dflash2.sh smoke    2 nodes, the 20k shard, 1 epoch
#   ~/pinedrift_dflash2.sh full     4 nodes, the full corpus, + reaper exemption
#   ~/pinedrift_dflash2.sh status   queue + the last milestone lines
#   ~/pinedrift_dflash2.sh logs     tail the newest run's log
#
# See tools/launcher/examples/PineDrift/README.md for what every setting is for and
# which of them are measured rather than inferred.
set -euo pipefail

REPO=$HOME/lustre/modelopt-pinedrift
LAUNCHER=$REPO/tools/launcher
YAML=examples/PineDrift/hf_streaming_dflash2_pinedrift.yaml
JOBDIR=$HOME/lustre/pinedrift-experiments
STATE=$HOME/.pinedrift_dflash2_last          # remembers the most recent cicd id + job id

export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy

submit() {                                   # submit <extra dotlist overrides...>
  cd "$LAUNCHER"
  mkdir -p "$JOBDIR"
  local out
  out=$(SLURM_HOST=$(hostname) SLURM_ACCOUNT=coreai_dlalgo_modelopt \
        SLURM_PARTITION=batch SLURM_QOS="${QOS:-interactive}" \
        SLURM_HF_LOCAL=$HOME/lustre/hf-local \
        SLURM_JOB_DIR=$JOBDIR NEMORUN_HOME=$PWD \
        uv run launch.py --yaml "$YAML" \
          identity=$HOME/.ssh/id_ed25519 detach=True --yes "$@" 2>&1 | tee /dev/stderr)
  local cicd job
  cicd=$(grep -oE 'cicd_[0-9]+' <<<"$out" | head -1)
  job=$(grep -oE 'Job id: [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)
  printf '%s %s\n' "$cicd" "$job" > "$STATE"
  echo
  echo "cicd=$cicd job=$job  (remembered in $STATE)"
}

logdir() {
  [ -s "$STATE" ] || { echo "no run recorded yet" >&2; exit 1; }
  read -r cicd _ < "$STATE"
  ls -d "$JOBDIR/cicd/$cicd"/*/ 2>/dev/null | head -1
}

case "${1:-}" in
  smoke)
    # Defaults in the yaml are already the smoke shape (2 nodes, 20k shard).
    submit
    ;;

  full)
    # nemo_run has no --comment field, so: let the launcher generate and submit the
    # sbatch, then cancel it and resubmit that same script with the reaper exemption.
    # Without it, tokenizing 1.96M records before step 1 looks like 30 min of idle
    # GPU and OccupiedIdleGPUsJobReaper SIGTERMs the job (sacct shows CANCELLED+,
    # exit 143:0, log ending right before the first step -- it does not look like a kill).
    submit \
      pipeline.task_0.args.4=data.data_path=/hf-local/Speculative-Decoding-Dataset-v1-Qwen3-8B/default-msgs.jsonl \
      pipeline.task_0.slurm_config.nodes=4 \
      pipeline.task_0.slurm_config.time=23:50:00 \
      pipeline.task_0.environment.6.SERVE_NODES=2
    read -r cicd job < "$STATE"
    local_sbatch=$(ls -t "$JOBDIR/cicd/$cicd"/*/*sbatch*.sh 2>/dev/null | head -1)
    if [ -z "$local_sbatch" ]; then
      echo "could not find the generated sbatch; job $job left as submitted (no reaper exemption)" >&2
      exit 1
    fi
    echo "cancelling $job and resubmitting $local_sbatch with the reaper exemption..."
    scancel "$job" || true
    sleep 10
    EXEMPT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"data_loading","description":"vLLM serve startup plus tokenizing the full spec-dec corpus before the first training step"}}'
    newjob=$(sbatch --comment="$EXEMPT" --parsable "$local_sbatch")
    printf '%s %s\n' "$cicd" "$newjob" > "$STATE"
    echo "resubmitted as $newjob"
    scontrol show job "$newjob" | grep -o 'Comment=.*' | head -c 200; echo
    ;;

  status)
    [ -s "$STATE" ] && { read -r cicd job < "$STATE"; echo "cicd=$cicd job=$job"; } || echo "no run recorded"
    squeue -u "$USER" -o "%.10i %.9P %.34j %.8T %.10M %R" || true
    d=$(logdir 2>/dev/null) || exit 0
    f=$(ls -t "$d"/log-*.out 2>/dev/null | head -1) || exit 0
    echo "--- milestones in $(basename "$f") ---"
    grep -nE "Launching vllm serve|Application startup|nixl|NIXL|LIBFABRIC|train_acc|'loss'|Traceback|AssertionError|Error|Saving|export" "$f" \
      | tail -20 | cut -c1-200
    ;;

  logs)
    d=$(logdir)
    f=$(ls -t "$d"/log-*.out | head -1)
    echo "$f"
    tail -f "$f"
    ;;

  *)
    sed -n '2,9p' "$0"
    exit 1
    ;;
esac
