#!/bin/bash
# One entry point for the PineDrift-820B DFlash2 training harness on PDX.
#
#   ~/pinedrift_dflash2.sh smoke    2 nodes, 8 shards, 1 epoch  -- pipe check
#   ~/pinedrift_dflash2.sh full     4 nodes (3 serve + 1 trainer), 583 shards, 5 epochs
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

# nemo_run has no --comment field, so every submit goes out twice: let the launcher
# generate and submit the sbatch, then cancel it and resubmit that same script with
# the reaper exemption. Without it, the 16 min weight load plus the Arrow build of
# the corpus looks like idle GPU and OccupiedIdleGPUsJobReaper SIGTERMs the job
# (sacct: CANCELLED+, exit 143:0, log stopping right before the first step -- it
# does not look like a kill). This applies to the smoke too, not just long runs.
submit() {                                   # submit <extra dotlist overrides...>
  cd "$LAUNCHER"
  mkdir -p "$JOBDIR"
  local out cicd job
  out=$(SLURM_HOST=$(hostname) SLURM_ACCOUNT=coreai_dlalgo_modelopt \
        SLURM_PARTITION="${PARTITION:-batch}" SLURM_QOS="${QOS:-interactive}" \
        SLURM_HF_LOCAL=$HOME/lustre/hf-local \
        SLURM_JOB_DIR=$JOBDIR NEMORUN_HOME=$PWD \
        uv run launch.py --yaml "$YAML" \
          identity=$HOME/.ssh/id_ed25519 detach=True --yes "$@" 2>&1 | tee /dev/stderr)
  cicd=$(grep -oE 'cicd_[0-9]+' <<<"$out" | head -1)
  job=$(grep -oE 'Job id: [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)
  printf '%s %s\n' "$cicd" "$job" > "$STATE"

  local sb
  sb=$(ls -t "$JOBDIR/cicd/$cicd"/*/*sbatch*.sh 2>/dev/null | head -1 || true)
  [ -z "$sb" ] && sb=$(ls -t "$LAUNCHER/experiments/cicd/$cicd"/*sbatch*.sh 2>/dev/null | head -1 || true)
  if [ -z "$sb" ]; then
    echo "WARNING: no generated sbatch found; job $job runs WITHOUT the reaper exemption" >&2
  else
    echo "cancelling $job, resubmitting $sb with the reaper exemption..."
    scancel "$job" || true
    sleep 10
    local exempt newjob
    exempt='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"data_loading","description":"vLLM serve cold start plus the corpus Arrow build before the first training step"}}'
    newjob=$(sbatch --comment="$exempt" --parsable "$sb")
    printf '%s %s\n' "$cicd" "$newjob" > "$STATE"
    job=$newjob
  fi
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
    # 8 shards (~12k rows) and one epoch: enough to prove serve -> NIXL -> fetch ->
    # step, without paying the Arrow build of all 583.
    submit \
      pipeline.task_0.args.4=data.data_path=/hf-local/pinedrift-xhigh-split-smoke8 \
      pipeline.task_0.args.7=training.num_train_epochs=1
    ;;

  full)
    # 4 nodes = 3 serve replicas (TP8 each) + 1 trainer.
    #
    # interactive QOS on the batch_long PARTITION. interactive is capped at 4
    # nodes/user but has priority 700 and, measured with --test-only, starts
    # immediately at this size. The partition matters: `batch` has
    # MaxTime=04:00:00, so 23:50 there is rejected outright ("Requested time limit
    # is invalid"); batch_long allows 7 days and accepts interactive.
    # normal@batch_long at the same size estimated a start three days out.
    #
    # NOTE: these dotlist overrides are POSITIONAL. args.4 = data.data_path,
    # args.7 = training.num_train_epochs, environment.2 = SERVE_NODES. Adding a
    # list entry above any of them silently retargets the override -- omegaconf
    # will happily set SERVE_NODES as a new key inside an unrelated env dict.
    PARTITION=batch_long submit \
      pipeline.task_0.slurm_config.nodes=4 \
      pipeline.task_0.slurm_config.time=23:50:00 \
      pipeline.task_0.environment.2.SERVE_NODES=3
    ;;

  status)
    [ -s "$STATE" ] && { read -r cicd job < "$STATE"; echo "cicd=$cicd job=$job"; } || echo "no run recorded"
    squeue -u "$USER" -o "%.10i %.9P %.34j %.8T %.10M %R" | grep -vE "synth" || true
    d=$(logdir 2>/dev/null) || exit 0
    f=$(ls -t "$d"/log-*.out 2>/dev/null | head -1) || exit 0
    echo "--- milestones in $(basename "$f") ---"
    grep -nE "Launching vllm serve|Application startup|Loading .* jsonl shards|map-style dataset|Start training|train_acc|'loss'|Traceback|AssertionError|no fetchable|Saving|export" "$f" \
      | tail -20 | cut -c1-200
    ;;

  logs)
    d=$(logdir)
    f=$(ls -t "$d"/log-*.out | head -1)
    echo "$f"
    tail -f "$f"
    ;;

  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
