#!/bin/bash
# Babysitter for the PineDrift DFlash2 run. Two jobs:
#
# 1. PRUNE checkpoints. Each is 56 GB (4 G params x fp32 weights + fp32 Adam moments,
#    per dflash_fp32_master_weights) and save_steps=500 at 0.82 s/step writes one every
#    ~7 min = 480 GB/h. /scratch/fsw is shared and was at 97% with 3.2 TB free, so this
#    fills the cluster's lustre in under 7 hours. Keep the newest KEEP, never touch one
#    written in the last 3 minutes (it may still be being written).
#
# 2. RESUBMIT on preemption. interactive QOS is PreemptMode=within,cancel with
#    PreemptExemptTime=04:05:00 -- past 4h05m another interactive job can take ours and
#    it is CANCELLED, not requeued, so nothing brings it back on its own. Resubmitting
#    the SAME sbatch keeps the same /scratchspace, and main.py calls
#    get_last_checkpoint(output_dir), so it resumes where it stopped.
#
# Only ever run ONE of these.
set -u
CICD=cicd_1789148807630758908
SB=$HOME/lustre/modelopt-pinedrift/tools/launcher/experiments/cicd/$CICD/PineDrift_820B_DFlash2_streaming_smoke_0_sbatch.sh
CKPT=$HOME/lustre/pinedrift-experiments/cicd/$CICD/dflash2
STATE=$HOME/.pinedrift_dflash2_last
LOG=$HOME/lustre/pinedrift_babysit.log
KEEP=3
MAX_RESTARTS=8
EXEMPT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"data_loading","description":"vLLM serve cold start plus the corpus Arrow build before the first training step"}}'
restarts=0

say() { printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG"; }
say "babysitter started (keep=$KEEP, max_restarts=$MAX_RESTARTS)"

while true; do
  # --- prune ---
  mapfile -t all < <(ls -dt "$CKPT"/checkpoint-* 2>/dev/null)
  if [ "${#all[@]}" -gt "$KEEP" ]; then
    now=$(date +%s)
    for d in "${all[@]:$KEEP}"; do
      m=$(stat -c %Y "$d" 2>/dev/null || echo 0)
      if [ $((now - m)) -gt 180 ]; then
        sz=$(du -sh "$d" 2>/dev/null | cut -f1)
        rm -rf "$d" && say "pruned $(basename "$d") ($sz)"
      fi
    done
  fi

  # --- resubmit if the job is gone ---
  read -r _ job < "$STATE"
  if ! squeue -j "$job" -h -o '%T' 2>/dev/null | grep -q .; then
    st=$(sacct -j "$job" -X -n --format=State%20 2>/dev/null | head -1 | tr -d ' ')
    say "job $job left the queue (sacct=$st)"
    case "$st" in
      COMPLETED) say "completed normally; babysitter exiting"; exit 0 ;;
    esac
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      say "hit MAX_RESTARTS=$MAX_RESTARTS; NOT resubmitting"; exit 1
    fi
    restarts=$((restarts + 1))
    new=$(sbatch --comment="$EXEMPT" --parsable "$SB" 2>&1)
    if [[ "$new" =~ ^[0-9]+$ ]]; then
      printf '%s %s\n' "$CICD" "$new" > "$STATE"
      say "resubmitted as $new (restart $restarts/$MAX_RESTARTS); resumes from $(ls -dt $CKPT/checkpoint-* 2>/dev/null | head -1 | xargs -r basename)"
    else
      say "RESUBMIT FAILED: $new"
    fi
    sleep 120
  fi
  sleep 60
done
