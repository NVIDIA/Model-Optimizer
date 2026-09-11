#!/bin/bash
# Durable throughput/curve sampler for the PineDrift DFlash2 run.
# Appends "iso_time n_logged_points last_loss last_acc epoch" every 5 min.
CICD=$1
OUT=$HOME/lustre/pinedrift_curve_samples.tsv
[ -f "$OUT" ] || printf "time\tpoints\tloss\tacc\tepoch\n" > "$OUT"
while true; do
  f=$(ls -t $HOME/lustre/pinedrift-experiments/cicd/$CICD/*/log-*.out 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    n=$(grep -c "'loss'" "$f")
    last=$(grep -oE "\{'loss':[^}]*\}" "$f" | tail -1)
    loss=$(sed -n "s/.*'loss': '\([^']*\)'.*/\1/p" <<<"$last")
    acc=$(sed -n "s/.*train_acc[^']*': '\([^']*\)'.*/\1/p" <<<"$last")
    ep=$(sed -n "s/.*'epoch': '\([^']*\)'.*/\1/p" <<<"$last")
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$n" "$loss" "$acc" "$ep" >> "$OUT"
  fi
  sleep 300
done
