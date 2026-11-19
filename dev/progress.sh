#!/usr/bin/env bash
# Live progress bar for a collect.sbatch run on the cluster.
#   dev/progress.sh [out_dir=data/r3] [total=250] [refresh_s=30]
dir=${1:-data/r3}; total=${2:-250}; every=${3:-30}
while :; do
  read -r n first last now run pend bad < <(ssh -o BatchMode=yes user@login "cd ~/bpc
    f=\$(ls $dir/*.npz 2>/dev/null)
    n=\$(echo \"\$f\" | grep -c .)
    t=\$(echo \"\$f\" | xargs -r stat -c %Y | sort -n)
    echo \$n \$(echo \"\$t\" | head -1) \$(echo \"\$t\" | tail -1) \$(date +%s) \
      \$(squeue -h -u $USER -n bpc-legs -t R | wc -l) \$(squeue -h -u $USER -n bpc-legs -t PD -r | wc -l) \
      \$(grep -l Traceback logs/legs_*.log 2>/dev/null | wc -l)")
  [ -z "$n" ] && { echo "ssh failed, retrying"; sleep "$every"; continue; }
  pct=$((n * 100 / total)); w=40; fill=$((n * w / total))
  bar=$(printf '%*s' "$fill" '' | tr ' ' '#')$(printf '%*s' $((w - fill)) '' | tr ' ' '.')
  eta="--"
  if [ "$n" -ge 2 ] && [ "$last" -gt "$first" ]; then
    # ponytail: steady-state rate from output mtimes; ignores the first wave's startup latency
    secs=$(( (total - n) * (last - first) / (n - 1) ))
    eta="$((secs / 60))m (~$(date -d @$((now + secs)) +%H:%M))"
  fi
  printf '\r[%s] %3d%%  %d/%d  run %d  pend %d  err %d  ETA %s   ' "$bar" "$pct" "$n" "$total" "$run" "$pend" "$bad" "$eta"
  [ "$n" -ge "$total" ] && [ "$run" -eq 0 ] && { echo; echo done; break; }
  sleep "$every"
done
