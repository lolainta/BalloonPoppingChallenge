#!/usr/bin/env bash
# PPO <-> SL generations of one lineage (sil13 -> sil14 -> ..., late1 -> late2 -> ...): when the current run's
# deterministic eval plateaus (PATIENCE evals without a new best) or the generation has run GEN_MIN minutes (with at
# least MIN_EVALS evals; user decision 2026-09-25: alternate often, let the two saturate), train a fresh frozen SL model from the newest
# decision logs, branch the next run from the current run's next checkpoint, and hand over its learner node and all
# worker slots. One instance per lineage, side by side, on the login node:
#   setsid nohup dev/sl_cycle.sh sil13 >> logs/sl_cycle.log 2>&1 < /dev/null &
# runs/<run>/next_gen_env (else runs/next_gen_env): one line of extra env for the NEXT generation's learner and
# workers, e.g. RL3_T_MIN=1,SIL_MIN=10,GNC_KW=k_att:3.0+max_rate:2.5 (read when the generation starts, so it can be
# changed without restarting this script; copied into the new run, so a lineage keeps its own settings).
# runs/next_env/<name> overrides both for the generation of that name (a schedule written in advance, e.g. late4);
# runs/next_env/<name>.teach likewise sets its teacher launch-time branches (runs/<run>/teach_launch_times).
# Lineage "late" records its current run in runs/late_current.
# Lineage "sil" (user decision 2026-09-25, sil already secures 11 pops): when silN with N >= SWITCH_AT plateaus and
# no sil generation from sil13 on reached MIN_GAIN best deterministic mean, the next sil is not started; its learner
# node and worker slots go to the live late run instead.
cd ~/bpc
run=${1:-sil13}; P=${PATIENCE:-5}; LOGS=runs/ppo5/t1data
SWITCH_AT=${SWITCH_AT:-15}; MIN_GAIN=${MIN_GAIN:-7.90}; GEN_MIN=${GEN_MIN:-90}; MIN_EVALS=${MIN_EVALS:-3}
while true; do
  pre=${run%%[0-9]*}
  [ "$pre" = late ] && echo $run > runs/late_current
  gen_t0=$(stat -c %Y runs/$run/next_gen_env 2>/dev/null || date +%s)  # written when the generation was created
  echo "$(date +%T) watching $run (plateau = $P evals without a new best, or ${GEN_MIN} min with >= $MIN_EVALS evals)"
  while true; do
    n=$(tail -n +2 runs/$run/det_eval.csv 2>/dev/null | wc -l)
    if [ "$n" -gt "$P" ]; then
      imp=$(tail -n +2 runs/$run/det_eval.csv | awk -F, -v p=$P '{m[NR]=$2} END{b=0; for(i=1;i<=NR-p;i++) if(m[i]>b) b=m[i]; x=0; for(i=NR-p+1;i<=NR;i++) if(m[i]>b) x=1; print x}')
      [ "$imp" = 0 ] && break
    fi
    if [ $(( $(date +%s) - gen_t0 )) -ge $((GEN_MIN * 60)) ] && [ "$n" -ge "$MIN_EVALS" ]; then
      echo "$(date +%T) $run: generation time limit ($GEN_MIN min, $n evals)"; break
    fi
    sleep 120
  done
  echo "$(date +%T) $run plateaued:"; tail -n $((P + 1)) runs/$run/det_eval.csv

  # ---- sil -> late switch (see header)
  if [ "$pre" = sil ] && [ "${run#sil}" -ge "$SWITCH_AT" ]; then
    best=$(for d in runs/sil[0-9]*; do k=${d#runs/sil}; [ "$k" -ge 13 ] 2>/dev/null && [ -f $d/det_eval.csv ] && tail -n +2 $d/det_eval.csv; done \
           | awk -F, 'BEGIN{b=0} $2 > b {b=$2} END{print b}')
    late=$(cat runs/late_current 2>/dev/null)
    if awk -v b="$best" -v m="$MIN_GAIN" 'BEGIN{exit !(b < m)}' && [ -n "$late" ] && [ -s runs/$late/worker_env ] \
       && squeue -h -u $USER -n bpc-async -o "%k" | grep -qx "$late"; then
      echo "$(date +%T) best sil13+ = $best < $MIN_GAIN: no sil$((${run#sil} + 1)); learner node and workers go to $late"
      lj=$(squeue -h -u $USER -n bpc-async -o "%i %k %N" | awk -v r=$run '$2==r'); lnode=$(echo "$lj" | awk '{print $3}')
      cp runs/workers.spec runs/workers.spec.bak_$run
      awk -v r="runs/$run" -v n="runs/$late" -v e="$(cat runs/$late/worker_env)" -v ln="$lnode" \
        '$1 == r {printf "%-12s %-5s %-3s %-5s %s\n", n, $2, $3, $4, e; next} {print}
         END {if (ln != "") printf "%-12s %-5s %-3s %-5s %s\n", n, ln, 95, "160G", e}' \
        runs/workers.spec > runs/workers.spec.tmp && mv runs/workers.spec.tmp runs/workers.spec
      [ "$(cat runs/teach_run 2>/dev/null)" = "$run" ] && echo $late > runs/teach_run
      t=$(squeue -h -u $USER -n bpc-teach -o "%i %k" | awk -v c="teach@$run" '$2 == c {print $1}'); [ -n "$t" ] && scancel $t
      w=$(squeue -h -u $USER -n bpc-aworker -o "%i %k" | awk -v r="$run@" 'index($2, r) == 1 {print $1}')
      scancel $(echo "$lj" | awk '{print $1}') $w 2>/dev/null
      echo "$(date +%T) $run handed over to $late: $(grep -c "^runs/$late " runs/workers.spec) worker lines; sil lineage stops"
      exit 0
    fi
    echo "$(date +%T) best sil13+ = $best (switch threshold $MIN_GAIN, late run '${late:-none}'): continuing the sil lineage"
  fi

  # ---- next generation
  num=${run#$pre}; next=$pre$((num + 1))
  while [ -d runs/$next ]; do num=$((num + 1)); next=$pre$((num + 1)); done  # skip existing side runs
  XF=runs/next_env/$next; [ -f $XF ] || XF=runs/$run/next_gen_env; [ -f $XF ] || XF=runs/next_gen_env
  X=$(head -1 $XF 2>/dev/null | sed 's/#.*//' | tr -d " \t\r\n\"'")  # extra env for the new generation
  case ",$X," in *,RL3_T_MIN=*) ;; *) X="RL3_T_MIN=1${X:+,$X}" ;; esac  # defaults of the sil lineage
  case ",$X," in *,SIL_MIN=*) ;; *) X="$X,SIL_MIN=10" ;; esac
  # validate BEFORE tearing anything down: a bad value would only crash the new learner after the handover
  g=$(echo "$X" | tr ',' '\n' | sed -n 's/^GNC_KW=//p' | tail -1)
  V='[A-Z_0-9]+=[A-Za-z0-9_.:+/-]+'  # NAME=value pairs, comma separated
  if ! echo "$X" | grep -qE "^$V(,$V)*\$" || { [ -n "$g" ] && ! .venv/bin/python -c \
      "import sys; from BalloonPoppingGymEnv.agents.rl2_agent import parse_gnc_kw as p; p(sys.argv[1])" "$g" 2>/dev/null; }; then
    echo "$(date +%T) bad $XF ($X): not starting $next, retry in 10 min"; sleep 600; continue
  fi
  E="T1DATA=$LOGS,SLMODEL=runs/$next/rl2_sl.npz,RL3_ATT=1,$X"  # learner AND workers (ppo11+: attitude head)
  mkdir -p runs/$next
  srun -p devq -c 4 --mem=16G --time=30 bash -c "OMP_NUM_THREADS=4 .venv/bin/python dev/train_t1sl.py '$LOGS/*.npy' runs/$next/rl2_sl_frozen.npz" 2>&1 | grep -E "rows|saved"
  [ -f runs/$next/rl2_sl_frozen.npz ] || { echo "SL training failed; retry in 10 min"; rmdir runs/$next 2>/dev/null; sleep 600; continue; }
  cp runs/$next/rl2_sl_frozen.npz runs/$next/rl2_sl.npz
  echo "$X" > runs/$next/next_gen_env; echo "$E" > runs/$next/worker_env
  TF=runs/next_env/$next.teach; [ -f $TF ] || TF=runs/$run/teach_launch_times; [ -f $TF ] && cp $TF runs/$next/teach_launch_times
  t0=$(date +%s); until [ $(stat -c %Y runs/$run/async_last.pt) -gt $t0 ]; do sleep 10; done
  cp runs/$run/async_last.pt runs/$next/async_last.pt
  mkdir -p runs/$next/demos_in; ls -t runs/$run/demos_in/*.npz 2>/dev/null | head -300 | xargs -r cp -t runs/$next/demos_in/  # keep recent teacher demos
  [ -f runs/$run/branch_step ] && cp runs/$run/branch_step runs/$next/  # step offset for the dashboard's shared axis
  # this run's learner only (job comment = run name; other lineages keep theirs)
  lj=$(squeue -h -u $USER -n bpc-async -o "%i %k %N" | awk -v r=$run '$2==r'); lnode=$(echo "$lj" | awk '{print $3}')
  scancel $(echo "$lj" | awk '{print $1}') 2>/dev/null
  cp runs/workers.spec runs/workers.spec.bak_$run
  # move this run's worker lines to the new run; the ENV column is rebuilt from E (no carried-over extras)
  awk -v r="runs/$run" -v n="runs/$next" -v e="$E" '$1 == r {printf "%-12s %-5s %-3s %-5s %s\n", n, $2, $3, $4, e; next} {print}' \
    runs/workers.spec > runs/workers.spec.tmp && mv runs/workers.spec.tmp runs/workers.spec
  echo "$(date +%T) workers.spec: $(grep -c "^runs/$next " runs/workers.spec) worker lines moved to $next"
  t=$(squeue -h -u $USER -n bpc-teach -o "%i %k" | awk -v c="teach@$run" '$2 == c {print $1}'); [ -n "$t" ] && scancel $t  # old run's teachers
  w=$(squeue -h -u $USER -n bpc-aworker -o "%i %k" | awk -v r="$run@" 'index($2, r) == 1 {print $1}'); [ -n "$w" ] && scancel $w
  sleep 12; rm -f runs/$next/learner.addr
  sbatch --comment=$next --export=ALL,RUN=runs/$next,LOCAL=92,LAUNCH=1,BATCH=19200,$E -w ${lnode:-cp18} dev/async.sbatch
  # deterministic eval: dev/local_services.sh (off-cluster) follows every live learner's comment
  [ "$(cat runs/teach_run 2>/dev/null)" = "$run" ] && echo $next > runs/teach_run  # teachers follow this lineage only
  echo "$(date +%T) $next started ($X) on ${lnode:-cp18} (eval: local), branched from $run checkpoint $(date -r runs/$run/async_last.pt +%T)"
  run=$next
done
