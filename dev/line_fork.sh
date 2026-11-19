#!/usr/bin/env bash
# Fork the single training line (DECISION_LOG §140): NEW continues from SRC's next checkpoint with one learner-env change.
#   dev/line_fork.sh <src run> <new run> '<sed expr on the learner submit line>' [min workers, default 280]
#   e.g. dev/line_fork.sh s4res2 s4tc 's/RL3_RES2=1,/RL3_RES2=1,RL3_TCOND=1,/'
# Runs on login02. Stops SRC (learner_stop, watchdog, learner, workers), copies async_last.pt + SL, rewrites workers.spec
# and SRC's learner_submit.sh for NEW, submits, starts NEW's watchdog, then checks NEW reaches MINW workers.
# 2026-10-02: the inline version once took the SRC worker list from a failed squeue (DNS) -> 127 of ~296 workers for 10 min,
# so every squeue here is checked and retried.
set -euo pipefail
cd ~/bpc
SRC=$1; NEW=$2; SEDX=$3; MINW=${4:-280}
fail() { echo "ABORT: $*"; exit 1; }
sq() { local i out; for i in 1 2 3 4 5 6; do out=$(squeue -h -u $USER "$@" 2>/dev/null) && { printf '%s\n' "$out"; return 0; }; sleep 5; done; return 1; }
jobs_of() { sq -o "%i %k" | awk -v r=$1 '$2 == r || index($2, r "@") == 1 {print $1}'; }

[ -f runs/$SRC/async_last.pt ] && [ -s runs/$SRC/learner_submit.sh ] && [ ! -e runs/$NEW ] || fail "preconditions (SRC files, NEW must not exist)"
SUB=$(sed -e "s/$SRC/$NEW/g" -e "$SEDX" runs/$SRC/learner_submit.sh)
[ "$SUB" != "$(sed "s/$SRC/$NEW/g" runs/$SRC/learner_submit.sh)" ] || fail "the sed change did not apply to the submit line"
L=$(sq -n bpc-async -o "%k") || fail "squeue keeps failing"
printf '%s\n' "$L" | grep -x "$SRC" > /dev/null || fail "no running learner for $SRC"
echo "new submit line: $SUB"
[ -z "${DRY:-}" ] || { echo "DRY: preconditions ok, nothing changed"; exit 0; }

touch runs/$SRC/learner_stop
wd=$(pgrep -f "[l]earner_watchdog.sh runs/$SRC\$" || true); [ -z "$wd" ] || kill $wd
for i in $(seq 90); do age=$(( $(date +%s) - $(stat -c %Y runs/$SRC/async_last.pt) )); [ $age -lt 45 ] && break; sleep 10; done
[ $age -lt 45 ] || fail "no fresh checkpoint of $SRC within 15 min"
J=$(jobs_of $SRC) || fail "squeue failed"; echo "$J" | xargs -r scancel
sleep 10
for i in 1 2 3; do left=$(jobs_of $SRC) || fail "squeue failed"; [ -z "$left" ] && break; echo "$left" | xargs -r scancel; sleep 10; done
[ -z "$left" ] || fail "jobs of $SRC still alive: $left"

mkdir -p runs/$NEW runs/s4data/t1data_$NEW ~/bpc_store/launchdata_s4_$NEW
cp runs/$SRC/async_last.pt runs/$NEW/async_last.pt; cp -p runs/$SRC/rl2_sl.npz runs/$NEW/rl2_sl.npz
[ -f runs/$SRC/branch_step ] && cp runs/$SRC/branch_step runs/$NEW/branch_step
CK=$(.venv-gpu/bin/python -c "import torch; s = torch.load('runs/$NEW/async_last.pt', map_location='cpu', weights_only=False); print(s['steps'], s['version'])") || fail "checkpoint copy unreadable"
echo "$SRC $CK $(date +%F_%T) sed: $SEDX" > runs/$NEW/branch_from; cat runs/$NEW/branch_from
cp runs/workers.spec runs/workers.spec.bak_$NEW
sed "/^runs\/$SRC /s/$SRC/$NEW/g" runs/workers.spec > runs/workers.spec.tmp && mv runs/workers.spec.tmp runs/workers.spec
echo "workers.spec lines for $NEW: $(grep -c "^runs/$NEW " runs/workers.spec)"
printf '%s\n' "$SUB" > runs/$NEW/learner_submit.sh
bash runs/$NEW/learner_submit.sh
setsid nohup dev/learner_watchdog.sh runs/$NEW >> logs/learner_watchdog.log 2>&1 < /dev/null &

for i in $(seq 60); do st=$(sq -n bpc-async -o "%k %T %i" | awk -v r=$NEW '$1 == r {print $2, $3}') || st=""; [ "${st%% *}" = RUNNING ] && break; sleep 10; done
echo "learner $NEW: ${st:-unknown}"
for i in $(seq 40); do  # up to ~13 min for the keeper to refill the nodes
  w=$(.venv/bin/python -c "import json; print(json.load(open('runs/$NEW/status.json')).get('workers', 0))" 2>/dev/null || echo 0)
  [ "$w" -ge "$MINW" ] && break
  left=$(jobs_of $SRC) && [ -n "$left" ] && { echo "cancelling leftover $SRC jobs: $left"; echo "$left" | xargs -r scancel; }
  sleep 20
done
echo "$NEW workers: $w (want >= $MINW)"; [ "$w" -ge "$MINW" ] || echo "WARNING: worker count below $MINW"
