#!/usr/bin/env bash
# Keep the 200 GB thome small (user 2026-09-26): decision logs older than MIN_AGE minutes move to shome
# (4 TB). The SL trainer only reads the newest 300k rows (~20-30 min of data), so thome keeps a few hours.
#   setsid nohup dev/archive_t1data.sh >> logs/archive_t1data.log 2>&1 < /dev/null &
cd ~/bpc
BASE=~/bpc_store; MIN_AGE=${MIN_AGE:-180}
while true; do
  pairs="runs/ppo5/t1data:t1data_archive"  # scenario 1; scenario 4: runs/s4data/t1data[_<run>] -> t1data_archive_s4[_<run>]
  for d in runs/s4data/t1data*; do pairs="$pairs $d:t1data_archive_s4${d#runs/s4data/t1data}"; done
  for pair in $pairs; do
    SRC=${pair%%:*}; DST=$BASE/${pair##*:}; [ -d $SRC ] || continue; mkdir -p $DST
    n=$(find $SRC -maxdepth 1 -name '*.npy' -mmin +$MIN_AGE | wc -l)
    find $SRC -maxdepth 1 -name '*.npy' -mmin +$MIN_AGE -print0 | xargs -0 -r mv -t $DST
    echo "$(date +%T) $SRC: moved $n files; thome $(du -sh $SRC | cut -f1) ($(ls $SRC | wc -l) files), shome $(du -sh $DST | cut -f1)"
  done
  sleep 1800
done
