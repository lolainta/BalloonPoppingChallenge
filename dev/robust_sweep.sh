#!/usr/bin/env bash
# Robustness of one policy to hidden-parameter changes (qualification params "might be different", #168):
# one unpaired 96-seed run per world, same seeds.  dev/robust_sweep.sh <policy npz> <sl npz> [first seed]  -> stdout
cd "$(dirname "$0")/.."
POL=$1; SL=$2; s0=${3:-22000}
while read -r w; do
  r=$(AGENT=rl3 .venv/bin/python dev/sweep_gnc.py --scenario 4 --seeds $s0-$((s0 + 95)) --base policy_path=$POL sl_path=$SL $w --cfg "" 2>&1 | grep -E '\[base\]|seeds [0-9]+$' | sed -E 's/.* (seeds [0-9]+)$/\1/' | tr '\n' ' ')
  echo "${w:-scenario 4 as published}: $r"
done <<'W'

P.atmosphere_data_filename=TW_Cup_250726_NetCDF4_Ensemble.nc P.date=2025,7,25,8
P.latitude_std=0.0016 P.longitude_std=0.0016
P.mass_std=0.4 P.volume_std=1.0
P.max_gust_speed=10
P.altitude_spacing=40
P.altitude_spacing=5
P.gust_decay_height=1000
P.enable=false
P.release_interval=0.25
P.gimbal_time_constant=0.128 P.roll_torque_time_constant=0.064 P.throttle_time_constant=0.16
W
date
