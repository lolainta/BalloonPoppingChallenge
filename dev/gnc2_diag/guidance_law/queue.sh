#!/bin/bash
# sequential simulation queue (<= 4 processes at a time): waits for the running batch-1 python to exit, then
# batch 2 (strict guidance variants), 3a (free-mode variants, 5 distinct seeds), 3b (strict forced speed levels).
cd ./
busy() { pgrep -f '^\.venv/bin/python dev/gnc2_diag/guidance_law/sim\.py' >/dev/null; }
while busy; do sleep 10; done
P=.venv/bin/python; S=dev/gnc2_diag/guidance_law
$P $S/sim.py --procs 4 'tol1:tol=1.0' 'ho3t1:ho=0.3,tol=1.0' 'ho4r1:ho=0.4,ho_r=1.0' 'ho2:ho=0.2' > $S/sim_batch2.log 2>&1
$P $S/sim.py --procs 4 --only 10_CoolBa,10_SpaceY,11_near_miss,12_Apex,14_SpaceBerry \
  'fho3:free=1,ho=0.3' 'ftol1:free=1,tol=1.0' 'fho3t1:free=1,ho=0.3,tol=1.0' > $S/sim_batch3a.log 2>&1
$P $S/sim.py --procs 4 's08:fix_s=1' 's13:fix_s=3' > $S/sim_batch3b.log 2>&1
echo done > $S/queue.done
