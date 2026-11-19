#!/bin/bash
# after batch 3a: consistent edge-aim tolerance + longer handoff (strict), scenario-4 robustness (free), forced 1.3.
cd ./
busy() { pgrep -f '^\.venv/bin/python dev/gnc2_diag/guidance_law/sim\.py' >/dev/null; }
while busy; do sleep 10; done
P=.venv/bin/python; S=dev/gnc2_diag/guidance_law
$P $S/sim.py --procs 4 'tolc1:tolc=1.0' 'ho5tc:ho=0.5,tolc=0.8' 'ho7:ho=0.7' > $S/sim_batch5.log 2>&1
$P $S/sim.py --procs 4 --scenario 4 --seeds 7100:7108 'f4base:free=1' 'f4ho5:free=1,ho=0.5' > $S/sim_batch4.log 2>&1
$P $S/sim.py --procs 4 's13:fix_s=3' > $S/sim_batch3b.log 2>&1
echo done > $S/queue5.done
