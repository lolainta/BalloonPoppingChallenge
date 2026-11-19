#!/bin/bash
# after queue5: handoff lead sweep upper end (strict).
cd ./
S=dev/gnc2_diag/guidance_law
until [ -f $S/queue5.done ]; do sleep 15; done
.venv/bin/python $S/sim.py --procs 4 'ho10:ho=1.0' 'ho7r5:ho=0.7,ho_r=0.5' > $S/sim_batch6.log 2>&1
echo done > $S/queue6.done
