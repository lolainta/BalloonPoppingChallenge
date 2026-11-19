#!/usr/bin/env bash
# C and E hashes side by side (2 + 2 processes), outputs in $T/C.out / $T/E.out
D=$(cd "$(dirname "$0")" && pwd); source $D/env.sh
bash $D/hashes.sh C > $T/C.out 2>&1 &
bash $D/hashes.sh E > $T/E.out 2>&1
wait; cat $T/C.out $T/E.out
