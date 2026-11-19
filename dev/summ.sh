#!/usr/bin/env bash
# Summarize an eval.sbatch job on the cluster: dev/summ.sh JOBID VARFILE [NCHUNK=12] [AGENT_LABEL]
job=$1; vf=$2; n=${3:-12}; lab=${4:-}
ssh user@login "cd ~/bpc; nv=\$(wc -l < $vf); for v in \$(seq 0 \$((nv-1))); do kw=\$(sed -n \"\$((v+1))p\" $vf); \
  cat \$(for c in \$(seq 0 $((n-1))); do echo logs/eval_${job}_\$((v*$n+c)).log; done) 2>/dev/null | \
  awk -v kw=\"$lab \$kw\" '/^seed/{s+=\$4;n++; if(\$4<m||n==1)m=\$4; if(\$4>M)M=\$4; if(\$4>=8)h++; if(\$0 !~ /terminated/)tr++} \
  END{printf \"%-50s n=%3d mean %.2f min %d max %d  >=8: %d  not_term %d\n\", kw, n, s/n, m, M, h, tr}'; done"
