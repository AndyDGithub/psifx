#!/bin/bash
# MediaPipe Tasks (psifx-next, worktree mediapipe-tasks) on the e2e masks, for comparison with the legacy outputs.
set -euo pipefail
source /home/psifx/conda/etc/profile.d/conda.sh && conda activate psifx-next
cd /home/psifx/psifx
for v in v1_example v2_ava v4_dance v3_talkshow; do
  out=.e2e_outputs/mp_tasks${MP_SUFFIX:-}/$v; mkdir -p $out
  s=$(date +%s)
  psifx video pose mediapipe multi-inference --video .e2e_outputs/$v/input.mp4 --masks .e2e_outputs/$v/masks \
    --poses_dir $out/poses_mp --overwrite --no-verbose > $out/run.log 2>&1 || { echo "FAILED $v"; tail -5 $out/run.log; exit 1; }
  echo "$v $(( $(date +%s) - s )) s"
done
