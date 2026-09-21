#!/usr/bin/env bash
# Runs the speed (time_psifx_pose.py) and accuracy (run_psifx_cli_benchmark.py)
# measurements of one tool for a list of "device|precision-kwargs-json|cli-extra" configs.
# Usage: bash run_optim_matrix.sh <tool> "<device>|<kwargs json>|<cli extra>" ...
set -euo pipefail
cd "$(dirname "$0")"
# onnxruntime-gpu needs the pip-installed cuDNN/cuBLAS; appended (not prepended) so torch keeps its own.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(python -c 'import nvidia.cudnn as m; print(m.__path__[0])')/lib:$(python -c 'import nvidia.cublas as m; print(m.__path__[0])')/lib"
tool="$1"; shift
for config in "$@"; do
  IFS='|' read -r device kwargs extra <<< "$config"
  echo "=== $tool device=$device kwargs=$kwargs"
  python time_psifx_pose.py --tool "$tool" --device "$device" --kwargs "$kwargs" 2>&1 | grep '^{' || echo "TIMING FAILED"
  python run_psifx_cli_benchmark.py --tool "$tool" --extra="--device $device $extra" 2>&1 | grep '^{' || echo "ACCURACY FAILED"
done
