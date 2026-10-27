#!/bin/bash
# Voice<->person linking, enhancement and evaluation of the e2e videos with another set of MediaPipe
# poses (.e2e_outputs/<poses_root>/<video>/poses_mp). Everything else is the e2e run (symlinks).
set -euo pipefail
source /home/psifx/conda/etc/profile.d/conda.sh && conda activate psifx-env
cd /home/psifx/psifx
E=.e2e_outputs; root=$1; shift
for v in "$@"; do
  out=$E/mp_e2e/${v}__$root${DIAR:+__$DIAR}; rm -rf "$out"; mkdir -p "$out"
  for f in $E/$v/*; do
    case $(basename "$f") in
      poses_mp|identification_lips.json|transcription_enhanced.vtt|metrics.json|render.mp4|subtitles.ass|contact_sheet.jpg|logs|run.log|relink.log|timings.json) ;;
      *) ln -s "$(realpath "$f")" "$out/";;
    esac
  done
  ln -s "$(realpath $E/$root/$v/poses_mp)" "$out/poses_mp"
  if [ -n "${DIAR:-}" ]; then rm "$out/diarization.rttm"; cp $E/diar/$v/$DIAR.rttm "$out/diarization.rttm"; fi
  python research/e2e/link_speakers.py $out > $out/link.log 2>&1
  psifx audio transcription enhance --transcription $out/transcription.vtt --diarization $out/diarization.rttm \
    --identification $out/identification_lips.json --enhanced_transcription $out/transcription_enhanced.vtt --no-verbose > $out/enhance.log 2>&1
  python research/e2e/evaluate.py $out > $out/evaluate.log 2>&1
  echo "done $out"
done
