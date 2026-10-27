#!/bin/bash
# Re-runs voice<->person linking, subtitle enhancement and evaluation of the e2e videos with each
# diarization variant from .e2e_outputs/diar/<video>/<tag>.rttm. Heavy outputs are symlinked.
set -euo pipefail
source /home/psifx/conda/etc/profile.d/conda.sh && conda activate psifx-env
cd /home/psifx/psifx
E=.e2e_outputs
for case in "$@"; do
  # "case" or "case:source": a noisy mix reuses the video, poses and transcription of its source.
  v=${case%%:*}; src=${case#*:}
  for rttm in $E/diar/$v/*.rttm; do
    tag=$(basename "$rttm" .rttm); [ "$tag" = reference ] && continue
    out=$E/diar_e2e/${v}__$tag; rm -rf "$out"; mkdir -p "$out"
    for f in $E/$src/*; do
      case $(basename "$f") in
        diarization.rttm|identification_*.json|transcription_enhanced.vtt|metrics.json|render.mp4|subtitles.ass|contact_sheet.jpg|logs|run.log|relink.log|timings.json) ;;
        *) ln -s "$(realpath "$f")" "$out/";;
      esac
    done
    cp "$rttm" "$out/diarization.rttm"
    if [ "$v" != "$src" ]; then rm "$out/audio.wav"; cp $E/diar/$v/audio.wav "$out/audio.wav"; fi
    if [ -f $E/$src/mic_left.wav ]; then
      psifx audio identification pyannote inference --mixed_audio $out/audio.wav --diarization $out/diarization.rttm \
        --mono_audios $out/mic_left.wav $out/mic_right.wav --identification $out/identification_mics.json \
        --device cuda --model_names pyannote/embedding --no-verbose > $out/ident.log 2>&1 || echo "identification failed: $out"
    fi
    python research/e2e/link_speakers.py $out > $out/link.log 2>&1
    psifx audio transcription enhance --transcription $out/transcription.vtt --diarization $out/diarization.rttm \
      --identification $out/identification_lips.json --enhanced_transcription $out/transcription_enhanced.vtt --no-verbose > $out/enhance.log 2>&1
    python research/e2e/evaluate.py $out > $out/evaluate.log 2>&1
    echo "done $out"
  done
done
