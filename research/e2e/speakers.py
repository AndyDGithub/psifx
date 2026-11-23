"""Who speaks, how many voices, and the subtitles: the whole speaker chain in one command.

Each step is a script or psifx command already measured on the talk-show (``research/15_visage_et_fluidite.md``,
section 7); this file only chains them, skips a step whose output exists (``--overwrite`` to redo all) and stops at
the first failure:

1. ``asd_speakers.py``: Light-ASD, frame by frame, is this face the one talking (per merged person of
   ``people.json``, or per SAM3 track without it);
2. ``psifx audio diarization pyannote``: diarization asked for ``--max_voices`` voices on purpose (an upper bound,
   not the answer: a voice cut in two is merged back at step 4, two voices in one cluster are not separable);
3. ``link_speakers.py --asd``: each cluster to the face that talks during it, or off-screen;
4. ``voice_ids.py``: voice prints keep one voice per person and group the off-screen clusters; gives the count;
5. ``psifx audio transcription enhance``: a speaker on each subtitle line;
6. ``render.py --subtitles_only``: ``subtitles.ass``, each person in the color of their avatar.

Inputs in the pipeline folder: ``input.mp4``, ``audio.wav``, ``poses_mp/``, ``transcription.vtt``, and
``people.json`` if it exists.

    python research/e2e/speakers.py .e2e_outputs/extrait_talkshow/pipeline --light_asd .e2e_outputs/ext/light-asd/repo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(name: str, command: list[str], outputs: list[Path], overwrite: bool) -> None:
    if not overwrite and all(o.exists() for o in outputs):
        print(f"[{name}] kept {', '.join(str(o) for o in outputs)}", flush=True)
        return
    start = time.time()
    print(f"[{name}] {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"error: step {name} failed ({result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}")
    missing = [str(o) for o in outputs if not o.exists()]
    if missing:
        sys.exit(f"error: step {name} ended without writing {', '.join(missing)}")
    print(f"[{name}] done in {time.time() - start:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--light_asd", type=Path, required=True, help="clone of github.com/Junhua-Liao/Light-ASD")
    parser.add_argument("--max_voices", type=int, default=6, help="voices asked to the diarization (upper bound)")
    parser.add_argument("--threshold", type=float, default=0.5, help="voice print similarity of one voice")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="redo every step")
    args = parser.parse_args()

    d = args.pipeline
    for path in (d / "input.mp4", d / "audio.wav", d / "poses_mp", d / "transcription.vtt", args.light_asd):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    people = d / "people.json"
    with_people = ["--people", str(people)] if people.is_file() else []
    print("people:", people if people.is_file() else "none, one person per SAM3 track", flush=True)
    py = sys.executable
    diarization = d / f"diarization_over{args.max_voices}.rttm"
    identification, voices = d / "identification_asd.json", d / "voices"
    vtt = d / "transcription_voices.vtt"

    run("asd", [py, str(HERE / "asd_speakers.py"), str(d), *with_people, "--light_asd", str(args.light_asd)],
        [d / "asd"], args.overwrite)
    run("diarization", ["psifx", "audio", "diarization", "pyannote", "inference", "--audio", str(d / "audio.wav"),
                        "--diarization", str(diarization), "--num_speakers", str(args.max_voices), "--exclusive",
                        "--device", args.device, "--overwrite"], [diarization], args.overwrite)
    run("link", [py, str(HERE / "link_speakers.py"), str(d), *with_people, "--asd", str(d / "asd"),
                 "--diarization", str(diarization), "--output", str(identification)], [identification], args.overwrite)
    run("voices", [py, str(HERE / "voice_ids.py"), str(d), "--diarization", str(diarization), "--identification",
                   str(identification), "--output", str(voices), "--threshold", str(args.threshold), "--device",
                   args.device], [voices.with_name(voices.name + ".json")], args.overwrite)
    run("enhance", ["psifx", "audio", "transcription", "enhance", "--transcription", str(d / "transcription.vtt"),
                    "--diarization", str(diarization), "--identification", str(voices.with_name(voices.name + ".json")),
                    "--enhanced_transcription", str(vtt), "--overwrite"], [vtt], args.overwrite)
    run("subtitles", [py, str(HERE / "render.py"), str(d), "--subtitles_only", "--subtitles", str(vtt), *with_people],
        [d / "subtitles.ass"], True)

    result = json.loads(voices.with_name(voices.name + ".json").read_text())
    print(json.dumps({"voices_counted": result["count"], "mapping": result["mapping"],
                      "off_screen_seconds": result["off_screen_seconds"]}))
    print("SPEAKERS_OK")


if __name__ == "__main__":
    main()
