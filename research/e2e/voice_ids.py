"""How many people speak, and which voice is which person? Diarization clusters checked by faces and voice prints.

Diarization alone guesses the number of voices and links nothing to the image; the active speaker scores alone
(``link_speakers.py --asd``) link a voice to a face even when the face belongs to someone else (on the talk-show,
the explainer of the inserted video sits in the host's pieces, so his voice went to the host). This step joins
both:

1. **Voice print per cluster**: mean speaker embedding (WeSpeaker ResNet34 of pyannote community-1, the model
   ``psifx audio diarization`` already uses) of the diarization turns of at least 1 s.
2. **Face link per cluster**: the active speaker mapping of ``link_speakers.py --asd`` (``identification`` json).
3. **One voice per person**: among the clusters linked to a person, the one with the most speech is that person's
   voice. Another cluster linked to the same person keeps the link only if its print is close to that voice
   (cosine >= ``--threshold``); otherwise it is someone else (a face of an inserted video, a voice-over).
4. **Off-screen voices**: clusters linked to no one are merged when their prints are close (average linkage,
   same threshold); each group is one off-screen voice (``off_1``, ...).
5. **Count**: people with a voice plus off-screen voices with at least ``--min_seconds`` of speech.

Writes ``<output>.json`` in the format of ``psifx audio identification`` (``mapping`` speaker -> label, usable by
``psifx audio transcription enhance``) with the count, the prints' similarities and the decisions.

    python research/e2e/voice_ids.py .e2e_outputs/extrait_talkshow/pipeline \\
        --identification asd_identification.json --output .e2e_outputs/extrait_talkshow/pipeline/voices
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.io import wavfile

from psifx.io import rttm

SAMPLE_RATE, MIN_TURN = 16000, 1.0


def cluster_prints(audio: np.ndarray, turns: list[dict], device: str) -> dict[str, np.ndarray]:
    """Mean L2-normalized voice print of each diarization cluster, from its turns of at least 1 s."""
    import torch
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    model = Model.from_pretrained("pyannote/speaker-diarization-community-1", subfolder="embedding")
    inference = Inference(model, window="whole", device=torch.device(device))
    file = {"waveform": torch.from_numpy(audio[None].astype(np.float32)), "sample_rate": SAMPLE_RATE}
    prints: dict[str, list] = {}
    for t in turns:
        if t["duration"] >= MIN_TURN:
            e = np.asarray(inference.crop(file, Segment(t["start"], t["start"] + t["duration"]))).reshape(-1)
            prints.setdefault(t["speaker_name"], []).append(e / (np.linalg.norm(e) + 1e-9))
    return {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in prints.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with audio.wav")
    parser.add_argument("--diarization", type=Path, help="default: <pipeline>/diarization.rttm")
    parser.add_argument("--identification", type=Path, required=True,
                        help="link_speakers.py --asd output for the same diarization")
    parser.add_argument("--output", type=Path, required=True, help="output path, .json added")
    parser.add_argument("--threshold", type=float, default=0.5, help="cosine similarity of the same voice")
    parser.add_argument("--min_seconds", type=float, default=2.0, help="speech needed to count an off-screen voice")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.pipeline
    diarization = args.diarization or d / "diarization.rttm"
    for path in (d / "audio.wav", diarization, args.identification):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    turns = rttm.RTTMReader.read(diarization, verbose=False)
    face_link = json.loads(args.identification.read_text())["mapping"]
    speech = {}
    for t in turns:
        speech[t["speaker_name"]] = speech.get(t["speaker_name"], 0) + t["duration"]
    if set(speech) - set(face_link):
        sys.exit(f"error: {args.identification} was not made from {diarization}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio16k.wav"
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "audio.wav"), "-ac", "1", "-ar",
                           str(SAMPLE_RATE), str(wav)]).returncode:
            sys.exit("error: ffmpeg could not resample the audio")
        _, audio = wavfile.read(wav)
    prints = cluster_prints(audio.astype(np.float32) / 32768.0, turns, args.device)

    def similarity(a: str, b: str) -> float | None:
        return round(float(prints[a] @ prints[b]), 3) if a in prints and b in prints else None

    mapping, decisions = {}, []
    # 3. One voice per person: the linked cluster with the most speech, the others checked against it.
    people = sorted({p for p in face_link.values() if p.startswith("person_")})
    for person in people:
        linked = sorted((s for s, p in face_link.items() if p == person), key=lambda s: -speech[s])
        main_voice = linked[0]
        mapping[main_voice] = person
        for s in linked[1:]:
            sim = similarity(s, main_voice)
            same = sim is not None and sim >= args.threshold
            mapping[s] = person if same else None
            decisions.append({"cluster": s, "face": person, "voice_of": main_voice, "similarity": sim,
                              "kept": same})
    # 4. Off-screen voices: unlinked clusters (and rejected links) merged by voice.
    rest = [s for s in speech if mapping.get(s) is None]
    with_print = [s for s in rest if s in prints]
    if len(with_print) >= 2:
        groups = fcluster(linkage(np.stack([prints[s] for s in with_print]), method="average", metric="cosine"),
                          t=1 - args.threshold, criterion="distance")
    else:
        groups = np.ones(len(with_print), int)
    group_of = {s: int(g) for s, g in zip(with_print, groups)}
    for s in rest:  # A cluster without a 1 s turn has no print: a voice of its own.
        group_of.setdefault(s, int(max(groups, default=0)) + 1 + rest.index(s))
    seconds = {}
    for s in rest:
        seconds[group_of[s]] = seconds.get(group_of[s], 0) + speech[s]
    rank = {g: k + 1 for k, g in enumerate(sorted(seconds, key=lambda g: -seconds[g]))}
    for s in rest:
        mapping[s] = f"off_{rank[group_of[s]]}"

    off = {f"off_{rank[g]}": round(v, 1) for g, v in seconds.items()}
    count = len(people) + int(sum(v >= args.min_seconds for v in off.values()))
    names = sorted(prints)
    result = {"mapping": mapping, "agreement": None, "count": count, "people": people, "off_screen_seconds": off,
              "speech_seconds": {s: round(v, 1) for s, v in speech.items()}, "decisions": decisions,
              "print_similarity": {a: {b: similarity(a, b) for b in names} for a in names},
              "params": {"diarization": str(diarization), "identification": str(args.identification),
                         "threshold": args.threshold, "min_seconds": args.min_seconds}}
    output = args.output.parent / (args.output.name + ".json")
    output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("mapping", "count", "off_screen_seconds", "decisions")}))
    print("VOICES_OK")


if __name__ == "__main__":
    main()
