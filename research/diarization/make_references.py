"""Builds the diarization references of the end-to-end videos (no manual RTTM exists for them).

- V1 (``example/data/Video.mp4``): one microphone per person. A 20 ms frame belongs to the person
  whose microphone is at least ``--dominance`` dB louder than the other one and above the noise
  floor; turns are smoothed (gaps < 0.3 s closed, turns < 0.3 s dropped). Overlapped speech is
  left out of the reference, so the DER is scored with a collar and read as a pseudo-reference.
- V3 (talk-show): the 40 transcription segments labeled from the text
  (``reference_speakers.txt``). Only these segments are scored (``reference.uem``).

    python research/diarization/make_references.py .e2e_outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

FRAME = 0.02


def frame_db(path: Path) -> np.ndarray:
    rate, signal = wavfile.read(path)
    if signal.ndim > 1:
        sys.exit(f"error: {path} should be mono")
    signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max
    hop = int(rate * FRAME)
    frames = signal[: len(signal) // hop * hop].reshape(-1, hop)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)


def runs(active: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame indices of the True runs."""
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smooth(active: np.ndarray, min_gap: int, min_turn: int) -> np.ndarray:
    out = active.copy()
    for start, end in runs(~out):
        if start > 0 and end < len(out) and end - start < min_gap:
            out[start:end] = True
    for start, end in runs(out):
        if end - start < min_turn:
            out[start:end] = False
    return out


def write_rttm(turns, uri: str, path: Path):
    lines = [f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
             for start, end, speaker in sorted(turns)]
    path.write_text("\n".join(lines) + "\n")


def mic_reference(d: Path, out: Path, dominance: float):
    levels = {name: frame_db(d / f"{name}.wav") for name in ("mic_left", "mic_right")}
    n = min(len(v) for v in levels.values())
    left, right = levels["mic_left"][:n], levels["mic_right"][:n]
    floor = np.percentile(np.maximum(left, right), 20) + 10  # 10 dB above the quietest fifth
    gap = int(0.3 / FRAME)
    turns = []
    for name, own, other in (("mic_left", left, right), ("mic_right", right, left)):
        active = smooth((own > floor) & (own - other > dominance), gap, gap)
        turns += [(s * FRAME, e * FRAME, name) for s, e in runs(active)]
    if not turns:
        sys.exit(f"error: no speech found in the microphones of {d}")
    write_rttm(turns, "audio", out / "reference.rttm")
    (out / "reference.uem").write_text(f"audio 1 0.000 {n * FRAME:.3f}\n")
    speech = {name: sum(e - s for s, e, sp in turns if sp == name) for name in levels}
    print(f"{d.name}: {len(turns)} turns, speech per mic {speech}")


def text_reference(d: Path, out: Path):
    from psifx.io import vtt

    subs = vtt.VTTReader.read(d / "transcription.vtt", verbose=False)
    labels = {}
    for line in (d / "reference_speakers.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            index, person = line.split()
            labels[int(index)] = person
    if max(labels) >= len(subs):
        sys.exit(f"error: labels refer to segment {max(labels)}, the transcription has {len(subs)}")
    turns = [(subs[i]["start"], subs[i]["end"], person) for i, person in labels.items()]
    write_rttm(turns, "audio", out / "reference.rttm")
    (out / "reference.uem").write_text(
        "".join(f"audio 1 {start:.3f} {end:.3f}\n" for start, end, _ in sorted(turns)))
    print(f"{d.name}: {len(turns)} labeled segments, {sum(e - s for s, e, _ in turns):.1f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("e2e_dir", type=Path)
    parser.add_argument("--dominance", type=float, default=6.0, help="dB margin between the microphones")
    args = parser.parse_args()
    for name, build in (("v1_example", lambda d, o: mic_reference(d, o, args.dominance)),
                        ("v3_talkshow", text_reference)):
        out = args.e2e_dir / "diar" / name
        out.mkdir(parents=True, exist_ok=True)
        build(args.e2e_dir / name, out)


if __name__ == "__main__":
    main()
