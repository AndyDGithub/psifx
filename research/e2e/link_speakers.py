"""Links each diarized voice (SPEAKER_00, ...) to a tracked person (mask 0, 1, ...) from lip motion.

psifx only links voices to people when each person wears a microphone (``audio identification``).
Here the link comes from the video: a person's mouth moves while their voice is heard.

1. Mouth opening per person and frame, from MediaPipe's face mesh: inner-lip gap (points 13/14)
   divided by the mouth width (corners 61/291), so the face size does not matter.
2. Lip activity: standard deviation of the opening over a sliding window of about 0.5 s. Talking
   makes the opening vary, a still or a constantly open mouth does not.
3. Score of (speaker, person): z-scored activity of the person while the speaker talks, minus while
   they do not.
4. Assignment: one person per speaker (Hungarian algorithm) when the score is convincing, then
   left-over speakers (diarization often splits one voice in two) go to their best person, else
   "offscreen" (voice-over, music, someone not visible).

Writes ``identification_lips.json`` in the format of ``psifx audio identification``
(``{"mapping": {speaker: person}, "agreement": ...}``) plus the scores, so that
``psifx audio transcription enhance`` can use it directly.

    python research/e2e/link_speakers.py .e2e_outputs/v1_example
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from psifx.io import rttm, video
from psifx.video.pose.tool import PoseEstimationTool

UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291
OFFSCREEN = "offscreen"


def mouth_opening(poses: Dict[int, dict], n_frames: int) -> np.ndarray:
    """Inner-lip gap over mouth width per frame, NaN where no face was found."""
    opening = np.full(n_frames, np.nan)
    for i, pose in poses.items():
        if i >= n_frames:
            continue
        face = np.asarray(pose.get("face_keypoints_2d", []), dtype=float).reshape(-1, 3)
        if len(face) <= MOUTH_RIGHT or not face[:, :2].any():
            continue
        width = np.linalg.norm(face[MOUTH_LEFT, :2] - face[MOUTH_RIGHT, :2])
        if width > 1.0:  # pixels; a degenerate mouth means a failed detection
            opening[i] = np.linalg.norm(face[UPPER_LIP, :2] - face[LOWER_LIP, :2]) / width
    return opening


def lip_activity(opening: np.ndarray, fps: float, window_s: float = 0.5) -> np.ndarray:
    window = max(3, int(round(window_s * fps)))
    series = pd.Series(opening)
    return series.rolling(window, center=True, min_periods=window // 2).std().to_numpy()


def speaking_frames(segments, speaker: str, n_frames: int, fps: float) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for seg in segments:
        if seg["speaker_name"] == speaker:
            start = int(np.floor(seg["start"] * fps))
            end = int(np.ceil((seg["start"] + seg["duration"]) * fps))
            mask[max(0, start):min(n_frames, end)] = True
    return mask


def score_matrix(activity: Dict[str, np.ndarray], segments, n_frames: int, fps: float,
                 min_coverage: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scores (speakers x people), and the share of each speaker's time where the person's lips are seen."""
    speakers = sorted({seg["speaker_name"] for seg in segments})
    scores = pd.DataFrame(np.nan, index=speakers, columns=sorted(activity, key=int))
    coverage = scores.copy()
    for person, act in activity.items():
        valid = ~np.isnan(act)
        if valid.sum() < 2:
            continue
        z = (act - np.nanmean(act)) / (np.nanstd(act) + 1e-9)
        for speaker in speakers:
            talking = speaking_frames(segments, speaker, n_frames, fps)
            seen = talking & valid
            coverage.loc[speaker, person] = seen.sum() / max(1, talking.sum())
            others = ~talking & valid
            if coverage.loc[speaker, person] >= min_coverage and others.any():
                scores.loc[speaker, person] = z[seen].mean() - z[others].mean()
    return scores, coverage


def assign(scores: pd.DataFrame, min_score: float) -> Dict[str, str]:
    mapping = {speaker: OFFSCREEN for speaker in scores.index}
    if scores.empty:
        return mapping
    filled = scores.fillna(-np.inf).to_numpy()
    cost = np.where(np.isfinite(filled), -filled, 1e6)
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if filled[r, c] >= min_score:
            mapping[scores.index[r]] = f"person_{scores.columns[c]}"
    # Diarization often splits one voice into several speakers: they may share a person.
    for r, speaker in enumerate(scores.index):
        if mapping[speaker] == OFFSCREEN and np.isfinite(filled[r]).any():
            c = int(np.argmax(filled[r]))
            if filled[r, c] >= min_score:
                mapping[speaker] = f"person_{scores.columns[c]}"
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min_score", type=float, default=0.2, help="minimum z-score difference to link")
    parser.add_argument("--min_coverage", type=float, default=0.2,
                        help="minimum share of a speaker's time where the person's lips are visible")
    args = parser.parse_args()
    d = args.out_dir

    poses_dir, diarization_path = d / "poses_mp", d / "diarization.rttm"
    for path in (poses_dir, diarization_path, d / "input.mp4"):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    output = d / "identification_lips.json"

    with video.VideoReader(path=d / "input.mp4") as reader:
        n_frames, fps = len(reader), float(Fraction(reader.frame_rate))  # "25/1" or "30000/1001"
    segments = rttm.RTTMReader.read(diarization_path, verbose=False)
    if not segments:
        sys.exit("error: empty diarization, nothing to link")

    decoder = PoseEstimationTool(device="cpu", verbose=False)
    activity = {}
    for archive in sorted(poses_dir.glob("*.tar.gz")):
        person = archive.name.removesuffix(".tar.gz")
        poses, _ = decoder.decode(archive)
        activity[person] = lip_activity(mouth_opening(poses, n_frames), fps)

    scores, coverage = score_matrix(activity, segments, n_frames, fps, args.min_coverage)
    mapping = assign(scores, args.min_score)
    linked = [scores.loc[s, p.removeprefix("person_")] for s, p in mapping.items() if p != OFFSCREEN]

    result = {
        "mapping": mapping,
        # Mean score of the kept links (z-score difference), the closer to 0 the weaker the evidence.
        "agreement": float(np.mean(linked)) if linked else 0.0,
        "scores": json.loads(scores.round(3).to_json()),
        "lip_coverage": json.loads(coverage.round(3).to_json()),
        "params": {"min_score": args.min_score, "min_coverage": args.min_coverage},
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["mapping"], indent=2))
    print(scores.round(2).to_string())


if __name__ == "__main__":
    main()
