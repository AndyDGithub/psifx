"""Who is speaking on screen? Light-ASD (audio-visual active speaker detection) on each merged person's face.

The lip opening of ``link_speakers.py`` fails when the mouth is hard to read (beard, profile, small face) or when a
person's track also holds someone else: on the talk-show the host's voice went "off-screen". An active speaker
detector looks at the face crop **and** the sound together and says, frame by frame, whether this face is the one
talking. Light-ASD (Liao et al., CVPR 2023, 94.1 % mAP on AVA-ActiveSpeaker, MIT licence) is used with its
published AVA weights.

Faces come from our own data, not from the demo's detector: for each merged person (``people.json``) the
MediaPipe face landmarks (``poses_mp``) of the pieces given to that person give the face box; pieces left out
(inserted video, dancers) are not used. The crop follows the authors' demo (``Columbia_test.py``): box size and
center smoothed by a median filter of 13, padding 0.4, 224 px resized, grey, center 112 px; video at 25 frames/s,
MFCC at 100 Hz from 16 kHz audio, scores averaged over several chunk lengths.

Writes ``<pipeline>/asd/<person>.npy``: one score per video frame (> 0 speaking, NaN where the face is not seen),
read by ``link_speakers.py --asd``.

    python research/e2e/asd_speakers.py .e2e_outputs/extrait_talkshow/pipeline \\
        --people .e2e_outputs/extrait_talkshow/pipeline/people.json --light_asd .e2e_outputs/ext/light-asd/repo
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

from psifx.io import video
from psifx.video.pose.tool import PoseEstimationTool

ASD_FPS, CROP_SCALE, MAX_GAP_S, MIN_RUN = 25, 0.40, 0.4, 10
DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)  # Chunk lengths (s) of the authors' demo, scores averaged.


def face_boxes(poses: dict, n_frames: int) -> np.ndarray:
    """(n_frames, 4) x1, y1, x2, y2 of the MediaPipe face landmarks, NaN where no face."""
    boxes = np.full((n_frames, 4), np.nan)
    for i, pose in poses.items():
        if i >= n_frames:
            continue
        face = np.asarray(pose.get("face_keypoints_2d", []), dtype=float).reshape(-1, 3)
        points = face[(face[:, :2] > 0).all(1), :2]
        if len(points) >= 100:  # A real face mesh, not a few stray points.
            boxes[i] = (*points.min(0), *points.max(0))
    return boxes


def person_runs(boxes: dict[str, np.ndarray], people: dict, n_frames: int, fps: float) -> dict[int, list]:
    """Per merged person, runs of 25 frames/s samples (sample index, video frame, box) inside one piece."""
    runs: dict[int, list] = {p: [] for p in range(people["people"])}
    for piece in people["assignment"]:
        person, track = piece["person"], str(piece["track"])
        if not isinstance(person, int) or track not in boxes:
            continue  # Left-out piece or a track without face archive.
        start, end = piece["frames"][0], min(piece["frames"][1], n_frames - 1)
        current, last_seen = [], None
        for k in range(math.ceil(start / fps * ASD_FPS), int(end / fps * ASD_FPS) + 1):
            f = min(int(round(k / ASD_FPS * fps)), end)
            box = boxes[track][f]
            if np.isnan(box).any():
                if last_seen is not None and (k - last_seen) / ASD_FPS > MAX_GAP_S:
                    runs[person].append(current)
                    current, last_seen = [], None
                continue
            current.append((k, f, box))
            last_seen = k
        if current:
            runs[person].append(current)
    return {p: [r for r in rs if len(r) >= MIN_RUN] for p, rs in runs.items()}


def crop_plan(run: list) -> list[tuple[int, int, float, float, float]]:
    """Every sample of a run with its smoothed crop (sample, frame, center x, center y, half size); short gaps
    in the run are filled by interpolation, as the demo's tracker does."""
    ks = np.array([k for k, _, _ in run])
    b = np.array([box for _, _, box in run])
    full = np.arange(ks[0], ks[-1] + 1)
    s = np.interp(full, ks, np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) / 2)
    x = np.interp(full, ks, (b[:, 0] + b[:, 2]) / 2)
    y = np.interp(full, ks, (b[:, 1] + b[:, 3]) / 2)
    kernel = 13 if len(full) >= 13 else (len(full) // 2) * 2 + 1
    s, x, y = (signal.medfilt(v, kernel_size=kernel) for v in (s, x, y))
    frames = {k: f for k, f, _ in run}
    fps_ratio = (run[-1][1] - run[0][1]) / max(1, run[-1][0] - run[0][0])
    return [(int(k), frames.get(int(k), int(round(run[0][1] + (k - ks[0]) * fps_ratio))), x[i], y[i], s[i])
            for i, k in enumerate(full)]


def crop(image: np.ndarray, x: float, y: float, s: float) -> np.ndarray:
    """The demo's face crop: padded, 224 px, grey, center 112 px."""
    pad = int(s * (1 + 2 * CROP_SCALE))
    frame = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))
    my, mx = y + pad, x + pad
    face = frame[int(my - s):int(my + s * (1 + 2 * CROP_SCALE)), int(mx - s * (1 + CROP_SCALE)):int(mx + s * (1 + CROP_SCALE))]
    face = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
    return face[56:168, 56:168]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4, audio.wav and poses_mp/")
    parser.add_argument("--people", type=Path, required=True, help="people.json of merge_people.py")
    parser.add_argument("--light_asd", type=Path, required=True, help="clone of github.com/Junhua-Liao/Light-ASD")
    parser.add_argument("--weights", default="weight/pretrain_AVA_CVPR.model", help="inside --light_asd")
    args = parser.parse_args()

    d = args.pipeline
    weights = args.light_asd / args.weights
    for path in (d / "input.mp4", d / "audio.wav", d / "poses_mp", args.people, weights):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    with video.VideoReader(path=d / "input.mp4") as reader:
        n_frames, fps = len(reader), float(Fraction(reader.frame_rate))
    people = json.loads(args.people.read_text())

    decoder = PoseEstimationTool(device="cpu", verbose=False)
    boxes = {a.name.removesuffix(".tar.gz"): face_boxes(decoder.decode(a)[0], n_frames)
             for a in sorted((d / "poses_mp").glob("*.tar.gz"))}
    runs = person_runs(boxes, people, n_frames, fps)
    plans = {(p, r): crop_plan(run) for p, rs in runs.items() for r, run in enumerate(rs)}
    print({p: f"{len(rs)} runs, {sum(len(x) for x in rs) / ASD_FPS:.0f} s" for p, rs in runs.items()}, flush=True)

    # One pass over the video: every needed crop is taken when its frame goes by.
    wanted: dict[int, list] = {}
    for key, plan in plans.items():
        for j, (_, f, x, y, s) in enumerate(plan):
            wanted.setdefault(f, []).append((key, j, x, y, s))
    crops = {key: np.zeros((len(plan), 112, 112), np.uint8) for key, plan in plans.items()}
    capture = cv2.VideoCapture(str(d / "input.mp4"))
    for f in range(max(wanted) + 1):
        ok, image = capture.read()
        if not ok:
            sys.exit(f"error: cannot read frame {f} of {d / 'input.mp4'}")
        for key, j, x, y, s in wanted.get(f, []):
            crops[key][j] = crop(image, x, y, s)
    capture.release()

    import python_speech_features  # noqa: E402
    import torch  # noqa: E402
    sys.path.insert(0, str(args.light_asd))
    from ASD import ASD  # noqa: E402
    model = ASD()
    model.loadParameters(str(weights))
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio16k.wav"
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "audio.wav"), "-ac", "1", "-ar", "16000",
                           str(wav)]).returncode:
            sys.exit("error: ffmpeg could not resample the audio to 16 kHz")
        _, audio = wavfile.read(wav)

    out = d / "asd"
    out.mkdir(exist_ok=True)
    scores = {p: np.full(n_frames, np.nan) for p in runs}
    with torch.no_grad():
        for (p, r), plan in plans.items():
            k0 = plan[0][0]
            start = int(k0 / ASD_FPS * 16000)
            piece = audio[start:start + int(len(plan) / ASD_FPS * 16000) + 400]
            mfcc = python_speech_features.mfcc(piece, 16000, numcep=13, winlen=0.025, winstep=0.010)
            length = min((len(mfcc) - len(mfcc) % 4) / 100, len(plan) / ASD_FPS)
            a, v = mfcc[:int(round(length * 100))], crops[(p, r)][:int(round(length * ASD_FPS))]
            per_duration = []
            for duration in DURATIONS:
                chunk = []
                for i in range(math.ceil(length / duration)):
                    ia = torch.FloatTensor(a[i * duration * 100:(i + 1) * duration * 100]).unsqueeze(0).cuda()
                    iv = torch.FloatTensor(v[i * duration * ASD_FPS:(i + 1) * duration * ASD_FPS]).unsqueeze(0).cuda()
                    if ia.shape[1] < 4 or iv.shape[1] < 1:
                        continue
                    embed = model.model.forward_audio_visual_backend(model.model.forward_audio_frontend(ia),
                                                                     model.model.forward_visual_frontend(iv))
                    chunk.extend(model.lossAV.forward(embed, labels=None))
                per_duration.append(np.array(chunk[:len(v)]))
            n = min(len(x) for x in per_duration)
            sample_scores = np.mean([x[:n] for x in per_duration], axis=0)
            # Each video frame takes the score of the 25 frames/s sample it falls in.
            for j in range(n):
                f0 = int(round((k0 + j) / ASD_FPS * fps))
                f1 = int(round((k0 + j + 1) / ASD_FPS * fps))
                scores[p][f0:max(f1, f0 + 1)] = sample_scores[j]
    for p, s in scores.items():
        np.save(out / f"{p}.npy", s)
        seen = ~np.isnan(s)
        print(f"person {p}: face scored on {seen.mean():.1%} of frames, speaking on {(s[seen] > 0).mean():.1%} of them")
    print("ASD_OK")


if __name__ == "__main__":
    main()
