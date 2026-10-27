"""Measures what the end-to-end pipeline produced on one video, to locate its limits.

- Tracking: number of SAM3 tracks and the share of frames where each one is present.
- Poses (per model and person): share of frames with a pose, mean keypoint confidence, normalized
  jerk (temporal stability, see tests/pose_eval/metrics.py).
- Audio: number of diarized speakers, speech time, number of subtitle segments and how many reach a
  visible person. With a ``reference.txt`` (song lyrics, manual transcript) also the WER.
- Voice to person linking: with AVA-ActiveSpeaker labels (``ava_active_speaker.csv``: speaking face
  boxes per instant), the share of speaking instants where the predicted speaker's head is in a
  speaking box. With per-person microphones (``identification_mics.json``), the share of
  speech time where the lip-based link agrees with the microphones. The microphone to person
  correspondence is the best one-to-one match, to check once by eye on the rendered video. With
  ``reference_speakers.txt`` (``<segment index> <person>`` per line, for the subtitle segments whose
  speaker is certain from the text alone), the share of their time given to the right person.
- Cost: time per stage and real-time factor (compute time / video duration).

Writes ``metrics.json`` and ``contact_sheet.jpg`` (8 frames of the rendered video).

    python research/e2e/evaluate.py .e2e_outputs/v1_example
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from psifx.io import rttm, video, vtt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.e2e.render import load_people  # noqa: E402
from tests.pose_eval.metrics import normalized_jerk_series  # noqa: E402

COCO_TRACKED = {"nose": 0, "left_wrist": 9, "right_wrist": 10}
MP_TRACKED = {"nose": 0, "left_wrist": 15, "right_wrist": 16}
SCALE_PAIRS = {"coco": (5, 12), "mp": (11, 24)}  # left shoulder - right hip


def mask_presence(mask_path: Path, n_frames: int) -> np.ndarray:
    """Share of frames where a mask video has any foreground, read at low resolution."""
    w, h = 64, 36
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mask_path), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w)[:n_frames]
    return (frames > 127).any(axis=(1, 2))


def pose_stats(parts, kind: str, fps: float, min_conf: float) -> dict:
    kp = parts["pose_keypoints_2d"]
    conf = kp[..., 2]
    detected = np.nanmax(np.nan_to_num(conf, nan=0.0), axis=1) >= min_conf
    tracked = COCO_TRACKED if kind == "coco" else MP_TRACKED
    a, b = SCALE_PAIRS[kind]
    scales = np.linalg.norm(kp[:, a, :2] - kp[:, b, :2], axis=1)
    scale = float(np.nanmedian(scales[detected])) if detected.any() else float("nan")
    jerks = {}
    for name, idx in tracked.items():
        track = kp[:, idx, :2].copy()
        track[~(np.nan_to_num(kp[:, idx, 2]) >= min_conf)] = np.nan
        jerks[name] = round(normalized_jerk_series(track, fps, scale), 1) if np.isfinite(scale) else None
    return {
        "detected_frames": round(float(detected.mean()), 3),
        "mean_confidence": round(float(np.nanmean(conf[detected])), 3) if detected.any() else None,
        "jerk": jerks,
    }


def normalize_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())  # [Chorus], [Drop]
    text = re.sub(r"[^\w' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def link_accuracy(segments, lips: dict, mics: dict) -> dict:
    """Share of speech time where lips and microphones point to the same person or to "offscreen"
    (best one-to-one match between microphones and these labels)."""
    durations = {}
    for seg in segments:
        durations[seg["speaker_name"]] = durations.get(seg["speaker_name"], 0.0) + float(seg["duration"])
    mic_names = sorted(set(mics.values()))
    # "offscreen" is a valid answer: a microphone can belong to someone outside the frame.
    people = sorted(set(lips.values()))
    weight = np.zeros((len(mic_names), len(people)))
    for speaker, duration in durations.items():
        if lips.get(speaker) in people and speaker in mics:
            weight[mic_names.index(mics[speaker]), people.index(lips[speaker])] += duration
    rows, cols = linear_sum_assignment(-weight)
    total = sum(durations.values())
    return {
        "accuracy": round(float(weight[rows, cols].sum() / total), 3),
        "mic_to_person": {mic_names[r]: people[c] for r, c in zip(rows, cols)},
        "speech_seconds": round(total, 1),
    }


def head_positions(d: Path, n_frames: int, width: int, height: int) -> dict:
    """Normalized head position per person and frame (MediaPipe face center, else RTMO nose), NaN if unseen."""
    heads = {}
    mp, body = load_people(d / "poses_mp", n_frames), load_people(d / "poses_body", n_frames)
    for person in set(mp) | set(body):
        pos = np.full((n_frames, 2), np.nan)
        if person in body:
            nose = body[person]["pose_keypoints_2d"][:, 0]
            ok = np.nan_to_num(nose[:, 2]) >= 0.3
            pos[ok] = nose[ok, :2]
        if person in mp:
            face = mp[person]["face_keypoints_2d"][..., :2]
            ok = ~np.isnan(face).all(axis=(1, 2))
            pos[ok] = np.nanmean(face[ok], axis=1)
        heads[person] = pos / [width, height]
    return heads


def active_speaker_accuracy(d: Path, segments, mapping: dict, n_frames: int, fps: float) -> dict:
    """AVA-ActiveSpeaker check: at each labeled instant where a face speaks audibly, is the person that
    diarization + lip linking says is speaking inside one of the speaking face boxes?"""
    import pandas as pd

    labels = pd.read_csv(d / "ava_active_speaker.csv")
    with video.VideoReader(path=d / "input.mp4") as reader:
        height, width = next(iter(reader)).shape[:2]
    heads = head_positions(d, n_frames, width, height)
    margin = 0.02
    correct = answered = total = 0
    chance = []
    for t, faces in labels.groupby("t"):
        speaking = faces[faces["label"] == "SPEAKING_AUDIBLE"]
        if speaking.empty:
            continue
        total += 1
        chance.append(len(speaking) / len(faces))
        frame = min(n_frames - 1, int(round(t * fps)))
        talkers = {mapping.get(s["speaker_name"]) for s in segments
                   if s["start"] <= t < s["start"] + s["duration"]}
        people = [p.removeprefix("person_") for p in talkers if p and p.startswith("person_")]
        if not people:
            continue
        answered += 1
        hit = False
        for person in people:
            x, y = heads.get(person, np.full((n_frames, 2), np.nan))[frame]
            inside = ((speaking["x1"] - margin <= x) & (x <= speaking["x2"] + margin)
                      & (speaking["y1"] - margin <= y) & (y <= speaking["y2"] + margin))
            hit |= bool(inside.any())
        correct += hit
    return {
        "labeled_speaking_instants": total,
        "answered": round(answered / total, 3) if total else None,
        "accuracy": round(correct / total, 3) if total else None,
        "accuracy_when_answered": round(correct / answered, 3) if answered else None,
        "chance": round(float(np.mean(chance)), 3) if chance else None,
    }


def text_label_accuracy(subs, labels_path: Path) -> dict:
    """Share of the labeled subtitle segments' time whose speaker matches the label."""
    labels = {}
    for line in labels_path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            index, person = line.split()
            labels[int(index)] = person
    if max(labels) >= len(subs):
        # The labels refer to another transcription run: refuse rather than score the wrong segments.
        raise ValueError(f"{labels_path} labels segment {max(labels)}, the transcription has {len(subs)}")
    duration = {i: subs[i]["end"] - subs[i]["start"] for i in labels}
    correct = [i for i, person in labels.items() if subs[i]["speaker"] == person]
    return {
        "segments": len(labels),
        "correct_segments": len(correct),
        "time_accuracy": round(sum(duration[i] for i in correct) / sum(duration.values()), 3),
    }


def contact_sheet(render: Path, out: Path, n: int = 8):
    cap = cv2.VideoCapture(str(render))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, frame = cap.read()
        if ok:
            tiles.append(cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
    cap.release()
    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(out), np.vstack(rows))


def busy_seconds(d: Path, timings: dict) -> float:
    """Wall-clock time of the run: union of the stage spans, as stages may overlap (run_pipeline.py).

    Gaps between two resumed runs are not counted. Without spans (older runs, one stage at a time),
    the sum of the stage timings.
    """
    spans_path = d / "stage_spans.json"
    if not spans_path.exists():
        return sum(timings.values())
    total, end = 0.0, float("-inf")
    for start, stop in sorted(json.loads(spans_path.read_text()).values()):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min_conf", type=float, default=0.3)
    args = parser.parse_args()
    d = args.out_dir
    if not (d / "input.mp4").is_file():
        sys.exit(f"error: missing {d / 'input.mp4'}")

    with video.VideoReader(path=d / "input.mp4") as reader:
        n_frames, fps = len(reader), float(Fraction(reader.frame_rate))  # "25/1" or "30000/1001"
    duration = n_frames / fps
    metrics = {"video": {"frames": n_frames, "fps": fps, "seconds": round(duration, 1)}}

    masks = sorted((d / "masks").glob("*.mp4"), key=lambda p: int(p.stem)) if (d / "masks").is_dir() else []
    metrics["tracking"] = {
        "tracks": len(masks),
        "presence": {m.stem: round(float(mask_presence(m, n_frames).mean()), 3) for m in masks},
    }

    metrics["poses"] = {}
    for name, kind in (("poses_body", "coco"), ("poses_vitpose", "coco"), ("poses_mp", "mp")):
        people = load_people(d / name, n_frames)
        if people:
            metrics["poses"][name] = {p: pose_stats(parts, kind, fps, args.min_conf) for p, parts in people.items()}

    audio = {}
    if (d / "diarization.rttm").exists():
        segments = rttm.RTTMReader.read(d / "diarization.rttm", verbose=False)
        audio["diarized_speakers"] = len({s["speaker_name"] for s in segments})
        audio["speech_seconds"] = round(sum(float(s["duration"]) for s in segments), 1)
        lips_path, mics_path = d / "identification_lips.json", d / "identification_mics.json"
        if lips_path.exists():
            lips = json.loads(lips_path.read_text())
            audio["lips_mapping"] = lips["mapping"]
            audio["lips_agreement"] = round(lips["agreement"], 3)
            if (d / "ava_active_speaker.csv").exists():
                audio["active_speaker_vs_ava"] = active_speaker_accuracy(d, segments, lips["mapping"], n_frames, fps)
            if mics_path.exists():
                audio["link_vs_mics"] = link_accuracy(segments, lips["mapping"],
                                                      json.loads(mics_path.read_text())["mapping"])
    if (d / "transcription_enhanced.vtt").exists():
        subs = vtt.VTTReader.read(d / "transcription_enhanced.vtt", verbose=False)
        seconds = sum(s["end"] - s["start"] for s in subs)
        on_person = sum(s["end"] - s["start"] for s in subs if (s["speaker"] or "").startswith("person_"))
        audio["subtitle_segments"] = len(subs)
        audio["subtitle_time_on_visible_person"] = round(on_person / seconds, 3) if seconds else None
        if (d / "reference_speakers.txt").exists():
            audio["attribution_vs_text_labels"] = text_label_accuracy(subs, d / "reference_speakers.txt")
        if (d / "reference.txt").exists():
            import jiwer

            reference = normalize_text((d / "reference.txt").read_text(encoding="utf-8"))
            hypothesis = normalize_text(" ".join(s["text"] for s in subs))
            audio["wer"] = round(float(jiwer.wer(reference, hypothesis)), 3) if hypothesis else 1.0
            audio["hypothesis"] = hypothesis[:500]
    metrics["audio"] = audio

    if (d / "timings.json").exists():
        timings = json.loads((d / "timings.json").read_text())
        metrics["timings_s"] = timings
        metrics["real_time_factor"] = round(busy_seconds(d, timings) / duration, 2)

    if (d / "render.mp4").exists():
        contact_sheet(d / "render.mp4", d / "contact_sheet.jpg")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
