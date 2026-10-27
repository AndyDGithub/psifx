"""Re-renders a video from the data psifx extracted: one colored skeleton per person, a simple 3D view
and subtitles colored by the person who speaks.

Layout (height 720):
- left, the 2D scene on a neutral background, with the same aspect ratio as the input:
  - body from RTMO, face contours and hands from MediaPipe;
  - the person who speaks is outlined;
  - subtitles at the bottom, in the speaker's color;
- right, a thumbnail of the original video and a 3D panel (MediaPipe world landmarks, one stick
  figure per person, slowly rotating to show depth).

Also writes ``subtitles.ass``, with one style per person, playable over the original video in VLC
or burnable with ffmpeg.

    python research/e2e/render.py .e2e_outputs/v1_example
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from psifx.io import video, vtt
from psifx.video.pose.tool import PoseEstimationTool

# Distinct colors (RGB), one per person in the order of their mask ids.
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 200, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (170, 110, 40), (128, 128, 0)]
NEUTRAL = (200, 200, 200)
BACKGROUND = (28, 28, 34)
HEIGHT, SIDE_WIDTH = 720, 400
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COCO_EDGES = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
              (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))


def load_archive(path: Path, n_frames: int) -> Dict[str, np.ndarray]:
    """Pose archive as {part: [n_frames, K, C] array}, NaN where the part was not detected."""
    poses, _ = PoseEstimationTool(device="cpu", verbose=False).decode(path)
    channels, sizes = {}, {}
    for pose in poses.values():
        for key, value in pose.items():
            channels[key] = 4 if key.endswith("_3d") else 3
            # MediaPipe writes 468 zeros for a missing face but 478 points (with irises) for a found one.
            sizes[key] = max(sizes.get(key, 0), len(value) // channels[key])
    parts = {key: np.full((n_frames, sizes[key], channels[key]), np.nan, dtype=np.float32) for key in sizes}
    for i, pose in poses.items():
        if i >= n_frames:
            continue
        for key, value in pose.items():
            arr = np.asarray(value, dtype=np.float32).reshape(-1, channels[key])
            if arr.any():
                parts[key][i, :len(arr)] = arr
    return parts


def load_people(directory: Path, n_frames: int) -> Dict[str, Dict[str, np.ndarray]]:
    if not directory.is_dir():
        return {}
    return {p.name.removesuffix(".tar.gz"): load_archive(p, n_frames) for p in sorted(directory.glob("*.tar.gz"))}


def face_contour_edges():
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS

    return tuple(FACEMESH_CONTOURS)


def draw_edges(img, points, edges, color, thickness, min_conf=None, scale=1.0, offset=(0, 0)):
    for a, b in edges:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        if min_conf is not None and (pa[2] < min_conf or pb[2] < min_conf):
            continue
        cv2.line(img, (int(pa[0] * scale + offset[0]), int(pa[1] * scale + offset[1])),
                 (int(pb[0] * scale + offset[0]), int(pb[1] * scale + offset[1])), color, thickness, cv2.LINE_AA)


def speaker_color(speaker: Optional[str], colors: Dict[str, tuple]) -> tuple:
    if speaker and speaker.startswith("person_"):
        return colors.get(speaker.removeprefix("person_"), NEUTRAL)
    return NEUTRAL


def speaker_label(speaker: Optional[str]) -> str:
    if speaker and speaker.startswith("person_"):
        return "P" + speaker.removeprefix("person_")
    return "?" if speaker in (None, "NA") else "off"


def draw_3d(panel: np.ndarray, people3d: Dict[str, np.ndarray], order: List[str], colors, edges, angle: float,
            speaking: Optional[str]):
    h, w = panel.shape[:2]
    present = [p for p in order if p in people3d]
    if not present:
        cv2.putText(panel, "no 3D pose", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEUTRAL, 1, cv2.LINE_AA)
        return
    slot = w / len(present)
    scale = min(slot * 0.45, h * 0.38)  # pixels per meter
    cos, sin = math.cos(angle), math.sin(angle)
    ground = int(h * 0.82)
    cv2.line(panel, (10, ground), (w - 10, ground), (70, 70, 80), 1, cv2.LINE_AA)
    for k, person in enumerate(present):
        pts = people3d[person]
        cx, cy = slot * (k + 0.5), h * 0.45
        proj = np.full((len(pts), 3), np.nan, dtype=np.float32)
        visible = pts[:, 3] >= 0.5
        # Orthographic view after a rotation around the vertical axis (MediaPipe: y points down).
        proj[visible, 0] = cx + scale * (pts[visible, 0] * cos - pts[visible, 2] * sin)
        proj[visible, 1] = cy + scale * pts[visible, 1]
        proj[visible, 2] = 1.0
        thickness = 4 if speaking == person else 2
        draw_edges(panel, proj, edges, colors[person], thickness)
        cv2.putText(panel, f"P{person}", (int(cx) - 12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[person], 2,
                    cv2.LINE_AA)


def draw_subtitles(canvas: Image.Image, segments, t: float, colors, box, font) -> Optional[str]:
    """Draws the last two segments active at time t; returns the current speaker."""
    active = [s for s in segments if s["start"] <= t < s["end"]][-2:]
    if not active:
        return None
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas)
    line_h = font.size + 8
    chars = max(20, int((x1 - x0) / (font.size * 0.55)))
    lines = []
    for seg in active:
        label = speaker_label(seg["speaker"])
        for j, text in enumerate(textwrap.wrap(f"{label}: {seg['text'].strip()}", chars)[:3]):
            lines.append((text, speaker_color(seg["speaker"], colors)))
    lines = lines[-4:]
    y = y1 - line_h * len(lines) - 6
    draw.rectangle((x0, y - 6, x1, y1), fill=(0, 0, 0))
    for text, color in lines:
        draw.text((x0 + 10, y), text, font=font, fill=color)
        y += line_h
    return active[-1]["speaker"]


def ass_color(rgb) -> str:
    r, g, b = rgb
    return f"&H00{b:02X}{g:02X}{r:02X}"


def ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(path: Path, segments, colors, width: int, height: int):
    styles = {"NEUTRAL": NEUTRAL, **{f"P{p}": c for p, c in colors.items()}}
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "",
             "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, "
             "Outline, Shadow, Alignment, MarginL, MarginR, MarginV"]
    for name, color in styles.items():
        lines.append(f"Style: {name},DejaVu Sans,{max(18, height // 22)},{ass_color(color)},&H00000000,&H80000000,"
                     f"1,1,2,1,2,20,20,30")
    lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for seg in segments:
        label = speaker_label(seg["speaker"])
        style = label if label in styles else "NEUTRAL"
        text = seg["text"].strip().replace("\n", " ")
        lines.append(f"Dialogue: 0,{ass_time(seg['start'])},{ass_time(seg['end'])},{style},{label},0,0,0,,"
                     f"{label}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--body_dir", default="poses_body", help="COCO-17 archives drawn as the body")
    parser.add_argument("--min_conf", type=float, default=0.3, help="body keypoint confidence threshold")
    args = parser.parse_args()
    d = args.out_dir

    src = d / "input.mp4"
    subtitles_path = d / "transcription_enhanced.vtt"
    for path in (src, subtitles_path, d / args.body_dir):
        if not path.exists():
            sys.exit(f"error: missing {path}")

    with video.VideoReader(path=src) as reader:
        n_frames, fps = len(reader), float(Fraction(reader.frame_rate))  # "25/1" or "30000/1001"
        first = next(iter(reader))
    in_h, in_w = first.shape[:2]

    body = load_people(d / args.body_dir, n_frames)
    mp = load_people(d / "poses_mp", n_frames)
    order = sorted(set(body) | set(mp), key=int)
    if not order:
        sys.exit("error: no pose archive to render")
    colors = {p: PALETTE[k % len(PALETTE)] for k, p in enumerate(order)}
    segments = vtt.VTTReader.read(subtitles_path, verbose=False)
    write_ass(d / "subtitles.ass", segments, colors, in_w, in_h)

    main_w = int(round(HEIGHT * in_w / in_h / 2)) * 2
    scale = HEIGHT / in_h
    thumb_h = min(int(SIDE_WIDTH * in_h / in_w), HEIGHT // 2)
    thumb_w = int(thumb_h * in_w / in_h)
    width = main_w + SIDE_WIDTH
    font = ImageFont.truetype(FONT_PATH, 20 if main_w >= 640 else 15)
    face_edges = face_contour_edges()
    hand_edges = [tuple(e) for e in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                                     (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                                     (17, 18), (18, 19), (19, 20))]
    from psifx.video.pose.mediapipe.skeleton import POSE_EDGES

    tmp = d / "render_noaudio.mp4"
    with (
        video.VideoReader(path=src) as reader,
        video.VideoWriter(path=tmp, input_dict={"-r": str(fps)},
                          output_dict={"-c:v": "libx264", "-crf": "18", "-pix_fmt": "yuv420p"},
                          overwrite=True) as writer,
    ):
        for i, frame in enumerate(tqdm(reader, total=n_frames, desc="Rendering")):
            t = i / fps
            canvas = np.full((HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
            scene = canvas[:, :main_w]
            active = [s for s in segments if s["start"] <= t < s["end"]]
            speaking = active[-1]["speaker"].removeprefix("person_") if active and active[-1]["speaker"] else None

            for person in order:
                color = colors[person]
                if person in mp:
                    draw_edges(scene, mp[person]["face_keypoints_2d"][i], face_edges, color, 1, scale=scale)
                    for hand in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
                        draw_edges(scene, mp[person][hand][i], hand_edges, color, 1, scale=scale)
                if person in body:
                    kp = body[person]["pose_keypoints_2d"][i]
                    draw_edges(scene, kp, COCO_EDGES, color, 4 if person == speaking else 2, args.min_conf, scale)
                    head = kp[0]
                    if not np.isnan(head).any() and head[2] >= args.min_conf:
                        center = (int(head[0] * scale), int(head[1] * scale))
                        if person == speaking:
                            cv2.circle(scene, center, 28, color, 2, cv2.LINE_AA)
                        cv2.putText(scene, f"P{person}", (center[0] - 14, center[1] - 34), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2, cv2.LINE_AA)

            canvas[:thumb_h, main_w:main_w + thumb_w] = cv2.resize(frame, (thumb_w, thumb_h))
            people3d = {p: mp[p]["pose_keypoints_3d"][i] for p in order
                        if p in mp and "pose_keypoints_3d" in mp[p] and not np.isnan(mp[p]["pose_keypoints_3d"][i]).all()}
            draw_3d(canvas[thumb_h:, main_w:], people3d, order, colors, POSE_EDGES, angle=0.35 * t, speaking=speaking)

            image = Image.fromarray(canvas)
            draw_subtitles(image, segments, t, colors, (0, 0, main_w, HEIGHT), font)
            writer.write(np.asarray(image))

    # Put the original sound back.
    out = d / "render.mp4"
    result = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(src), "-map", "0:v", "-map",
                             "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)])
    if result.returncode != 0:
        sys.exit("error: ffmpeg could not mux the audio")
    tmp.unlink()
    print(f"wrote {out} and {d / 'subtitles.ass'}")


if __name__ == "__main__":
    main()
