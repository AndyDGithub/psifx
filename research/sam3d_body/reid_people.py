"""Who is who across shots: re-identify the pieces of people.json by appearance, under "two people seen together are
two different people".

people_from_tracks.py gives one person per SAM3 track. SAM3 keeps a track id across a cut by its own memory, and
it is wrong when the shot changes a lot: on V5, the man entering the push shot (frames 281-291) takes the girl's
track and the girl gets a new one, so the man is drawn in her color and she in a third one. Colors of the clothes
and the body shape of SAM 3D Body do not separate the dancers (both in black, the flash changes every color).

Each piece is described by DINOv2 features (self-supervised, robust across domains; Oquab et al. 2024) of

- the person crop: the largest part of its SAM3 mask on a gray background, squared;
- the head crop: a square around the head points of SAM 3D Body (nose, eyes, ears), when the head is inside the
  picture. Faces and hair change much less than clothes from shot to shot (long-term re-ID, Wan et al. 2024).

on up to ``--samples`` frames. The similarity of two pieces is the mean, over the frames of each, of the best
cosine match in the other (symmetric), averaged over the crops both have.

Pieces of at least ``--min_frames`` frames are grouped by average-linkage clustering with a cannot-link rule: two
groups never merge if they share a frame (a frame cannot show the same person twice). Groups merge while their
similarity is above ``--threshold`` (or down to ``--people`` groups if the number is known). Shorter pieces (a
dancer entering or leaving the frame, a few pixels on the border) are too small to judge: they take the person of
the adjacent piece of the same track in the same shot when it is free on their frames, otherwise the most similar
free group above the threshold, otherwise they are set aside as ``"other"`` (no avatar): a few frames like nobody
else are a poster or a passing picture more often than a new person (talk-show, frame 646: an actress's photo). Person numbers follow the input where possible (the group
holding most frames of person k gets number k), so colors stay the same for unchanged videos.

    python research/sam3d_body/reid_people.py .e2e_outputs/v5_danse_montage --output people_reid.json

Sources: Oquab et al., DINOv2 (TMLR 2024), arXiv 2304.07193; Wan et al., long-term person re-identification with a
head stream, arXiv 2403.02892; Person Re-ID in 2025: what works?, arXiv 2601.20598.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tracks import largest_part  # noqa: E402

HEAD = [0, 1, 2, 3, 4]  # MHR70: nose, eyes, ears


def read_frame(capture, f: int, path: Path) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, image = capture.read()
    if not ok:
        sys.exit(f"error: cannot read frame {f} of {path}")
    return image


def square(crop: np.ndarray) -> np.ndarray:
    """crop padded with gray to a square, centered."""
    s = max(crop.shape[:2])
    out = np.full((s, s, 3), 128, np.uint8)
    y, x = (s - crop.shape[0]) // 2, (s - crop.shape[1]) // 2
    out[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return out


class Embedder:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch, self.device = torch, device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        """(n, d) unit features of BGR crops."""
        if not crops:
            return np.zeros((0, 1), np.float32)
        with self.torch.no_grad():
            x = self.processor(images=[np.ascontiguousarray(c[..., ::-1]) for c in crops],
                               return_tensors="pt")["pixel_values"].to(self.device)
            e = self.model(pixel_values=x).pooler_output.float()
        return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def crops_of_piece(pipeline: Path, piece: dict, video, mask_video, keypoints: dict | None, samples: int,
                   min_area: int, head_score: float) -> tuple[list, list]:
    """(person crops, head crops) on up to ``samples`` frames of the piece."""
    first, last = piece["frames"]
    bodies, heads = [], []
    for f in np.unique(np.linspace(first, last, samples).astype(int)):
        image = read_frame(video, f, pipeline / "input.mp4")
        mask = largest_part(read_frame(mask_video, f, pipeline / "masks")[..., 0] > 127)
        if mask.sum() < min_area:
            continue
        ys, xs = np.nonzero(mask)
        bodies.append(square(np.where(mask[..., None], image, 128).astype(np.uint8)
                             [ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
        if keypoints is None or f not in keypoints:
            continue
        head = keypoints[f][HEAD]
        h, w = mask.shape
        size = 2.2 * max(np.ptp(head[:, 0]), np.ptp(head[:, 1]), 1.0)
        cx, cy = head.mean(0)
        x0, y0, x1, y1 = int(cx - size / 2), int(cy - size / 2), int(cx + size / 2), int(cy + size / 2)
        # Only heads fully in the picture, large enough, and on this person's mask (not another dancer's head).
        if size < head_score or x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        if not mask[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))]:
            continue
        heads.append(square(image[y0:y1, x0:x1]))
    return bodies, heads


def match(a: np.ndarray, b: np.ndarray) -> float | None:
    """Symmetric mean of best cosine matches, None if a side has no feature."""
    if len(a) == 0 or len(b) == 0:
        return None
    s = a @ b.T
    return float(0.5 * (s.max(1).mean() + s.max(0).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4, masks/ and people.json")
    parser.add_argument("--people_in", default="people.json", help="people.json to re-identify (inside the pipeline)")
    parser.add_argument("--params", default="sam3d_params", help="per-track SAM 3D Body archives (head points)")
    parser.add_argument("--output", default="people_reid.json", help="output, inside the pipeline")
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--samples", type=int, default=12, help="frames described per piece")
    parser.add_argument("--min_frames", type=int, default=5, help="shortest piece judged by its appearance")
    parser.add_argument("--min_area", type=int, default=2000, help="smallest mask (pixels) worth a crop")
    parser.add_argument("--min_head", type=float, default=24, help="smallest head crop (pixels)")
    parser.add_argument("--threshold", type=float, default=0.55, help="lowest similarity of one person's groups")
    parser.add_argument("--people", type=int, help="number of people, if known: merge down to it")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.pipeline
    source = d / args.people_in
    for path in [source, d / "input.mp4", d / "masks"]:
        if not path.exists():
            sys.exit(f"error: missing {path}")
    people = json.loads(source.read_text())
    # Pieces set aside before ("picture", "other": not a person of the scene) stay as they are.
    pieces = [p for p in people["assignment"] if isinstance(p["person"], int)]
    if not pieces:
        sys.exit(f"error: no piece in {source}")

    embed = Embedder(args.model, args.device)
    video = cv2.VideoCapture(str(d / "input.mp4"))
    masks, points = {}, {}
    features = []
    for piece in pieces:
        track = str(piece["track"])
        if track not in masks:
            masks[track] = cv2.VideoCapture(str(d / "masks" / f"{track}.mp4"))
            archive = d / args.params / f"{track}.npz"
            if archive.is_file():
                z = np.load(archive)
                points[track] = {int(f): k for f, k in zip(z["frames"], z["keypoints_2d"])}
        bodies, heads = crops_of_piece(d, piece, video, masks[track], points.get(track), args.samples,
                                       args.min_area, args.min_head)
        features.append((embed(bodies), embed(heads)))
    video.release()
    for capture in masks.values():
        capture.release()

    n = len(pieces)
    names = [f"t{p['track']}:{p['frames'][0]}-{p['frames'][1]}" for p in pieces]
    sim = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            parts = [m for m in (match(features[i][0], features[j][0]), match(features[i][1], features[j][1]))
                     if m is not None]
            if parts:
                sim[i, j] = float(np.mean(parts))
    span = [set(range(p["frames"][0], p["frames"][1] + 1)) for p in pieces]
    length = np.array([len(s) for s in span])
    judged = [i for i in range(n) if length[i] >= args.min_frames and len(features[i][0])]

    # Average-linkage clustering with cannot-link (shared frames).
    groups = [[i] for i in judged]

    def linkage(a, b):
        if any(span[i] & span[j] for i in a for j in b):
            return -np.inf
        s = [sim[i, j] for i in a for j in b if np.isfinite(sim[i, j])]
        return float(np.mean(s)) if s else -np.inf

    while len(groups) > 1:
        best, pair = -np.inf, None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                s = linkage(groups[x], groups[y])
                if s > best:
                    best, pair = s, (x, y)
        enough = args.people is not None and len(groups) <= args.people
        if pair is None or not np.isfinite(best) or enough or (args.people is None and best < args.threshold):
            break
        print(f"merge {[names[i] for i in groups[pair[0]]]} + {[names[i] for i in groups[pair[1]]]}: {best:.2f}")
        groups[pair[0]] += groups.pop(pair[1])

    group_of = {i: g for g, members in enumerate(groups) for i in members}
    # Short pieces, longest first.
    for i in sorted(set(range(n)) - set(group_of), key=lambda i: -length[i]):
        def free(g):
            return not any(span[i] & span[j] for j, h in group_of.items() if h == g)
        p = pieces[i]
        neighbours = [j for j in group_of if pieces[j]["track"] == p["track"] and pieces[j]["shot"] == p["shot"]
                      and min(abs(pieces[j]["frames"][0] - p["frames"][1]), abs(p["frames"][0] - pieces[j]["frames"][1])) <= 2]
        choice = next((group_of[j] for j in neighbours if free(group_of[j])), None)
        how = "next to its own track"
        if choice is None:
            scores = [(linkage([i], groups[g]) if free(g) else -np.inf, g) for g in range(len(groups))]
            s, g = max(scores) if scores else (-np.inf, None)
            if g is not None and s >= args.threshold:
                choice, how = g, f"looks like it ({s:.2f})"
        if choice is None:
            print(f"short piece {names[i]}: like nobody, set aside (other)")
            continue
        groups[choice].append(i)
        group_of[i] = choice
        print(f"short piece {names[i]} -> group {choice}: {how}")

    # Number the groups after the people of the input they hold most frames of (Hungarian on frame counts), so
    # colors stay the same for videos already right.
    from scipy.optimize import linear_sum_assignment

    before = sorted({p["person"] for p in pieces})
    count = np.zeros((len(groups), len(before)))
    for i, g in group_of.items():
        count[g, before.index(pieces[i]["person"])] += length[i]
    rows, cols = linear_sum_assignment(-count)
    number = {int(g): int(before[c]) for g, c in zip(rows, cols) if count[g, c] > 0}
    spare = max(before) + 1
    for g in range(len(groups)):
        if g not in number:
            number[g], spare = spare, spare + 1

    changed = 0
    for i, p in enumerate(pieces):
        new = number[group_of[i]] if i in group_of else "other"
        changed += new != p["person"]
        p["person_track"] = p["person"]
        p["person"] = new
    # A frame cannot show one person twice (cannot-link kept the groups apart; check it anyway).
    seen = {}
    for p in pieces:
        if not isinstance(p["person"], int):
            continue
        for f in range(p["frames"][0], p["frames"][1] + 1):
            if (f, p["person"]) in seen:
                sys.exit(f"error: person {p['person']} twice on frame {f} ({seen[f, p['person']]} and t{p['track']})")
            seen[f, p["person"]] = f"t{p['track']}"
    people["method"] += f"; re-identified across shots by {args.model} person and head features, cannot-link on " \
                        f"shared frames (reid_people.py)"
    people["people"] = len({p["person"] for p in pieces if isinstance(p["person"], int)})
    people["reid"] = {"model": args.model, "threshold": args.threshold, "people": args.people,
                      "similarity": {names[i]: {names[j]: (None if not np.isfinite(sim[i, j]) else round(float(sim[i, j]), 3))
                                                for j in range(n)} for i in range(n)}}
    out = d / args.output
    out.write_text(json.dumps(people, indent=1))
    for i, p in enumerate(pieces):
        print(f"{names[i]:>16}  track person {p['person_track']} -> person {p['person']}")
    print(f"{changed} pieces changed person, {people['people']} people; wrote {out}")
    print("REID_OK")


if __name__ == "__main__":
    main()
