"""One avatar per person: the SAM3 tracks cut by shot, then the pieces grouped into people by body shape.

On an edited video, a SAM3 track is not a person: its number can pass from one person to another between shots, and
people of an inserted video (split screen) get tracks too. So:
1. each track is cut at the shot cuts into pieces (track, shot);
2. a piece is described by its median identity parameters and limb scales (``shape_params`` and the last 68
   ``mhr_model_params``), standardized over the pieces;
3. ``--people`` main people are found by k-means on the pieces, weighted by their length (the hosts are on screen
   most of the time);
4. inside each shot, pieces are assigned to the main people by the Hungarian method on the shape distance: two
   pieces of one shot are two people. A piece farther than ``--max_distance`` from every main person, or left over,
   is ``other`` (an inserted video, a passer-by): it gets no avatar.
This is ``--method pieces``. ``--method tracks`` trusts SAM3 more: whole tracks are merged, closest median shapes
first, until ``--people`` remain; then, in a shot where two pieces of one person are visible together, the piece
closest to that person's median shape keeps it and the other becomes ``other``.
``--method appearance`` (default) does the same with the colors of the person instead of the body shape: an HSV
histogram of the pixels inside the SAM3 mask, on 5 frames of each piece (Hellinger distance). A piece whose colors
are farther than ``--max_color`` from its person's is also ``other``. The colors only serve to build the avatars:
they are never shared.

Evaluation (``--labels``): each annotated person of the talk-show annotations is matched to the piece whose 2D
keypoints box overlaps its box most, then compared with its label (man, woman, incrustation). Reported: the share of
labeled people correctly attributed, with the raw track numbers as baseline (each track mapped to its best label).

Writes ``<pipeline>/people.json`` and the archives of each main person in ``<pipeline>/<output>/`` (same format,
for ``render_avatars.py --params``).

    python research/scene3d/merge_people.py .e2e_outputs/extrait_talkshow/pipeline --people 2 \\
        --labels .e2e_outputs/extrait_talkshow/identity_labels/labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sam3d_body"))
from run_keyframes import cut_labels  # noqa: E402

POSE = 136  # mhr_model_params: 136 pose values, then 68 scale values.


def kmeans(x: np.ndarray, weight: np.ndarray, k: int, rng, rounds: int = 20) -> np.ndarray:
    """Weighted k-means, best of ``rounds`` starts (lowest weighted inertia)."""
    best, best_inertia = None, np.inf
    for _ in range(rounds):
        centers = x[rng.choice(len(x), k, replace=False, p=weight / weight.sum())]
        for _ in range(50):
            assign = np.linalg.norm(x[:, None] - centers[None], axis=2).argmin(1)
            new = np.stack([np.average(x[assign == c], axis=0, weights=weight[assign == c]) if (assign == c).any()
                            else centers[c] for c in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        inertia = (weight * np.linalg.norm(x - centers[assign], axis=1) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = centers, inertia
    return best


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4 and the archives")
    parser.add_argument("--params", default="sam3d_params")
    parser.add_argument("--people", type=int, required=True, help="number of main people (e.g. the speakers)")
    parser.add_argument("--max_distance", type=float, default=None,
                        help="farther from every main person: other (default: the 95th percentile of the main pieces)")
    parser.add_argument("--labels", type=Path, help="labels.json of the annotated people")
    parser.add_argument("--annotations", type=Path, help="folder with frames.csv (default: next to labels)")
    parser.add_argument("--method", choices=("appearance", "tracks", "pieces"), default="appearance")
    parser.add_argument("--max_color", type=float, default=0.5, help="Hellinger distance to the person's colors")
    parser.add_argument("--output", default="sam3d_people")
    parser.add_argument("--cut", type=float, default=18)
    args = parser.parse_args()

    folder, video = args.pipeline / args.params, args.pipeline / "input.mp4"
    for path in (folder, video):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    archives = {}
    for path in sorted(folder.glob("*.npz"), key=lambda p: int(p.stem)):
        with np.load(path) as npz:
            archives[path.stem] = {k: npz[k] for k in npz.files}
    labels = cut_labels(video, args.cut)

    pieces = []  # (track, shot, rows)
    for track, a in archives.items():
        shots = labels[np.clip(a["frames"].astype(int), 0, len(labels) - 1)]
        for shot in np.unique(shots):
            pieces.append((track, int(shot), np.where(shots == shot)[0]))
    features = np.stack([np.r_[np.median(archives[t]["shape_params"][r], 0),
                               np.median(archives[t]["mhr_model_params"][r][:, POSE:], 0)] for t, _, r in pieces])
    features = (features - features.mean(0)) / (features.std(0) + 1e-9)
    length = np.array([len(r) for *_, r in pieces], float)
    rng = np.random.default_rng(0)
    if args.method == "appearance":
        colors = color_features(args.pipeline, pieces, archives)
        person, limit = by_tracks(archives, pieces, colors, length, args.people, hellinger, args.max_color)
    elif args.method == "tracks":
        person, limit = by_tracks(archives, pieces, features, length, args.people)
    else:
        person, limit = by_pieces(pieces, features, length, args.people, args.max_distance, rng)
    print(f"{len(pieces)} pieces in {len(set(s for _, s, _ in pieces))} shots; "
          f"main people get {length[person >= 0].sum() / length.sum():.1%} of the person-frames", flush=True)
    report = {"method": args.method, "pieces": len(pieces), "people": args.people, "max_distance": limit,
              "kept_share": round(float(length[person >= 0].sum() / length.sum()), 4)}
    if args.method != "pieces":
        report["track_to_person"] = {t: int(person[[i for i, p in enumerate(pieces) if p[0] == t]].max())
                                     for t in archives}
    _evaluate_and_write(args, archives, pieces, person, report)


def by_pieces(pieces, features, length, people, max_distance, rng):
    centers = kmeans(features, length, people, rng)
    distance = np.linalg.norm(features[:, None] - centers[None], axis=2)
    nearest = distance.min(1)
    # Default limit: 95 % of the person-frames are at most this far from their nearest main person.
    limit = max_distance or float(np.percentile(np.repeat(nearest, length.astype(int)), 95))
    person = np.full(len(pieces), -1)
    for shot in sorted({s for _, s, _ in pieces}):
        index = [i for i, (_, s, _) in enumerate(pieces) if s == shot]
        rows, cols = linear_sum_assignment(distance[index])
        for r, c in zip(rows, cols):
            if distance[index[r], c] <= limit:
                person[index[r]] = c
    return person, round(limit, 3)


def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1 - np.sqrt(a * b).sum())))


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def color_features(pipeline: Path, pieces, archives, samples: int = 5) -> np.ndarray:
    """Normalized HSV histogram (8 x 4 x 4) of the masked pixels, averaged over a few frames of each piece."""
    import cv2
    out = []
    videos = {"input": cv2.VideoCapture(str(pipeline / "input.mp4"))}
    for track, _, rows in pieces:
        if track not in videos:
            path = pipeline / "masks" / f"{track}.mp4"
            if not path.is_file():
                sys.exit(f"error: missing {path}")
            videos[track] = cv2.VideoCapture(str(path))
        hist = np.zeros(8 * 4 * 4)
        for r in rows[np.linspace(0, len(rows) - 1, min(samples, len(rows))).astype(int)]:
            frame = int(archives[track]["frames"][r])
            images = []
            for v in (videos["input"], videos[track]):
                v.set(cv2.CAP_PROP_POS_FRAMES, frame)  # Exact seeks on these files (checked earlier).
                ok, image = v.read()
                if not ok:
                    sys.exit(f"error: cannot read frame {frame} of track {track}")
                images.append(image)
            mask = (images[1].max(axis=2) > 127).astype(np.uint8)
            hsv = cv2.cvtColor(images[0], cv2.COLOR_BGR2HSV)
            hist += cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        out.append(hist / max(hist.sum(), 1))
    return np.stack(out)


def by_tracks(archives, pieces, features, length, people, distance=euclid, max_distance=None):
    """Merge whole tracks by their median features, then one piece per person and shot."""
    tracks = list(archives)
    group = {t: i for i, t in enumerate(tracks)}
    median = {t: np.average(features[[i for i, p in enumerate(pieces) if p[0] == t]], axis=0,
                            weights=length[[i for i, p in enumerate(pieces) if p[0] == t]]) for t in tracks}
    while len(set(group.values())) > people:
        ids = sorted(set(group.values()))
        centers = {g: np.mean([median[t] for t in tracks if group[t] == g], axis=0) for g in ids}
        a, b = min(((x, y) for x in ids for y in ids if x < y), key=lambda xy: distance(centers[xy[0]], centers[xy[1]]))
        for t in tracks:
            if group[t] == b:
                group[t] = a
    names = {g: k for k, g in enumerate(sorted(set(group.values())))}
    person = np.array([names[group[t]] for t, _, _ in pieces])
    centers = {k: np.average(features[person == k], axis=0, weights=length[person == k]) for k in range(people)}
    for shot in {s for _, s, _ in pieces}:
        for k in range(people):
            index = [i for i, (_, s, _) in enumerate(pieces) if s == shot and person[i] == k]
            if len(index) > 1:  # Two pieces of one person in one shot: the farther one is someone else.
                keep = min(index, key=lambda i: distance(features[i], centers[k]))
                for i in index:
                    if i != keep:
                        person[i] = -1
    if max_distance is not None:  # Someone else (an inserted video, a passer-by) inside a person's track.
        for i in range(len(pieces)):
            if person[i] >= 0 and distance(features[i], centers[person[i]]) > max_distance:
                person[i] = -1
    return person, max_distance


def _evaluate_and_write(args, archives, pieces, person, report):
    if args.labels:
        data = json.loads(args.labels.read_text())
        notes = args.annotations or args.labels.parent.parent / "pose_annotation"
        frame_of = {}
        for line in (notes / "frames.csv").read_text().splitlines()[1:]:
            name, frame, _ = line.split(",")
            frame_of[name] = int(frame)
        piece_of = {(t, int(r)): i for i, (t, _, rows) in enumerate(pieces) for r in rows}
        matched = []  # (label, raw track, merged person)
        for e in data["annotations"]:
            if e["person"] == "unknown":
                continue
            x, y, w, h = e["bbox"]
            box, best = (x, y, x + w, y + h), (0.3, None)
            for track, a in archives.items():
                position = {int(f): i for i, f in enumerate(a["frames"])}
                f = frame_of[e["image"]]
                i = position.get(f, position.get(f - 1))
                if i is None:
                    continue
                k = a["keypoints_2d"][i]
                score = iou(box, (*k.min(0), *k.max(0)))
                if score > best[0]:
                    best = (score, (track, i))
            if best[1] is None:
                matched.append((e["person"], None, None))
                continue
            p = person[piece_of[best[1]]]
            matched.append((e["person"], best[1][0], "other" if p < 0 else int(p)))
        found = [m for m in matched if m[1] is not None]

        def accuracy(column: int) -> float:
            """Each predicted id takes its majority label (``other`` stands for incrustation): the share right."""
            mapping = {i: Counter(m[0] for m in found if m[column] == i).most_common(1)[0][0]
                       for i in {m[column] for m in found}}
            mapping["other"] = "incrustation"
            return float(np.mean([mapping[m[column]] == m[0] for m in found]))

        report["labels"] = {"labeled": len(matched), "matched_to_a_mesh": len(found),
                            "by_label": dict(Counter(m[0] for m in matched)),
                            "raw_tracks_accuracy": round(accuracy(1), 3),
                            "merged_accuracy": round(accuracy(2), 3),
                            "raw_track_labels": {t: dict(Counter(m[0] for m in found if m[1] == t))
                                                 for t in sorted({m[1] for m in found})},
                            "merged_labels": {str(p): dict(Counter(m[0] for m in found if m[2] == p))
                                              for p in {m[2] for m in found}}}
        print(json.dumps(report["labels"]), flush=True)

    out = args.pipeline / args.output
    out.mkdir(exist_ok=True)
    for p in range(args.people):
        chosen = [(pieces[i][0], pieces[i][2]) for i in np.where(person == p)[0]]
        merged = {k: np.concatenate([archives[t][k][r] for t, r in chosen]) for k in next(iter(archives.values())) if k != "faces"}
        order = np.argsort(merged["frames"], kind="stable")
        merged = {k: v[order] for k, v in merged.items()}
        merged["faces"] = archives[next(iter(archives))]["faces"]
        np.savez_compressed(out / f"{p}.npz", **merged)
    report["assignment"] = [{"track": t, "shot": s, "frames": [int(archives[t]["frames"][r[0]]), int(archives[t]["frames"][r[-1]])],
                             "person": int(p) if p >= 0 else "other"} for (t, s, r), p in zip(pieces, person)]
    (args.pipeline / "people.json").write_text(json.dumps(report, indent=1))
    print("MERGE_OK")


if __name__ == "__main__":
    main()
