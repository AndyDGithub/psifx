"""Which chips of the test split are people, and what SAM 3D Body gives on them, by crop height.

The filter (``keep``) uses ViTPose only, run on the whole chip: a chip is a person if the mean of its 17 scores is at
least ``--min_mean`` and both shoulders and both hips score above ``--min_torso``. It removes empty chips (road,
plant, white image) and partial ones (legs only); it does not see a chip with two people.
A tracklet is kept if at least ``--min_chips`` of its chips are.

Per height bin (height of the first chip of the tracklet), without and with the filter:
- ``chips`` and ``kept`` share;
- ``oks_vs_vitpose_p50`` and ``failed`` (OKS < 0.5), as in ``sam3d_small_crops.py``;
- ``accel_mm_p50``: 3D jitter around the pelvis, on triples of consecutive chips (all kept, with the filter);
- ``shape_within_between``: identity parameters, variance inside a tracklet over variance between tracklets.
Then what the privacy measure can use: kept tracklets, people, queries (``query_IDX.txt``) and gallery.

Writes ``<folder>/summary.json`` and ``<folder>/filter_sheet.jpg``: random chips sorted by ViTPose mean score, with
the score and the decision, to check the thresholds by eye.

    python research/mevid/summarize_test_split.py .e2e_outputs/ext/mevid/test_split \\
        --annotations .e2e_outputs/ext/mevid/annotations/x/mevid-v1-annotation-data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sam3d_body"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tracks import MHR70_TO_COCO17  # noqa: E402
from sam3d_small_crops import LEFT_HIP, RIGHT_HIP, oks  # noqa: E402

TORSO = [5, 6, 11, 12]  # COCO shoulders and hips.


def keep(vit_s: np.ndarray, min_mean: float, min_torso: float) -> np.ndarray:
    """(N, 17) ViTPose scores -> (N,) True where the chip shows a person."""
    return (vit_s.mean(1) >= min_mean) & (vit_s[:, TORSO].min(1) > min_torso)


def accelerations(k3: np.ndarray, kept: np.ndarray) -> list[float]:
    relative = k3 - k3[:, [LEFT_HIP, RIGHT_HIP]].mean(1, keepdims=True)
    values = np.linalg.norm(relative[2:] - 2 * relative[1:-1] + relative[:-2], axis=2).mean(1) * 1000
    return values[kept[2:] & kept[1:-1] & kept[:-2]].tolist()


def bin_summary(tracks: list[dict], filtered: bool, min_chips: int) -> dict:
    o, accel, shapes, means, chips, kept_chips = [], [], [], [], 0, 0
    for t in tracks:
        kept = t["kept"] if filtered else np.ones(len(t["oks"]), bool)
        chips += len(kept)
        kept_chips += int(kept.sum())
        o += t["oks"][kept].tolist()
        accel += accelerations(t["k3"], kept)
        if kept.sum() >= min_chips:  # Identity variance needs a few chips.
            shapes.append(t["shape"][kept].var(0))
            means.append(t["shape"][kept].mean(0))
    o = np.array(o)
    result = {"chips": chips, "kept_share": round(kept_chips / max(chips, 1), 3)}
    if len(o):
        result.update(oks_vs_vitpose_p50=round(float(np.nanmedian(o)), 3), failed=round(float(np.nanmean(o < 0.5)), 3))
    if accel:
        result["accel_mm_p50"] = round(float(np.median(accel)), 1)
    if len(means) > 1:
        within, between = np.mean(shapes, axis=0), np.var(np.stack(means), axis=0)
        result["shape_within_between"] = round(float(within.sum() / max(between.sum(), 1e-9)), 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json, chips/ and tracks/")
    parser.add_argument("--annotations", type=Path, required=True, help="folder with query_IDX.txt")
    parser.add_argument("--min_mean", type=float, default=0.5)
    parser.add_argument("--min_torso", type=float, default=0.3)
    parser.add_argument("--min_chips", type=int, default=8)
    parser.add_argument("--partial", action="store_true", help="summarize the tracklets done so far")
    args = parser.parse_args()

    for path in (args.folder / "sample.json", args.annotations / "query_IDX.txt"):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    sample = json.loads((args.folder / "sample.json").read_text())
    done = [t for t in sample if (args.folder / "tracks" / f"{t['track']}.npz").is_file()]
    if len(done) < len(sample) and not args.partial:
        sys.exit(f"error: {len(sample) - len(done)} tracklets not done (use --partial)")
    queries = set(np.loadtxt(args.annotations / "query_IDX.txt").astype(int).tolist())

    for t in done:
        a = np.load(args.folder / "tracks" / f"{t['track']}.npz")
        area = a["size"].prod(1).astype(float)
        t["oks"] = np.array([oks(k[MHR70_TO_COCO17], vk, vs, s) for k, vk, vs, s in
                             zip(a["keypoints_2d"], a["vit_k"], a["vit_s"], area)])
        t["k3"], t["shape"], t["vit_s"] = a["keypoints_3d"], a["shape_params"], a["vit_s"]
        t["kept"] = keep(a["vit_s"], args.min_mean, args.min_torso)

    report = {"tracklets": len(done), "filter": {"min_mean": args.min_mean, "min_torso": args.min_torso,
                                                 "min_chips": args.min_chips}, "bins": {}}
    for low, high in sorted({tuple(t["bin"]) for t in done}):
        tracks = [t for t in done if tuple(t["bin"]) == (low, high)]
        report["bins"][f"{low}-{high}"] = {"tracklets": len(tracks),
                                           "all": bin_summary(tracks, False, args.min_chips),
                                           "filtered": bin_summary(tracks, True, args.min_chips)}
    usable = [t for t in done if t["kept"].sum() >= args.min_chips]
    report["usable"] = {"tracklets": len(usable), "people": len({t["person"] for t in usable}),
                        "queries": sum(t["track"] in queries for t in usable),
                        "gallery": sum(t["track"] not in queries for t in usable),
                        "people_all": len({t["person"] for t in done}),
                        "queries_all": sum(t["track"] in queries for t in done)}
    (args.folder / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    # Filter sheet: 16 random chips per score range (low, around the threshold, high), sorted by mean score,
    # red border when dropped. Uniform drawing would show almost only good chips.
    rng = np.random.default_rng(0)
    pool = [(t, i, float(t["vit_s"][i].mean())) for t in done for i in range(len(t["kept"]))]
    picked = []
    for low, high in ((0, args.min_mean - 0.1), (args.min_mean - 0.1, args.min_mean + 0.1), (args.min_mean + 0.1, 1.1)):
        inside = [c for c in pool if low <= c[2] < high]
        picked += [inside[j] for j in rng.choice(len(inside), min(16, len(inside)), replace=False)]
    cells = []
    for t, i, _ in sorted(picked, key=lambda c: c[2]):
        name = t["frames"][i]
        image = cv2.imread(str(args.folder / "chips" / "bbox_test" / name[:4] / name))
        image = cv2.resize(image, (96, 192))
        cv2.putText(image, f"{t['vit_s'][i].mean():.2f}", (2, 186), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        color = (0, 200, 0) if t["kept"][i] else (0, 0, 255)
        cells.append(cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=color))
    cells += [np.zeros_like(cells[0])] * (-len(cells) % 12)
    sheet = np.vstack([np.hstack(cells[r:r + 12]) for r in range(0, len(cells), 12)])
    cv2.imwrite(str(args.folder / "filter_sheet.jpg"), sheet)
    print("SUMMARY_OK")


if __name__ == "__main__":
    main()
