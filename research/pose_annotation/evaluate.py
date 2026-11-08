"""Measures pose models against the hand-corrected keypoints exported by ``fetch_annotations.py``.

Predictions come from two sources, compared on the same people:

- ``--models``: a psifx pose model run on each annotated image, every person of the frame
  (``rtmo:balanced``, ``vitpose:usyd-community/vitpose-base-simple``, ...). Measures the model alone.
- ``--archives NAME=DIR``: the per-person ``.tar.gz`` archives of a psifx ``multi-inference`` run on
  the whole video (SAM3 masks, then RTMO, ViTPose or MediaPipe), read at the annotated frames
  (``frames.csv``). Measures what a user gets, tracking and temporal smoothing included.

``--preannotation`` adds the uncorrected pre-annotation as a model: its distance to the ground truth
is the amount of correction, and a high score means the reference stays close to ViTPose++.

Per image, predicted people are matched one to one to the annotated ones, maximizing the OKS.
Per model:

- ``oks``: mean COCO OKS over the annotated people, 0 for a person without a match. The scale is the
  area of the box of the labeled points (the COCO segmentation area is not available);
- ``found``: share of annotated people matched with an OKS of at least 0.5;
- ``pck@t``: share of labeled points predicted (score >= ``--min_score``) within ``t`` times the box
  diagonal, apart for visible and occluded points, and per keypoint;
- ``phantom``: share of the points annotated absent (out of frame, under the table) that the model
  predicts with a score >= ``--min_score``: the points drawn where there is nothing;
- ``absent_auc`` and ``phantom@keep95``: the same question without a fixed threshold, see
  ``score_separation``;
- ``extra_people``: predicted people left without a match. The small and blurry people are not
  annotated on purpose, so it is not an error count.

    python research/pose_annotation/evaluate.py .e2e_outputs/extrait_talkshow/pose_annotation \\
        --preannotation --models rtmo:balanced rtmo:performance vitpose:usyd-community/vitpose-base-simple
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.pose_eval.coco import COCO_KEYPOINT_NAMES, COCO_OKS_SIGMAS, MEDIAPIPE_TO_COCO17  # noqa: E402

N = len(COCO_KEYPOINT_NAMES)
PCK_THRESHOLDS = (0.05, 0.1)


def load_ground_truth(path: Path) -> dict[str, list[dict]]:
    """``{file_name: [{"xy" (17, 2), "v" (17,), "area", "diagonal"}]}``, people with a labeled point."""
    coco = json.loads(path.read_text())
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    people = {name: [] for name in names.values()}
    for a in coco["annotations"]:
        k = np.asarray(a["keypoints"], dtype=np.float64).reshape(N, 3)
        labeled = k[:, 2] > 0
        if not labeled.any():
            continue
        w, h = np.ptp(k[labeled, :2], axis=0)
        people[names[a["image_id"]]].append(
            {"xy": k[:, :2], "v": k[:, 2].astype(int), "area": float(w * h), "diagonal": float(np.hypot(w, h))})
    return people


def oks(pred: np.ndarray, gt: dict) -> float:
    """COCO OKS of a (17, 3) prediction, over the labeled points of the ground truth."""
    labeled = gt["v"] > 0
    d2 = np.sum((pred[:, :2] - gt["xy"]) ** 2, axis=1)
    e = d2 / ((2 * COCO_OKS_SIGMAS) ** 2 * (gt["area"] + np.spacing(1)) * 2)
    e = np.where(np.isnan(e), np.inf, e)  # an undetected point counts as a miss
    return float(np.mean(np.exp(-e[labeled])))


def match(preds: list[np.ndarray], gts: list[dict]) -> list[tuple[int, int, float]]:
    """``(gt index, pred index, oks)`` pairs maximizing the total OKS, pairs at 0 left out."""
    if not preds or not gts:
        return []
    scores = np.array([[oks(p, g) for p in preds] for g in gts])
    rows, cols = linear_sum_assignment(-scores)
    return [(r, c, scores[r, c]) for r, c in zip(rows, cols) if scores[r, c] > 0]


def score_separation(labeled: np.ndarray, absent: np.ndarray) -> dict:
    """Measures that do not depend on the scale of a model's scores (SimCC scores of RTMPose and RTMW are
    not in [0, 1], so a fixed ``--min_score`` means something else for each model):

    - ``absent_auc``: probability that a labeled point scores higher than a point annotated absent
      (ROC AUC, Mann-Whitney); 0.5 means the score cannot tell them apart;
    - ``phantom@keep95``: share of absent points kept at the threshold that keeps 95 % of the labeled ones.
    """
    if not len(labeled) or not len(absent):
        return {"absent_auc": None, "phantom@keep95": None}
    ranks = rankdata(np.concatenate([labeled, absent]))
    auc = (ranks[:len(labeled)].sum() - len(labeled) * (len(labeled) + 1) / 2) / (len(labeled) * len(absent))
    threshold = np.quantile(labeled, 0.05)
    return {"absent_auc": round(float(auc), 3), "phantom@keep95": round(float(np.mean(absent >= threshold)), 3)}


def evaluate(predictions: dict[str, list[np.ndarray]], truth: dict[str, list[dict]], min_score: float) -> dict:
    oks_all, found, extra = [], 0, 0
    # Per keypoint and visibility (1 occluded, 2 visible): correct and total counts per threshold.
    correct = {t: np.zeros((3, N)) for t in PCK_THRESHOLDS}
    total = np.zeros((3, N))
    phantom, absent = 0, 0
    # Scores of the matched people's points, labeled or annotated absent: the threshold-free measures.
    labeled_scores, absent_scores = [], []
    for name, gts in truth.items():
        preds = predictions.get(name, [])
        pairs = match(preds, gts)
        matched = {g: (p, s) for g, p, s in pairs}
        extra += len(preds) - len(pairs)
        for g, gt in enumerate(gts):
            if g not in matched:
                oks_all.append(0.0)
                total[gt["v"], np.arange(N)] += gt["v"] > 0
                continue
            p, s = matched[g]
            oks_all.append(s)
            found += s >= 0.5
            pred = preds[p]
            predicted = ~np.isnan(pred[:, 0]) & (np.nan_to_num(pred[:, 2]) >= min_score)
            dist = np.linalg.norm(pred[:, :2] - gt["xy"], axis=1)
            score = np.where(np.isnan(pred[:, 0]), -np.inf, np.nan_to_num(pred[:, 2], nan=-np.inf))
            labeled_scores += list(score[gt["v"] > 0])
            absent_scores += list(score[gt["v"] == 0])
            for k in range(N):
                if gt["v"][k] == 0:
                    absent += 1
                    phantom += predicted[k]
                    continue
                total[gt["v"][k], k] += 1
                for t in PCK_THRESHOLDS:
                    correct[t][gt["v"][k], k] += predicted[k] and dist[k] <= t * gt["diagonal"]
    n_people = len(oks_all)
    result = {"people": n_people, "oks": round(float(np.mean(oks_all)), 3), "found": round(found / n_people, 3),
              "phantom": round(phantom / absent, 3) if absent else None, "extra_people": extra}
    for t in PCK_THRESHOLDS:
        result[f"pck@{t}"] = round(float(correct[t][1:].sum() / total[1:].sum()), 3)
        for v, label in ((2, "visible"), (1, "occluded")):
            result[f"pck@{t}_{label}"] = round(float(correct[t][v].sum() / total[v].sum()), 3) if total[v].sum() else None
    result.update(score_separation(np.array(labeled_scores), np.array(absent_scores)))
    t = PCK_THRESHOLDS[-1]
    result[f"pck@{t}_per_keypoint"] = {
        COCO_KEYPOINT_NAMES[k]: round(float(correct[t][1:, k].sum() / total[1:, k].sum()), 3)
        for k in range(N) if total[1:, k].sum()}
    return result


def from_preannotation(path: Path) -> dict[str, list[np.ndarray]]:
    coco = json.loads(path.read_text())
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    out = {}
    for a in coco["annotations"]:
        k = np.asarray(a["keypoints"], dtype=np.float64).reshape(N, 3)
        # v=0 points were left out by prepare.py: no prediction there.
        k[k[:, 2] == 0, :2] = np.nan
        k[:, 2] = (k[:, 2] > 0).astype(float)
        out.setdefault(names[a["image_id"]], []).append(k)
    return out


def from_model(spec: str, images: list[Path], device: str) -> dict[str, list[np.ndarray]]:
    """Runs ``family:option`` on each image, every person kept."""
    import cv2
    # Before RTMO: the CUDA libraries it preloads for onnxruntime break a later torch import.
    import torch  # noqa: F401
    from tqdm import tqdm

    family, _, option = spec.partition(":")
    if family == "rtmo":
        from psifx.video.pose.rtmo.tool import RTMOPoseEstimationTool
        tool = RTMOPoseEstimationTool(mode=option or "balanced", device=device, verbose=False)
    elif family == "vitpose":
        from psifx.video.pose.vitpose.tool import ViTPosePoseEstimationTool
        tool = ViTPosePoseEstimationTool(model_name=option or "usyd-community/vitpose-base-simple", device=device,
                                         verbose=False)
    else:
        sys.exit(f"error: unknown model family {family!r} in {spec!r} (rtmo or vitpose)")
    out = {}
    for path in tqdm(images, desc=spec):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        out[path.name] = [np.concatenate([np.asarray(kp, float), np.asarray(sc, float)[:, None]], axis=1)
                          for kp, sc, _ in tool.predict(rgb)]
    return out


def from_archives(directory: Path, frames: dict[int, str]) -> dict[str, list[np.ndarray]]:
    """One person per ``.tar.gz`` of a multi-inference run, read at the annotated frames."""
    from psifx.io.tar import TarReader

    archives = sorted(directory.glob("*.tar.gz"))
    if not archives:
        sys.exit(f"error: no .tar.gz archive in {directory}")
    out = {name: [] for name in frames.values()}
    for archive in archives:
        files = TarReader.read(archive, verbose=False)
        for frame, name in frames.items():
            content = files.get(f"{frame: 015d}.json")
            if content is None:
                sys.exit(f"error: {archive} has no frame {frame}, is it a run on the same video?")
            points = np.asarray(json.loads(content)["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
            if not np.any(points):
                continue  # psifx writes zeros when the person is not found (or masked out)
            if len(points) == N:
                out[name].append(points)
            else:
                out[name].append(np.array([points[MEDIAPIPE_TO_COCO17[k]] for k in COCO_KEYPOINT_NAMES]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder of prepare.py and fetch_annotations.py")
    parser.add_argument("--models", nargs="*", default=[], help="family:option, run on the images")
    parser.add_argument("--archives", nargs="*", default=[], help="NAME=DIR of a multi-inference run")
    parser.add_argument("--preannotation", action="store_true", help="also score the uncorrected pre-annotation")
    parser.add_argument("--min_score", type=float, default=0.3, help="a point under it counts as not predicted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None, help="default: out_dir/pose_eval.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gt_path = args.out_dir / "ground_truth.json"
    pre_path = args.out_dir / "annotations" / "person_keypoints_default.json"
    frames_path = args.out_dir / "frames.csv"
    output = args.output or args.out_dir / "pose_eval.json"
    for path in (gt_path, frames_path) + ((pre_path,) if args.preannotation else ()):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    if output.exists() and not args.overwrite:
        sys.exit(f"error: {output} exists, pass --overwrite to replace it")
    archives = {}
    for spec in args.archives:
        name, sep, directory = spec.partition("=")
        if not sep or not Path(directory).is_dir():
            sys.exit(f"error: --archives expects NAME=DIR with an existing DIR, got {spec!r}")
        archives[name] = Path(directory)
    if not (args.models or archives or args.preannotation):
        sys.exit("error: nothing to evaluate, give --models, --archives or --preannotation")

    truth = load_ground_truth(gt_path)
    with open(frames_path, newline="") as f:
        frames = {int(row["frame"]): row["file"] for row in csv.DictReader(f)}
    if set(frames.values()) != set(truth):
        sys.exit(f"error: the images of {frames_path} and {gt_path} differ")
    images = [args.out_dir / "images" / name for name in frames.values()]
    if missing := [p for p in images if not p.is_file()]:
        sys.exit(f"error: {len(missing)} images missing, first {missing[0]}")

    sources = {}
    if args.preannotation:
        sources["preannotation"] = lambda: from_preannotation(pre_path)
    for spec in args.models:
        sources[spec] = lambda spec=spec: from_model(spec, images, args.device)
    for name, directory in archives.items():
        sources[name] = lambda directory=directory: from_archives(directory, frames)

    results = {"ground_truth": {"images": len(truth), "people": sum(map(len, truth.values())),
                                "min_score": args.min_score}}
    for name, predict in sources.items():
        results[name] = evaluate(predict(), truth, args.min_score)
        r = results[name]
        print(f"{name:45s} OKS {r['oks']:.3f}  found {r['found']:.3f}  PCK@0.1 {r['pck@0.1']:.3f} "
              f"(occluded {r['pck@0.1_occluded']})  phantom {r['phantom']}  extra {r['extra_people']}")
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
