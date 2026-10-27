"""MediaPipe accuracy on COCO val2017 images with a single annotated person, through the psifx CLI.

``prepare`` picks ``--n`` such images (seeded) from the annotation zip and downloads them. ``run``
letterboxes them into one video, runs ``psifx video pose mediapipe single-inference`` with the
psifx installed in the current environment and scores the archive (COCO OKS, PCK@0.2) with the
same code as the integration test (tests/pose_eval/psifx_eval.py). Run it once per environment
to compare the legacy Solutions API with the Tasks API on the same images.

    python research/mediapipe_tasks/coco_accuracy.py prepare research/benchmark/data/annotations_trainval2017.zip out/coco500
    python research/mediapipe_tasks/coco_accuracy.py run out/coco500 --tag tasks
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def prepare(args):
    with zipfile.ZipFile(args.zip) as zf, zf.open("annotations/person_keypoints_val2017.json") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    # Exactly one annotated person (any keypoint count), with enough labeled keypoints to score.
    ids = sorted(i for i, anns in by_image.items() if len(anns) == 1 and anns[0].get("num_keypoints", 0) >= 5)
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        name = images[image_id]["file_name"]
        dest = args.out / "images" / name
        if not dest.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{name}", dest)
    subset = {"images": [images[i] for i in ids], "annotations": [by_image[i][0] for i in ids]}
    (args.out / "annotations.json").write_text(json.dumps(subset))
    print(f"{len(ids)} images in {args.out}")


def run(args):
    from tests.pose_eval.psifx_eval import build_coco_video, evaluate_archive, load_coco_samples

    samples = load_coco_samples(args.data / "annotations.json", args.data / "images")
    video_path = args.data / "coco.mp4"
    ground_truth = build_coco_video(samples, video_path)
    poses = args.data / f"mediapipe_{args.tag}.tar.gz"
    start = time.perf_counter()
    subprocess.run(["psifx", "video", "pose", "mediapipe", "single-inference", "--video", str(video_path),
                    "--poses", str(poses), "--overwrite", "--no-verbose"], check=True)
    scores = evaluate_archive(poses, ground_truth, "mediapipe")
    scores["seconds"] = round(time.perf_counter() - start, 1)
    scores["images"] = len(samples)
    (args.data / f"scores_{args.tag}.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("zip", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("run")
    p.add_argument("data", type=Path)
    p.add_argument("--tag", required=True)
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else run(args)


if __name__ == "__main__":
    main()
