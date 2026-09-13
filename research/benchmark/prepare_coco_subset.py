"""Download a small, fixed, seeded subset of COCO val2017 person-keypoints
annotations + only the corresponding images (not the full 1GB val2017 set).

Usage: python prepare_coco_subset.py [--n 50] [--seed 0]
Writes: data/coco_subset/annotations.json, data/coco_subset/images/*.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"
ANNOTATION_MEMBER = "annotations/person_keypoints_val2017.json"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "coco_subset"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of images to keep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-keypoints", type=int, default=5,
        help="skip GT persons with fewer than this many labeled (visible) keypoints",
    )
    args = parser.parse_args()

    zip_path = HERE / "data" / "annotations_trainval2017.zip"
    download(ANNOTATIONS_ZIP_URL, zip_path)

    print("Extracting person_keypoints_val2017.json ...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ANNOTATION_MEMBER) as f:
            coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann.get("num_keypoints", 0) < args.min_keypoints:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only keep images with exactly one qualifying annotated person: this
    # benchmark's match_prediction_to_gt (metrics.py) is a nearest-prediction
    # heuristic, not full COCO multi-person assignment, so ambiguous
    # multi-person images would silently bias the accuracy numbers.
    single_person_image_ids = [
        image_id for image_id, anns in anns_by_image.items() if len(anns) == 1
    ]

    random.Random(args.seed).shuffle(single_person_image_ids)
    chosen_ids = single_person_image_ids[: args.n]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    subset = {"images": [], "annotations": []}
    for image_id in chosen_ids:
        img = images_by_id[image_id]
        dest = images_dir / img["file_name"]
        download(IMAGE_BASE_URL + img["file_name"], dest)
        subset["images"].append(img)
        subset["annotations"].extend(anns_by_image[image_id])

    out_path = DATA_DIR / "annotations.json"
    out_path.write_text(json.dumps(subset))
    print(f"Wrote {len(subset['images'])} images / {len(subset['annotations'])} annotations to {out_path}")


if __name__ == "__main__":
    main()
