"""SAM 3D Body (``body`` mode) and ViTPose on every chip of ``sample.json``: the avatars of the MEVID test split.

Each chip is given whole (box = the chip, no mask), as in ``sam3d_small_crops.py``. Nothing is filtered here: ViTPose
scores are stored with each chip, and ``summarize_test_split.py`` decides which chips are people.

Writes one archive per tracklet, ``<folder>/tracks/<track>.npz``:
- ``frames`` (N,) chip names, ``size`` (N, 2) height and width;
- ``vit_k`` (N, 17, 2), ``vit_s`` (N, 17): ViTPose COCO keypoints and scores;
- ``keypoints_2d`` (N, 70, 2), ``keypoints_3d`` (N, 70, 3), ``cam_t`` (N, 3), ``focal`` (N,);
- the MHR parameters of ``run_tracks.PARAMETERS``, enough to rebuild the mesh.
An archive is written to a temporary name and renamed when complete, so a stopped run resumes where it was: tracklets
with an archive are skipped.

    python research/mevid/sam3d_test_split.py .e2e_outputs/ext/mevid/test_split \\
        --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sam3d_body"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tracks import PARAMETERS, load_estimator  # noqa: E402
from sam3d_small_crops import vitpose  # noqa: E402


def sam3d(estimator, image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    with open(os.devnull, "w") as null:  # The estimator prints a line per call.
        stdout, sys.stdout = sys.stdout, null
        try:
            out = estimator.process_one_image(image, bboxes=np.array([[0, 0, w, h]], np.float32), inference_type="body")
        finally:
            sys.stdout = stdout
    if len(out) != 1:
        sys.exit(f"error: {len(out)} results for one box")
    o = out[0]
    result = {"keypoints_2d": o["pred_keypoints_2d"][:, :2], "keypoints_3d": o["pred_keypoints_3d"],
              "cam_t": o["pred_cam_t"], "focal": float(o["focal_length"])}
    result.update({name: o[name] for name in PARAMETERS})
    return {k: np.asarray(v, np.float32) for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json and chips/")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vitpose", default="usyd-community/vitpose-base-simple")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sample_path = args.folder / "sample.json"
    if not sample_path.is_file():
        sys.exit(f"error: missing {sample_path}")
    sample = json.loads(sample_path.read_text())
    # Every chip is checked before the long run starts, so a bad extraction stops it at once.
    missing = [n for t in sample for n in t["frames"] if not (args.folder / "chips" / "bbox_test" / n[:4] / n).is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} chips missing, first {missing[0]}")
    out = args.folder / "tracks"
    out.mkdir(exist_ok=True)
    todo = [t for t in sample if not (out / f"{t['track']}.npz").is_file()]
    print(f"{len(sample)} tracklets, {len(sample) - len(todo)} already done", flush=True)

    from transformers import AutoProcessor, VitPoseForPoseEstimation
    processor = AutoProcessor.from_pretrained(args.vitpose)
    pose_model = VitPoseForPoseEstimation.from_pretrained(args.vitpose).to(args.device).eval()
    estimator = load_estimator(args.repo, args.checkpoint, args.device)

    start, chips, total = time.time(), 0, sum(len(t["frames"]) for t in todo)
    for n, track in enumerate(todo):
        rows = []
        for name in track["frames"]:
            image = np.asarray(Image.open(args.folder / "chips" / "bbox_test" / name[:4] / name).convert("RGB"))
            row = sam3d(estimator, image)
            row["vit_k"], row["vit_s"] = vitpose(processor, pose_model, image, args.device)
            row["size"] = np.array(image.shape[:2], np.int32)
            rows.append(row)
        archive = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        temporary = out / f"{track['track']}.tmp.npz"
        np.savez_compressed(temporary, frames=np.array(track["frames"]), **archive)
        temporary.rename(out / f"{track['track']}.npz")  # Complete archives only, for the resume.
        chips += len(rows)
        if n % 20 == 0 or n == len(todo) - 1:
            elapsed = time.time() - start
            print(f"{n + 1}/{len(todo)} tracklets, {chips}/{total} chips, {elapsed / chips:.3f} s/chip, "
                  f"{(total - chips) * elapsed / chips / 60:.0f} min left", flush=True)
    print("TEST_SPLIT_OK")


if __name__ == "__main__":
    main()
