"""Runs SAM 3D Body on the people tracked by SAM3 in an end-to-end output folder.

For every ``stride``-th frame, each person's SAM3 mask (``masks/<id>.mp4``) gives the box and the mask
prompt; SAM 3D Body returns a full-body mesh (MHR, 18k vertices), 70 3D keypoints and the camera.
No detector, segmentor or field-of-view model is loaded: the boxes and masks come from psifx, and
the camera uses the default intrinsics (focal = image diagonal).

Writes ``<out_dir>/sam3d_body/<id>.npz`` with, per processed frame:
- ``frames`` (N,), ``keypoints_2d`` (N, 70, 2) in pixels, ``keypoints_3d`` (N, 70, 3) in meters,
  camera frame minus ``cam_t``;
- ``cam_t`` (N, 3), ``focal`` (N,), ``vertices`` (N, V, 3) float16 (with ``--save_vertices``);
- ``faces`` (F, 3), shared by all frames.

The SAM 3D Body code is not a pip package: pass its clone with ``--repo``. Checkpoints are loaded
with ``torch.load(weights_only=True)``, unlike the upstream loader, so no pickle code runs.

    python research/sam3d_body/run_tracks.py .e2e_outputs/dev_2026-10-27/v1_example \\
        --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# MHR mesh: use the TorchScript model shipped with the checkpoint, not the pymomentum package.
os.environ.setdefault("MOMENTUM_ENABLED", "0")

# MHR-70 keypoints in COCO-17 order (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
MHR70_TO_COCO17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def load_estimator(repo: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(repo))
    import torch
    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body.utils.config import get_config

    cfg = get_config(str(checkpoint / "model_config.yaml"))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(checkpoint / "assets" / "mhr_model.pt")
    cfg.freeze()
    model = SAM3DBody(cfg)
    state = torch.load(checkpoint / "model.ckpt", map_location="cpu", weights_only=True)
    load_state_dict(model, state.get("state_dict", state), strict=False)
    model = model.to(device).eval()
    return SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)


def mask_box(mask: np.ndarray, min_area: int):
    ys, xs = np.nonzero(mask)
    if len(xs) < min_area:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="end-to-end folder with input.mp4 and masks/")
    parser.add_argument("--repo", type=Path, required=True, help="clone of facebookresearch/sam-3d-body")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with model.ckpt and assets/")
    parser.add_argument("--stride", type=int, default=1, help="process one frame out of N")
    parser.add_argument("--inference_type", default="full", choices=["full", "body"],
                        help="full adds the hand decoder (slower)")
    parser.add_argument("--min_area", type=int, default=400, help="smallest mask, in pixels, worth a mesh")
    parser.add_argument("--save_vertices", action="store_true", help="store meshes (about 110 kB per frame)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.out_dir
    src, mask_dir, dst = d / "input.mp4", d / "masks", d / "sam3d_body"
    for path in (src, mask_dir, args.checkpoint / "model.ckpt", args.checkpoint / "assets" / "mhr_model.pt",
                 args.repo / "sam_3d_body"):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    people = sorted((p.stem for p in mask_dir.glob("*.mp4")), key=int)
    if not people:
        sys.exit(f"error: no mask in {mask_dir}")
    if dst.exists() and any(dst.iterdir()) and not args.overwrite:
        sys.exit(f"error: {dst} is not empty, pass --overwrite to replace it")
    dst.mkdir(exist_ok=True)

    from psifx.io import video
    from tqdm import tqdm

    estimator = load_estimator(args.repo, args.checkpoint, args.device)
    results = {p: {"frames": [], "keypoints_2d": [], "keypoints_3d": [], "cam_t": [], "focal": [], "vertices": []}
               for p in people}
    readers = [video.VideoReader(path=src)] + [video.VideoReader(path=mask_dir / f"{p}.mp4") for p in people]
    n_frames = len(readers[0])
    start, calls = time.time(), 0
    try:
        for i, (frame, *masks) in enumerate(tqdm(zip(*readers), total=n_frames, desc="SAM 3D Body")):
            if i % args.stride:
                continue
            boxes, prompts, who = [], [], []
            for person, mask in zip(people, masks):
                binary = mask[..., 0] > 127
                box = mask_box(binary, args.min_area)
                if box is not None:
                    boxes.append(box), prompts.append(binary), who.append(person)
            if not boxes:
                continue
            # The estimator prints a line per call; keep the progress bar readable.
            with open(os.devnull, "w") as null:
                stdout, sys.stdout = sys.stdout, null
                try:
                    out = estimator.process_one_image(frame, bboxes=np.stack(boxes), masks=np.stack(prompts),
                                                      inference_type=args.inference_type)
                finally:
                    sys.stdout = stdout
            calls += 1
            for person, o in zip(who, out):
                r = results[person]
                r["frames"].append(i)
                r["keypoints_2d"].append(o["pred_keypoints_2d"][:, :2])
                r["keypoints_3d"].append(o["pred_keypoints_3d"])
                r["cam_t"].append(o["pred_cam_t"])
                r["focal"].append(float(o["focal_length"]))
                if args.save_vertices:
                    r["vertices"].append(o["pred_vertices"].astype(np.float16))
    finally:
        for reader in readers:
            reader.close()

    elapsed = time.time() - start
    for person, r in results.items():
        arrays = {k: np.asarray(v) for k, v in r.items() if v}
        np.savez_compressed(dst / f"{person}.npz", faces=estimator.faces, **arrays)
        print(f"person {person}: {len(r['frames'])} frames")
    print(f"{calls} frames processed in {elapsed:.0f} s ({elapsed / max(calls, 1):.2f} s per frame)")


if __name__ == "__main__":
    main()
