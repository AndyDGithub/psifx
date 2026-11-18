"""MEVID chips with the person replaced by its avatar, for the appearance attacker (CAL) on avatars.

For every tracklet archive of ``sam3d_test_split.py``, each chip is redrawn at its own size: grey background (the
decor of the chip would let the place and the camera identify the tracklet), the MHR mesh rebuilt from the
parameters, placed with ``cam_t`` and projected with the archive focal, drawn by ``render_video.draw_mesh`` in the
**same color for everyone** (a color per person would give the identity away).

``--body own`` keeps the identity and scale parameters (level N2); ``--body neutral`` sets identity to 0 and the
scales to the model's mean (level N3, as in ``render_avatars.py``).

Writes ``<output>/bbox_test/<id>/<chip name>``, the tree of the original chips, so that ``cal_eval.py --window
--frames_root <output>`` reads them in place of the originals.

    python research/mevid/render_chips.py .e2e_outputs/ext/mevid/test_split \\
        --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith --body own --output .e2e_outputs/ext/mevid/avatars_own \\
        --faces .e2e_outputs/extrait_talkshow/pipeline/sam3d_params/0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sam3d_body"))
from render_video import draw_mesh  # noqa: E402

GREY, COLOR = (128, 128, 128), (200, 170, 140)  # BGR: one light color for every avatar.
POSE = 136


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json and tracks/")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with model.ckpt and assets/mhr_model.pt")
    parser.add_argument("--body", choices=("own", "neutral"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True,
                        help="an archive of run_tracks.py with the MHR triangles (faces), e.g. sam3d_params/0.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = args.checkpoint / "assets" / "mhr_model.pt"
    for path in (args.folder / "sample.json", model, args.faces):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    sample = json.loads((args.folder / "sample.json").read_text())
    missing = [t["track"] for t in sample if not (args.folder / "tracks" / f"{t['track']}.npz").is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} tracklets without archive, first {missing[0]}")
    scale_mean = None
    if args.body == "neutral":
        state = torch.load(args.checkpoint / "model.ckpt", map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        if "head_pose.scale_mean" not in state:
            sys.exit("error: head_pose.scale_mean not in the checkpoint")
        scale_mean = state["head_pose.scale_mean"].numpy().astype(np.float32)
        del state
    mhr = torch.jit.load(str(model), map_location=args.device)
    with np.load(args.faces) as npz:
        if "faces" not in npz.files:
            sys.exit(f"error: no faces in {args.faces}")
        faces = npz["faces"]

    start, done = time.time(), 0
    for n, t in enumerate(sample):
        a = dict(np.load(args.folder / "tracks" / f"{t['track']}.npz"))
        shape, params = a["shape_params"].copy(), a["mhr_model_params"].copy()
        if scale_mean is not None:
            shape[:] = 0
            params[:, POSE:] = scale_mean
        with torch.inference_mode():
            vertices = mhr(*(torch.from_numpy(x).to(args.device).float() for x in (shape, params, a["expr_params"])))
            vertices = vertices[0].cpu().numpy() / 100
        vertices[..., [1, 2]] *= -1  # MHR axes to the camera axes, as in mesh_eval.py.
        for i, name in enumerate(a["frames"]):
            height, width = (int(v) for v in a["size"][i])
            canvas = np.full((height, width, 3), GREY, np.uint8)
            draw_mesh(canvas, vertices[i] + a["cam_t"][i], faces, float(a["focal"][i]), COLOR)
            path = args.output / "bbox_test" / str(name)[:4] / str(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), canvas):
                sys.exit(f"error: cannot write {path}")
            done += 1
        if n % 100 == 0:
            print(f"{n + 1}/{len(sample)} tracklets, {done} chips, {(time.time() - start) / done * 1000:.0f} ms/chip",
                  flush=True)
    print("RENDER_CHIPS_OK")


if __name__ == "__main__":
    main()
