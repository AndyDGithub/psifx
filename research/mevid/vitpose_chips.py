"""ViTPose on every chip of a chip tree (originals or avatar renders of ``render_chips.py``), one archive per tracklet.

What an attacker who only has the video reads of the body: the 17 COCO keypoints, in chip pixels, as
``sam3d_test_split.py`` stores them for the originals (``vit_k``, ``vit_s``). The gait attacker then works the same
way on the originals and on the avatars.

Writes ``<output>/<track>.npz`` with ``frames``, ``vit_k`` (N, 17, 2), ``vit_s`` (N, 17); resumable (an archive is
written to a temporary name, then renamed: a stopped run leaves no half archive).

    python research/mevid/vitpose_chips.py .e2e_outputs/ext/mevid/test_split \\
        --chips .e2e_outputs/ext/mevid/avatars_own --output .e2e_outputs/ext/mevid/avatars_own_vitpose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3d_small_crops import vitpose  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json")
    parser.add_argument("--chips", type=Path, required=True, help="tree with bbox_test/<id>/<chip>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vitpose", default="usyd-community/vitpose-base-simple")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sample = json.loads((args.folder / "sample.json").read_text())
    # Every chip is checked before the long run, so a missing render stops it at once.
    missing = [n for t in sample for n in t["frames"] if not (args.chips / "bbox_test" / n[:4] / n).is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} chips missing in {args.chips}, first {missing[0]}")
    args.output.mkdir(parents=True, exist_ok=True)
    todo = [t for t in sample if not (args.output / f"{t['track']}.npz").is_file()]
    print(f"{len(todo)}/{len(sample)} tracklets to do", flush=True)

    from transformers import AutoProcessor, VitPoseForPoseEstimation
    processor = AutoProcessor.from_pretrained(args.vitpose)
    model = VitPoseForPoseEstimation.from_pretrained(args.vitpose).to(args.device).eval()

    start, chips = time.time(), 0
    with torch.inference_mode():
        for n, t in enumerate(todo):
            k, s = [], []
            for name in t["frames"]:
                image = np.asarray(Image.open(args.chips / "bbox_test" / name[:4] / name).convert("RGB"))
                kk, ss = vitpose(processor, model, image, args.device)
                k.append(kk)
                s.append(ss)
            tmp = args.output / f"{t['track']}.tmp.npz"
            np.savez(tmp, frames=np.array(t["frames"]), vit_k=np.stack(k).astype(np.float32),
                     vit_s=np.stack(s).astype(np.float32))
            tmp.rename(args.output / f"{t['track']}.npz")
            chips += len(t["frames"])
            if (n + 1) % 100 == 0 or n + 1 == len(todo):
                elapsed = time.time() - start
                print(f"{n + 1}/{len(todo)} tracklets, {elapsed / chips:.3f} s/chip", flush=True)
    print("VITPOSE_CHIPS_OK")


if __name__ == "__main__":
    main()
