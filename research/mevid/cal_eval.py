"""CAL, the appearance attacker of MEVID, evaluated on one GPU with the published MEVID weights.

Reuses the code of the MEVID paper (``mevid-checkout/CAL``): its dataset (``data/datasets/ccvid.py``, the MEVID
class), its C2D ResNet-50, its test clips (8 frames, one out of 4: every frame is used) and its metrics
(``tools/eval_metrics.py``). Only the distributed launch is replaced: the features are extracted here by a plain
DataLoader, then averaged per tracklet as in ``test.py``.

Patches, all at import time and without changing their files:
- ``h5py`` is imported by the dataset module but not used for MEVID: an empty module stands for it;
- ``np.int`` (removed from numpy) is set to ``int``, ``collections.Iterable`` (moved in Python 3.10) to
  ``collections.abc.Iterable``;
- ``torchvision.models.resnet50`` builds the network without downloading ImageNet weights: the checkpoint replaces
  them all.

``--window``: each tracklet is reduced to its frames in ``sample.json`` (our 32-frame windows), so that the result is
comparable to the avatar renders. ``--frames_root``: read the chips from another folder (the renders), same tree.

Writes ``<output>.npz`` (distance matrix and ids) and ``<output>.json`` (Rank-1, 5, 10, mAP per setting).

    python research/mevid/cal_eval.py --cal .e2e_outputs/ext/mevid-code/mevid-checkout/CAL \\
        --root .e2e_outputs/ext/mevid/cal_root --checkpoint .e2e_outputs/ext/ckpt/cal-mevid/CAL_best_model.pth.tar \\
        --output .e2e_outputs/ext/mevid/cal_eval/full
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def patched_imports(cal: Path):
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    np.int = int  # noqa: NPY001 (old code)
    import collections
    import collections.abc
    collections.Iterable = collections.abc.Iterable
    import torchvision
    original = torchvision.models.resnet50

    @functools.wraps(original)
    def resnet50(pretrained=False, **kwargs):  # The checkpoint brings every weight.
        return original(weights=None, **kwargs)

    torchvision.models.resnet50 = resnet50
    sys.path.insert(0, str(cal))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cal", type=Path, required=True, help="the CAL folder of mevid-checkout")
    parser.add_argument("--root", type=Path, required=True,
                        help="folder with bbox_test/ and the annotation files (test_name.txt, track_*_info.txt, query_IDX.txt)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path without extension")
    parser.add_argument("--window", type=Path, help="sample.json: keep only these frames per tracklet")
    parser.add_argument("--frames_root", type=Path, help="read the chips from this folder instead of <root>")
    # Batches go from the workers through shared memory, 64 MB in a default Docker container: 4 clips are 12.6 MB
    # (4 x 8 x 3 x 256 x 128 float32), and each worker keeps one batch ahead.
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for path in (args.cal / "models", args.root / "bbox_test", args.root / "query_IDX.txt", args.checkpoint):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    patched_imports(args.cal.resolve())
    from configs.default_vid import _C
    from data import build_vid_transforms
    from data.dataset_loader import VideoDataset
    from data.datasets.ccvid import CCVID
    from models import build_model
    from tools.eval_metrics import evaluate, evaluate_with_clothes

    config = _C.clone()
    config.defrost()
    config.merge_from_file(str(args.cal / "configs" / "c2dres50_ce_cal.yaml"))
    dataset = CCVID(root=str(args.root), sampling_step=config.DATA.SAMPLING_STEP, seq_len=config.AUG.SEQ_LEN,
                    stride=config.AUG.SAMPLING_STRIDE)

    # Tracklets in the order of the dataset: query_IDX, then the others, as in ccvid.py.
    info = np.loadtxt(args.root / "track_test_info.txt").astype(int)
    query_index = np.loadtxt(args.root / "query_IDX.txt").astype(int)
    gallery_index = np.array([i for i in range(len(info)) if i not in set(query_index.tolist())])
    sets = {"query": (dataset.query, query_index), "gallery": (dataset.gallery, gallery_index)}
    if args.window:
        windows = {t["track"]: t["frames"] for t in json.loads(args.window.read_text())}
        for name, (tracklets, index) in sets.items():
            reduced = []
            for (paths, pid, camid, clothes), t in zip(tracklets, index):
                folder = os.path.dirname(paths[0])
                reduced.append(([os.path.join(folder, f) for f in windows[int(t)]], pid, camid, clothes))
            sets[name] = (reduced, index)
    if args.frames_root:
        old, new = str(args.root), str(args.frames_root)
        for name, (tracklets, index) in sets.items():
            sets[name] = ([([p.replace(old, new, 1) for p in paths], *rest) for paths, *rest in tracklets], index)
    missing = [p for tracklets, _ in sets.values() for paths, *_ in tracklets for p in paths if not os.path.isfile(p)]
    if missing:
        sys.exit(f"error: {len(missing)} frames missing, first {missing[0]}")

    model, _, _ = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # Kitware's published file.
    model.load_state_dict(state["model_state_dict"])
    model = model.to(args.device).eval()
    _, spatial, _, _ = build_vid_transforms(config)

    features = {}
    start = time.time()
    for name, (tracklets, _) in sets.items():
        clips, vid2clip = dataset._recombination_for_testset(tracklets, seq_len=config.AUG.SEQ_LEN,
                                                               stride=config.AUG.SAMPLING_STRIDE)
        # Workers only read and resize images, no random transform; shuffle=False keeps the order of the clips.
        loader = torch.utils.data.DataLoader(VideoDataset(clips, spatial, None), batch_size=args.batch,
                                             num_workers=args.workers, shuffle=False, pin_memory=True,
                                             prefetch_factor=1 if args.workers else None)
        out = []
        with torch.inference_mode():
            for b, (videos, *_) in enumerate(loader):
                out.append(model(videos.to(args.device)).float().cpu())
                if b % 5000 == 0:
                    print(f"{name}: {b}/{len(loader)} batches, {time.time() - start:.0f} s", flush=True)
        clip_features = torch.cat(out)
        tracks = torch.stack([F.normalize(clip_features[a:b].mean(0), dim=0) for a, b in vid2clip])
        ids = np.array([[pid, camid, clothes] for _, pid, camid, clothes in tracklets])
        features[name] = (tracks, ids)
    seconds = time.time() - start

    (qf, q), (gf, g) = features["query"], features["gallery"]
    distance = (-qf @ gf.T).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{args.output}.npz", distance=distance, query=q, gallery=g, query_index=query_index,
             gallery_index=gallery_index)
    report = {"seconds": round(seconds), "window": bool(args.window), "frames_root": str(args.frames_root or "")}
    for setting in ("general", "SC", "CC"):
        if setting == "general":
            cmc, mean_ap = evaluate(distance, q[:, 0], g[:, 0], q[:, 1], g[:, 1])
        else:
            cmc, mean_ap = evaluate_with_clothes(distance, q[:, 0], g[:, 0], q[:, 1], g[:, 1], q[:, 2], g[:, 2],
                                                 mode=setting)
        report[setting] = {"rank1": round(100 * float(cmc[0]), 1), "rank5": round(100 * float(cmc[4]), 1),
                           "rank10": round(100 * float(cmc[9]), 1), "map": round(100 * float(mean_ap), 1)}
        print(setting, report[setting], flush=True)
    Path(f"{args.output}.json").write_text(json.dumps(report, indent=1))
    print("CAL_OK")


if __name__ == "__main__":
    main()
