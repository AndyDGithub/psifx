"""Learned gait attacker: GaitGraph2 (pretrained, CASIA-B) on the 2D keypoints of MEVID tracklets.

GaitGraph2 (Teepe et al., CVPR-W 2022) turns a sequence of COCO-17 keypoints into an identity embedding (ResGCN on
joints, velocities and bones). The published CASIA-B checkpoint is used as is, without training: it learned which
movements tell people apart on 124 people of a lab. Two steps:

1. ``--casia``: the published result is reproduced first on CASIA-B test (subjects 75-124, gallery NM#1-4, probes
   NM#5-6 / BG / CL, same view left out), from the keypoints of the GaitGraph release; this checks the loading and
   the preprocessing before anything is said about MEVID.
2. MEVID: for each source of keypoints, one embedding per tracklet, then the MEVID protocol of
   ``attack_params.evaluate`` (same tracklets, queries, settings and intervals as the other attackers):
   - ``original``: ViTPose on the original chips (``vit_k`` of ``tracks/``): the gait seen on the real video;
   - ``avatar_own`` / ``avatar_neutral``: ViTPose on the avatar renders (``vitpose_chips.py``): the gait an
     attacker reads on the published avatar video, own body (N2) or neutral body (N3);
   - ``sam3d``: the SAM 3D Body keypoints (MHR-70 reduced to COCO-17), the movement the avatar parameters carry.
   Control for each source: the other half of the same tracklet found from the first half (``halves``).

The network sees pixels: CASIA-B frames are 320x240 with the person crossing the frame. MEVID chips have the person
standing in a box of any size, so each tracklet is rescaled to the median CASIA-B body height and its hips moved to
the median CASIA-B hip position (one scale and shift per tracklet: the movement itself is not changed). Chips show
the person in place, so the walk across the frame is lost: a limit of the chips, not of the attacker.

    python research/mevid/gait_attack.py .e2e_outputs/ext/mevid/test_split \\
        --annotations .e2e_outputs/ext/mevid/annotations/x/mevid-v1-annotation-data \\
        --gaitgraph .e2e_outputs/ext/gaitgraph2/src/repo --weights .e2e_outputs/ext/gaitgraph2/weights/casia-b \\
        --casia .e2e_outputs/ext/gaitgraph2/data/casia-b_pose_coco.csv \\
        --avatars own=.e2e_outputs/ext/mevid/avatars_own_vitpose neutral=.e2e_outputs/ext/mevid/avatars_neutral_vitpose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_params import chance, evaluate  # noqa: E402
from summarize_test_split import keep  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "sam3d_body"))
SEQUENCE = 60  # Frames the CASIA-B checkpoint was trained on (config.yaml, data.sequence_length).


def load_file(path: Path, name: str):
    """Imports one file of the GaitGraph2 repository without its package (whose __init__ needs Lightning)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GaitGraph:
    """The CASIA-B model of GaitGraph2 at test time: preprocessing, embedding, test-time augmentation."""

    def __init__(self, repo: Path, weights: Path, device: str):
        package = repo / "GaitGraph"
        sys.path.insert(0, str(package))  # ResGCN imports its blocks as models.ResGCNv1.*
        from models.ResGCNv1 import create  # noqa: E402  (pure PyTorch)
        graph = load_file(package / "datasets" / "graph.py", "gg_graph").Graph("coco")
        augmentation = load_file(package / "transforms" / "augmentation.py", "gg_augmentation")
        self.multi_input = load_file(package / "transforms" / "multi_input.py", "gg_multi").MultiInput(
            graph.connect_joint, graph.center, enabled=True)
        self.pad = augmentation.PadSequence(SEQUENCE)
        self.center = augmentation.SelectSequenceCenter(SEQUENCE)
        self.flip_idx = graph.flip_idx
        # The released checkpoint predates the current repository code: three input branches (joints, velocities,
        # bones) of 5 channels, not one branch of 15; its two main-stream modules are the structure of n21.
        self.backbone = create("resgcn-n21-r8", A=torch.tensor(graph.A, dtype=torch.float32),
                               num_class=128, num_input=3, num_channel=5, parts=graph.parts)
        checkpoint = sorted((weights / "checkpoints").glob("gaitgraph-*.ckpt"))
        if not checkpoint:
            sys.exit(f"error: no checkpoint in {weights / 'checkpoints'}")
        # Lightning checkpoint from the authors' GitHub release (pickled hyperparameters, hence weights_only=False).
        state = torch.load(checkpoint[0], map_location="cpu", weights_only=False)["state_dict"]
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            sys.exit(f"error: checkpoint does not match the network, missing {missing[:3]}, unexpected {unexpected[:3]}")
        self.backbone.to(device).eval()
        self.device = device

    def prepare(self, keypoints: np.ndarray) -> torch.Tensor:
        """(T, 17, 3) x, y, confidence -> (60, 17, 3, 5), as the validation transform of GaitGraph2."""
        data = SimpleNamespace(x=torch.as_tensor(keypoints, dtype=torch.float32))
        for step in (self.pad, self.center, self.multi_input):
            data = step(data)
        return data.x

    @torch.inference_mode()
    def embed(self, sequences: list[np.ndarray], batch: int = 256) -> np.ndarray:
        """One embedding per sequence: original, time-reversed and left-right flipped, concatenated (their TTA)."""
        out = []
        for i in range(0, len(sequences), batch):
            x = torch.stack([self.prepare(s) for s in sequences[i:i + batch]]).to(self.device)
            flipped = torch.stack([self.multi_input(SimpleNamespace(x=d[:, :, 0, :3].flip(0))).x for d in x])
            mirrored = torch.stack([self.multi_input(SimpleNamespace(x=d[:, self.flip_idx, 0, :3])).x for d in x])
            n = len(x)
            y = self.backbone(torch.cat([x, flipped, mirrored]))[0]
            out.append(torch.cat(torch.split(y, [n, n, n]), dim=1).cpu().numpy())
        return np.concatenate(out)


def casia_sequences(csv: Path) -> dict[tuple, np.ndarray]:
    """CASIA-B test sequences (subjects 75-124) of the GaitGraph release: key (subject, status, number, angle)."""
    status = {"nm": 0, "bg": 1, "cl": 2}
    sequences: dict[tuple, list] = {}
    with open(csv) as f:
        next(f)
        for line in f:
            row = line.rstrip().split(",")
            _, sequence, _ = row[0].split("/")
            subject, walk, number, angle = sequence.split("-")
            if int(subject) < 75:
                continue
            key = int(subject), status[walk], int(number), int(angle)
            sequences.setdefault(key, []).append(np.array(row[1:], np.float32).reshape(17, 3))
    return {k: np.stack(v) for k, v in sequences.items()}


def casia_accuracy(keys: list[tuple], x: np.ndarray) -> dict:
    """GaitGraph2's test: per probe view and gallery view, nearest gallery (NM#1-4) sequence, same view excluded."""
    x = torch.nn.functional.normalize(torch.as_tensor(x), dim=1)  # LpDistance normalizes the embeddings.
    keys = np.array(keys)
    gallery = (keys[:, 1] == 0) & (keys[:, 2] <= 4)
    probes = {"NM#5-6": (keys[:, 1] == 0) & (keys[:, 2] >= 5), "BG#1-2": keys[:, 1] == 1, "CL#1-2": keys[:, 1] == 2}
    result = {}
    for name, probe in probes.items():
        table = np.zeros((11, 11))
        for g, ga in enumerate(range(0, 181, 18)):
            gi = np.where(gallery & (keys[:, 3] == ga))[0]
            for p, pa in enumerate(range(0, 181, 18)):
                pi = np.where(probe & (keys[:, 3] == pa))[0]
                nearest = torch.cdist(x[pi], x[gi]).argmin(1).numpy()
                table[g, p] = (keys[gi[nearest], 0] == keys[pi, 0]).mean()
        np.fill_diagonal(table, 0)
        result[name] = round(100 * float((table.sum(0) / 10).mean()), 1)  # Mean over probe views, same view out.
    return result


def halves(model: GaitGraph, sequences: list[np.ndarray]) -> float:
    """Control: Rank-1 of the second half of the same tracklet, searched from the first half among all second halves.

    Says whether the embedding carries something stable along a tracklet (identity, but also view and activity), so
    that a failure across cameras is not a broken pipeline. Each tracklet is cut at its own middle: an empty half
    would make GaitGraph2's PadSequence loop forever.
    """
    first = model.embed([s[:len(s) // 2] for s in sequences])
    second = model.embed([s[len(s) // 2:] for s in sequences])
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    return round(100 * float(((first @ second.T).argmax(1) == np.arange(len(first))).mean()), 1)


def normalize(k: np.ndarray, s: np.ndarray, reference: dict) -> np.ndarray:
    """One scale and shift per tracklet so that its median body height and hip position match CASIA-B's."""
    confident = s > 0.3
    heights = [np.ptp(f[c, 1]) for f, c in zip(k, confident) if c.sum() >= 5]
    hips = k[:, [11, 12]].mean(1)
    if not heights:
        return None
    scale = reference["height"] / max(np.median(heights), 1e-6)
    out = np.empty((len(k), 17, 3), np.float32)
    out[..., :2] = (k - np.median(hips, axis=0)) * scale + reference["hips"]
    out[..., 2] = s
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json and tracks/")
    parser.add_argument("--annotations", type=Path, required=True, help="folder with query_IDX.txt")
    parser.add_argument("--gaitgraph", type=Path, required=True, help="clone of github.com/tteepe/GaitGraph2")
    parser.add_argument("--weights", type=Path, required=True, help="casia-b folder of the release model_weights.zip")
    parser.add_argument("--casia", type=Path, required=True, help="casia-b_pose_coco.csv of the GaitGraph release")
    parser.add_argument("--avatars", nargs="*", default=[], help="name=folder of vitpose_chips.py archives")
    parser.add_argument("--min_mean", type=float, default=0.5)
    parser.add_argument("--min_torso", type=float, default=0.3)
    parser.add_argument("--min_chips", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    avatars = dict(a.split("=", 1) for a in args.avatars)
    for path in (args.folder / "sample.json", args.annotations / "query_IDX.txt", args.casia,
                 *(Path(p) for p in avatars.values())):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    model = GaitGraph(args.gaitgraph, args.weights, args.device)
    report = {"model": "GaitGraph2 CASIA-B (resgcn-n21-r8), published weights", "sequence": SEQUENCE}

    # 1. Reproduction on CASIA-B, and the size and place of a CASIA-B walker for step 2.
    start = time.time()
    casia = casia_sequences(args.casia)
    keys = list(casia)
    report["casia_b"] = casia_accuracy(keys, model.embed([casia[k] for k in keys]))
    every = np.concatenate([v for v in casia.values()])
    visible = every[..., 2] > 0.3
    report["casia_reference"] = reference = {
        "height": float(np.median([np.ptp(f[c, 1]) for f, c in zip(every, visible) if c.sum() >= 5])),
        "hips": np.median(every[:, [11, 12], :2].mean(1), axis=0).tolist()}
    print("CASIA-B", report["casia_b"], reference, f"{time.time() - start:.0f} s", flush=True)

    # 2. MEVID: the tracklets of attack_params.py, one sequence per keypoint source.
    from run_tracks import MHR70_TO_COCO17
    sample = json.loads((args.folder / "sample.json").read_text())
    queries = set(np.loadtxt(args.annotations / "query_IDX.txt").astype(int).tolist())
    rows, sources = [], {"original": [], "sam3d": [], **{f"avatar_{n}": [] for n in avatars}}
    for t in sample:
        a = np.load(args.folder / "tracks" / f"{t['track']}.npz")
        kept = keep(a["vit_s"], args.min_mean, args.min_torso)
        if kept.sum() < args.min_chips:
            continue
        sequences = {"original": normalize(a["vit_k"], a["vit_s"], reference),
                     # SAM 3D Body gives no confidence: every keypoint counts as seen.
                     "sam3d": normalize(a["keypoints_2d"][:, MHR70_TO_COCO17], np.ones((len(kept), 17)), reference)}
        for name, folder in avatars.items():
            v = np.load(Path(folder) / f"{t['track']}.npz")
            if list(v["frames"]) != list(a["frames"]):
                sys.exit(f"error: frames of {folder}/{t['track']}.npz differ from the original tracklet")
            sequences[f"avatar_{name}"] = normalize(v["vit_k"], v["vit_s"], reference)
        if any(s is None or len(s) < 8 for s in sequences.values()):
            continue  # No frame with enough keypoints in one source: left out of every source, for a fair comparison.
        rows.append(t)
        for name, s in sequences.items():
            sources[name].append(s)
    is_query = np.array([t["track"] in queries for t in rows])
    print(f"{len(rows)} tracklets, {is_query.sum()} queries", flush=True)

    rng = np.random.default_rng(0)
    report.update(tracklets=len(rows), queries=int(is_query.sum()), chance_rank1=chance(rows, is_query), mevid={})
    for name, sequences in sources.items():
        entry = evaluate(model.embed(sequences), rows, is_query, args.resamples, rng)
        entry["halves_rank1"] = halves(model, sequences)
        report["mevid"][name] = entry
        print(name, json.dumps({s: (e["rank1"], e["map"]) for s, e in entry.items() if s != "halves_rank1"}),
              "halves", entry["halves_rank1"], flush=True)
    print("chance", report["chance_rank1"])
    (args.folder / "gait_attack.json").write_text(json.dumps(report, indent=1))
    print("GAIT_OK")


if __name__ == "__main__":
    main()
