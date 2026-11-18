"""What identity is left in the avatar parameters? Re-identification on MEVID test from the MHR parameters alone.

Training-free attackers, nearest neighbor between tracklets (cosine distance on features standardized over the
tracklets); each tracklet is described by its chips kept by the ViTPose filter (``summarize_test_split.keep``):
- ``shape``: mean identity parameters (45), what an avatar with its own body (level N2) keeps;
- ``shape_scale``: plus the mean scale values (68, the end of ``mhr_model_params``: limb proportions);
- ``motion``: body pose angles (``body_pose_params``, independent of the body shape and of the camera, since the
  global rotation is left out): mean, spread and mean change between consecutive chips. A low bound of what the
  movement says; it also sees the activity (walking, sitting), which is not identity;
- ``all``: the three together.
A neutral avatar (N3) has the same identity and scale for everyone: the ``shape`` attackers fall to chance by
construction, and only ``motion`` is left.

MEVID protocol: queries from ``query_IDX.txt``, gallery the other tracklets; gallery tracklets of the same person
and the same camera are left out. Three settings: ``general`` (all of the rest), ``same_outfit`` (the same person
in another outfit left out), ``other_outfit`` (the same person in the same outfit left out: clothes change).
A query without any correct gallery tracklet in a setting is not counted there.

Reports Rank-1, Rank-5, mAP, with a 95 % interval by resampling the query identities, per setting and per crop
height of the query. Writes ``<folder>/attack_params.json``.

    python research/mevid/attack_params.py .e2e_outputs/ext/mevid/test_split \\
        --annotations .e2e_outputs/ext/mevid/annotations/x/mevid-v1-annotation-data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_test_split import keep  # noqa: E402

POSE = 136  # mhr_model_params: 136 pose values, then 68 scale values.
HEIGHTS = ((0, 96), (96, 192), (192, 10 ** 6))


def features(a, kept: np.ndarray) -> dict[str, np.ndarray]:
    pose = a["body_pose_params"][kept]
    # Angles: sine and cosine, so that a turn through +-pi is not a jump.
    angles = np.concatenate([np.sin(pose), np.cos(pose)], axis=1)
    change = np.abs(np.diff(angles, axis=0)).mean(0) if len(angles) > 1 else np.zeros(angles.shape[1])
    f = {"shape": a["shape_params"][kept].mean(0),
         "scale": a["mhr_model_params"][kept][:, POSE:].mean(0),
         "motion": np.concatenate([angles.mean(0), angles.std(0), change])}
    return {"shape": f["shape"], "shape_scale": np.r_[f["shape"], f["scale"]], "motion": f["motion"],
            "all": np.r_[f["shape"], f["scale"], f["motion"]]}


def rank(distance: np.ndarray, correct: np.ndarray, allowed: np.ndarray) -> tuple[int, float] | None:
    """Rank of the first correct gallery tracklet (0 = first) and average precision; None if nothing to find."""
    d, c = distance[allowed], correct[allowed]
    if not c.any():
        return None
    order = np.argsort(d, kind="stable")
    hits = c[order]
    positions = np.where(hits)[0]
    precision = np.arange(1, len(positions) + 1) / (positions + 1)
    return int(positions[0]), float(precision.mean())


def scores(ranks: list[tuple[int, float, int]], resamples: int, rng) -> dict:
    """Rank-1, Rank-5, mAP and their 95 % interval over the query identities."""
    r = np.array([x[0] for x in ranks]), np.array([x[1] for x in ranks]), np.array([x[2] for x in ranks])

    def value(index):
        return [float((r[0][index] < 1).mean()), float((r[0][index] < 5).mean()), float(r[1][index].mean())]

    people = np.unique(r[2])
    by_person = {p: np.where(r[2] == p)[0] for p in people}
    draws = np.array([value(np.concatenate([by_person[p] for p in rng.choice(people, len(people))]))
                      for _ in range(resamples)])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    result = {"queries": len(r[0])}
    for k, name in enumerate(("rank1", "rank5", "map")):
        result[name] = [round(100 * value(np.arange(len(r[0])))[k], 1), round(100 * low[k], 1), round(100 * high[k], 1)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json and tracks/")
    parser.add_argument("--annotations", type=Path, required=True, help="folder with query_IDX.txt")
    parser.add_argument("--min_mean", type=float, default=0.5)
    parser.add_argument("--min_torso", type=float, default=0.3)
    parser.add_argument("--min_chips", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()

    for path in (args.folder / "sample.json", args.annotations / "query_IDX.txt"):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    sample = json.loads((args.folder / "sample.json").read_text())
    missing = [t["track"] for t in sample if not (args.folder / "tracks" / f"{t['track']}.npz").is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} tracklets without archive, first {missing[0]}")
    queries = set(np.loadtxt(args.annotations / "query_IDX.txt").astype(int).tolist())

    rows, feats = [], {}
    for t in sample:
        a = np.load(args.folder / "tracks" / f"{t['track']}.npz")
        kept = keep(a["vit_s"], args.min_mean, args.min_torso)
        if kept.sum() < args.min_chips:
            continue
        rows.append(t)
        for name, value in features(a, kept).items():
            feats.setdefault(name, []).append(value)
    person = np.array([t["person"] for t in rows])
    outfit = np.array([t["outfit"] for t in rows])
    camera = np.array([t["camera"] for t in rows])
    height = np.array([t["first_height"] for t in rows])
    is_query = np.array([t["track"] in queries for t in rows])
    print(f"{len(rows)} tracklets kept, {is_query.sum()} queries, {len(set(person))} people", flush=True)

    rng = np.random.default_rng(0)
    report = {"tracklets": len(rows), "queries": int(is_query.sum()), "filter": vars(args) | {"folder": None,
                                                                                            "annotations": None},
              "attackers": {}}
    gallery = np.where(~is_query)[0]
    for name, matrix in feats.items():
        x = np.stack(matrix)
        x = (x - x.mean(0)) / (x.std(0) + 1e-9)  # Standardized over the tracklets: no unit dominates.
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        ranks = {s: [] for s in ("general", "same_outfit", "other_outfit")}
        for q in np.where(is_query)[0]:
            distance = 1 - x[gallery] @ x[q]
            same_person = person[gallery] == person[q]
            base = ~(same_person & (camera[gallery] == camera[q]))
            settings = {"general": base,
                        "same_outfit": base & ~(same_person & (outfit[gallery] != outfit[q])),
                        "other_outfit": base & ~(same_person & (outfit[gallery] == outfit[q]))}
            for s, allowed in settings.items():
                result = rank(distance, same_person, allowed)
                if result:
                    ranks[s].append((*result, person[q], height[q]))
        entry = {}
        for s, values in ranks.items():
            entry[s] = scores([v[:3] for v in values], args.resamples, rng)
            entry[s]["by_height"] = {f"{lo}-{hi}": scores([v[:3] for v in values if lo <= v[3] < hi], args.resamples,
                                                          rng)["rank1"] for lo, hi in HEIGHTS
                                     if any(lo <= v[3] < hi for v in values)}
        report["attackers"][name] = entry
        print(name, json.dumps({s: (e["rank1"], e["map"]) for s, e in entry.items()}), flush=True)

    # Chance: Rank-1 of a random order, the same queries and allowed galleries.
    chance = {}
    for s in ("general", "other_outfit"):
        hits = []
        for q in np.where(is_query)[0]:
            same_person = person[gallery] == person[q]
            allowed = ~(same_person & (camera[gallery] == camera[q]))
            if s == "other_outfit":
                allowed &= ~(same_person & (outfit[gallery] == outfit[q]))
            if same_person[allowed].any():
                hits.append(same_person[allowed].mean())
        chance[s] = round(100 * float(np.mean(hits)), 1)
    report["chance_rank1"] = chance
    print("chance", chance)
    (args.folder / "attack_params.json").write_text(json.dumps(report, indent=1))
    print("ATTACK_OK")


if __name__ == "__main__":
    main()
