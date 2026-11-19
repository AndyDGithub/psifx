"""Expressions on the avatars: MediaPipe's 52 ARKit blendshapes turned into the 72 FACS expressions of MHR.

MHR names its 72 expressions (``docs/face-expressions.md`` of facebookresearch/MHR: ``jawDrop``,
``lipCornerPuller_L``, ``eyesClosed_R``...), close to the ARKit names of MediaPipe, so the conversion is a table
(``ARKIT_TO_MHR``) and not a learned model. ``_L`` / ``_R`` are the person's own left and right in both. Then:

1. **Rest face**: each person has a rest value per blendshape (glasses read as a half blink, a mouth always a bit
   open): the ``REST_PERCENTILE`` of the person's scores is taken away, below it is 0.
2. **Gain** per MHR expression (``--gains``: a json written by ``--reread``), 1 by default.
3. **Smoothing** per run of frames (no gap, no cut): the robust smoother of ``robust_smooth.py`` (Hampel then
   Rauch-Tung-Striebel) with ``CUTOFF_HZ`` per region; no outlier rejection for the eyelids (a blink is a real
   spike). A frame where the face was not read is filled by the smoother and goes back to the rest face when the
   face stays unread (a weak pull to 0).
4. The expressions are written into ``expr_params`` of a copy of the parameter archives (``--params`` ->
   ``--output``), which ``render_avatars.py --params`` draws.

``--reread <folder>`` measures instead what the avatar's face carries: the blendshapes read by
``face_blendshapes.py --video`` on the avatar video against those of the original, per blendshape (Pearson
correlation and slope), and writes the gains that make the slopes 1.

    python research/sam3d_body/face_expression.py .e2e_outputs/extrait_talkshow/pipeline \\
        --params sam3d_robust --output sam3d_face
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_smooth import channel_noise, hampel_mask, rts_smooth, HIDDEN_NOISE  # noqa: E402
from run_keyframes import cut_labels  # noqa: E402

MHR_EXPRESSIONS = [
    "browLowerer_L", "browLowerer_R", "cheekPuff_L", "cheekPuff_R", "cheekRaiser_L", "cheekRaiser_R", "cheekSuck_L",
    "cheekSuck_R", "chinRaiser_B", "chinRaiser_T", "dimpler_L", "dimpler_R", "eyesClosed_L", "eyesClosed_R",
    "eyesLookDown_L", "eyesLookDown_R", "eyesLookLeft_L", "eyesLookLeft_R", "eyesLookRight_L", "eyesLookRight_R",
    "eyesLookUp_L", "eyesLookUp_R", "innerBrowRaiser_L", "innerBrowRaiser_R", "jawDrop", "jawSidewaysLeft",
    "jawSidewaysRight", "jawThrust", "lidTightener_L", "lidTightener_R", "lipCornerDepressor_L",
    "lipCornerDepressor_R", "lipCornerPuller_L", "lipCornerPuller_R", "lipFunneler_LB", "lipFunneler_LT",
    "lipFunneler_RB", "lipFunneler_RT", "lipPressor_L", "lipPressor_R", "lipPucker_L", "lipPucker_R",
    "lipStretcher_L", "lipStretcher_R", "lipSuck_LB", "lipSuck_LT", "lipSuck_RB", "lipSuck_RT", "lipTightener_L",
    "lipTightener_R", "lipsToward_LB", "lipsToward_LT", "lipsToward_RB", "lipsToward_RT", "lowerLipDepressor_L",
    "lowerLipDepressor_R", "mouthLeft", "mouthRight", "nasolabialFurrow_L", "nasolabialFurrow_R", "noseWrinkler_L",
    "noseWrinkler_R", "nostrilCompressor_L", "nostrilCompressor_R", "nostrilDilator_L", "nostrilDilator_R",
    "outerBrowRaiser_L", "outerBrowRaiser_R", "upperLidRaiser_L", "upperLidRaiser_R", "upperLipRaiser_L",
    "upperLipRaiser_R"]
# ARKit blendshape -> MHR expressions it drives (eye "in" looks towards the nose: the left eye then looks right).
ARKIT_TO_MHR = {
    "browDownLeft": ["browLowerer_L"], "browDownRight": ["browLowerer_R"],
    "browInnerUp": ["innerBrowRaiser_L", "innerBrowRaiser_R"],
    "browOuterUpLeft": ["outerBrowRaiser_L"], "browOuterUpRight": ["outerBrowRaiser_R"],
    "cheekPuff": ["cheekPuff_L", "cheekPuff_R"],
    "cheekSquintLeft": ["cheekRaiser_L"], "cheekSquintRight": ["cheekRaiser_R"],
    "eyeBlinkLeft": ["eyesClosed_L"], "eyeBlinkRight": ["eyesClosed_R"],
    "eyeLookDownLeft": ["eyesLookDown_L"], "eyeLookDownRight": ["eyesLookDown_R"],
    "eyeLookInLeft": ["eyesLookRight_L"], "eyeLookInRight": ["eyesLookLeft_R"],
    "eyeLookOutLeft": ["eyesLookLeft_L"], "eyeLookOutRight": ["eyesLookRight_R"],
    "eyeLookUpLeft": ["eyesLookUp_L"], "eyeLookUpRight": ["eyesLookUp_R"],
    "eyeSquintLeft": ["lidTightener_L"], "eyeSquintRight": ["lidTightener_R"],
    "eyeWideLeft": ["upperLidRaiser_L"], "eyeWideRight": ["upperLidRaiser_R"],
    "jawForward": ["jawThrust"], "jawLeft": ["jawSidewaysLeft"], "jawRight": ["jawSidewaysRight"],
    "jawOpen": ["jawDrop"],
    "mouthDimpleLeft": ["dimpler_L"], "mouthDimpleRight": ["dimpler_R"],
    "mouthFrownLeft": ["lipCornerDepressor_L"], "mouthFrownRight": ["lipCornerDepressor_R"],
    "mouthFunnel": ["lipFunneler_LB", "lipFunneler_LT", "lipFunneler_RB", "lipFunneler_RT"],
    "mouthLeft": ["mouthLeft"], "mouthRight": ["mouthRight"],
    "mouthLowerDownLeft": ["lowerLipDepressor_L"], "mouthLowerDownRight": ["lowerLipDepressor_R"],
    "mouthPressLeft": ["lipPressor_L"], "mouthPressRight": ["lipPressor_R"],
    "mouthPucker": ["lipPucker_L", "lipPucker_R"],
    "mouthRollLower": ["lipSuck_LB", "lipSuck_RB"], "mouthRollUpper": ["lipSuck_LT", "lipSuck_RT"],
    "mouthShrugLower": ["chinRaiser_B"], "mouthShrugUpper": ["chinRaiser_T"],
    "mouthSmileLeft": ["lipCornerPuller_L"], "mouthSmileRight": ["lipCornerPuller_R"],
    "mouthStretchLeft": ["lipStretcher_L"], "mouthStretchRight": ["lipStretcher_R"],
    "mouthUpperUpLeft": ["upperLipRaiser_L"], "mouthUpperUpRight": ["upperLipRaiser_R"],
    "noseSneerLeft": ["noseWrinkler_L"], "noseSneerRight": ["noseWrinkler_R"],
}  # Not used: _neutral, mouthClose (lips together while the jaw opens), tongueOut (MHR has no tongue shape).
REST_PERCENTILE = 20
MOUTH_GAIN = 3.0  # jawDrop per unit of lip gap over mouth width (above the rest); calibrated by --reread.
CUTOFF_HZ = {"eye": 8.0, "mouth": 6.0, "other": 4.0}
HAMPEL_S, HAMPEL_FLOOR = 0.15, 0.25


def region(name: str) -> str:
    if name.startswith("eye"):
        return "eye"
    if name == "mouthGap":
        return "mouth"
    if name.startswith(("mouth", "jaw")):
        return "mouth"
    return "other"


def mhr_expressions(scores: np.ndarray, names: list[str], gains: dict) -> np.ndarray:
    """(n, 52) ARKit scores (rest face already removed) -> (n, 72) MHR expressions (NaN rows stay NaN)."""
    out = np.zeros((len(scores), len(MHR_EXPRESSIONS)), np.float64)
    for arkit, targets in ARKIT_TO_MHR.items():
        if arkit not in names:
            continue
        for target in targets:
            out[:, MHR_EXPRESSIONS.index(target)] += scores[:, names.index(arkit)] * gains.get(target, 1.0)
    out[np.isnan(scores).any(1)] = np.nan
    return out


def smooth_scores(scores: np.ndarray, names: list[str], frames: np.ndarray, labels: np.ndarray,
                  rate: float) -> np.ndarray:
    """Hampel (not on the eyes) and RTS per run of consecutive frames of one shot; unread faces filled."""
    out = np.zeros_like(scores)
    cutoff = np.array([CUTOFF_HZ[region(n)] for n in names])
    no_hampel = np.array([region(n) == "eye" for n in names])
    breaks = np.where((np.diff(frames) != 1) | (labels[frames[1:]] != labels[frames[:-1]]))[0] + 1
    for rows in np.split(np.arange(len(frames)), breaks):
        x = scores[rows].astype(np.float64)
        read = ~np.isnan(x[:, 0])
        if read.sum() < 4:
            continue  # Too few faces read in this run: the rest face (0).
        noise = channel_noise(x[read])
        r = np.broadcast_to(noise ** 2, x.shape).copy()
        if len(rows) >= 5:
            filled = np.where(read[:, None], x, 0.0)
            outliers = hampel_mask(filled, max(3, int(round(HAMPEL_S * rate)) | 1), noise, np.full(len(names), HAMPEL_FLOOR))
            r[outliers & ~no_hampel[None] & read[:, None]] = np.inf
        # Unread face: a weak measurement of the rest face (0), so a long gap fades back to it.
        x[~read] = 0.0
        r[~read] = noise ** 2 * HIDDEN_NOISE
        q = noise ** 2 * (2 * np.pi * cutoff / rate) ** 4
        out[rows] = np.clip(rts_smooth(x, r, q), 0.0, None)
    return out


def reread(pipeline: Path, original: str, avatar: str) -> None:
    """Correlation and slope per blendshape, avatar read vs original read, on the frames read in both."""
    report, gains = {"blendshapes": {}}, {}
    pairs: dict[str, list] = {}
    for path in sorted((pipeline / avatar).glob("*.npz")):
        a, o = np.load(path), np.load(pipeline / original / path.name)
        names = list(o["names"])
        index = {int(f): i for i, f in enumerate(o["frames"])}
        rows = [(index[int(f)], j) for j, f in enumerate(a["frames"]) if int(f) in index]
        oi, ai = (np.array(v) for v in zip(*rows))
        both = ~np.isnan(o["scores"][oi, 0]) & ~np.isnan(a["scores"][ai, 0])
        for k, n in enumerate(names):
            pairs.setdefault(n, []).append(np.stack([o["scores"][oi[both], k], a["scores"][ai[both], k]], 1))
        if "mouth" in o.files and "mouth" in a.files:
            pairs.setdefault("mouthGap", []).append(np.stack([o["mouth"][oi[both]], a["mouth"][ai[both]]], 1))
        report.setdefault("frames_read_both", {})[path.stem] = int(both.sum())
        report.setdefault("avatar_read_share", {})[path.stem] = round(float(np.mean(~np.isnan(a["scores"][ai, 0]))), 3)
    for n, p in pairs.items():
        p = np.concatenate(p)
        if len(p) < 20 or p[:, 0].std() < 1e-3:
            continue
        corr = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]) if p[:, 1].std() > 1e-6 else 0.0
        slope = float(np.polyfit(p[:, 0], p[:, 1], 1)[0])
        report["blendshapes"][n] = {"corr": round(corr, 3), "slope": round(slope, 3), "std": round(float(p[:, 0].std()), 3)}
        if corr > 0.3 and slope > 0.05:
            for target in ARKIT_TO_MHR.get(n, ["jawDrop"] if n == "mouthGap" else []):
                gains[target] = round(min(4.0, 1.0 / slope), 2)
    weights = {n: v["std"] for n, v in report["blendshapes"].items()}
    total = sum(weights.values())
    report["corr_weighted_by_std"] = round(sum(v["corr"] * weights[n] for n, v in report["blendshapes"].items()) / total, 3)
    for group in ("eye", "mouth", "other"):
        values = [v["corr"] for n, v in report["blendshapes"].items() if region(n) == group]
        report[f"corr_median_{group}"] = round(float(np.median(values)), 3) if values else None
    (pipeline / f"face_reread_{avatar}.json").write_text(json.dumps(report, indent=1))
    (pipeline / f"face_gains_{avatar}.json").write_text(json.dumps(gains, indent=1))
    top = sorted(report["blendshapes"].items(), key=lambda kv: -kv[1]["std"])[:12]
    print(json.dumps({k: report[k] for k in report if k != "blendshapes"}))
    print("most moving:", {n: (v["corr"], v["slope"]) for n, v in top})
    print("REREAD_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--blendshapes", default="face_blendshapes", help="face_blendshapes.py output folder")
    parser.add_argument("--params", default="sam3d_robust", help="parameter archives to copy")
    parser.add_argument("--output", default="sam3d_face", help="output folder of the archives with expressions")
    parser.add_argument("--gains", type=Path, help="json MHR expression -> gain (from --reread)")
    parser.add_argument("--reread", help="avatar blendshapes folder (face_blendshapes.py --video): measure only")
    parser.add_argument("--cut", type=float, default=18)
    args = parser.parse_args()

    d = args.pipeline
    if args.reread:
        for path in (d / args.blendshapes, d / args.reread):
            if not path.is_dir():
                sys.exit(f"error: missing {path}")
        reread(d, args.blendshapes, args.reread)
        return
    for path in (d / args.blendshapes, d / args.params, d / "input.mp4"):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    out = d / args.output
    if out.exists() and any(out.iterdir()):
        sys.exit(f"error: {out} is not empty")
    gains = json.loads(args.gains.read_text()) if args.gains else {}
    labels = cut_labels(d / "input.mp4", args.cut)
    import cv2
    capture = cv2.VideoCapture(str(d / "input.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    out.mkdir(exist_ok=True)
    report = {"rest": {}, "active_share": {}}
    for path in sorted((d / args.params).glob("*.npz"), key=lambda p: int(p.stem)):
        with np.load(path) as npz:
            a = {k: npz[k] for k in npz.files}
        face_path = d / args.blendshapes / path.name
        if not face_path.is_file():
            sys.exit(f"error: missing {face_path}")
        face = np.load(face_path)
        names, frames, scores = list(face["names"]), face["frames"].astype(int), face["scores"].astype(np.float64)
        # The lip gap drives the jaw (jawOpen barely moves on these faces while they talk): it joins the scores as
        # a 53rd channel, smoothed with them, then replaces jawOpen's share of jawDrop.
        if "mouth" in face.files:
            names = names + ["mouthGap"]
            scores = np.concatenate([scores, face["mouth"].astype(np.float64)[:, None]], axis=1)
        rest = np.nanpercentile(scores, REST_PERCENTILE, axis=0)
        report["rest"][path.stem] = {n: round(float(v), 3) for n, v in zip(names, rest) if v > 0.05}
        relative = np.clip(scores - rest, 0.0, None)  # NaN stays NaN.
        smoothed = smooth_scores(relative, names, frames, labels, fps)
        expressions = mhr_expressions(smoothed, names, gains)
        if "mouthGap" in names:
            jaw = MHR_EXPRESSIONS.index("jawDrop")
            expressions[:, jaw] = smoothed[:, names.index("mouthGap")] * MOUTH_GAIN * gains.get("jawDrop", 1.0)
        index = {int(f): i for i, f in enumerate(frames)}
        rows = np.array([index.get(int(f), -1) for f in a["frames"]])
        a["expr_params"] = np.zeros((len(a["frames"]), len(MHR_EXPRESSIONS)), np.float32)
        a["expr_params"][rows >= 0] = expressions[rows[rows >= 0]]
        report["active_share"][path.stem] = round(float(np.mean(np.abs(a["expr_params"]).max(1) > 0.1)), 3)
        np.savez_compressed(out / path.name, **a)
        print(f"person {path.stem}: {int((rows >= 0).sum())} of {len(rows)} frames with a face", flush=True)
    (d / f"{args.output}.json").write_text(json.dumps(report, indent=1))
    print("FACE_OK")


if __name__ == "__main__":
    main()
