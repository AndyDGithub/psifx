"""Spike, no GPU: what if SAM 3D Body only ran on key frames? Uses the night-run archives (one frame out of 2) as the
model's answer: hide frames, interpolate them inside each shot, compare with what the model said and with the annotations.
Adaptive: an interval whose picture changed too much (person box, mean gray difference) is computed by the model."""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(R), str(R/"research/pose_annotation")]
from evaluate import evaluate, load_ground_truth
if len(sys.argv) != 2:
    sys.exit("usage: keyframe_simulation.py <folder with pipeline/ and pose_annotation/>")
B = Path(sys.argv[1]); P = B/"pipeline"; A = B/"pose_annotation"
C17 = [0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14]; WRISTS = [62, 41]
W, H, SCALE = 320, 180, 1920/320
p = subprocess.Popen(["ffmpeg","-v","error","-i",str(P/"input.mp4"),"-vf",f"scale={W}:{H}","-f","rawvideo","-pix_fmt","gray","-"],stdout=subprocess.PIPE)
frames = []
while len(b := p.stdout.read(W*H)) == W*H: frames.append(np.frombuffer(b, np.uint8).reshape(H, W))
video = np.stack(frames); n = len(video); print("frames", n, flush=True)
diff = np.abs(video[1:].astype(np.int16) - video[:-1]).mean(axis=(1, 2))
cuts = np.r_[0, np.where(diff > 18)[0] + 1, n]; print("cuts", len(cuts) - 2)
shot_of = np.searchsorted(cuts, np.arange(n), side="right")
people = {f.stem: np.load(f) for f in sorted((P/"sam3d_body").glob("*.npz"))}
data = {k: dict(f=a["frames"], k2=a["keypoints_2d"], k3=a["keypoints_3d"]) for k, a in people.items()}
def segments(f):
    """Runs of consecutive processed frames (step 2) inside one shot."""
    breaks = np.where((np.diff(f) != 2) | (shot_of[f[1:]] != shot_of[f[:-1]]))[0] + 1
    return np.split(np.arange(len(f)), breaks)
truth = load_ground_truth(A/"ground_truth.json")
rows = list(csv.DictReader(open(A/"frames.csv")))
def oks(interp2d):
    pred = {}
    for r in rows:
        fr = int(r["frame"]); people_ = []
        for k, d in data.items():
            i = np.searchsorted(d["f"], fr)
            if i >= len(d["f"]) or d["f"][i] != fr:
                i = np.searchsorted(d["f"], fr-1)
                if i >= len(d["f"]) or d["f"][i] != fr-1: continue
            people_.append(np.c_[interp2d[k][i][C17].astype(float), np.ones(17)])
        pred[r["file"]] = people_
    e = evaluate(pred, truth, 0.3); return float(e["oks"]), float(e["pck@0.05"])
def run(K, tau=None):
    """K: spacing of key frames, in video frames. tau: picture change above which the model computes the interval."""
    out2d = {k: d["k2"].copy() for k, d in data.items()}; computed = total = 0
    err3, errw, err2 = [], [], []
    for k, d in data.items():
        f, k2, k3 = d["f"], d["k2"], d["k3"]
        rel = k3 - ((k3[:, 9] + k3[:, 10]) / 2)[:, None]
        for seg in segments(f):
            total += len(seg)
            step = max(K // 2, 1)
            keys = list(seg[::step])
            if keys[-1] != seg[-1]: keys.append(seg[-1])
            computed += len(keys)
            for a, b in zip(keys, keys[1:]):
                if b - a < 2: continue
                if tau is not None:
                    pts = np.r_[k2[a], k2[b]] / SCALE
                    x0, y0 = np.maximum(pts.min(0).astype(int), 0); x1, y1 = pts.max(0).astype(int) + 1
                    crop = (video[f[a], y0:y1, x0:x1].astype(np.int16) - video[f[b], y0:y1, x0:x1])
                    if crop.size == 0 or np.abs(crop).mean() > tau:
                        computed += b - a - 1; continue
                w = ((np.arange(a+1, b) - a) / (b - a))[:, None, None]
                est3 = rel[a] * (1 - w) + rel[b] * w
                est2 = k2[a] * (1 - w) + k2[b] * w
                out2d[k][a+1:b] = est2
                e3 = np.linalg.norm(est3 - rel[a+1:b], axis=2) * 1000
                err3.append(e3[:, C17].ravel()); errw.append(e3[:, WRISTS].ravel())
                diag = np.linalg.norm(k2[a+1:b].max(1) - k2[a+1:b].min(1), axis=1, keepdims=True)
                err2.append((np.linalg.norm(est2 - k2[a+1:b], axis=2) / diag)[:, C17].ravel())
    cat = lambda v: np.concatenate(v) if v else np.zeros(1)
    e3, ew, e2 = cat(err3), cat(errw), cat(err2)
    o, pck = oks(out2d)
    return dict(K=K, tau=tau, computed=round(computed/total, 3), speedup=round(total/computed, 2), e3d_mm_med=round(float(np.median(e3)), 1),
                e3d_mm_p95=round(float(np.percentile(e3, 95)), 1), wrist_mm_med=round(float(np.median(ew)), 1), wrist_mm_p95=round(float(np.percentile(ew, 95)), 1),
                e2d_diag_p95=round(float(np.percentile(e2, 95)), 4), oks=round(o, 4), pck05=round(pck, 4))
# Picture changes between key frames, to choose thresholds from the data.
deltas = []
for k, d in data.items():
    for seg in segments(d["f"]):
        for a, b in zip(seg[::5], seg[5::5]):
            pts = np.r_[d["k2"][a], d["k2"][b]] / SCALE
            x0, y0 = np.maximum(pts.min(0).astype(int), 0); x1, y1 = pts.max(0).astype(int) + 1
            c = video[d["f"][a], y0:y1, x0:x1].astype(np.int16) - video[d["f"][b], y0:y1, x0:x1]
            if c.size: deltas.append(np.abs(c).mean())
q = np.percentile(deltas, [50, 75, 90]); print("delta p50/75/90 for 10-frame gaps", np.round(q, 2), flush=True)
res = [run(2)] + [run(K) for K in (4, 6, 10, 20, 40)]
for K in (10, 20, 40):
    res += [run(K, float(t)) for t in q]
for r in res: print(r, flush=True)
json.dump(res, open(B/"keyframe_spike.json", "w"), indent=1)
