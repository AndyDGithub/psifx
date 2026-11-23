"""Can a standard pose tool still read the body on the avatar video? ViTPose on each video, against the annotations.

For each annotated frame of the talk-show, the frame is read from each video (the original and the renders of
``render_avatars.py``) and ViTPose runs on the annotated people's boxes (the same boxes for every video: only the
reading of the pose is measured, not the detection). The keypoints are compared with the annotations as in the
other talk-show measures (OKS, PCK, wrists).

The original gives the reference. An avatar video close to it keeps what a pose tool, and probably an observer,
can read of the body; the gap is what the avatar loses.

    python research/scene3d/utility_pose.py .e2e_outputs/extrait_talkshow/pipeline \\
        --annotations .e2e_outputs/extrait_talkshow/pose_annotation \\
        --videos input.mp4 avatars_plate_own.mp4 avatars_plate_neutral.mp4 avatars_grey_neutral.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "pose_annotation"))
from evaluate import evaluate, load_ground_truth  # noqa: E402


@torch.inference_mode()
def vitpose(processor, model, image: np.ndarray, boxes: list, device: str) -> list[np.ndarray]:
    if not boxes:
        return []
    inputs = processor(image, boxes=[np.array(boxes, np.float32)], return_tensors="pt").to(device)
    results = processor.post_process_pose_estimation(model(**inputs), boxes=[np.array(boxes, np.float32)])[0]
    return [np.concatenate([r["keypoints"].cpu().numpy()[:17], r["scores"].cpu().numpy()[:17, None]], axis=1)
            for r in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with the videos")
    parser.add_argument("--annotations", type=Path, required=True, help="folder with frames.csv and ground_truth.json")
    parser.add_argument("--videos", nargs="+", required=True, help="video files in <pipeline>, the original first")
    parser.add_argument("--model", default="usyd-community/vitpose-base-simple")
    parser.add_argument("--output", type=Path, help="default: <pipeline>/utility_pose.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    truth_path, frames_path = args.annotations / "ground_truth.json", args.annotations / "frames.csv"
    videos = [args.pipeline / v for v in args.videos]
    for path in (truth_path, frames_path, *videos):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    coco = json.loads(truth_path.read_text())
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    boxes: dict[str, list] = {name: [] for name in names.values()}
    for a in coco["annotations"]:
        # COCO [x, y, width, height]; a flat box breaks ViTPose's crop and is skipped (its person counts as missed,
        # the same for every video).
        if a["num_keypoints"] > 0 and min(a["bbox"][2:]) >= 2:
            boxes[names[a["image_id"]]].append(a["bbox"])
    frame_of = {}
    for line in frames_path.read_text().splitlines()[1:]:
        name, frame, _ = line.split(",")
        frame_of[name] = int(frame)
    truth = load_ground_truth(truth_path)

    from transformers import AutoProcessor, VitPoseForPoseEstimation
    processor = AutoProcessor.from_pretrained(args.model)
    model = VitPoseForPoseEstimation.from_pretrained(args.model).to(args.device).eval()

    report = {"model": args.model, "videos": {}}
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        predictions = {}
        for name, frame in frame_of.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)  # Exact seeks on these files (checked earlier).
            ok, image = capture.read()
            if not ok:
                sys.exit(f"error: cannot read frame {frame} of {video}")
            predictions[name] = vitpose(processor, model, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes[name],
                                        args.device)
        capture.release()
        result = evaluate(predictions, truth, min_score=0.3)
        per_keypoint = next(v for k, v in result.items() if k.endswith("_per_keypoint"))
        entry = {"oks": result["oks"], "pck@0.05": result["pck@0.05"], "pck@0.1": result["pck@0.1"],
                 "wrists_pck": round((per_keypoint["left_wrist"] + per_keypoint["right_wrist"]) / 2, 3),
                 "head_pck": round(np.mean([per_keypoint[k] for k in ("nose", "left_eye", "right_eye")]), 3)}
        report["videos"][video.name] = entry
        print(video.name, json.dumps(entry), flush=True)
    (args.output or args.pipeline / "utility_pose.json").write_text(json.dumps(report, indent=1))
    print("UTILITY_OK")


if __name__ == "__main__":
    main()
