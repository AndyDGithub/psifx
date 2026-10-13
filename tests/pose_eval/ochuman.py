"""COCO keypoint AP of a psifx pose tool on OCHuman, a dataset of heavily occluded people.

OCHuman (Zhang et al., "Pose2Seg", CVPR 2019) uses the same 17 keypoints as COCO, but its
images show several people who hide each other, and the RTM models' training mix (body7)
does not contain it. It is not redistributable here: download it once from
https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman and point ``OCHUMAN_ROOT``
at the folder holding ``images/`` and the COCO-format annotation files
(``ochuman_coco_format_{val,test}_range_0.00_1.00.json``).

Every person of every image is evaluated with the official COCO evaluation (pycocotools),
so the scores are comparable to published OCHuman results. The tools are called through
``predict()``, which returns all people, because the CLI keeps a single person per video.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

ENV_VAR = "OCHUMAN_ROOT"
SPLITS = {
    "val": "ochuman_coco_format_val_range_0.00_1.00.json",
    "test": "ochuman_coco_format_test_range_0.00_1.00.json",
}


def find_split(split: str, root: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Returns the image folder and annotation file of an OCHuman split.

    :param split: 'val' or 'test'.
    :param root: Dataset folder; defaults to ``$OCHUMAN_ROOT``.
    :return: ``(images_dir, annotations_path)``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split}")
    root = root or os.environ.get(ENV_VAR)
    # An empty value must not fall back to the current directory (Path("") is ".").
    if not root or not Path(root).is_dir():
        raise FileNotFoundError(f"set {ENV_VAR} to the OCHuman folder (see tests/pose_eval/ochuman.py)")
    root = Path(root)
    annotations = root / SPLITS[split]
    images = root / "images"
    for path in (annotations, images):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
    return images, annotations


def predictions_to_coco(image_id: int, people) -> List[Dict]:
    """
    Converts ``predict()`` output to COCO keypoint results.

    :param image_id: COCO image id.
    :param people: ``[(keypoints [17, 2], scores [17], bbox [4]), ...]``.
    :return: One COCO result per person; the person score is the mean keypoint score.
    """
    results = []
    for keypoints, scores, _ in people:
        keypoints = np.asarray(keypoints, dtype=float).reshape(17, 2)
        scores = np.asarray(scores, dtype=float).reshape(17)
        flat = np.concatenate([keypoints, scores[:, None]], axis=1).flatten()
        results.append({
            "image_id": int(image_id),
            "category_id": 1,
            "keypoints": [round(v, 2) for v in flat.tolist()],
            "score": float(np.mean(scores)),
        })
    return results


def evaluate_tool(
    predict: Callable[[np.ndarray], list],
    images_dir: Path,
    annotations: Path,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """
    Runs ``predict`` on every image of a split and computes the COCO keypoint metrics.

    :param predict: A pose tool's ``predict`` (RGB uint8 image -> people).
    :param images_dir: Folder of the OCHuman images.
    :param annotations: COCO-format annotation file of the split.
    :param limit: Only use the first ``limit`` images (sorted by id), for quick runs.
    :return: AP, AP50, AP75, AR (COCO keypoint evaluation) and the number of images.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(annotations))
    image_ids = sorted(gt.getImgIds())[:limit]
    results = []
    for image_id in image_ids:
        info = gt.loadImgs(image_id)[0]
        image = np.asarray(Image.open(images_dir / info["file_name"]).convert("RGB"))
        results.extend(predictions_to_coco(image_id, predict(image)))
    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0, "n_images": len(image_ids)}

    evaluator = COCOeval(gt, gt.loadRes(results), iouType="keypoints")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {"AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]), "AR": float(stats[5]),
            "n_images": len(image_ids)}
