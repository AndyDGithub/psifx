"""ViTPose pose estimation tool."""

from typing import List, Union

import numpy as np
import torch

from psifx.video.pose.coco17.tool import Coco17PoseEstimationTool, Person

# COCO class index of "person" in RT-DETR's label map.
PERSON_LABEL = 0


class ViTPosePoseEstimationTool(Coco17PoseEstimationTool):
    """
    ViTPose pose estimation tool, run with ``transformers``.

    ViTPose is top-down: it needs person boxes, which come from an RT-DETR
    detector (also from ``transformers``, Apache 2.0).

    :param model_name: Hugging Face ViTPose checkpoint.
    :param detector_name: Hugging Face RT-DETR checkpoint used to find people.
    :param detection_threshold: Minimum detector score for a person box.
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        model_name: str = "usyd-community/vitpose-base-simple",
        detector_name: str = "PekingU/rtdetr_r50vd_coco_o365",
        detection_threshold: float = 0.3,
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError(f"detection_threshold must be in [0, 1], got {detection_threshold}")
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

        self.detection_threshold = detection_threshold
        self.detector_processor = AutoProcessor.from_pretrained(detector_name)
        self.detector = RTDetrForObjectDetection.from_pretrained(detector_name).to(device).eval()
        self.pose_processor = AutoProcessor.from_pretrained(model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the people in one frame.

        :param image: [H, W, 3] RGB uint8 frame.
        :return: [N, 4] person boxes as ``[x1, y1, x2, y2]``.
        """
        h, w = image.shape[:2]
        inputs = self.detector_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        result = self.detector_processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=self.detection_threshold
        )[0]
        boxes = result["boxes"][result["labels"] == PERSON_LABEL]
        return boxes.cpu().numpy()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> List[Person]:
        """
        Estimates the poses of all the people in one frame.

        :param image: [H, W, 3] RGB uint8 frame.
        :return: One ``(keypoints [17, 2], scores [17], bbox [4])`` tuple per person.
        """
        boxes_xyxy = self.detect(image)
        if len(boxes_xyxy) == 0:
            return []
        # The ViTPose processor expects COCO-style [x, y, width, height] boxes.
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]

        inputs = self.pose_processor(image, boxes=[boxes_xywh], return_tensors="pt").to(self.device)
        outputs = self.pose_model(**inputs)
        results = self.pose_processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
        return [
            (r["keypoints"].cpu().numpy()[:17], r["scores"].cpu().numpy()[:17], box)
            for r, box in zip(results, boxes_xyxy)
        ]
