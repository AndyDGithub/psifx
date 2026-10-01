"""RTMO pose estimation tool."""

from typing import List, Union

import cv2
import numpy as np

from psifx.video.pose.coco17.tool import Coco17PoseEstimationTool, Person, bbox_from_keypoints


class RTMOPoseEstimationTool(Coco17PoseEstimationTool):
    """
    RTMO pose estimation tool (one-stage: no separate person detector), run with ``rtmlib``.

    :param mode: rtmlib preset: 'lightweight' (RTMO-s), 'balanced' (RTMO-m) or 'performance' (RTMO-l).
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )
        import onnxruntime
        from rtmlib import Body

        # onnxruntime silently falls back to CPU when its CUDA provider is missing
        # (CPU-only wheel, or cuDNN not found): say it rather than hide it.
        if device != "cpu" and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            print(f"onnxruntime has no CUDA provider, running RTMO on CPU instead of {device}.")
            device = "cpu"
        # pose="rtmo" makes Body load only the RTMO model, without the YOLOX detector.
        # The ONNX weights are downloaded to ~/.cache/rtmlib on first use.
        self.model = Body(pose="rtmo", mode=mode, backend="onnxruntime", device=device)

    def predict(self, image: np.ndarray) -> List[Person]:
        """
        Estimates the poses of all the people in one frame.

        :param image: [H, W, 3] RGB uint8 frame.
        :return: One ``(keypoints [17, 2], scores [17], bbox [4])`` tuple per person.
        """
        # rtmlib follows OpenCV's BGR convention.
        keypoints, scores = self.model(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        people = []
        for kp, sc in zip(keypoints, scores):
            kp = np.asarray(kp)[:17]
            # RTMO's rtmlib wrapper returns no boxes: use the keypoints' extent.
            people.append((kp, np.asarray(sc)[:17], bbox_from_keypoints(kp)))
        return people
