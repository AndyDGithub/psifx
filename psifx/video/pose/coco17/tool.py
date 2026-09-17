"""Base tool for pose models that output COCO-17 body keypoints."""

from typing import List, Tuple, Union

import json
from pathlib import Path
from tqdm import tqdm

import numpy as np

from psifx.video.pose.tool import PoseEstimationTool
from psifx.video.pose.coco17 import skeleton
from psifx.io import tar, video

# One detected person: (keypoints [17, 2] in pixels, scores [17], bbox [x1, y1, x2, y2]).
Person = Tuple[np.ndarray, np.ndarray, np.ndarray]


class Coco17PoseEstimationTool(PoseEstimationTool):
    """
    Base class for COCO-17 pose estimation tools (RTMPose, RTMO, ViTPose, ...).

    Subclasses only implement ``predict()`` on one RGB frame; this class runs it
    over the video and writes the same archive layout as the MediaPipe tool
    (one ``pose_keypoints_2d`` entry per frame, flattened ``[x, y, confidence]``),
    so a COCO-17 tool can replace MediaPipe in any psifx pipeline.

    Like MediaPipe, a single person is kept per frame: the one with the largest
    bounding box. Use a mask (``--mask`` / multi-inference) to choose which
    person is followed when several are visible.

    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def predict(self, image: np.ndarray) -> List[Person]:
        """
        Estimates the poses of all the people in one frame.

        :param image: [H, W, 3] RGB uint8 frame.
        :return: One ``(keypoints [17, 2], scores [17], bbox [4])`` tuple per person.
        """
        raise NotImplementedError

    @staticmethod
    def select_person(people: List[Person]) -> List[float]:
        """
        Keeps the person with the largest bounding box and flattens it to ``[x, y, c] * 17``.

        :param people: Output of ``predict()``.
        :return: Flattened keypoints, all zeros when nobody is detected (MediaPipe convention).
        """
        if not people:
            return [0.0] * (skeleton.N_POSE_LANDMARKS * 3)

        def area(person: Person) -> float:
            x1, y1, x2, y2 = person[2]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        keypoints, scores, _ = max(people, key=area)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if keypoints.shape != (skeleton.N_POSE_LANDMARKS, 2) or scores.shape != (skeleton.N_POSE_LANDMARKS,):
            raise ValueError(f"expected keypoints [17, 2] and scores [17], got {keypoints.shape} and {scores.shape}")
        return np.concatenate([keypoints, scores[:, None]], axis=1).flatten().tolist()

    def inference(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
    ):
        """
        Runs the pose estimation model on a video.

        :param video_path: Path to the video file.
        :param poses_path: Path to the pose archive.
        :return:
        """
        video_path = Path(video_path)
        poses_path = Path(poses_path)

        if self.verbose:
            print(f"video   =   {video_path}")
            print(f"poses   =   {poses_path}")

        # Fail before the (long) inference rather than after it.
        tar.TarWriter.check(path=poses_path, overwrite=self.overwrite)

        poses = {"edges": {"pose_keypoints_2d": skeleton.POSE_EDGES}}
        with video.VideoReader(path=video_path) as video_reader:
            for i, image in enumerate(
                tqdm(
                    video_reader,
                    desc="Processing",
                    disable=not self.verbose,
                )
            ):
                poses[f"{i: 015d}"] = {"pose_keypoints_2d": self.select_person(self.predict(image))}

        poses = {
            f"{k}.json": json.dumps(v)
            for k, v in tqdm(
                poses.items(),
                desc="Encoding",
                disable=not self.verbose,
            )
        }
        tar.TarWriter.write(
            dictionary=poses,
            path=poses_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )
