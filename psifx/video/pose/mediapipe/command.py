"""MediaPipe pose estimation command-line interface."""

import argparse
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from pathlib import Path

import torch

from psifx.utils.command import Command, register_command
from psifx.video.pose.mediapipe.tool import MediaPipePoseEstimationTool
from psifx.utils.constants import MEDIAPIPE_HOLISTIC_PATH
from psifx.video.tracking.tool import TrackingTool


class MediaPipeCommand(Command):
    """
    Command-line interface for running MediaPipe.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        from psifx.video.pose.command import VisualizationCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "single-inference", MediaPipeSingleInferenceCommand)
        register_command(subparsers, "multi-inference", MediaPipeMultiInferenceCommand)
        register_command(subparsers, "visualization", VisualizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class MediaPipeSingleInferenceCommand(Command):
    """
    Command-line interface for inferring human pose with MediaPipe Holistic.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--video",
            type=Path,
            required=True,
            help="path to the input video file, such as ``/path/to/video.mp4`` (or .avi, .mkv, etc.)",
        )
        parser.add_argument(
            "--poses",
            type=Path,
            required=True,
            help="path to the output pose archive, such as ``/path/to/poses.tar.gz``",
        )
        parser.add_argument(
            "--mask",
            type=Path,
            help="path to the input .mp4 mask file",
        )
        parser.add_argument(
            "--mask_threshold",
            type=float,
            default=0.1,
            help="threshold for the binarization of the segmentation mask",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=MEDIAPIPE_HOLISTIC_PATH,
            help="URL or local path of the MediaPipe Holistic Landmarker ``.task`` bundle",
        )
        parser.add_argument(
            "--face_fallback",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="look for the face on a crop placed from the body pose when the Holistic model misses it:"
            " more faces, but noisier ones",
        )
        parser.add_argument(
            "--min_detection_confidence",
            type=float,
            default=0.5,
            help="minimum score of the body and face detections that start a track",
        )
        parser.add_argument(
            "--min_presence_confidence",
            type=float,
            default=0.5,
            help="minimum score of the body, face and hand landmarks to keep them",
        )
        parser.add_argument(
            "--model_complexity",
            type=int,
            default=None,
            help="ignored since the move to the MediaPipe Tasks API, which has a single Holistic model",
        )
        parser.add_argument(
            "--smooth",
            default=None,
            action=argparse.BooleanOptionalAction,
            help="ignored since the move to the MediaPipe Tasks API, whose video mode always smooths",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
        )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """

        def use_mediapipe(args, video_path):
            mediapipe_tool = MediaPipePoseEstimationTool(
                model_path=args.model_path,
                face_fallback=args.face_fallback,
                min_detection_confidence=args.min_detection_confidence,
                min_presence_confidence=args.min_presence_confidence,
                model_complexity=args.model_complexity,
                smooth=args.smooth,
                device=args.device,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            mediapipe_tool.inference(
                video_path=video_path,
                poses_path=args.poses,
            )
            del mediapipe_tool

        if args.mask is None:
            use_mediapipe(args, args.video)
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)

                tracking_tool = TrackingTool(
                    device=args.device,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                )

                tracking_tool.visualize(
                    video_path=args.video,
                    mask_paths=args.mask,
                    visualization_path=tmp_dir / args.mask.name,
                    blackout=True,
                    color=False,
                    labels=False,
                )
                del tracking_tool
                use_mediapipe(args, tmp_dir / args.mask.name)


def default_num_workers() -> int:
    """One worker per 4 cores: MediaPipe already runs a few threads per instance."""
    return max(1, (os.cpu_count() or 1) // 4)


def infer_one_person(video: Path, mask: Path, blackout_path: Path, poses_path: Path, tool_kwargs: dict):
    """Blacks out everything but one person, then runs MediaPipe on that video. Module level so it pickles."""
    TrackingTool(device=tool_kwargs["device"], overwrite=tool_kwargs["overwrite"],
                 verbose=tool_kwargs["verbose"]).visualize(
        video_path=video,
        mask_paths=[mask],
        visualization_path=blackout_path,
        blackout=True,
        color=False,
        labels=False,
    )
    MediaPipePoseEstimationTool(**tool_kwargs).inference(video_path=blackout_path, poses_path=poses_path)


class MediaPipeMultiInferenceCommand(Command):
    """
    Command-line interface for inferring human pose with MediaPipe Holistic.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--video",
            type=Path,
            required=True,
            help="path to the input video file, such as ``/path/to/video.mp4`` (or .avi, .mkv, etc.)",
        )
        parser.add_argument(
            "--poses_dir",
            type=Path,
            required=True,
            help="path to the output pose directory, such as ``/path/to/poses``",
        )
        parser.add_argument(
            "--masks",
            type=Path,
            nargs='+',
            required=True,
            help="list of path to mask directories or individual .mp4 mask files",
        )

        def custom_error(message):
            parser.print_usage(sys.stderr)
            args = {'prog': parser.prog, 'message': message}
            sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
            if 'masks' in message:
                sys.stderr.write("run psifx tracking Sam3 inference to get masks\n")
            parser.exit(2)

        parser.error = custom_error

        parser.add_argument(
            "--mask_threshold",
            type=float,
            default=0.1,
            help="threshold for the binarization of the segmentation mask",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=MEDIAPIPE_HOLISTIC_PATH,
            help="URL or local path of the MediaPipe Holistic Landmarker ``.task`` bundle",
        )
        parser.add_argument(
            "--face_fallback",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="look for the face on a crop placed from the body pose when the Holistic model misses it:"
            " more faces, but noisier ones",
        )
        parser.add_argument(
            "--min_detection_confidence",
            type=float,
            default=0.5,
            help="minimum score of the body and face detections that start a track",
        )
        parser.add_argument(
            "--min_presence_confidence",
            type=float,
            default=0.5,
            help="minimum score of the body, face and hand landmarks to keep them",
        )
        parser.add_argument(
            "--model_complexity",
            type=int,
            default=None,
            help="ignored since the move to the MediaPipe Tasks API, which has a single Holistic model",
        )
        parser.add_argument(
            "--smooth",
            default=None,
            action=argparse.BooleanOptionalAction,
            help="ignored since the move to the MediaPipe Tasks API, whose video mode always smooths",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=None,
            help="number of people processed at the same time, in separate processes"
            " (default: one per 4 CPU cores, at most one per person)",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
        )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """

        if args.poses_dir.exists() and any(args.poses_dir.iterdir()):
            if args.overwrite:
                print(f"Poses directory {args.poses_dir} is non-empty")
            else:
                raise FileExistsError(f"Poses directory {args.poses_dir} is non-empty")

        mask_files = []
        for mask in args.masks:
            if mask.is_dir():
                dir_files = list(mask.iterdir())
                non_mp4_files = [str(f) for f in dir_files if not (f.is_file() and f.suffix == '.mp4')]
                if non_mp4_files:
                    raise ValueError(f"Directory {mask} contains non-.mp4 files: {non_mp4_files}")
                mask_files.extend(dir_files)
            elif mask.is_file() and mask.suffix == '.mp4':
                mask_files.append(mask)
            else:
                raise FileNotFoundError(f"{mask} is not a directory or a .mp4 file")

        num_workers = min(args.num_workers or default_num_workers(), len(mask_files))
        tool_kwargs = dict(
            model_path=args.model_path,
            face_fallback=args.face_fallback,
            min_detection_confidence=args.min_detection_confidence,
            min_presence_confidence=args.min_presence_confidence,
            model_complexity=args.model_complexity,
            smooth=args.smooth,
            device=args.device,
            overwrite=args.overwrite,
            # Progress bars of parallel workers would interleave.
            verbose=args.verbose if num_workers == 1 else False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs = [(args.video, mask, Path(tmp_dir) / mask.name, args.poses_dir / f"{mask.stem}.tar.gz",
                     tool_kwargs) for mask in mask_files]
            if num_workers == 1:
                for job in jobs:
                    infer_one_person(*job)
                return
            # People are independent: one process each, every one writing its own files only.
            # spawn, not fork: a forked child would inherit the parent's threads and any CUDA state.
            with ProcessPoolExecutor(num_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(infer_one_person, *job): job[1] for job in jobs}
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                failed = [f for f in done if f.exception() is not None]
                if failed:
                    # Stop before starting the remaining people; running ones finish their own file.
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise RuntimeError(f"MediaPipe failed on {futures[failed[0]]}") from failed[0].exception()
                for future in futures:
                    future.result()
                    if args.verbose:
                        print(f"Poses written for {futures[future].name}")
