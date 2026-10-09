"""Reusable command-line interfaces for COCO-17 pose estimation tools.

Each model only subclasses these commands, setting ``tool_class`` and, if it
has options of its own, overriding ``add_model_arguments()`` / ``tool_kwargs()``.
The video, mask and output handling is identical to the MediaPipe commands.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Type

import torch

from psifx.utils.command import Command
from psifx.video.pose.coco17.tool import Coco17PoseEstimationTool
from psifx.video.tracking.tool import TrackingTool


def add_common_arguments(parser: argparse.ArgumentParser):
    """
    Adds the arguments shared by every pose inference command.

    :param parser: The argument parser.
    :return:
    """
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.1,
        help="threshold for the binarization of the segmentation mask",
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


def collect_mask_files(masks: List[Path]) -> List[Path]:
    """
    Expands mask directories into their ``.mp4`` files.

    :param masks: Mask directories or individual ``.mp4`` mask files.
    :return: The list of ``.mp4`` mask files.
    """
    mask_files = []
    for mask in masks:
        if mask.is_dir():
            dir_files = sorted(mask.iterdir())
            non_mp4_files = [str(f) for f in dir_files if not (f.is_file() and f.suffix == ".mp4")]
            if non_mp4_files:
                raise ValueError(f"Directory {mask} contains non-.mp4 files: {non_mp4_files}")
            mask_files.extend(dir_files)
        elif mask.is_file() and mask.suffix == ".mp4":
            mask_files.append(mask)
        else:
            raise FileNotFoundError(f"{mask} is not a directory or a .mp4 file")
    return mask_files


def blackout_video(args: argparse.Namespace, mask: Path, output_path: Path):
    """
    Writes a copy of the video where everything outside the mask is black.

    :param args: The arguments, providing ``video``, ``device``, ``overwrite`` and ``verbose``.
    :param mask: The ``.mp4`` mask file.
    :param output_path: Path to the masked video.
    :return:
    """
    tracking_tool = TrackingTool(device=args.device, overwrite=args.overwrite, verbose=args.verbose)
    tracking_tool.visualize(
        video_path=args.video,
        mask_paths=[mask],
        visualization_path=output_path,
        blackout=True,
        color=False,
        labels=False,
    )
    del tracking_tool


class Coco17ModelCommand(Command):
    """
    Model-specific hooks shared by the single- and multi-inference commands.
    """

    tool_class: Type[Coco17PoseEstimationTool] = Coco17PoseEstimationTool

    @classmethod
    def add_model_arguments(cls, parser: argparse.ArgumentParser):
        """
        Adds the model-specific arguments, none by default.

        :param parser: The argument parser.
        :return:
        """

    @classmethod
    def tool_kwargs(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Maps the model-specific arguments to the tool constructor's keyword arguments.

        :param args: The arguments.
        :return: Keyword arguments for ``tool_class``.
        """
        return {}

    @classmethod
    def make_tool(cls, args: argparse.Namespace) -> Coco17PoseEstimationTool:
        return cls.tool_class(
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
            **cls.tool_kwargs(args),
        )


class Coco17SingleInferenceCommand(Coco17ModelCommand):
    """
    Command-line interface for inferring the pose of one person with a COCO-17 model.
    """

    @classmethod
    def setup(cls, parser: argparse.ArgumentParser):
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
        cls.add_model_arguments(parser)
        add_common_arguments(parser)

    @classmethod
    def execute(cls, parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        tool = cls.make_tool(args)
        if args.mask is None:
            tool.inference(video_path=args.video, poses_path=args.poses)
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                masked_video = Path(tmp_dir) / args.mask.name
                blackout_video(args, args.mask, masked_video)
                tool.inference(video_path=masked_video, poses_path=args.poses)
        del tool


class Coco17MultiInferenceCommand(Coco17ModelCommand):
    """
    Command-line interface for inferring the pose of each masked person with a COCO-17 model.
    """

    @classmethod
    def setup(cls, parser: argparse.ArgumentParser):
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
            nargs="+",
            required=True,
            help="list of path to mask directories or individual .mp4 mask files",
        )

        def custom_error(message):
            parser.print_usage(sys.stderr)
            sys.stderr.write(f"{parser.prog}: error: {message}\n")
            if "masks" in message:
                sys.stderr.write("run psifx tracking Sam3 inference to get masks\n")
            parser.exit(2)

        parser.error = custom_error

        cls.add_model_arguments(parser)
        add_common_arguments(parser)

    @classmethod
    def execute(cls, parser: argparse.ArgumentParser, args: argparse.Namespace):
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

        # Validate every mask before loading the model.
        mask_files = collect_mask_files(args.masks)
        tool = cls.make_tool(args)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for mask in mask_files:
                masked_video = Path(tmp_dir) / mask.name
                blackout_video(args, mask, masked_video)
                tool.inference(
                    video_path=masked_video,
                    poses_path=args.poses_dir / f"{mask.stem}.tar.gz",
                )
        del tool
