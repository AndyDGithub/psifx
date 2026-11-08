"""Body pose estimation command-line interface, with a choice between RTMO and ViTPose."""

import argparse
from typing import Any, Dict

from psifx.utils.command import Command, register_command
from psifx.video.pose.coco17.command import (
    Coco17ModelCommand,
    Coco17MultiInferenceCommand,
    Coco17SingleInferenceCommand,
)
from psifx.video.pose.coco17.tool import Coco17PoseEstimationTool

MODELS = ("rtmo", "vitpose")
DEFAULT_MODEL = "rtmo"


class BodyCommand(Command):
    """
    Command-line interface for estimating 17-keypoint body poses with RTMO (default) or ViTPose.
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

        register_command(subparsers, "single-inference", BodySingleInferenceCommand)
        register_command(subparsers, "multi-inference", BodyMultiInferenceCommand)
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


class BodyModelCommand(Coco17ModelCommand):
    """
    Model choice and model-specific arguments.
    """

    @classmethod
    def add_model_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model",
            type=str,
            default=DEFAULT_MODEL,
            choices=MODELS,
            help="pose model: 'rtmo' (fast, needs psifx[rtm]) or 'vitpose' (most accurate, no extra dependency)",
        )
        rtmo = parser.add_argument_group("RTMO options (--model rtmo)")
        rtmo.add_argument(
            "--mode",
            type=str,
            default="balanced",
            choices=["lightweight", "balanced", "performance"],
            help="rtmlib preset: RTMO-s, RTMO-m or RTMO-l",
        )
        rtmo.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp32", "fp16"],
            help="model precision: fp16 needs CUDA, same accuracy and about 40%% faster",
        )
        vitpose = parser.add_argument_group("ViTPose options (--model vitpose)")
        vitpose.add_argument(
            "--model_name",
            type=str,
            default="usyd-community/vitpose-base-simple",
            help="Hugging Face ViTPose checkpoint",
        )
        vitpose.add_argument(
            "--detector_name",
            type=str,
            default="PekingU/rtdetr_r50vd_coco_o365",
            help="Hugging Face RT-DETR checkpoint used to detect people",
        )
        vitpose.add_argument(
            "--detection_threshold",
            type=float,
            default=0.3,
            help="minimum detector score for a person box",
        )

    @classmethod
    def tool_kwargs(cls, args: argparse.Namespace) -> Dict[str, Any]:
        if args.model == "rtmo":
            return {"mode": args.mode, "precision": args.precision}
        # Fail before loading anything: these options would be silently ignored otherwise.
        if args.precision != "fp32":
            raise ValueError("--precision only applies to --model rtmo")
        return {
            "model_name": args.model_name,
            "detector_name": args.detector_name,
            "detection_threshold": args.detection_threshold,
        }

    @classmethod
    def make_tool(cls, args: argparse.Namespace) -> Coco17PoseEstimationTool:
        kwargs = cls.tool_kwargs(args)
        # Tools are imported here so that each model's dependencies are only needed when it is used.
        if args.model == "rtmo":
            from psifx.video.pose.rtmo.tool import RTMOPoseEstimationTool as tool_class
        else:
            from psifx.video.pose.vitpose.tool import ViTPosePoseEstimationTool as tool_class
        try:
            return tool_class(device=args.device, overwrite=args.overwrite, verbose=args.verbose, **kwargs)
        except ImportError as error:
            if args.model != "rtmo":
                raise
            raise ImportError(f"{error} Or use --model vitpose, which needs no extra dependency.") from error


class BodySingleInferenceCommand(BodyModelCommand, Coco17SingleInferenceCommand):
    """
    Command-line interface for inferring the body pose of one person with RTMO (default) or ViTPose.
    """


class BodyMultiInferenceCommand(BodyModelCommand, Coco17MultiInferenceCommand):
    """
    Command-line interface for inferring the body pose of each masked person with RTMO (default) or ViTPose.
    """
