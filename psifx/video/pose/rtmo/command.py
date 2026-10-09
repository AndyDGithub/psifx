"""RTMO pose estimation command-line interface."""

import argparse
from typing import Any, Dict

from psifx.utils.command import Command, register_command
from psifx.video.pose.coco17.command import (
    Coco17ModelCommand,
    Coco17MultiInferenceCommand,
    Coco17SingleInferenceCommand,
)
from psifx.video.pose.rtmo.tool import RTMOPoseEstimationTool


class RTMOCommand(Command):
    """
    Command-line interface for running RTMO.
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

        register_command(subparsers, "single-inference", RTMOSingleInferenceCommand)
        register_command(subparsers, "multi-inference", RTMOMultiInferenceCommand)
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


class RTMOModelCommand(Coco17ModelCommand):
    """
    RTMO-specific arguments.
    """

    tool_class = RTMOPoseEstimationTool

    @classmethod
    def add_model_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--mode",
            type=str,
            default="balanced",
            choices=["lightweight", "balanced", "performance"],
            help="rtmlib preset: RTMO-s, RTMO-m or RTMO-l",
        )
        parser.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp32", "fp16"],
            help="model precision: fp16 needs CUDA, same accuracy and about 40%% faster",
        )

    @classmethod
    def tool_kwargs(cls, args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "mode": args.mode,
            "precision": args.precision,
        }


class RTMOSingleInferenceCommand(RTMOModelCommand, Coco17SingleInferenceCommand):
    """
    Command-line interface for inferring the pose of one person with RTMO.
    """


class RTMOMultiInferenceCommand(RTMOModelCommand, Coco17MultiInferenceCommand):
    """
    Command-line interface for inferring the pose of each masked person with RTMO.
    """
