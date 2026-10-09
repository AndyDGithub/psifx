"""ViTPose pose estimation command-line interface."""

import argparse
from typing import Any, Dict

from psifx.utils.command import Command, register_command
from psifx.video.pose.coco17.command import (
    Coco17ModelCommand,
    Coco17MultiInferenceCommand,
    Coco17SingleInferenceCommand,
)
from psifx.video.pose.vitpose.tool import ViTPosePoseEstimationTool


class ViTPoseCommand(Command):
    """
    Command-line interface for running ViTPose.
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

        register_command(subparsers, "single-inference", ViTPoseSingleInferenceCommand)
        register_command(subparsers, "multi-inference", ViTPoseMultiInferenceCommand)
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


class ViTPoseModelCommand(Coco17ModelCommand):
    """
    ViTPose-specific arguments.
    """

    tool_class = ViTPosePoseEstimationTool

    @classmethod
    def add_model_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model_name",
            type=str,
            default="usyd-community/vitpose-base-simple",
            help="Hugging Face ViTPose checkpoint",
        )
        parser.add_argument(
            "--detector_name",
            type=str,
            default="PekingU/rtdetr_r50vd_coco_o365",
            help="Hugging Face RT-DETR checkpoint used to detect people",
        )
        parser.add_argument(
            "--detection_threshold",
            type=float,
            default=0.3,
            help="minimum detector score for a person box",
        )

    @classmethod
    def tool_kwargs(cls, args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "model_name": args.model_name,
            "detector_name": args.detector_name,
            "detection_threshold": args.detection_threshold,
        }


class ViTPoseSingleInferenceCommand(ViTPoseModelCommand, Coco17SingleInferenceCommand):
    """
    Command-line interface for inferring the pose of one person with ViTPose.
    """


class ViTPoseMultiInferenceCommand(ViTPoseModelCommand, Coco17MultiInferenceCommand):
    """
    Command-line interface for inferring the pose of each masked person with ViTPose.
    """
