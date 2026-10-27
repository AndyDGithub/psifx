"""pyannote speaker diarization command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.diarization.pyannote.tool import PyannoteDiarizationTool, default_model_name


class PyannoteCommand(Command):
    """
    Command-line interface for running pyannote diarization tool.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        from psifx.audio.diarization.command import VisualizationCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", PyannoteInferenceCommand)
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


class PyannoteInferenceCommand(Command):
    """
    Command-line interface for diarizing an audio track with pyannote.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--audio",
            type=Path,
            required=True,
            help="path to the input audio file, such as ``/path/to/audio.wav``",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the output diarization file, such as ``/path/to/diarization.rttm``",
        )
        parser.add_argument(
            "--num_speakers",
            type=int,
            default=None,
            help="number of speaking participants, if ignored the model will try to"
            " guess it, it is advised to specify it",
        )
        parser.add_argument(
            "--min_speakers",
            type=int,
            default=None,
            help="lower bound on the number of speaking participants, when it is not known exactly",
        )
        parser.add_argument(
            "--max_speakers",
            type=int,
            default=None,
            help="upper bound on the number of speaking participants, when it is not known exactly",
        )
        parser.add_argument(
            "--exclusive",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="keep a single speaker at a time, which simplifies the alignment with a"
            " transcription (pyannote.audio 4 only)",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default=default_model_name(),
            help="name of the diarization model used, pyannote/speaker-diarization-community-1 with"
            " pyannote.audio 4, pyannote/speaker-diarization-3.1 otherwise",
        )
        parser.add_argument(
            "--api_token",
            type=str,
            default=None,
            help="API token for the downloading the models from HuggingFace",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
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
        tool = PyannoteDiarizationTool(
            model_name=args.model_name,
            api_token=args.api_token,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            audio_path=args.audio,
            diarization_path=args.diarization,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            exclusive=args.exclusive,
        )
        del tool
