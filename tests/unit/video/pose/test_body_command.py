"""Unit tests for psifx video pose body: argument parsing and model dispatch (no model loaded)."""

import argparse

import pytest

from psifx.video.pose.body import command as body_command
from psifx.video.pose.body.command import BodySingleInferenceCommand


def parse(*argv):
    parser = argparse.ArgumentParser()
    BodySingleInferenceCommand.setup(parser)
    return parser.parse_args(["--video", "v.mp4", "--poses", "p.tar.gz", "--device", "cpu", *argv])


class FakeTool:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_tools(monkeypatch):
    """Replace both model tools by a recorder, so make_tool loads nothing."""
    import psifx.video.pose.rtmo.tool as rtmo_tool
    import psifx.video.pose.vitpose.tool as vitpose_tool

    rtmo = type("FakeRTMO", (FakeTool,), {})
    vitpose = type("FakeViTPose", (FakeTool,), {})
    monkeypatch.setattr(rtmo_tool, "RTMOPoseEstimationTool", rtmo)
    monkeypatch.setattr(vitpose_tool, "ViTPosePoseEstimationTool", vitpose)
    return rtmo, vitpose


@pytest.mark.unit
def test_rtmo_is_the_default(fake_tools):
    tool = BodySingleInferenceCommand.make_tool(parse())
    assert isinstance(tool, fake_tools[0])
    assert tool.kwargs["mode"] == "balanced" and tool.kwargs["precision"] == "fp32"


@pytest.mark.unit
def test_vitpose_option(fake_tools):
    tool = BodySingleInferenceCommand.make_tool(parse("--model", "vitpose", "--detection_threshold", "0.5"))
    assert isinstance(tool, fake_tools[1])
    assert tool.kwargs["detection_threshold"] == 0.5
    assert "precision" not in tool.kwargs


@pytest.mark.unit
def test_rtmo_only_option_rejected_for_vitpose(fake_tools):
    with pytest.raises(ValueError, match="--precision"):
        BodySingleInferenceCommand.make_tool(parse("--model", "vitpose", "--precision", "fp16"))


@pytest.mark.unit
def test_unknown_model_rejected():
    with pytest.raises(SystemExit):
        parse("--model", "mediapipe")


@pytest.mark.unit
def test_missing_rtmlib_suggests_vitpose(monkeypatch):
    import psifx.video.pose.rtmo.tool as rtmo_tool

    def missing(**kwargs):
        raise ImportError("RTMO needs the optional rtmlib dependencies: pip install 'psifx[rtm]'.")

    monkeypatch.setattr(rtmo_tool, "RTMOPoseEstimationTool", missing)
    with pytest.raises(ImportError, match="--model vitpose"):
        BodySingleInferenceCommand.make_tool(parse())


@pytest.mark.unit
def test_models_list_matches_choices():
    assert body_command.DEFAULT_MODEL in body_command.MODELS
