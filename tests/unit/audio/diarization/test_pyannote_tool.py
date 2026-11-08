"""Unit tests for the pyannote diarization tool across pyannote.audio 3 and 4 (no model download)."""

from dataclasses import dataclass
import os
import subprocess
import sys

import numpy as np

import pytest
from pyannote.core import Annotation, Segment
from scipy.io import wavfile

from psifx.audio.diarization.pyannote.tool import PyannoteDiarizationTool
from psifx.io import rttm
from psifx.utils.huggingface import pyannote_token_kwargs


def annotation(*turns) -> Annotation:
    out = Annotation()
    for start, end, speaker in turns:
        out[Segment(start, end)] = speaker
    return out


@dataclass
class DiarizeOutput:
    """Shape of the pyannote.audio 4 pipeline output."""

    speaker_diarization: Annotation
    exclusive_speaker_diarization: Annotation


REGULAR = annotation((0.0, 2.0, "A"), (1.5, 3.0, "B"))
EXCLUSIVE = annotation((0.0, 1.5, "A"), (1.5, 3.0, "B"))


@pytest.mark.unit
def test_annotation_from_pyannote3_output():
    assert PyannoteDiarizationTool._annotation(REGULAR, exclusive=False) is REGULAR
    with pytest.raises(ValueError, match="pyannote.audio 4"):
        PyannoteDiarizationTool._annotation(REGULAR, exclusive=True)


@pytest.mark.unit
def test_annotation_from_pyannote4_output():
    output = DiarizeOutput(REGULAR, EXCLUSIVE)
    assert PyannoteDiarizationTool._annotation(output, exclusive=False) is REGULAR
    assert PyannoteDiarizationTool._annotation(output, exclusive=True) is EXCLUSIVE


@pytest.mark.unit
def test_token_keyword_matches_installed_pyannote():
    import pyannote.audio

    expected = "token" if pyannote.audio.__version__.startswith("4") else "use_auth_token"
    assert pyannote_token_kwargs("hf_x") == {expected: "hf_x"}


@pytest.mark.unit
def test_inference_writes_exclusive_turns(tmp_path):
    calls = {}

    def pipeline(file, **kwargs):
        calls.update(kwargs)
        return DiarizeOutput(REGULAR, EXCLUSIVE)

    tool = PyannoteDiarizationTool.__new__(PyannoteDiarizationTool)
    tool.model, tool.overwrite, tool.verbose = pipeline, False, False
    audio = tmp_path / "audio.wav"
    wavfile.write(audio, 16000, np.zeros(16000 * 3, dtype=np.int16))
    tool.inference(audio, tmp_path / "out.rttm", min_speakers=1, max_speakers=3, exclusive=True)

    assert calls == {"num_speakers": None, "min_speakers": 1, "max_speakers": 3}
    turns = [(s["start"], s["duration"], s["speaker_name"]) for s in rttm.RTTMReader.read(tmp_path / "out.rttm", verbose=False)]
    assert turns == [(0.0, 1.5, "A"), (1.5, 1.5, "B")]


@pytest.mark.unit
def test_psifx_turns_pyannote_telemetry_off_by_default():
    # Fresh interpreter without the variable: importing psifx first must keep pyannote's metrics off.
    env = {k: v for k, v in os.environ.items() if k != "PYANNOTE_METRICS_ENABLED"}
    code = (
        "import os, psifx, pyannote.audio\n"
        "from importlib.util import find_spec\n"
        "if find_spec('pyannote.audio.telemetry'):\n"
        "    from pyannote.audio.telemetry.metrics import is_metrics_enabled\n"
        "    assert not is_metrics_enabled()\n"
        "assert os.environ['PYANNOTE_METRICS_ENABLED'] == '0'\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@pytest.mark.unit
def test_default_model_follows_pyannote_major_version(monkeypatch):
    from psifx.audio.diarization.pyannote import tool

    monkeypatch.setattr(tool, "version", lambda name: "4.0.7")
    assert tool.default_model_name() == "pyannote/speaker-diarization-community-1"
    monkeypatch.setattr(tool, "version", lambda name: "3.3.2")
    assert tool.default_model_name() == "pyannote/speaker-diarization-3.1"
