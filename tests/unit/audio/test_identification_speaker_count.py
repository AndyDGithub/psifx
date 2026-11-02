"""More speakers in the diarization than mono audios must stop with a clear error, before any embedding."""

import wave

import pytest

pytest.importorskip("pyannote.audio")
from psifx.audio.identification.pyannote.tool import PyannoteIdentificationTool


def write_silence(path, seconds=1.0, rate=16000):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * int(seconds * rate))


@pytest.mark.unit
def test_more_speakers_than_mono_audios(tmp_path):
    for name in ("Audio.wav", "Left.wav", "Right.wav"):
        write_silence(tmp_path / name)
    rttm = tmp_path / "Diarization.rttm"
    rttm.write_text(
        "".join(
            f"SPEAKER Audio 1 {start:.3f} 0.500 <NA> <NA> SPEAKER_0{k} <NA> <NA>\n"
            for k, start in enumerate((0.0, 0.2, 0.4))
        )
    )
    # Skips the model loading: the check must come before the models are used.
    tool = PyannoteIdentificationTool.__new__(PyannoteIdentificationTool)
    tool.overwrite, tool.verbose, tool.models = False, False, {}

    with pytest.raises(ValueError, match="found 3 speakers but 2 mono audios"):
        tool.inference(
            mixed_audio_path=tmp_path / "Audio.wav",
            diarization_path=rttm,
            mono_audio_paths=[tmp_path / "Left.wav", tmp_path / "Right.wav"],
            identification_path=tmp_path / "Identification.json",
        )
    assert not (tmp_path / "Identification.json").exists()
