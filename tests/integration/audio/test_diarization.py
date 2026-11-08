import os
from pathlib import Path

import pytest
from huggingface_hub import HfApi

from tests.integration.conftest import run_command


def _require_hf_model_access(*model_ids: str):
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not available")

    api = HfApi(token=token)
    for model_id in model_ids:
        repo_id = model_id.split("@", 1)[0]
        try:
            api.model_info(repo_id=repo_id)
        except Exception as exc:
            pytest.skip(f"HF_TOKEN lacks access to required model '{repo_id}': {exc}")


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not available")
@pytest.mark.integration
def test_audio_diarization(audio_path: Path, output_dir: Path, diarization_rttm: Path):
    """Test audio diarization."""

    pytest.importorskip("pyannote.audio", reason="Pyannote not installed")
    from psifx.audio.diarization.pyannote.tool import default_model_name

    _require_hf_model_access(default_model_name())

    diarization_path = output_dir / "diarization.rttm"

    try:
        run_command(
            "psifx", "audio", "diarization", "pyannote", "inference",
            "--audio", audio_path,
            "--diarization", diarization_path
        )
    except PermissionError as exc:
        pytest.skip(str(exc))

    assert audio_path.exists(), "Audio extraction failed"
    assert diarization_path.exists(), "Audio diarization failed"

    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def load_rttm(path: Path) -> Annotation:
        annotation = Annotation()
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            assert len(parts) == 10, f"Abnormal rttm line: {parts}"
            start, duration = float(parts[3]), float(parts[4])
            annotation[Segment(start, start + duration)] = parts[7]
        return annotation

    # Speaker names are mapped optimally, and recent models split turns at pauses that the
    # reference counts as speech: compare with the diarization error rate, not turn by turn.
    der = DiarizationErrorRate(collar=0.25)(load_rttm(diarization_rttm), load_rttm(diarization_path))
    assert der < 0.10, f"Diarization error rate {der:.3f} against the reference"
