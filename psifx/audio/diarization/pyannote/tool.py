"""pyannote speaker diarization tool."""
from collections import OrderedDict, defaultdict
from importlib.metadata import version
import os
from typing import Optional, Union
from typing import Any

from pathlib import Path
from tqdm import tqdm

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from psifx.audio.diarization.tool import DiarizationTool
from psifx.io import rttm, wav
from psifx.utils.huggingface import patch_hf_hub_download_use_auth_token, pyannote_token_kwargs


def default_model_name() -> str:
    """
    community-1 (2025) with pyannote.audio 4, which needs it; the legacy 3.1 pipeline otherwise.

    community-1 counts speakers better in noise and on films (see research/08_diarization_community1.md).
    """
    if int(version("pyannote.audio").split(".")[0]) >= 4:
        return "pyannote/speaker-diarization-community-1"
    return "pyannote/speaker-diarization-3.1"


class PyannoteDiarizationTool(DiarizationTool):
    """
    pyannote diarization tool.

    :param model_name: The name of the model to use, see ``default_model_name`` when None.
    :param api_token: The HuggingFace API token to use.
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_token: Optional[str] = None,
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.model_name = model_name = model_name or default_model_name()
        self.api_token = api_token or os.environ.get('HF_TOKEN')
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        self._register_torch_safe_globals()
        patch_hf_hub_download_use_auth_token()

        model = Pipeline.from_pretrained(
            model_name,
            **pyannote_token_kwargs(self.api_token),
        )
        if model is None:
            raise PermissionError(
                "Could not load pyannote diarization model. Ensure HF_TOKEN is valid and "
                "the token account accepted pyannote model terms on Hugging Face."
            )
        self.model: Pipeline = model.to(device=torch.device(device))

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        exclusive: bool = False,
    ):
        """
        Implementation of pyannote's diarization inference method.

        :param audio_path: Path to the audio track.
        :param diarization_path: Path to the diarization file.
        :param num_speakers: Number of speaking participants, if ignored the model will try to guess it, it is advised to specify it.
        :param min_speakers: Lower bound on the number of speakers, when it is not known exactly.
        :param max_speakers: Upper bound on the number of speakers, when it is not known exactly.
        :param exclusive: Whether to keep a single speaker at a time (pyannote.audio 4 only), which
            simplifies the alignment with a transcription.
        :return:
        """
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"diarization     =   {diarization_path}")

        wav.WAVReader.check(path=audio_path)
        rttm.RTTMWriter.check(path=diarization_path, overwrite=self.overwrite)

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        for _ in tqdm(
            range(1),
            desc="Processing",
            disable=not self.verbose,
        ):
            output = self.model(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        results = self._annotation(output, exclusive)

        segments = [
            {
                "type": "SPEAKER",
                "file_stem": audio_path.stem,
                "channel": 1,
                "start": segment.start,
                "duration": segment.duration,
                "orthography": "<NA>",
                "speaker_type": "<NA>",
                "speaker_name": speaker_name,
                "confidence_score": "<NA>",
                "signal_lookahead_time": "<NA>",
            }
            for segment, track_name, speaker_name in tqdm(
                results.itertracks(yield_label=True),
                desc="Parsing",
                disable=not self.verbose,
            )
        ]
        rttm.RTTMWriter.write(
            segments=segments, path=diarization_path, overwrite=self.overwrite
        )

    @staticmethod
    def _annotation(output: Any, exclusive: bool) -> Annotation:
        """
        Speaker turns from the pipeline output.

        pyannote.audio 3 returns an Annotation, pyannote.audio 4 a DiarizeOutput holding the
        regular and the exclusive (one speaker at a time) diarizations.
        """
        if isinstance(output, Annotation):
            if exclusive:
                raise ValueError("exclusive diarization needs pyannote.audio 4 or newer")
            return output
        return output.exclusive_speaker_diarization if exclusive else output.speaker_diarization

    @staticmethod
    def _register_torch_safe_globals() -> None:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is None:
            return

        safe_globals = []
        try:
            from omegaconf.listconfig import ListConfig

            safe_globals.append(ListConfig)
        except Exception:
            pass

        try:
            from omegaconf.dictconfig import DictConfig

            safe_globals.append(DictConfig)
        except Exception:
            pass

        try:
            from omegaconf.base import ContainerMetadata

            safe_globals.append(ContainerMetadata)
        except Exception:
            pass

        try:
            import omegaconf.base as omegaconf_base

            safe_globals.extend(
                value for value in vars(omegaconf_base).values() if isinstance(value, type)
            )
        except Exception:
            pass

        try:
            import omegaconf.nodes as omegaconf_nodes

            safe_globals.extend(
                value
                for name, value in vars(omegaconf_nodes).items()
                if name.endswith("Node") and isinstance(value, type)
            )
        except Exception:
            pass

        safe_globals.extend([Any, list, dict, tuple, set, int, float, bool, str, bytes, OrderedDict, defaultdict])

        if safe_globals:
            add_safe_globals(safe_globals)
