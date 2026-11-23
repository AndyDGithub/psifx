"""Video I/O module."""

import subprocess
import warnings
from typing import Dict, Optional, Union

from pathlib import Path

from numpy import ndarray

from skvideo.io import FFmpegReader, FFmpegWriter


class VideoReader(FFmpegReader):
    """
    Video reader object.

    :param path: The path to the video file.
    :param input_dict: Input options.
    :param output_dict: Output options.
    """

    def __init__(
            self,
            path: Union[str, Path],
            input_dict: Optional[Dict[str, str]] = None,
            output_dict: Optional[Dict[str, str]] = None,
    ):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File missing at path {path}")

        super().__init__(
            filename=str(path),
            inputdict=input_dict,
            outputdict=output_dict,
        )

        self.num_frames = self.inputframenum
        self.frame_rate = self.probeInfo["video"][self.INFO_AVERAGE_FRAMERATE]

    def __len__(self):
        return self.num_frames


class VideoWriter(FFmpegWriter):
    """
    Video writer object.

    :param path: The path to the video file.
    :param input_dict: Input options.
    :param output_dict: Output options.
    :param overwrite: Whether to overwrite existing files.
    :param audio_source: Video or audio file whose sound is added to the written video when the writer is closed
        (a ``with`` block that ends without error). The frames are written without sound, so a visualization
        would otherwise lose the sound of the video it is drawn on. Nothing is added if the source has no sound.
    """

    def __init__(
            self,
            path: Union[str, Path],
            input_dict: Optional[Dict[str, str]] = None,
            output_dict: Optional[Dict[str, str]] = None,
            overwrite: bool = False,
            audio_source: Optional[Union[str, Path]] = None,
    ):
        path = Path(path)
        self.path = path
        self.audio_source = Path(audio_source) if audio_source is not None else None
        if self.audio_source is not None and not self.audio_source.exists():
            raise FileNotFoundError(f"File missing at path {self.audio_source}")

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"File {path} already exists.")

        super().__init__(
            filename=str(path),
            inputdict=input_dict,
            outputdict=output_dict,
        )

    def write(self, image: ndarray):
        """
        Appends an image to the existing video.

        :param image: [H, W, 3] ndarray.
        :return:
        """
        self.writeFrame(im=image)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is None and self.audio_source is not None:
            add_audio(self.path, self.audio_source)


def has_audio(path: Union[str, Path]) -> bool:
    """Whether the file has an audio stream (ffprobe)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def add_audio(video_path: Union[str, Path], audio_source: Union[str, Path]):
    """
    Adds the sound of ``audio_source`` to ``video_path`` in place, the video stream copied as is.

    The sound is encoded in AAC (accepted by every video container) and cut at the end of the video. If the source
    has no sound, the video is left unchanged. If ffmpeg fails, the video is left unchanged and a warning is given.

    :param video_path: Path to the video without sound.
    :param audio_source: Path to the video or audio file with the sound.
    :return:
    """
    video_path, audio_source = Path(video_path), Path(audio_source)
    if not has_audio(audio_source):
        return
    tmp = video_path.with_name(video_path.stem + ".audio_tmp" + video_path.suffix)
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-i", str(audio_source), "-map", "0:v", "-map",
         "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        warnings.warn(f"Could not add the sound of {audio_source} to {video_path}: {result.stderr.strip()}")
        return
    tmp.replace(video_path)
