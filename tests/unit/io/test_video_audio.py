"""VideoWriter keeps the sound of the source video (visualizations used to lose it)."""

import subprocess

import numpy as np
import pytest

from psifx.io.video import VideoReader, VideoWriter, has_audio


def make_video(path, with_audio: bool):
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(command, check=True)


def copy_frames(source, target, **kwargs):
    with VideoReader(path=source) as reader, VideoWriter(path=target, input_dict={"-r": reader.frame_rate},
                                                         output_dict={"-pix_fmt": "yuv420p"}, **kwargs) as writer:
        for frame in reader:
            writer.write(frame)


def test_sound_of_the_source_is_kept(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    make_video(source, with_audio=True)
    copy_frames(source, target, audio_source=source)
    assert has_audio(target)
    with VideoReader(path=target) as reader:
        assert len(reader) == 10
    assert not list(tmp_path.glob("*audio_tmp*"))


def test_without_audio_source_no_sound(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    make_video(source, with_audio=True)
    copy_frames(source, target)
    assert not has_audio(target)


def test_silent_source_leaves_video_unchanged(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    make_video(source, with_audio=False)
    copy_frames(source, target, audio_source=source)
    assert not has_audio(target)
    with VideoReader(path=target) as reader:
        assert len(reader) == 10


def test_missing_audio_source_fails_before_writing(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoWriter(path=tmp_path / "target.mp4", audio_source=tmp_path / "missing.mp4")
    assert not (tmp_path / "target.mp4").exists()


def test_error_inside_block_adds_nothing(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    make_video(source, with_audio=True)
    with pytest.raises(RuntimeError):
        with VideoWriter(path=target, input_dict={"-r": "10"}, audio_source=source) as writer:
            writer.write(np.zeros((48, 64, 3), np.uint8))
            raise RuntimeError("stop")
    assert target.exists() and not has_audio(target)
