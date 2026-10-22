"""End-to-end psifx pipeline on one video: data extraction, speaker-person linking and re-rendering.

Expects ``<out_dir>/input.mp4`` and writes every product next to it (masks, pose archives, RTTM, VTT,
mapping, rendered video). Each stage is a psifx command (or one of the research scripts of this
folder), is timed into ``timings.json`` and is skipped when its outputs already exist, so a run can
be resumed. The first failing stage stops the run.

Stages run one after the other on purpose: they share an 8 GB GPU, and most of them already use
the whole device.

    python research/e2e/run_pipeline.py .e2e_outputs/v1_example --num_speakers 2 --stereo_mics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, NamedTuple

HERE = Path(__file__).resolve().parent


class Stage(NamedTuple):
    name: str
    outputs: List[Path]  # The stage is skipped when all of them exist.
    command: List[str]


def psifx(*args: str) -> List[str]:
    return ["psifx", *args]


def script(name: str, *args: str) -> List[str]:
    return [sys.executable, str(HERE / name), *args]


def build_stages(d: Path, args: argparse.Namespace) -> List[Stage]:
    video = str(d / "input.mp4")
    masks, poses_body, poses_mp = d / "masks", d / "poses_body", d / "poses_mp"
    audio_raw, audio = d / "audio_raw.wav", d / "audio.wav"
    speakers = ["--num_speakers", str(args.num_speakers)] if args.num_speakers else []

    tracking = ["--video", video, "--mask_dir", str(masks), "--text_prompt", args.text_prompt,
                "--chunk_size", str(args.chunk_size), "--device", "cuda"]
    if args.max_objects:
        tracking += ["--max_num_objects", str(args.max_objects)]

    stages = [
        Stage("tracking_sam3", [masks], psifx("video", "tracking", "sam3", "inference", *tracking)),
        Stage("pose_body_rtmo", [poses_body], psifx(
            "video", "pose", "body", "multi-inference", "--video", video, "--masks", str(masks),
            "--poses_dir", str(poses_body), "--model", "rtmo", "--device", "cuda")),
    ]
    if args.vitpose:
        stages.append(Stage("pose_body_vitpose", [d / "poses_vitpose"], psifx(
            "video", "pose", "body", "multi-inference", "--video", video, "--masks", str(masks),
            "--poses_dir", str(d / "poses_vitpose"), "--model", "vitpose", "--device", "cuda")))
    stages += [
        Stage("pose_mediapipe", [poses_mp], psifx(
            "video", "pose", "mediapipe", "multi-inference", "--video", video, "--masks", str(masks),
            "--poses_dir", str(poses_mp))),
        Stage("audio_extraction", [audio_raw], psifx(
            "audio", "manipulation", "extraction", "--video", video, "--audio", str(audio_raw))),
        Stage("audio_mono", [d / "audio_mono.wav"], psifx(
            "audio", "manipulation", "conversion", "--audio", str(audio_raw), "--mono_audio", str(d / "audio_mono.wav"))),
        Stage("audio_normalization", [audio], psifx(
            "audio", "manipulation", "normalization", "--audio", str(d / "audio_mono.wav"), "--normalized_audio", str(audio))),
        Stage("diarization", [d / "diarization.rttm"], psifx(
            "audio", "diarization", "pyannote", "inference", "--audio", str(audio), "--diarization",
            str(d / "diarization.rttm"), "--model_name", "pyannote/speaker-diarization-3.1", "--device", "cuda", *speakers)),
        Stage("transcription", [d / "transcription.vtt"], psifx(
            "audio", "transcription", "whisperx", "inference", "--audio", str(audio), "--transcription",
            str(d / "transcription.vtt"), "--model_name", args.whisper_model, "--device", "cuda",
            *(["--language", args.language] if args.language else []))),
    ]
    if args.stereo_mics:
        # One lavalier microphone per person on each stereo channel: reference speaker identities.
        left, right = d / "mic_left.wav", d / "mic_right.wav"
        stages += [
            Stage("mics_split", [left, right], psifx(
                "audio", "manipulation", "split", "--stereo_audio", str(audio_raw), "--left_audio", str(left),
                "--right_audio", str(right))),
            Stage("identification_mics", [d / "identification_mics.json"], psifx(
                "audio", "identification", "pyannote", "inference", "--mixed_audio", str(audio), "--diarization",
                str(d / "diarization.rttm"), "--mono_audios", str(left), str(right), "--identification",
                str(d / "identification_mics.json"), "--device", "cuda",
                # The default also loads speechbrain/spkrec-ecapa-voxceleb, which fails with recent
                # huggingface_hub (its optional custom.py lookup raises RemoteEntryNotFoundError).
                "--model_names", "pyannote/embedding")),
        ]
    stages += [
        Stage("link_speakers", [d / "identification_lips.json"], script("link_speakers.py", str(d))),
        Stage("enhance", [d / "transcription_enhanced.vtt"], psifx(
            "audio", "transcription", "enhance", "--transcription", str(d / "transcription.vtt"), "--diarization",
            str(d / "diarization.rttm"), "--identification", str(d / "identification_lips.json"),
            "--enhanced_transcription", str(d / "transcription_enhanced.vtt"))),
        Stage("render", [d / "render.mp4"], script("render.py", str(d))),
        Stage("evaluate", [d / "metrics.json"], script("evaluate.py", str(d))),
    ]
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder holding input.mp4")
    parser.add_argument("--num_speakers", type=int, default=None, help="known number of speakers, if any")
    parser.add_argument("--stereo_mics", action="store_true", help="one lavalier mic per person on L/R channels")
    parser.add_argument("--vitpose", action="store_true", help="also run ViTPose, for comparison")
    parser.add_argument("--text_prompt", default="person")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument("--whisper_model", default="large-v3-turbo", help="multilingual, unlike distil-large-v3")
    parser.add_argument("--language", default=None, help="forced language, auto-detected otherwise")
    parser.add_argument("--only", nargs="*", default=None, help="run only these stages (by name)")
    args = parser.parse_args()

    d = args.out_dir.resolve()
    if not (d / "input.mp4").is_file():
        sys.exit(f"error: {d / 'input.mp4'} not found")

    timings_path = d / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    for stage in build_stages(d, args):
        if args.only is not None and stage.name not in args.only:
            continue
        if all(p.exists() for p in stage.outputs):
            print(f"[skip] {stage.name}: outputs exist")
            continue
        print(f"[run ] {stage.name}: {' '.join(stage.command)}", flush=True)
        start = time.perf_counter()
        log_path = d / "logs" / f"{stage.name}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("w") as log:
            result = subprocess.run(stage.command, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            # Stop before later stages consume missing or partial outputs.
            tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-20:])
            sys.exit(f"error: stage {stage.name} failed (exit {result.returncode}) after {elapsed:.0f}s, "
                     f"see {log_path}\n{tail}")
        timings[stage.name] = round(elapsed, 2)
        timings_path.write_text(json.dumps(timings, indent=2))
        print(f"[done] {stage.name}: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
