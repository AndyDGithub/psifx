"""End-to-end psifx pipeline on one video: data extraction, speaker-person linking and re-rendering.

Expects ``<out_dir>/input.mp4`` and writes every product next to it (masks, pose archives, RTTM, VTT,
mapping, rendered video). Each stage is a psifx command (or one of the research scripts of this
folder), is timed into ``timings.json`` and is skipped when its outputs already exist, so a run can
be resumed. The first failing stage stops the run.

Stages run as soon as the stages they read from are done, several at a time, with one rule: at most
one GPU stage at a time, because they share an 8 GB GPU and most of them already fill it (one more
CUDA context alone takes 0.5 to 1 GB). So the audio branch and MediaPipe (CPU only) run while SAM3,
RTMO or ViTPose hold the GPU. ``--sequential`` restores the one-after-the-other order.

    python research/e2e/run_pipeline.py .e2e_outputs/v1_example --num_speakers 2 --stereo_mics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

HERE = Path(__file__).resolve().parent


class Stage(NamedTuple):
    name: str
    outputs: List[Path]  # The stage is skipped when all of them exist.
    command: List[str]
    after: Tuple[str, ...] = ()  # Stages whose outputs it reads.
    gpu: bool = False


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
        Stage("tracking_sam3", [masks], psifx("video", "tracking", "sam3", "inference", *tracking), gpu=True),
        Stage("filter_tracks", [d / "tracks.json"], script(
            "filter_tracks.py", str(d), "--min_presence", str(args.min_presence)), after=("tracking_sam3",)),
        Stage("pose_body_rtmo", [poses_body], psifx(
            "video", "pose", "body", "multi-inference", "--video", video, "--masks", str(masks),
            "--poses_dir", str(poses_body), "--model", "rtmo", "--device", "cuda"), after=("filter_tracks",), gpu=True),
    ]
    if args.vitpose:
        stages.append(Stage("pose_body_vitpose", [d / "poses_vitpose"], psifx(
            "video", "pose", "body", "multi-inference", "--video", video, "--masks", str(masks),
            "--poses_dir", str(d / "poses_vitpose"), "--model", "vitpose", "--device", "cuda"),
            after=("filter_tracks",), gpu=True))
    stages += [
        # MediaPipe runs on the CPU only (psifx/video/pose/mediapipe/tool.py).
        Stage("pose_mediapipe", [poses_mp], psifx(
            "video", "pose", "mediapipe", "multi-inference", "--video", video, "--masks", str(masks),
            "--poses_dir", str(poses_mp), "--device", "cpu"), after=("filter_tracks",)),
        Stage("audio_extraction", [audio_raw], psifx(
            "audio", "manipulation", "extraction", "--video", video, "--audio", str(audio_raw))),
    ]
    speech, speech_stage = audio_raw, "audio_extraction"
    if args.separate_vocals:
        # Music hides the voice from WhisperX's voice activity detection: keep only the vocals (Demucs).
        speech, speech_stage = d / "separated" / "htdemucs" / "audio_raw" / "vocals.wav", "vocals_separation"
        stages.append(Stage("vocals_separation", [speech], [
            sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(d / "separated"),
            str(audio_raw)], after=("audio_extraction",), gpu=True))
    stages += [
        Stage("audio_mono", [d / "audio_mono.wav"], psifx(
            "audio", "manipulation", "conversion", "--audio", str(speech), "--mono_audio", str(d / "audio_mono.wav")),
            after=(speech_stage,)),
        Stage("audio_normalization", [audio], psifx(
            "audio", "manipulation", "normalization", "--audio", str(d / "audio_mono.wav"), "--normalized_audio", str(audio)),
            after=("audio_mono",)),
        Stage("diarization", [d / "diarization.rttm"], psifx(
            "audio", "diarization", "pyannote", "inference", "--audio", str(audio), "--diarization",
            # psifx default model: pyannote community-1 (research/08_diarization_community1.md).
            str(d / "diarization.rttm"), "--device", "cuda", *speakers), after=("audio_normalization",), gpu=True),
        Stage("transcription", [d / "transcription.vtt"], psifx(
            "audio", "transcription", "whisperx", "inference", "--audio", str(audio), "--transcription",
            str(d / "transcription.vtt"), "--model_name", args.whisper_model, "--device", "cuda",
            *(["--language", args.language] if args.language else [])), after=("audio_normalization",), gpu=True),
    ]
    if args.stereo_mics:
        # One lavalier microphone per person on each stereo channel: reference speaker identities.
        left, right = d / "mic_left.wav", d / "mic_right.wav"
        stages += [
            Stage("mics_split", [left, right], psifx(
                "audio", "manipulation", "split", "--stereo_audio", str(audio_raw), "--left_audio", str(left),
                "--right_audio", str(right)), after=("audio_extraction",)),
            Stage("identification_mics", [d / "identification_mics.json"], psifx(
                "audio", "identification", "pyannote", "inference", "--mixed_audio", str(audio), "--diarization",
                str(d / "diarization.rttm"), "--mono_audios", str(left), str(right), "--identification",
                str(d / "identification_mics.json"), "--device", "cuda",
                # The default also loads speechbrain/spkrec-ecapa-voxceleb, which fails with recent
                # huggingface_hub (its optional custom.py lookup raises RemoteEntryNotFoundError).
                "--model_names", "pyannote/embedding"), after=("mics_split", "diarization", "audio_normalization"),
                gpu=True),
        ]
    pose_stages = tuple(s.name for s in stages if s.name.startswith("pose_body"))
    stages += [
        Stage("link_speakers", [d / "identification_lips.json"], script("link_speakers.py", str(d)),
              after=("diarization", "pose_mediapipe")),
        Stage("enhance", [d / "transcription_enhanced.vtt"], psifx(
            "audio", "transcription", "enhance", "--transcription", str(d / "transcription.vtt"), "--diarization",
            str(d / "diarization.rttm"), "--identification", str(d / "identification_lips.json"),
            "--enhanced_transcription", str(d / "transcription_enhanced.vtt")),
            after=("transcription", "diarization", "link_speakers")),
        Stage("render", [d / "render.mp4"], script("render.py", str(d)), after=(*pose_stages, "pose_mediapipe", "enhance")),
    ]
    # evaluate reads every product, timings.json included: it runs last.
    stages.append(Stage("evaluate", [d / "metrics.json"], script("evaluate.py", str(d)),
                        after=tuple(s.name for s in stages)))
    return stages


def check_graph(stages: List[Stage]) -> None:
    """Every dependency must be an earlier stage, so the graph has no cycle and the run cannot hang."""
    seen = set()
    for stage in stages:
        unknown = [name for name in stage.after if name not in seen]
        if unknown:
            raise ValueError(f"stage {stage.name} waits for {unknown}, which are not earlier stages")
        seen.add(stage.name)


class Running(NamedTuple):
    process: subprocess.Popen
    start: float
    log_path: Path


def run_stages(stages: List[Stage], d: Path, timings: Dict[str, float], max_parallel: int) -> None:
    """Starts each stage once its dependencies are done; stops starting new ones after a failure.

    Single-threaded on purpose: the scheduler only starts subprocesses and polls them, so it shares no
    state between threads. Each child writes to its own log file, never to a pipe, so it cannot block
    on a full pipe buffer while the scheduler waits.
    """
    timings_path, spans_path = d / "timings.json", d / "stage_spans.json"
    # Start and end (Unix time) of each stage: with stages overlapping, the sum of the timings is no
    # longer the wall-clock time, evaluate.py measures it on these spans.
    spans = json.loads(spans_path.read_text()) if spans_path.exists() else {}
    pending = list(stages)
    done = set()
    running: Dict[str, Running] = {}
    failures = []
    try:
        while pending or running:
            if not failures:
                gpu_busy = any(s.gpu for s in stages if s.name in running)
                for stage in list(pending):
                    if len(running) >= max_parallel:
                        break
                    if not all(name in done for name in stage.after) or (stage.gpu and gpu_busy):
                        continue
                    pending.remove(stage)
                    print(f"[run ] {stage.name}: {' '.join(stage.command)}", flush=True)
                    log_path = d / "logs" / f"{stage.name}.log"
                    log_path.parent.mkdir(exist_ok=True)
                    with log_path.open("w") as log:
                        # The child keeps its own copy of the file descriptor.
                        process = subprocess.Popen(stage.command, stdout=log, stderr=subprocess.STDOUT)
                    running[stage.name] = Running(process, time.perf_counter(), log_path)
                    gpu_busy = gpu_busy or stage.gpu
            elif not running:
                break

            time.sleep(0.5)
            for name, job in list(running.items()):
                code = job.process.poll()
                if code is None:
                    continue
                del running[name]
                elapsed = time.perf_counter() - job.start
                if code != 0:
                    tail = "".join(job.log_path.read_text(errors="replace").splitlines(keepends=True)[-20:])
                    failures.append(f"error: stage {name} failed (exit {code}) after {elapsed:.0f}s, "
                                    f"see {job.log_path}\n{tail}")
                    # Later stages would read missing or partial outputs: start no new one. Running
                    # stages are independent of this one and finish, so they leave no partial output.
                    print(failures[-1], file=sys.stderr, flush=True)
                    if running:
                        print(f"waiting for {', '.join(running)} to finish (Ctrl-C to stop them)", flush=True)
                    continue
                done.add(name)
                timings[name] = round(elapsed, 2)
                timings_path.write_text(json.dumps(timings, indent=2))
                spans[name] = [round(time.time() - elapsed, 2), round(time.time(), 2)]
                spans_path.write_text(json.dumps(spans, indent=2))
                print(f"[done] {name}: {elapsed:.1f}s", flush=True)
    except KeyboardInterrupt:
        for name, job in running.items():
            print(f"stopping {name}; its outputs may be partial, delete them before resuming", file=sys.stderr)
            job.process.terminate()
        for job in running.values():
            try:
                job.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                job.process.kill()
        raise
    if failures:
        sys.exit(failures[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder holding input.mp4")
    parser.add_argument("--num_speakers", type=int, default=None, help="known number of speakers, if any")
    parser.add_argument("--stereo_mics", action="store_true", help="one lavalier mic per person on L/R channels")
    parser.add_argument("--vitpose", action="store_true", help="also run ViTPose, for comparison")
    parser.add_argument("--separate_vocals", action="store_true",
                        help="diarize and transcribe the vocals only (Demucs), for music videos")
    parser.add_argument("--text_prompt", default="person")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument("--min_presence", type=float, default=0.05,
                        help="tracks present in fewer frames are set aside before pose estimation")
    parser.add_argument("--whisper_model", default="large-v3-turbo", help="multilingual, unlike distil-large-v3")
    parser.add_argument("--language", default=None, help="forced language, auto-detected otherwise")
    parser.add_argument("--only", nargs="*", default=None, help="run only these stages (by name)")
    parser.add_argument("--max_parallel", type=int, default=4, help="stages running at the same time")
    parser.add_argument("--sequential", action="store_true", help="one stage at a time, in the listed order")
    args = parser.parse_args()

    d = args.out_dir.resolve()
    if not (d / "input.mp4").is_file():
        sys.exit(f"error: {d / 'input.mp4'} not found")

    timings_path = d / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    stages = build_stages(d, args)
    check_graph(stages)
    names = {stage.name for stage in stages}
    if args.only is not None and set(args.only) - names:
        sys.exit(f"error: unknown stages {sorted(set(args.only) - names)}, choose from {sorted(names)}")
    to_run = []
    for stage in stages:
        if args.only is not None and stage.name not in args.only:
            continue
        if all(p.exists() for p in stage.outputs):
            print(f"[skip] {stage.name}: outputs exist")
            continue
        to_run.append(stage)
    # Skipped stages are done: drop them from the dependencies of the others.
    selected = {stage.name for stage in to_run}
    to_run = [stage._replace(after=tuple(n for n in stage.after if n in selected)) for stage in to_run]
    run_stages(to_run, d, timings, max_parallel=1 if args.sequential else args.max_parallel)


if __name__ == "__main__":
    main()
