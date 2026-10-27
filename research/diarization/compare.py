"""Runs pyannote diarization pipelines through psifx and scores them against the references.

    python research/diarization/compare.py run .e2e_outputs --model pyannote/speaker-diarization-community-1
    python research/diarization/compare.py score .e2e_outputs

``run`` writes ``.e2e_outputs/diar/<case>/<tag>.rttm`` for every case: the noisy V1 mixes and the
four videos. Each case is diarized with the known number of speakers when there is one and
without it (``_auto``), the realistic setting. ``--exclusive`` also writes the one-speaker-at-a-time
output (pyannote.audio 4). ``score`` writes ``diar/results.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Known number of speakers per video; None when the count itself is part of the test.
VIDEOS = {"v1_example": 2, "v2_ava": None, "v3_talkshow": 2, "v4_dance": 1}
TAGS = {"pyannote/speaker-diarization-3.1": "legacy31",
        "pyannote/speaker-diarization-community-1": "community1"}


def cases(e2e: Path) -> dict[str, tuple[Path, int | None]]:
    """case name -> (audio, known number of speakers)."""
    out = {name: (e2e / name / "audio.wav", count) for name, count in VIDEOS.items()}
    for d in sorted((e2e / "diar").glob("v1_*db")):
        out[d.name] = (d / "audio.wav", 2)
    return out


def run(args):
    import pyannote.audio
    from psifx.audio.diarization.pyannote.tool import PyannoteDiarizationTool

    tag = TAGS.get(args.model, args.model.replace("/", "_"))
    if not pyannote.audio.__version__.startswith("4"):
        tag += "_pya3"
    tool = PyannoteDiarizationTool(model_name=args.model, device=args.device, overwrite=True, verbose=False)
    timings_path = args.e2e_dir / "diar" / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    for case, (audio, count) in cases(args.e2e_dir).items():
        if not audio.exists():
            sys.exit(f"error: {audio} is missing")
        out = args.e2e_dir / "diar" / case
        out.mkdir(parents=True, exist_ok=True)
        settings = [("", count), ("_auto", None)] if count else [("_auto", None)]
        for suffix, num_speakers in settings:
            for exclusive in ([False, True] if args.exclusive else [False]):
                name = f"{tag}{suffix}{'_excl' if exclusive else ''}"
                start = time.perf_counter()
                tool.inference(audio, out / f"{name}.rttm", num_speakers=num_speakers, exclusive=exclusive)
                timings[f"{case}/{name}"] = round(time.perf_counter() - start, 2)
                print(f"{case}/{name}: {timings[f'{case}/{name}']} s", flush=True)
        timings_path.write_text(json.dumps(timings, indent=1, sort_keys=True))


def load(path: Path):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri="audio")
    for line in path.read_text().splitlines():
        if line.strip():
            fields = line.split()
            start, duration = float(fields[3]), float(fields[4])
            annotation[Segment(start, start + duration)] = fields[7]
    return annotation


def load_uem(path: Path):
    from pyannote.core import Segment, Timeline

    return Timeline([Segment(float(f[2]), float(f[3]))
                     for f in (line.split() for line in path.read_text().splitlines() if line.strip())]).support()


def score(args):
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    diar = args.e2e_dir / "diar"
    timings = json.loads((diar / "timings.json").read_text()) if (diar / "timings.json").exists() else {}
    results = {}
    for case in cases(args.e2e_dir):
        d = diar / case
        reference = load(d / "reference.rttm") if (d / "reference.rttm").exists() else None
        uem = load_uem(d / "reference.uem") if reference is not None else None
        hypotheses = sorted(d.glob("*.rttm")) if d.exists() else []
        if case in VIDEOS and (args.e2e_dir / case / "diarization.rttm").exists():
            hypotheses.append(args.e2e_dir / case / "diarization.rttm")  # the e2e run: 3.1 on pyannote 3.3.2
        for path in hypotheses:
            if path.stem == "reference":
                continue
            name = "e2e_legacy31_pya3" if path.name == "diarization.rttm" else path.stem
            hypothesis = load(path)
            row = {"speakers": len(hypothesis.labels()),
                   "speech_s": round(hypothesis.get_timeline().support().duration(), 1),
                   "time_s": timings.get(f"{case}/{name}")}
            if reference is not None:
                der = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
                detail = der(reference, hypothesis, uem=uem, detailed=True)
                total = detail["total"]
                row.update({
                    "der": round(detail["diarization error rate"], 3),
                    "missed": round(detail["missed detection"] / total, 3),
                    "false_alarm": round(detail["false alarm"] / total, 3),
                    "confusion": round(detail["confusion"] / total, 3),
                    "jer": round(JaccardErrorRate(collar=args.collar)(reference, hypothesis, uem=uem), 3),
                })
            results.setdefault(case, {})[name] = row
    (diar / "results.json").write_text(json.dumps(results, indent=1))
    for case, rows in results.items():
        print(f"\n{case}")
        for name, row in sorted(rows.items()):
            print(f"  {name:28s} " + "  ".join(f"{k}={v}" for k, v in row.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    p.add_argument("e2e_dir", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--exclusive", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("e2e_dir", type=Path)
    p.add_argument("--collar", type=float, default=0.25, help="seconds forgiven around each reference boundary")
    args = parser.parse_args()
    if os.environ.get("PYANNOTE_METRICS_ENABLED") not in (None, "0", "false"):
        print("warning: pyannote telemetry is enabled", file=sys.stderr)
    run(args) if args.action == "run" else score(args)


if __name__ == "__main__":
    main()
