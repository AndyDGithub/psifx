"""Compares transcriptions and diarizations with hand-made subtitles (who said what, every few seconds).

The reference is a text file in the format of ``ExtraitTalkShow_Sous-titre.txt``:

- ``CODE = Name`` lines declare the speakers;
- a ``m:ss`` line (``m.ss`` accepted) starts a block, which lasts until the next one;
- ``CODE : text`` lines are what that speaker said in the block, in order;
- ``[...]`` (laughs, cuts), ``(...)`` (comments) and lines without a speaker are not speech.

Measures, per transcription:

- WER and CER on the whole text, after the normalization of ``evaluate.py`` and without the hesitations
  (``euh``, ``hm``) that Whisper usually drops;
- with a speaker per word (from the transcription's ``<v ...>`` tags, or from a ``--diarization`` RTTM,
  each segment taking the speaker it overlaps most), the share of aligned words given to the right
  person. The labels are matched to the reference speakers one to one, maximizing the agreement:
  a speaker split in two by the diarization counts as an error for one of the halves.

The block times are coarse (a few seconds), so the speaker check goes through the word alignment,
not through the times. Writes ``reference.vtt`` (blocks with their speakers) and
``subtitle_eval.json`` next to the first transcription, and prints a table.

    python research/e2e/subtitle_eval.py example/data/ExtraitTalkShow_Sous-titre.txt \\
        .e2e_outputs/extrait_talkshow/asr/*.vtt --diarization .e2e_outputs/extrait_talkshow/asr/diar_*.rttm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

from psifx.io import rttm, vtt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.e2e.evaluate import normalize_text  # noqa: E402

HESITATIONS = {"euh", "heu", "hm", "hmm", "mh"}
TIME = re.compile(r"^(\d+)[:.](\d{2})$")
LEGEND = re.compile(r"^(\w+)\s*=\s*(.+)$")
LINE = re.compile(r"^(\w+)\s*:\s*(.*)$")


def words(text: str) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", text.replace("’", "'"))  # (with a strong British accent)
    return [w for w in normalize_text(text).split() if w not in HESITATIONS]


def parse_reference(path: Path, end: float | None) -> tuple[dict, list[dict]]:
    """Speaker names and utterances ``{start, end, speaker, text}`` of the hand-made subtitles."""
    names, utterances, start = {}, [], None
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if m := TIME.match(line):
            start = int(m[1]) * 60 + int(m[2])
            continue
        if start is None:
            if m := LEGEND.match(line):
                names[m[1]] = m[2].strip()
            continue
        m = LINE.match(line)
        if m and m[1] in names:
            utterances.append({"start": float(start), "speaker": m[1], "text": m[2].strip()})
        elif m and not line.startswith(("[", "(")):
            sys.exit(f"error: {path}:{number}: unknown speaker {m[1]!r} (declared: {', '.join(names)})")
    if not utterances:
        sys.exit(f"error: no utterance found in {path}")
    # A block ends where the next one starts; its utterances share it in order.
    starts = sorted({u["start"] for u in utterances})
    last = end if end is not None else starts[-1] + 5.0
    for u in utterances:
        k = starts.index(u["start"])
        u["end"] = starts[k + 1] if k + 1 < len(starts) else last
    return names, utterances


def labeled_words(segments: list[dict]) -> tuple[list[str], list[str]]:
    """Normalized words of the segments and the speaker of each one."""
    ws, labels = [], []
    for s in segments:
        w = words(s["text"])
        ws += w
        labels += [s["speaker"] or "NA"] * len(w)
    return ws, labels


def speakers_from_diarization(segments: list[dict], diarization: Path) -> list[dict]:
    turns = rttm.RTTMReader.read(path=diarization, verbose=False)
    out = []
    for s in segments:
        overlap = Counter()
        for t in turns:
            o = min(s["end"], t["start"] + t["duration"]) - max(s["start"], t["start"])
            if o > 0:
                overlap[t["speaker_name"]] += o
        out.append({**s, "speaker": overlap.most_common(1)[0][0] if overlap else "NA"})
    return out


def attribution(ref_labels, hyp_labels, chunks) -> dict:
    """Share of aligned words (same or substituted) whose speaker maps to the reference speaker."""
    pairs = []
    for c in chunks:
        if c.type in ("equal", "substitute"):
            pairs += zip(ref_labels[c.ref_start_idx:c.ref_end_idx], hyp_labels[c.hyp_start_idx:c.hyp_end_idx])
    refs, hyps = sorted({r for r, _ in pairs}), sorted({h for _, h in pairs})
    counts = np.zeros((len(hyps), len(refs)))
    for r, h in pairs:
        counts[hyps.index(h), refs.index(r)] += 1
    rows, cols = linear_sum_assignment(-counts)
    mapping = {hyps[i]: refs[j] for i, j in zip(rows, cols)}
    per_speaker = {}
    for r in refs:
        mine = [(rr, h) for rr, h in pairs if rr == r]
        per_speaker[r] = round(sum(mapping.get(h) == r for _, h in mine) / len(mine), 3)
    return {
        "accuracy": round(sum(mapping.get(h) == r for r, h in pairs) / len(pairs), 3) if pairs else None,
        "aligned_words": len(pairs),
        "mapping": mapping,
        "per_speaker": per_speaker,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="hand-made subtitles (see above)")
    parser.add_argument("transcriptions", type=Path, nargs="+", help="WebVTT transcriptions to compare")
    parser.add_argument("--diarization", type=Path, nargs="*", default=[],
                        help="RTTM files, each one paired with every transcription for the speaker check")
    parser.add_argument("--end", type=float, default=None, help="end of the last block, in seconds (video length)")
    parser.add_argument("--top", type=int, default=15, help="most frequent substitutions to list")
    args = parser.parse_args()

    for path in [args.reference, *args.transcriptions, *args.diarization]:
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    names, reference = parse_reference(args.reference, args.end)
    ref_words, ref_labels = labeled_words(reference)
    ref_text = " ".join(ref_words)
    out_dir = args.transcriptions[0].parent
    vtt.VTTWriter.write(segments=reference, path=out_dir / "reference.vtt", overwrite=True, verbose=False)

    results = {"reference": {"speakers": names, "utterances": len(reference), "words": len(ref_words),
                             "words_per_speaker": dict(Counter(ref_labels))}}
    for path in args.transcriptions:
        segments = vtt.VTTReader.read(path, verbose=False)
        hyp_words, hyp_labels = labeled_words(segments)
        hyp_text = " ".join(hyp_words)
        output = jiwer.process_words(ref_text, hyp_text)
        substitutions, _, _ = jiwer.collect_error_counts(output)
        result = {
            "wer": round(output.wer, 3),
            "cer": round(jiwer.cer(ref_text, hyp_text), 3),
            "words": len(hyp_words),
            "substitutions": output.substitutions, "deletions": output.deletions,
            "insertions": output.insertions,
            "top_substitutions": [f"{r} -> {h} ({n})" for (r, h), n in
                                  sorted(substitutions.items(), key=lambda kv: -kv[1])[:args.top]],
        }
        chunks = output.alignments[0]
        if any(s["speaker"] for s in segments):
            result["speakers_from_vtt"] = attribution(ref_labels, hyp_labels, chunks)
        for diarization in args.diarization:
            _, labels = labeled_words(speakers_from_diarization(segments, diarization))
            result[f"speakers_{diarization.stem}"] = attribution(ref_labels, labels, chunks)
        results[path.name] = result

    (out_dir / "subtitle_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"reference: {len(reference)} utterances, {len(ref_words)} words, {results['reference']['words_per_speaker']}")
    for name, r in results.items():
        if name == "reference":
            continue
        speakers = "  ".join(f"{k.removeprefix('speakers_')}={v['accuracy']}" for k, v in r.items()
                             if k.startswith("speakers_"))
        print(f"{name:32s} WER {r['wer']:.3f}  CER {r['cer']:.3f}  {speakers}")
    print(f"wrote {out_dir / 'subtitle_eval.json'}, {out_dir / 'reference.vtt'}")


if __name__ == "__main__":
    main()
