"""Mixes noise into the V1 audio at fixed signal-to-noise ratios, for the robustness test.

Noises:
- ``pink``: stationary pink noise (fan, room tone);
- ``music``: the V4 song, vocals included (singing is the hardest music for a speech model);
- ``babble``: three time-shifted copies of the V3 talk-show summed, i.e. unknown voices
  talking at the same time (cafeteria, open space).

The SNR is measured on the reference speech frames of V1, so silence does not dilute it.

    python research/diarization/make_noisy.py .e2e_outputs --snr 10 5 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read(path: Path) -> tuple[int, np.ndarray]:
    rate, signal = wavfile.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return rate, signal.astype(np.float64) / 32768.0


def fit(noise: np.ndarray, n: int) -> np.ndarray:
    """Loops or cuts the noise to n samples."""
    return np.tile(noise, int(np.ceil(n / len(noise))))[:n]


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))  # 1/f power
    return np.fft.irfft(spectrum, n)


def speech_mask(reference: Path, rate: int, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for line in reference.read_text().split("\n"):
        if line.strip():
            fields = line.split()
            start, duration = float(fields[3]), float(fields[4])
            mask[int(start * rate): int((start + duration) * rate)] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("e2e_dir", type=Path)
    parser.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0])
    args = parser.parse_args()
    e2e = args.e2e_dir
    rate, speech = read(e2e / "v1_example" / "audio.wav")
    n = len(speech)
    reference = e2e / "diar" / "v1_example" / "reference.rttm"
    if not reference.exists():
        sys.exit(f"error: {reference} is missing, run make_references.py first")
    active = speech_mask(reference, rate, n)

    noises = {"pink": pink(n, np.random.default_rng(0))}
    for name, video in (("music", "v4_dance"), ("babble", "v3_talkshow")):
        noise_rate, noise = read(e2e / video / "audio.wav")
        if noise_rate != rate:
            noise = resample_poly(noise, rate, noise_rate)
        if name == "babble":
            shift = len(noise) // 3
            noise = sum(np.roll(noise, k * shift) for k in range(3))
        noises[name] = fit(noise, n)

    speech_power = np.mean(speech[active] ** 2)
    for name, noise in noises.items():
        noise_power = np.mean(noise[active] ** 2)
        for snr in args.snr:
            mix = speech + noise * np.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))
            peak = np.abs(mix).max()
            if peak > 1:
                mix /= peak  # avoid clipping, the SNR is unchanged
            out = e2e / "diar" / f"v1_{name}_{snr:g}db"
            out.mkdir(parents=True, exist_ok=True)
            wavfile.write(out / "audio.wav", rate, (mix * 32767).astype(np.int16))
            for ext in ("rttm", "uem"):
                (out / f"reference.{ext}").write_text((reference.parent / f"reference.{ext}").read_text())
            print(f"{out.name}: written")


if __name__ == "__main__":
    main()
