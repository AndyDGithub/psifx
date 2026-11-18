"""Charts of report v7, from the result files in .e2e_outputs (PDF, vector).

- ``mevid_quality.pdf``: SAM 3D Body on the MEVID test split by crop height, three small multiples (one measure
  each, one axis each): agreement with ViTPose, jitter, failed chips without and with the filter;
- ``identity_attacks.pdf``: Rank-1 of each attacker with its 95 % interval, general and other outfit settings,
  with the chance level of each setting.
Palette: categorical slots 1 and 2 of the dataviz reference palette, validated for color vision deficiency.

    python research/report/figures/charts_v7.py .e2e_outputs/ext/mevid research/report/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
BIN_LABELS = {"0-64": "< 64", "64-96": "64–96", "96-128": "96–128", "128-192": "128–192", "192-288": "192–288",
              "288-1000000": "> 288"}


def mevid_quality(summary: dict, out: Path) -> None:
    bins = [b for b in BIN_LABELS if b in summary["bins"] and b != "0-64"]  # 2 tracklets under 64 px: left out.
    labels = [BIN_LABELS[b] for b in bins]
    x = range(len(bins))
    rows = [summary["bins"][b] for b in bins]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    axes[0].plot(x, [r["filtered"]["oks_vs_vitpose_p50"] for r in rows], color=BLUE, lw=2, marker="o", ms=5)
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_title("Accord avec ViTPose (OKS)", fontsize=9, color=INK)
    axes[1].bar(x, [r["filtered"]["accel_mm_p50"] for r in rows], color=BLUE, width=0.6)
    axes[1].set_title("Saccades (mm, médiane)", fontsize=9, color=INK)
    width = 0.38
    axes[2].bar([i - width / 2 - 0.01 for i in x], [100 * r["all"]["failed"] for r in rows], width, color=ORANGE,
                label="sans filtre")
    axes[2].bar([i + width / 2 + 0.01 for i in x], [100 * r["filtered"]["failed"] for r in rows], width, color=BLUE,
                label="avec filtre")
    axes[2].set_title("Découpes en échec (%)", fontsize=9, color=INK)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(list(x), labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_xlabel("Hauteur de la découpe (px)", fontsize=8, color=MUTED)
    fig.savefig(out / "mevid_quality.pdf")


def identity_attacks(attacks: dict, cal: dict | None, out: Path) -> None:
    names = {"shape": "Morphologie", "shape_scale": "Morphologie + échelle", "motion": "Mouvement (simple)",
             "all": "Les trois"}
    rows = [(names[k], v["general"]["rank1"], v["other_outfit"]["rank1"]) for k, v in attacks["attackers"].items()]
    if cal:
        rows.append(("CAL, vidéo d'origine", [cal["general"]["rank1"]] * 3, [cal["CC"]["rank1"]] * 3))
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(rows) + 1.0), constrained_layout=True)
    for i, (name, general, other) in enumerate(rows):
        for value, color, shift in ((general, BLUE, -0.12), (other, ORANGE, 0.12)):
            ax.errorbar(value[0], i + shift, xerr=[[value[0] - value[1]], [value[2] - value[0]]], fmt="o",
                        color=color, ms=6, lw=2, capsize=0)
    for value, color, label in ((attacks["chance_rank1"]["general"], BLUE, "hasard, général"),
                                (attacks["chance_rank1"]["other_outfit"], ORANGE, "hasard, autre tenue")):
        ax.axvline(value, color=color, lw=1, ls="--", label=label)
    ax.plot([], [], "o", color=BLUE, label="général")
    ax.plot([], [], "o", color=ORANGE, label="autre tenue")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)  # First attacker on top.
    ax.set_xlabel("Rank-1 (%), intervalle à 95 % par rééchantillonnage des identités", fontsize=8, color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "identity_attacks.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mevid", type=Path, help=".e2e_outputs/ext/mevid")
    parser.add_argument("out", type=Path, help="figures folder")
    args = parser.parse_args()
    summary, attacks = args.mevid / "test_split" / "summary.json", args.mevid / "test_split" / "attack_params.json"
    for path in (summary, attacks):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    cal_path = args.mevid / "cal_eval" / "full.json"
    cal = json.loads(cal_path.read_text()) if cal_path.is_file() else None
    mevid_quality(json.loads(summary.read_text()), args.out)
    identity_attacks(json.loads(attacks.read_text()), cal, args.out)
    print("CHARTS_OK")


if __name__ == "__main__":
    main()
