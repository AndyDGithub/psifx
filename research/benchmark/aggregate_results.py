"""Merge accuracy + video results into one table, compute the Pareto front
(precision vs speed) and a simple weighted ranking as a fallback when a
single dominant backend doesn't exist.

Usage: python aggregate_results.py
Writes: results/final_table.csv, results/pareto.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless devcontainer, no display
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def pareto_front(df: pd.DataFrame, maximize: list[str]) -> pd.Series:
    """Boolean mask: True for rows not dominated by any other row on all
    `maximize` columns simultaneously (standard Pareto dominance)."""
    is_dominated = pd.Series(False, index=df.index)
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i == j:
                continue
            better_or_equal = all(row_j[c] >= row_i[c] for c in maximize)
            strictly_better = any(row_j[c] > row_i[c] for c in maximize)
            if better_or_equal and strictly_better:
                is_dominated[i] = True
                break
    return ~is_dominated


def main() -> None:
    acc = pd.read_csv(RESULTS_DIR / "accuracy_summary.csv")
    vid = pd.read_csv(RESULTS_DIR / "video_summary.csv")
    df = acc.merge(vid, on="backend", suffixes=("_coco", "_video"))

    # Lower jerk is better -- invert so every column in `maximize` follows
    # "higher is better", which is what pareto_front() assumes.
    df["temporal_stability"] = 1.0 / df["mean_normalized_jerk"]

    df.to_csv(RESULTS_DIR / "final_table.csv", index=False)
    print(df.to_string(index=False))

    maximize = ["mean_oks", "fps_achieved", "temporal_stability"]
    df["on_pareto_front"] = pareto_front(df, maximize)
    df.to_csv(RESULTS_DIR / "final_table.csv", index=False)
    print("\nPareto-optimal backends:", df.loc[df.on_pareto_front, "backend"].tolist())

    # Simple weighted ranking as a fallback / sanity check alongside the
    # Pareto front (equal weights -- documented in 03_resultats.md as a
    # starting point, not a claim that these weights are "correct").
    normalized = df[maximize].apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9))
    df["weighted_score"] = normalized.mean(axis=1)
    df.sort_values("weighted_score", ascending=False).to_csv(
        RESULTS_DIR / "ranking.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in df.iterrows():
        marker = "o" if row.on_pareto_front else "x"
        ax.scatter(row.fps_achieved, row.mean_oks, s=80, marker=marker, label=row.backend)
        ax.annotate(row.backend, (row.fps_achieved, row.mean_oks), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Vitesse (FPS, GPU)")
    ax.set_ylabel("Précision (OKS moyen, sous-ensemble COCO)")
    ax.set_title("Front de Pareto -- précision vs vitesse\n(o = non dominé, x = dominé)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pareto.png", dpi=150)
    print(f"\nWrote {RESULTS_DIR / 'pareto.png'}")


if __name__ == "__main__":
    main()
