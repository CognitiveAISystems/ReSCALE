import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONFIGS = [
    ("alpha-zero", "#7DC4BB", "s", "AlphaZero"),
    ("rescale",    "#8A7FC6", "o", "ReSCALE"),
]


def load_tsv(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    tokens = np.array([float(r["tokens"]) for r in rows])
    accs = np.array([float(r["accuracy"]) for r in rows])
    return tokens, accs


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs tokens.")
    args = parser.parse_args()

    name = "gsm8k"

    fig, ax = plt.subplots(figsize=(6, 4))

    for method, color, marker, label in CONFIGS:
        tsv_path = DATA_DIR / f"{name}_{method}.tsv"
        if not tsv_path.exists():
            continue
        tokens, accs = load_tsv(tsv_path)
        ax.scatter(tokens, accs, c=color, marker=marker, label=label,
                   alpha=0.5, edgecolors="black", linewidths=0.8, s=40, zorder=3)
        if len(tokens) >= 2:
            coeffs = np.polyfit(tokens, accs, 1)
            t_sorted = np.sort(tokens)
            ax.plot(t_sorted, np.polyval(coeffs, t_sorted),
                    color=color, linewidth=2, linestyle="--", zorder=2)

    ax.set_xlabel("Token Budget", fontsize=18)
    ax.set_ylabel("Accuracy", fontsize=18)
    ax.set_title(name.upper(), fontsize=18, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelsize=12)

    leg = ax.legend(fontsize=15, ncol=1, loc="lower left", framealpha=0.6,
                     borderpad=0.4, handletextpad=0.3, columnspacing=0.5,
                     fancybox=True, shadow=False, edgecolor="0.75",
                     markerscale=1.3, bbox_to_anchor=(0.09, 0))
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    fig.savefig(f"{name}.png", dpi=450, bbox_inches="tight")
    print(f"Saved {name}.png")


if __name__ == "__main__":
    main()
