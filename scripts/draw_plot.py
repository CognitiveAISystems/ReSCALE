import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CONFIGS = [
    ("alpha-zero", False, "#7DC4BB", "s", "AlphaZero"),
    ("rescale",     False, "#8A7FC6", "o", "ReSCALE"),
]


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs tokens.")
    parser.add_argument("file", nargs="?", default="results/GSM8K.json",
                        help="Path to JSON data file (default: GSM8K.json)")
    args = parser.parse_args()

    path = Path(args.file)
    name = path.stem
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))

    for method, cs, color, marker, label in CONFIGS:
        subset = [d for d in data if d["method"] == method and d["clear_subtrees"] == cs]
        if not subset:
            continue
        tokens = np.array([d["tokens"] for d in subset])
        accs = np.array([d["accuracy"] for d in subset])
        ax.scatter(tokens, accs, c=color, marker=marker, label=label,
                   alpha=0.5, edgecolors="black", linewidths=0.8, s=40, zorder=3)
        if len(tokens) >= 2:
            coeffs = np.polyfit(tokens, accs, 1)
            t_sorted = np.sort(tokens)
            ax.plot(t_sorted, np.polyval(coeffs, t_sorted),
                    color=color, linewidth=2, linestyle="--", zorder=2)

    ax.set_xlabel("Token Budget", fontsize=18)
    ax.set_ylabel("Accuracy", fontsize=18)
    ax.set_title(name, fontsize=18, fontweight="bold")
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
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=450, bbox_inches="tight")
    print(f"Saved {name}.pdf and {name}.png")


if __name__ == "__main__":
    main()
