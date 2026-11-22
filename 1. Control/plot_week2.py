"""Plot cross-seed mean +/- standard deviation of *evaluation* learning curves."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "run_log.csv"
OUT = Path(__file__).resolve().parent / "figures"


def main():
    if not LOG.exists():
        raise SystemExit("No results/run_log.csv yet; train models first")
    groups = defaultdict(list)
    with LOG.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "control":
                continue
            try:
                metrics = json.loads(row.get("other_metrics") or "{}")
                path = ROOT / metrics["learning_curve_path"]
                with path.open(newline="", encoding="utf-8") as curve_file:
                    points = [(float(p["timesteps"]), float(p["mean_reward"])) for p in csv.DictReader(curve_file)]
                if points:
                    groups[(row["algorithm"], row["environment"])].append(points)
            except (KeyError, OSError, ValueError, TypeError):
                continue
    OUT.mkdir(parents=True, exist_ok=True)
    for (algo, env), curves in sorted(groups.items()):
        # Common x-range prevents extrapolating shorter seeds beyond their observations.
        start = max(p[0][0] for p in curves)
        end = min(p[-1][0] for p in curves)
        if start > end:
            continue
        xs = np.linspace(start, end, 100) if end > start else np.array([start])
        values = np.array([np.interp(xs, [v[0] for v in curve], [v[1] for v in curve]) for curve in curves])
        avg, sd = values.mean(axis=0), values.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        for curve in curves:
            ax.plot([p[0] for p in curve], [p[1] for p in curve], alpha=0.15)
        ax.plot(xs, avg, linewidth=2, label="Mean across seeds")
        ax.fill_between(xs, avg-sd, avg+sd, alpha=0.2, label="+/- 1 std dev")
        ax.axhline(475 if env == "CartPole-v1" else -100, linestyle="--", label="Solved threshold")
        ax.set(title=f"{algo} / {env} (n={len(curves)} seeds)", xlabel="Training timesteps", ylabel="Mean episode reward (10 evaluation episodes)")
        ax.legend()
        fig.tight_layout()
        path = OUT / f"{algo.lower()}_{env.lower().replace('-v1', '')}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(path)
    if not groups:
        print("No valid learning-curve files found")


if __name__ == "__main__":
    main()
