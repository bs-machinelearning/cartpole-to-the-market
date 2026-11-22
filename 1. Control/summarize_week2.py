"""Report median seed performance for each 4 experiment groups."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "run_log.csv"


def main():
    if not LOG.exists():
        raise SystemExit("No results/run_log.csv exists yet")
    groups = defaultdict(dict)
    with LOG.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "control":
                continue
            try:
                m = json.loads(row.get("other_metrics") or "{}")
                if m.get("eval_episodes") == 100:
                    key = (row["algorithm"], row["environment"])
                    groups[key][int(row["seed"])] = float(m["mean_return"])
            except (KeyError, ValueError, TypeError):
                continue
    print("Algorithm | Environment | Seeds | Median 100-episode reward | Solved?")
    for algorithm in ("PPO", "DQN"):
        for environment, threshold in (("CartPole-v1", 475), ("Acrobot-v1", -100)):
            scores = groups[(algorithm, environment)]
            if not scores:
                print(f"{algorithm} | {environment} | 0/10 | - | no results")
                continue
            median = statistics.median(scores.values())
            print(f"{algorithm} | {environment} | {len(scores)}/10 | {median:.2f} | {median >= threshold}")


if __name__ == "__main__":
    main()
