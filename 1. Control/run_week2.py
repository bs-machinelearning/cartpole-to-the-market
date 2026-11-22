"""Run the complete 2 x 2 x 10 seed control-stage matrix; skip already completed runs."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIGS = ["config_ppo_cartpole.json", "config_dqn_cartpole.json", "config_ppo_acrobot.json", "config_dqn_acrobot.json"]


def completed(log_path):
    if not log_path.exists():
        return set()
    with log_path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {(r["algorithm"], r["environment"], int(r["seed"]))
                for r in rows if r.get("status") == "control" and r.get("seed", "").isdigit()
                and json.loads(r.get("other_metrics") or "{}").get("eval_episodes", 0) == 100}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", choices=["ppo_cartpole", "dqn_cartpole", "ppo_acrobot", "dqn_acrobot"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=0)
    args = p.parse_args()
    if not 1 <= args.seeds <= 100:
        p.error("--seeds must be between 1 and 100")
    done = completed(ROOT / "results" / "run_log.csv")
    tasks = []
    for filename in CONFIGS:
        key = filename.removeprefix("config_").removesuffix(".json")
        if args.only and key != args.only:
            continue
        cfg = json.loads((HERE / filename).read_text(encoding="utf-8"))
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            if (cfg["algorithm"], cfg["environment"], seed) not in done:
                tasks.append((filename, seed))
    print(f"Pending runs: {len(tasks)}", flush=True)
    for i, (filename, seed) in enumerate(tasks, 1):
        cmd = [sys.executable, str(HERE / "train.py"), "--config", str(HERE / filename), "--seed", str(seed)]
        print(f"[{i}/{len(tasks)}] {filename} seed={seed}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("Batch finished" if not args.dry_run else "Dry run only: no training performed")


if __name__ == "__main__":
    main()
