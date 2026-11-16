"""Feasibility check for the trading environment, on a made-up market. Owner: Env.

Run from the repo root:

    uv run python "2. Market Environment/feasibility.py"

The "market" here is a series of random numbers, never real prices, so nothing
in this script can tune the project on the data it will later be judged on.
Because the series is random, the right answers are known in advance:

1. Positive control. With the one-bar peek ON, tomorrow's return is in the
   observation. An agent that works must learn to use it and get close to the
   oracle. If it does not, the environment or the training set-up is broken.
2. Negative control. With the peek OFF there is nothing to learn. On fresh
   random data the agent must end near zero, whatever it scored in training.

It also records how long one training run takes on this environment.
Every run adds a row to the shared run log with status "control".
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Reference.envs import ENV_ID, TradingEnv, run_policy  # noqa: E402
from Reference.evaluation.check_run_log import COLUMNS  # noqa: E402

ALGORITHMS = {"PPO": PPO, "DQN": DQN}


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def repo_relative(path):
    """Path relative to the repo root, so the log reads the same on every laptop."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def score(model, returns, config, run):
    """Total log return of the trained agent on a series, acting deterministically."""
    env = TradingEnv(returns, k=config["k"], cost_bps=config["cost_bps"], days="all", peek=run["peek"],
                     obs_scale=run["obs_scale"])
    history = run_policy(env, lambda obs: model.predict(obs, deterministic=True)[0])
    return {
        "total_log_return": round(sum(row["net_return"] for row in history), 4),
        "switches": int(sum(row["trade"] for row in history)),
        "share_of_days_long": round(float(np.mean([row["position"] for row in history])), 3),
    }


def scripted(returns, config):
    """What the simple policies earn on the same series: the yardsticks."""
    k, cost_bps = config["k"], config["cost_bps"]
    traded = returns[k:]
    policies = {
        "flat": np.zeros(len(traded), dtype=int),
        "always_long": np.ones(len(traded), dtype=int),
        "oracle": (traded > 0).astype(int),  # long exactly on the days that go up
    }
    return {
        name: round(sum(row["net_return"] for row in run_policy(TradingEnv(returns, k=k, cost_bps=cost_bps, days="all"), actions.tolist())), 4)
        for name, actions in policies.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Trading environment feasibility check on synthetic data")
    parser.add_argument("--config", default=str(Path(__file__).with_name("feasibility.json")))
    parser.add_argument("--run-log", default=None, help="write rows here instead of the path in the config")
    parser.add_argument("--budget", type=int, default=None, help="override the training budget (for a quick try)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    budget = args.budget or int(config["budget"])
    seed = int(config["seed"])
    log_path = Path(args.run_log).resolve() if args.run_log else REPO_ROOT / config["run_log"]

    # The made-up market: independent daily log returns, no drift, no pattern.
    # Train and test are separate draws, so the test series is new to the agent.
    k, sigma = config["k"], config["daily_volatility"]
    train = np.random.default_rng(seed).normal(0.0, sigma, config["train_days"] + k)
    test = np.random.default_rng(seed + 1).normal(0.0, sigma, config["test_days"] + k)
    yardsticks = {"train": scripted(train, config), "test": scripted(test, config)}

    started = datetime.now(timezone.utc)
    out_dir = REPO_ROOT / config["output_dir"] / started.strftime("%Y%m%dT%H%M%S%fZ")
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    uncommitted = bool(git("status", "--porcelain", "--untracked-files=no"))
    results = []

    for run in config["runs"]:
        algorithm = run["algorithm"].upper()
        run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                  + f"_{algorithm}_s{seed}_peek{int(run['peek'])}_obs{run['obs_scale']:g}_rew{run['reward_scale']:g}")
        row = {"run_id": run_id, "date": started.date().isoformat(), "owner": config["owner"], "git_commit": commit,
               "config": repo_relative(config_path), "environment": f"{ENV_ID}-synthetic",
               "algorithm": algorithm, "seed": seed, "fold": "none", "status": "abandoned",
               "test_sharpe_per_period": "", "other_metrics": "", "returns_path": ""}
        try:
            env = TradingEnv(train, k=k, cost_bps=config["cost_bps"], days="all", peek=run["peek"],
                             obs_scale=run["obs_scale"])
            if run["reward_scale"] != 1:  # the agent trains on rescaled rewards; the accounting below never does
                env = gym.wrappers.TransformReward(env, lambda r, s=run["reward_scale"]: r * s)
            env = Monitor(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            model = ALGORITHMS[algorithm]("MlpPolicy", env, seed=seed, verbose=0)  # library defaults, as in 1. Control
            start = time.perf_counter()
            model.learn(total_timesteps=budget)
            seconds = time.perf_counter() - start
            env.close()

            on_train, on_test = score(model, train, config, run), score(model, test, config, run)
            metrics = {
                "peek": run["peek"], "obs_scale": run["obs_scale"], "reward_scale": run["reward_scale"],
                "k": k, "cost_bps": config["cost_bps"], "budget": budget,
                "training_seconds": round(seconds, 2), "steps_per_second": round(budget / seconds),
                "train": on_train, "test": on_test, "test_oracle": yardsticks["test"]["oracle"],
                "test_share_of_oracle": round(on_test["total_log_return"] / yardsticks["test"]["oracle"], 3),
            }
            if uncommitted:
                metrics["note"] = "run with uncommitted changes in the working tree"
            row["status"], row["other_metrics"] = "control", json.dumps(metrics)
            results.append({"run_id": run_id, "algorithm": algorithm, **metrics})
            print(f"{algorithm} peek={run['peek']!s:5} obs x{run['obs_scale']:<3g} reward x{run['reward_scale']:<3g} | "
                  f"{seconds:5.1f}s | train {on_train['total_log_return']:+.3f} | test {on_test['total_log_return']:+.3f} "
                  f"= {metrics['test_share_of_oracle']:+.0%} of oracle | {on_test['switches']} switches")
        except Exception as exc:
            row["status"], row["other_metrics"] = "crashed", json.dumps({"error": str(exc)})
            raise
        finally:
            append_log(log_path, row)

    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "what": "TradingEnv feasibility on synthetic i.i.d. returns (no market data)",
        "saved_at_utc": started.isoformat(), "git_commit": commit, "config": config, "budget": budget,
        "versions": {"gymnasium": gym.__version__, "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__},
        "noise_band_test": round(2 * sigma * config["test_days"] ** 0.5, 3),
        "yardsticks": yardsticks, "runs": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nYardsticks on the test series: {yardsticks['test']}")
    print(f"A total within +/-{summary['noise_band_test']} on the test series is indistinguishable from luck.")
    print(f"Saved: {out_dir / 'summary.json'}\nLogged {len(config['runs'])} rows to: {log_path}")


if __name__ == "__main__":
    main()
