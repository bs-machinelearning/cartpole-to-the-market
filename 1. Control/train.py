import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

ALGORITHMS = {"PPO": PPO, "DQN": DQN}
# Column names and rules: see results/README.md. Every run goes in the shared log.
FIELDS = ["run_id", "date", "owner", "git_commit", "config", "environment", "algorithm", "seed", "fold", "status", "test_sharpe_per_period", "other_metrics", "returns_path"]
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ENVS = {"CartPole-v1", "Acrobot-v1"}


def repo_relative(path):
    """Path relative to the repo root, so the log reads the same on every laptop."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an RL agent from a JSON config")
    parser.add_argument("--config", default="config.json", help="Path to JSON config")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = config["environment"]
    algorithm = config["algorithm"].upper()
    seed = int(config["seed"])
    budget = int(config["budget"])
    eval_episodes = int(config.get("eval_episodes", 20))
    owner = config["owner"]  # first name, e.g. "Stefan"
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm must be one of: {', '.join(ALGORITHMS)}")
    if budget <= 0 or eval_episodes <= 0:
        raise ValueError("budget and eval_episodes must be positive")

    # Resolve paths relative to the config, so commands work from any directory.
    base = config_path.parent
    log_path = REPO_ROOT / config.get("run_log", "results/run_log.csv")  # relative to repo root
    model_dir = base / config.get("model_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{algorithm}_s{seed}"
    model_path = model_dir / run_id

    is_control = environment in CONTROL_ENVS
    # Status stays "abandoned" if the run is stopped by hand (Ctrl+C) before it finishes.
    row = {"run_id": run_id, "date": datetime.now(timezone.utc).date().isoformat(),
           "owner": owner, "git_commit": git_commit(), "config": repo_relative(config_path),
           "environment": environment, "algorithm": algorithm, "seed": seed,
           "fold": "none" if is_control else "", "status": "abandoned",
           "test_sharpe_per_period": "", "other_metrics": "", "returns_path": ""}

    env = None
    eval_env = None
    start = time.perf_counter()
    try:
        env = Monitor(gym.make(environment))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        model = ALGORITHMS[algorithm]("MlpPolicy", env, seed=seed, verbose=0)
        model.learn(total_timesteps=budget)
        training_seconds = time.perf_counter() - start
        model.save(str(model_path))

        eval_env = Monitor(gym.make(environment))
        eval_env.reset(seed=seed + 1000)
        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes,
                                    deterministic=True, warn=False)
        row["status"] = "control" if is_control else "completed"
        row["other_metrics"] = json.dumps({
            "training_seconds": round(training_seconds, 2),
            "mean_return": round(float(mean), 2),
            "std_return": round(float(std), 2),
            "eval_episodes": eval_episodes,
            "model_path": repo_relative(str(model_path) + ".zip"),
        })
        print(f"{algorithm} on {environment} | seed={seed} | steps={budget}")
        print(f"Training seconds: {training_seconds:.2f}")
        print(f"Final mean return over {eval_episodes} episodes: {mean:.2f} +/- {std:.2f}")
        print(f"Saved model: {model_path}.zip")
    except Exception as exc:
        row["status"] = "crashed"
        row["other_metrics"] = json.dumps({"error": str(exc), "elapsed_seconds": round(time.perf_counter() - start, 2)})
        raise
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
        append_log(log_path, row)
        print(f"Logged run to: {log_path}")


if __name__ == "__main__":
    main()
