"""Config-driven PPO/DQN control training, evaluation, learning-curve logging.
Run from repository root: uv run python '1. Control/train.py' --config '1. Control/config_ppo_cartpole.json'
"""
import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ALGORITHMS = {"PPO": PPO, "DQN": DQN}
FIELDS = ["run_id", "date", "owner", "git_commit", "config", "environment", "algorithm", "seed", "fold", "status", "test_sharpe_per_period", "other_metrics", "returns_path"]
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ENVS = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0}


def repo_relative(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
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


def linear_schedule(initial):
    return lambda progress_remaining: float(initial) * progress_remaining


def make_env(environment, seed):
    def _factory():
        env = Monitor(gym.make(environment))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _factory


class ProgressEvaluation(BaseCallback):
    """Independent evaluation checkpoints, separate from final 100-episode test."""
    def __init__(self, eval_env, csv_path, interval, n_episodes=10):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.csv_path = Path(csv_path)
        self.interval = int(interval)
        self.n_episodes = int(n_episodes)
        self.next_eval = self.interval

    def _on_training_start(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timesteps", "mean_reward", "std_reward", "episodes"])

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        if isinstance(self.training_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms
        mean, std = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_episodes,
                                    deterministic=True, warn=False)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self.num_timesteps, float(mean), float(std), self.n_episodes])
        print(f"  progress {self.num_timesteps}: reward={mean:.1f} +/- {std:.1f}", flush=True)
        self.next_eval += self.interval
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seed", type=int, help="Overrides the config's seed")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = config["environment"]
    algorithm = config["algorithm"].upper()
    seed = int(args.seed if args.seed is not None else config["seed"])
    budget = int(config["budget"])
    eval_episodes = int(config.get("eval_episodes", 100))
    if algorithm not in ALGORITHMS or environment not in CONTROL_ENVS:
        raise ValueError("Week 2 supports PPO/DQN on CartPole-v1/Acrobot-v1 only")
    if budget <= 0 or eval_episodes <= 0:
        raise ValueError("budget and eval_episodes must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = int(config.get("n_envs", 1))
    if algorithm == "DQN" and n_envs != 1:
        raise ValueError("DQN Zoo configs use one environment")
    normalize = bool(config.get("normalize", False))
    hp = config.get("hyperparameters", {}).copy()
    for name in ("learning_rate", "clip_range"):
        value = hp.get(name)
        if isinstance(value, str) and value.startswith("lin_"):
            hp[name] = linear_schedule(float(value[4:]))
    if "policy_kwargs" in hp and "net_arch" in hp["policy_kwargs"]:
        hp["policy_kwargs"] = dict(hp["policy_kwargs"])

    base = config_path.parent
    log_path = REPO_ROOT / config.get("run_log", "results/run_log.csv")
    model_dir = base / config.get("model_dir", "models")
    curve_dir = base / config.get("curve_dir", "curves")
    model_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{environment}_{algorithm}_s{seed}"
    model_path = model_dir / run_id
    curve_path = curve_dir / (run_id + ".csv")
    row = {"run_id": run_id, "date": datetime.now(timezone.utc).date().isoformat(),
           "owner": config.get("owner", "Stefan"), "git_commit": git_commit(),
           "config": repo_relative(config_path), "environment": environment,
           "algorithm": algorithm, "seed": seed, "fold": "none", "status": "abandoned",
           "test_sharpe_per_period": "", "other_metrics": "", "returns_path": ""}
    train_env = eval_env = None
    started = time.perf_counter()
    try:
        train_env = DummyVecEnv([make_env(environment, seed + i) for i in range(n_envs)])
        eval_env = DummyVecEnv([make_env(environment, seed + 1000)])
        if normalize:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
        model = ALGORITHMS[algorithm]("MlpPolicy", train_env, seed=seed, verbose=0, **hp)
        callback = ProgressEvaluation(eval_env, curve_path, config.get("curve_interval", 10000),
                                      config.get("curve_eval_episodes", 10))
        print(f"Starting {algorithm} / {environment} / seed {seed} / {budget} steps", flush=True)
        model.learn(total_timesteps=budget, callback=callback)
        training_seconds = time.perf_counter() - started
        model.save(str(model_path))
        norm_path = None
        if normalize:
            norm_path = str(model_path) + "_vecnormalize.pkl"
            train_env.save(norm_path)
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes,
                                    deterministic=True, warn=False)
        row["status"] = "control"
        row["other_metrics"] = json.dumps({
            "training_seconds": round(training_seconds, 2),
            "mean_return": round(float(mean), 2), "std_return": round(float(std), 2),
            "eval_episodes": eval_episodes, "training_timesteps": model.num_timesteps,
            "solved": bool(mean >= CONTROL_ENVS[environment]),
            "model_path": repo_relative(str(model_path) + ".zip"),
            "normalization_path": repo_relative(norm_path) if norm_path else None,
            "learning_curve_path": repo_relative(curve_path),
            "hyperparameters": config.get("hyperparameters", {}),
            "n_envs": n_envs, "normalize": normalize,
        })
        print(f"Finished: {mean:.2f} +/- {std:.2f}; solved={mean >= CONTROL_ENVS[environment]}")
    except Exception as exc:
        row["status"] = "crashed"
        row["other_metrics"] = json.dumps({"error": str(exc), "elapsed_seconds": round(time.perf_counter() - started, 2)})
        raise
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        append_log(log_path, row)
        print(f"Logged: {log_path}")


if __name__ == "__main__":
    main()
