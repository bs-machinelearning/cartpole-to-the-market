# Week 2: RL control experiments

These files **augment** the existing `1. Control/` folder. Keep the old `config.json`, `config_dqn.json`, and the existing shared `results/run_log.csv`. Back up and commit the original `train.py` before replacing it with the updated version.

## Source of the configurations

The PPO and DQN settings were transcribed from RL Baselines3 Zoo's `master` branch on November 22, 2026. The exact commit used when the settings were copied was not recorded.

For a pinned reference snapshot, the RL Zoo v2.9.2 publishing commit is `61956d90d9f8ac126043ed4024efa12a462d97e1`:

- [PPO hyperparameters at the pinned commit](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo.yml)
- [DQN hyperparameters at the pinned commit](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/dqn.yml)

This is a reproducible reference, not proof that these exact files were used for the original runs. Compare the pinned YAML values with the saved configs before attributing the original runs to this commit. The commit is recorded in the official [RL Zoo v2.9.2 publication provenance](https://pypi.org/project/rl-zoo3/2.9.2/).

The CartPole DQN entry is labeled **“Almost Tuned”** by the Zoo. The PPO Acrobot entry uses Zoo's default PPO learning rate, clip range, and batch size, plus 16 parallel environments and VecNormalize. Linear schedules are implemented as functions rather than passed as raw strings.

## Set up

At the root of your cloned repository:

```bash
uv sync
uv run python '1. Control/run_week2.py' --dry-run
