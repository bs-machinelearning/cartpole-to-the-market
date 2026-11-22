# Week 2: RL control experiments

These files **augment** the existing `1. Control/` folder. Keep the old `config.json`, `config_dqn.json`, and the existing shared `results/run_log.csv`. Replace the original `train.py` with this updated version after committing/backing it up.

## Source of the configurations

These configurations are transcribed from RL Baselines3 Zoo `master` on November 22, 2026:
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

The CartPole DQN entry is labeled **"Almost Tuned"** by the Zoo. The PPO Acrobot entry uses Zoo's default PPO learning rate, clip range, and batch size, plus 16 parallel environments and VecNormalize. Linear schedules are implemented as functions rather than passed as raw strings.

## Set up

At the root of your cloned repository:

```bash
uv sync
uv run python '1. Control/run_week2.py' --dry-run
```

Check `uv run python -c "import matplotlib"`; if matplotlib is missing, install it in your project with `uv add matplotlib`, committing the `pyproject.toml` and lock-file changes.

## Run

One run first:

```bash
uv run python '1. Control/run_week2.py' --only ppo_cartpole --seeds 1
```

Then run all **remaining** experiments (up to 40 total, seeds 0–9). The runner skips finished 100-episode control runs recorded in the shared run log:

```bash
uv run python '1. Control/run_week2.py'
```

Run the summary and figures:

```bash
uv run python '1. Control/summarize_week2.py'
uv run python '1. Control/plot_week2.py'
uv run python -m Reference.evaluation.check_run_log
```

`1. Control/figures/` will contain four mean-and-spread learning curves. `1. Control/curves/` contains raw checkpoint evaluation curves, and `1. Control/models/` contains trained checkpoints and any VecNormalize statistics. Avoid committing huge model files unless the team wants them stored; keep metrics, configs, code and final figures in Git.

## Key details

- Four groups: PPO/DQN x CartPole-v1/Acrobot-v1, ten seeds each.
- Each **final** test is 100 deterministic episodes in a separate evaluation environment.
- Progress plots use 10 deterministic episodes per checkpoint and show mean +/- std **across seeds**; they are not the final 100-episode scores.
- Score thresholds: CartPole >= 475; Acrobot >= -100. The checkpoint requires the **median seed** in each group to solve.
- Training budgets follow the Zoo configs: 100k, 50k, 1M and 100k timesteps respectively. Full 40-run training may take significant CPU time, especially PPO Acrobot; budget time and disk accordingly.
- Do not edit hyperparameters after viewing results without logging the change as another trial.
- If a run fails, its failure is logged. The runner stops and can be resumed after fixing the issue.
- For full reproducibility, record package versions and the precise RL Zoo commit in the report, because the Zoo master branch can change.

## Note on experiment state

No complete 40-run training has been executed in this delivered bundle. Training and result scores must be generated on your own machine/remote runtime. This package contains scripts/configs, not fabricated experimental results.
