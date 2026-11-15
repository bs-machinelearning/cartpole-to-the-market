# CartPole RL training starter

## Install

Nothing extra: the project's packages come from `uv sync` at the repo root (see `readme_code.md`).

## Train and log

```bash
uv run python train.py --config config.json
uv run python train.py --config config_dqn.json
```

Each run adds one row to the shared log, `results/run_log.csv` (one row **per attempt**, including crashed and stopped runs), and saves a model checkpoint under `models/` (not uploaded to GitHub). The output gives training time in seconds and mean/std episode reward on the evaluation episodes. Configurable fields: `environment`, `algorithm`, `seed`, `budget` (training timesteps), `eval_episodes`, `owner` (your first name), `run_log` (path from the repo root), and `model_dir`.

Run it from inside the repo so the git commit is recorded. Column rules are in `results/README.md`; check the log with `uv run python -m Reference.evaluation.check_run_log`.

In CartPole, mean episode reward measures how many steps the agent balances the pole (up to 500). CartPole is not a trading environment: these runs get status `control`, fold `none`, and an empty Sharpe and returns path. The goal of these first two runs is a **feasibility check**, not necessarily reaching the later 475/500 solved threshold at this budget. If the score is low, increase budget and log the additional run.
