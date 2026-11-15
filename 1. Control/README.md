# CartPole RL training starter

## Install

From this folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import gymnasium, stable_baselines3; print('Both imports work')"
```

## Train and log

```bash
python train.py --config config.json
python train.py --config config_dqn.json
```

This creates `run_log.csv` (one row **per attempt**, including failed runs), and model checkpoints under `models/`. The output gives training time in seconds and mean/std episode reward on 20 evaluation episodes. Configurable fields: `environment`, `algorithm`, `seed`, `budget` (training timesteps), `eval_episodes`, `owner`, `run_log`, and `model_dir`.

In CartPole, mean episode reward measures how many steps the agent balances the pole (up to 500). CartPole is not a trading environment: Sharpe, fold/window, and saved financial returns are marked not applicable. The goal of these first two runs is a **feasibility check**, not necessarily reaching the later 475/500 solved threshold at this budget. If the score is low, increase budget and log the additional run.

For the project's shared run log, `run_log.csv` has the requested column names, so the CSV can be imported into a spreadsheet. Each completed run includes the score and timing in the `other metrics` JSON cell. `git commit` is recorded if running within a Git repository.
