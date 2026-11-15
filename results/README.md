# Results

`run_log.csv` has one row per scored run, abandoned runs included. Nothing is ever deleted: abandoned runs get a status. The Deflated Sharpe Ratio counts every row.

## How to fill in each column

| Column | Rule | Example |
|---|---|---|
| `run_id` | Unique ID: date, algorithm, environment, seed | `20261115_ppo_cartpole_s0` |
| `date` | YYYY-MM-DD | `2026-11-15` |
| `owner` | First name of whoever launched the run | `Stefan` |
| `git_commit` | Short hash of the code that produced the run (`git rev-parse --short HEAD`); `unknown` only for runs made outside the repo (gives a warning) | `a1b2c3d` |
| `config` | Path to the config file the run started from | `configs/ppo_cartpole.yaml` |
| `environment` | Gymnasium ID or our market env name | `CartPole-v1` |
| `algorithm` | `PPO` or `DQN` (or a baseline name: `buy_hold`, `flat`, `random_turnover`) | `PPO` |
| `seed` | Integer seed used for Python, NumPy, PyTorch and the environment | `0` |
| `fold` | Walk-forward fold number; `none` for CartPole and Acrobot runs | `none` |
| `status` | One of `completed`, `abandoned`, `crashed`, `control` | `control` |
| `test_sharpe_per_period` | Daily Sharpe ratio on the test window, **not annualised**; empty for control runs | `0.042` |
| `other_metrics` | Short JSON string, no commas outside the JSON quotes | `{"mean_return": 500}` |
| `returns_path` | Path to the saved test-period daily return series; empty for control runs | `results/returns/20261115_ppo_spy_s0.csv` |

Daily, non-annualised Sharpe ratios matter: the Deflated Sharpe Ratio needs every run in the same units.

## Checking the log

From the repo root, run:

```bash
uv run python -m Reference.evaluation.check_run_log
```

It lists every row that breaks a rule above, with its line number. Eval runs it at every weekly run-log check.
