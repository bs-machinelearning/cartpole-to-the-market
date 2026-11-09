# From CartPole to the Market: code guide

This repository contains the code, notebooks and report material for project 16. A reinforcement-learning agent is first proven correct on CartPole and Acrobot, then moved to a market and evaluated honestly. Common backtesting shortcuts are switched on one at a time to measure how much fake performance each adds, and the results are deflated for the number of configurations tried.

## Repository structure

```text
1. Control/               Step 1: PPO and DQN on CartPole and Acrobot
2. Market Environment/    Step 2: building and testing the trading simulator
3. Honest Evaluation/     Step 3: walk-forward splits, metrics, baselines, honest runs
4. Break It/              Step 4: the shortcut factorial and the decomposition
5. Deflation/             Step 5: trial counting and the Deflated Sharpe Ratio
Reference/                Final shared code: envs, agents, data, evaluation, experiments
results/                  run_log.csv and saved results
img/                      Figures used by the README and the report
resources/                Proposal and background reading
Report.typ                Main Typst report
pyproject.toml            Python project metadata and dependencies
```

## Setup

Install `uv`, then from the repository root:

```bash
uv sync
```

Open notebooks with the environment created by `uv`. In VS Code, select the corresponding Jupyter kernel before running cells. Price data is downloaded into `data/`, which is not committed.

## Rules

- One notebook per person in each stage folder, named `Firstname.ipynb` (for example `Christina.ipynb`).
- Code others depend on moves to `Reference/` and gets a test.
- Every scored run gets a row in `results/run_log.csv`, abandoned runs included.

## Report

```bash
typst compile Report.typ Report.pdf
```
