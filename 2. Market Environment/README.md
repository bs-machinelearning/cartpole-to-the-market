# 2. Market Environment

Market environment (Step 2). Development and testing of the trading simulator: position, fee and spread, reward equal to the change in wealth after costs. Owner: Env.

One notebook per person, named `Firstname.ipynb` (for example `Christina.ipynb`). Code that works and that others rely on moves to `Reference/`.

## What is here

- [`interface.md`](interface.md): what the environment takes in, what it gives back, what it assumes, and the open questions for each role.
- [`protocol_proposal.md`](protocol_proposal.md): Env's proposal for the protocol (observation window, actions, cost levels), with the reason for each value.
- [`Francesco.ipynb`](Francesco.ipynb): a tour of the environment on a toy price series and on SPY.
- [`feasibility.py`](feasibility.py) and [`feasibility.json`](feasibility.json): train PPO and DQN on a synthetic market to check that an agent can learn in this environment.

The environment itself is in `Reference/envs/`, next to its tests. Run them with `uv run python -m pytest Reference`.
