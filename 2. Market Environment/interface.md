# Trading environment: interface (v1)

**Owner:** Env (Francesco) · **Status:** draft for the team to review. Values that belong to the protocol are proposals until it is frozen on 22 Nov; see [`protocol_proposal.md`](protocol_proposal.md).

`Reference/envs/trading_env.py` is the simulator the agents trade in: one asset, long or flat, one decision per trading day. It follows the Gymnasium template (`reset()`, `step(action)`), so the training code that solves CartPole runs on it unchanged.

## The timing rule

```
day             j-2         j-1          j
close            |           |           |
                             |
the agent sees the returns up to day j-1 and its current position
the agent picks flat or long at this close ...... holds it ...... earns the return of day j
the cost is charged here, only if the position changed
```

`log_returns[j]` is the log return of day j, from the close of day j-1 to the close of day j. The position that earns it is chosen at the close of day j-1 and can depend only on returns up to day j-1.

## What goes in and what comes out

| | |
|---|---|
| Data in | `log_returns`: daily log returns of one asset, in date order |
| Window | `days`: the positions in `log_returns` whose return can be earned (the train or the test window), or `"all"` |
| Observation | the last `k` log returns times `obs_scale` (default 100, so returns are in percent), then the current position (0 or 1). With `peek=True`, one more number: the return about to be earned |
| Action | 0 = flat, 1 = long. It is the position to hold, so repeating it costs nothing |
| Reward | `position × log_returns[j] − (cost_bps / 10 000) × |position − previous position|` |
| Wealth | starts at 1; `wealth = exp(sum of the rewards so far)` |
| Episode | starts flat on the first day of `days`; ends with `truncated=True` after the last one |
| Records | one row per day, in `info` and in `env.history`: `day`, `date`, `position`, `trade`, `cost`, `gross_return`, `net_return`, `wealth` |

`k`, `cost_bps` and `days` have no default. A forgotten cost would be the zero-cost shortcut and a forgotten window would mix train and test, so the environment refuses to guess.

Rewards are log returns, so adding them up compounds exactly: with a position of 0 or 1, `position × log return` is the change in log wealth.

## Worked example

Prices 100 → 100 → 110 → 121 → 108.9 → 108.9 → 119.79, `k = 1`, cost 10 bps (0.001). The first move is only history. Five days are traded and they return +10%, +10%, −10%, 0 and +10%; in log terms +10% is 0.09531 and −10% is −0.10536.

| Policy | Positions | Total at zero cost | Changes | Total at 10 bps |
|---|---|---|---|---|
| Always flat | 0 0 0 0 0 | 0 | 0 | 0 |
| Always long | 1 1 1 1 1 | 0.18057 | 1 | 0.17957 |
| Switch every day | 1 0 1 0 1 | 0.08526 | 5 | 0.08026 |
| Oracle (long on the up days) | 1 1 0 0 1 | 0.28593 | 3 | 0.28293 |

Three checks you can redo by hand. `exp(0.18057) = 1.1979 = 119.79 / 100`, so always long at zero cost is buy and hold. Each difference between the two totals is `0.001 × changes`. None of the 32 ways of playing the five days beats the oracle.

## Using it

```python
from Reference.envs import TradingEnv, run_policy

env = TradingEnv(log_returns, k=20, cost_bps=2, days=test_days)
history = run_policy(env, lambda obs: model.predict(obs, deterministic=True)[0])  # a trained agent
history = run_policy(env, [1] * len(test_days))                                   # buy and hold
daily_net_returns = [row["net_return"] for row in history]
```

`gym.make("TradingEnv-v0", log_returns=..., k=20, cost_bps=2, days=...)` builds the same environment by name.

## How we know it is right

- `uv run python -m pytest Reference` runs 57 environment tests in a few seconds: the scripted policies above, the timing rule, the cost rule, the peek option, and a leakage check (rewriting every return from the day about to be traded onwards leaves the observation unchanged).
- We planted 55 bugs, one at a time: reward shifted by a day in either direction, window including the traded day, cost charged on the wrong thing, state surviving a reset, and so on. The tests catch all 55.
- A second implementation written from the plan, without reading this code, agrees with the environment on 20,150 random cases. A copy of it is in the tests.
- `check_env` from Gymnasium and from Stable-Baselines3 pass with no warnings.
- The environment runs about 330,000 steps per second, so it is never the bottleneck.
- On a synthetic market made of random numbers, PPO and DQN find the answer when the peek shows it to them and make nothing on fresh data when it does not. See [`feasibility.py`](feasibility.py) and `results/francesco_feasibility/`.

These are Env's own checks. The plan's two outside checks are still to come in week 4: the leakage test written by Data and the review of the timing rule by Eval.

## Assumptions to state in the report

1. **Fill at the close we decide on.** The decision uses the close of day j-1 and the trade is filled at that same price. No return is ever visible at the step that earns it, but a real order has to be sent before the close is known. This is the usual convention for daily data and the one in the plan.
2. **Constant proportional cost.** Every change of position costs the same fraction of the value traded. No market impact.
3. **Cash earns nothing.** Flat is exactly zero.
4. **No exit cost at the end.** The position held after the last day is not sold. Always long pays the cost once, and each walk-forward fold pays an entry again.
5. **Cost in log units.** Wealth is multiplied by `exp(−c)` per change instead of `(1 − c)`. At 2 bps the two differ by 2 parts in 100 million per trade.
6. **Gaps are stepped over.** If `days` is not contiguous, a position held across a gap pays nothing for crossing it and neither earns nor loses the skipped returns.

## Choices we made, and what we set aside

| Choice | Alternative | Why not |
|---|---|---|
| Log returns everywhere | Simple returns on a fixed stake | Sums of simple returns do not compound, so "always long equals buy and hold" would hold only roughly |
| `days` as a list of positions | A start date and an end date | A range cannot describe the random split of Step 4, where train and test days are interleaved |
| Reward in natural units | Reward already multiplied by 100 | The reward must match the wealth path. Rescaling for training is one wrapper line on the RL side |
| Fill at the same close | Fill one day later | More conservative, but not the rule in the plan. It can become an option; after the freeze it would count as a trial |

## Risks

| Risk | What we do about it |
|---|---|
| Positions in `days` are off by one against the price file, because `np.diff(np.log(prices))` shifts the index by one | Agree the format with Data. `dates` are copied into every record, so a join by date exposes a shift |
| A policy is scored from `env.history` after Stable-Baselines3 has reset the environment by itself, which leaves the record empty | Score with `run_policy` on a bare environment, as above |
| The gap rule changes what scattered days cost, so the "split × cost" interaction of Step 4 partly measures the rule | Decide with Data and Exp before the January runs (last question below) |
| The peek is switched on by accident | It is off by default, and it changes the size of the observation, so a model trained without it cannot be run with it |

## Open questions

- **Data format** (Data, by 22 Nov). Proposed: one array of log returns per asset and, for each fold, the positions of its train days and of its test days. Are positions counted in the returns array or in the price file?
- **More than one ETF** (Lead and Data, by 22 Nov). The environment trades one asset. With several ETFs, one environment per ETF is enough. Is that what the protocol wants?
- **Records** (Eval, by 22 Nov). Are the fields above what the metrics need? Is the per-period Sharpe ratio computed on `net_return`, which is a log return?
- **Baselines** (Eval and Exp). Running buy and hold, always flat and the random policy through `run_policy` makes them pay the same costs under the same rules as the agent. Agreed?
- **Scaling** (RL). The observation shows returns in percent. On the synthetic market, training on rewards multiplied by 100 let both agents learn within 3,000 steps instead of needing far more. Does RL want that in the training script?
- **Random split** (Data and Exp, before January). Keep the gap rule, or run the whole series in order and count only the test days?
