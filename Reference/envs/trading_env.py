"""Long/flat daily trading environment (Gymnasium interface). Owner: Env.

One asset, one position (0 = flat, 1 = long), one decision per trading day.

Timing rule
-----------
``log_returns[j]`` is the log return realised on day j, from the close of day
j-1 to the close of day j. To earn it, the position has to be chosen at the
close of day j-1, knowing only the returns up to day j-1. So, for every day j
in ``days``, one step does this:

    the agent sees    log_returns[j-k : j] and its current position
    the agent picks   a position p (0 or 1), held from close j-1 to close j
    the reward is     p * log_returns[j] - cost * |p - previous position|

Rewards are log returns net of costs. Their sum is the log of final wealth over
initial wealth, so ``wealth = exp(sum of rewards)``; for a long/flat position
this compounds exactly. Always flat earns exactly 0. Always long at zero cost
earns exactly the asset's own log return.

Assumptions (state them in the report): the trade is filled at the same close
the decision is based on; the cost is a constant fraction of the value traded;
cash earns nothing; no leverage and no short selling.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

FLAT, LONG = 0, 1


class TradingEnv(gym.Env):
    """Trade one asset, long or flat, one step per day.

    Parameters
    ----------
    log_returns : daily log returns of one asset in date order; entry j is the
        return of day j (close j-1 to close j).
    k : observation window, the number of past daily returns the agent sees.
    cost_bps : cost of changing position, in basis points of the value traded
        (fee plus half the bid-ask spread). No default on purpose: a forgotten
        cost would silently become the zero-cost shortcut.
    days : indices of the days whose return can be earned (the train or the
        test window). They need not be contiguous. Default: every day from k on.
    peek : if True, the observation also contains log_returns[j], the return the
        position is about to earn. This is the deliberate one-bar look-ahead
        (shortcut P). It must stay False in honest runs.
    obs_scale : constant multiplying the returns in the observation (100 shows
        them in percent). A fixed number, not a statistic estimated from data,
        so it cannot leak information.
    dates : optional labels, one per entry of log_returns, copied into the records.

    Observation: ``[r(j-k), ..., r(j-1), position]``, plus ``r(j)`` when peek is on.
    Action: 0 = flat, 1 = long. It is the position to hold, so repeating it is free.
    Episode: starts flat on the first day of ``days`` and ends (truncated) after
    the last one. The last position is not sold: no cost is charged at the end.
    """

    metadata = {"render_modes": []}

    def __init__(self, log_returns, *, k, cost_bps, days=None, peek=False, obs_scale=100.0, dates=None):
        super().__init__()
        returns = np.asarray(log_returns, dtype=np.float64)
        if returns.ndim != 1:
            raise ValueError("log_returns must be a one-dimensional sequence")
        if not np.all(np.isfinite(returns)):
            raise ValueError("log_returns contains NaN or infinite values")
        if int(k) != k or k < 1:
            raise ValueError("k must be a whole number of days, at least 1")
        if not cost_bps >= 0:
            raise ValueError("cost_bps cannot be negative")
        n, k = len(returns), int(k)
        days = np.arange(k, n) if days is None else np.asarray(days)
        if days.ndim != 1 or len(days) == 0 or not np.issubdtype(days.dtype, np.integer):
            raise ValueError("days must be a non-empty list of whole-number indices")
        if np.any(np.diff(days) <= 0):
            raise ValueError("days must be strictly increasing")
        if days[0] < k or days[-1] > n - 1:
            raise ValueError(f"days must lie between k={k} and {n - 1}: each day needs k earlier returns")
        if dates is not None and len(dates) != n:
            raise ValueError("dates must have one entry per log return")

        self._returns = returns
        self.days = days.astype(np.int64)
        self.k = k
        self.cost = float(cost_bps) / 10_000
        self.peek = bool(peek)
        self.obs_scale = float(obs_scale)
        self.dates = dates

        self.action_space = spaces.Discrete(2)
        size = k + 1 + int(self.peek)
        limit = float(np.finfo(np.float32).max)  # no real bound on a return; finite keeps check_env quiet
        self.observation_space = spaces.Box(-limit, limit, shape=(size,), dtype=np.float32)

        self._step = 0  # how many days of the window have been traded
        self._position = FLAT
        self._log_wealth = 0.0
        self.history: list[dict] = []  # one record per day traded in the current episode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # no randomness here: same data and actions give the same rewards
        self._step = 0
        self._position = FLAT
        self._log_wealth = 0.0
        self.history = []
        return self._observation(), {}

    def step(self, action):
        if self._step >= len(self.days):
            raise RuntimeError("the episode is over: call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be 0 (flat) or 1 (long), got {action!r}")
        target = int(action)
        day = int(self.days[self._step])

        trade = abs(target - self._position)
        cost = self.cost * trade
        gross = target * float(self._returns[day])
        reward = gross - cost

        self._position = target
        self._log_wealth += reward
        self._step += 1

        info = {
            "day": day,
            "position": target,
            "trade": trade,
            "cost": cost,
            "gross_return": gross,
            "net_return": reward,
            "wealth": math.exp(self._log_wealth),
        }
        if self.dates is not None:
            info["date"] = self.dates[day]
        self.history.append(dict(info))

        truncated = self._step >= len(self.days)  # out of data: a time limit, not a terminal state
        return self._observation(), reward, False, truncated, info

    def _observation(self):
        """What is known when the next position is chosen: past returns and the current position."""
        if self._step < len(self.days):
            day = int(self.days[self._step])  # the day whose return is about to be earned
            end, next_return = day, self._returns[day]
        else:  # after the last day: the window ending on it, and nothing left to peek at
            end, next_return = int(self.days[-1]) + 1, 0.0
        obs = np.empty(self.observation_space.shape, dtype=np.float32)
        obs[: self.k] = self._returns[end - self.k : end] * self.obs_scale
        obs[self.k] = self._position
        if self.peek:
            obs[self.k + 1] = next_return * self.obs_scale
        return obs


def run_policy(env, policy):
    """Play one episode and return the per-day records (``env.history``).

    ``policy`` is either a function ``observation -> action`` or a ready-made
    sequence of actions, one per day. Baselines and scripted checks should go
    through here, so they pay the same costs under the same rules as an agent.
    """
    actions = None if callable(policy) else iter(policy)
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy(obs) if actions is None else next(actions)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.unwrapped.history
