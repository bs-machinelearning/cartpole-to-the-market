"""Tests for the trading environment. Run with: uv run python -m pytest Reference

Every expected number below can be redone by hand from the toy price series.
"""

import math
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from Reference.envs import ENV_ID, TradingEnv, run_policy

# Toy series: seven closing prices, six daily log returns.
#   day      0     1     2     3      4      5      6
#   price  100   100   110   121   108.9  108.9  119.79
#   move          0   +10%  +10%   -10%     0    +10%
PRICES = [100.0, 100.0, 110.0, 121.0, 108.9, 108.9, 119.79]
UP = math.log(1.1)  # 0.0953101798...
DOWN = math.log(0.9)  # -0.1053605157...
RETURNS = np.diff(np.log(PRICES))  # [0, UP, UP, DOWN, 0, UP]; entry j is the return of day j+1

# With k = 1 the first return is only history. Five days are traded; they earn UP, UP, DOWN, 0, UP.
K = 1
COST_BPS = 10.0  # 10 basis points = 0.001 per change of position
COST = 0.001

ALWAYS_FLAT = [0, 0, 0, 0, 0]
ALWAYS_LONG = [1, 1, 1, 1, 1]
SWITCH_DAILY = [1, 0, 1, 0, 1]  # earns UP + DOWN + UP, changes position 5 times
ORACLE = [1, 1, 0, 0, 1]  # long exactly on the up days: earns 3 * UP, changes position 3 times
ANTI_ORACLE = [0, 0, 1, 0, 0]  # long exactly on the down day: earns DOWN


def make(cost_bps=0.0, returns=RETURNS, k=K, **kwargs):
    return TradingEnv(returns, k=k, cost_bps=cost_bps, **kwargs)


def total(history):
    return sum(row["net_return"] for row in history)


def test_toy_returns_are_what_the_comments_say():
    assert RETURNS == pytest.approx([0.0, UP, UP, DOWN, 0.0, UP], abs=1e-12)


def test_always_flat_earns_exactly_zero():
    for cost_bps in (0.0, COST_BPS):
        history = run_policy(make(cost_bps), ALWAYS_FLAT)
        assert [row["net_return"] for row in history] == [0.0] * 5
        assert history[-1]["wealth"] == 1.0
        assert sum(row["trade"] for row in history) == 0


def test_always_long_at_zero_cost_equals_buy_and_hold():
    history = run_policy(make(0.0), ALWAYS_LONG)
    assert total(history) == pytest.approx(3 * UP + DOWN, abs=1e-12)
    assert total(history) == pytest.approx(math.log(119.79 / 100.0), abs=1e-12)
    assert history[-1]["wealth"] == pytest.approx(1.1979, abs=1e-12)  # 119.79 / 100


def test_always_long_pays_the_cost_once():
    history = run_policy(make(COST_BPS), ALWAYS_LONG)
    assert [row["cost"] for row in history] == pytest.approx([COST, 0, 0, 0, 0], abs=1e-15)
    assert total(history) == pytest.approx(3 * UP + DOWN - COST, abs=1e-12)


def test_switching_every_day_costs_exactly_cost_times_switches():
    free = run_policy(make(0.0), SWITCH_DAILY)
    paid = run_policy(make(COST_BPS), SWITCH_DAILY)
    switches = sum(row["trade"] for row in paid)
    assert switches == 5
    assert total(free) == pytest.approx(2 * UP + DOWN, abs=1e-12)
    assert total(free) - total(paid) == pytest.approx(COST * switches, abs=1e-12)


def test_cost_is_charged_only_when_the_position_changes():
    history = run_policy(make(COST_BPS), [1, 1, 0, 0, 1])
    assert [row["trade"] for row in history] == [1, 0, 1, 0, 1]
    assert [row["cost"] for row in history] == pytest.approx([COST, 0, COST, 0, COST], abs=1e-15)


def test_oracle_beats_every_other_scripted_policy():
    for cost_bps in (0.0, COST_BPS):
        scores = {
            name: total(run_policy(make(cost_bps), actions))
            for name, actions in [
                ("flat", ALWAYS_FLAT), ("long", ALWAYS_LONG), ("switch", SWITCH_DAILY),
                ("oracle", ORACLE), ("anti", ANTI_ORACLE),
            ]
        }
        assert all(scores["oracle"] > score for name, score in scores.items() if name != "oracle")
    assert total(run_policy(make(0.0), ORACLE)) == pytest.approx(3 * UP, abs=1e-12)
    assert total(run_policy(make(COST_BPS), ORACLE)) == pytest.approx(3 * UP - 3 * COST, abs=1e-12)


def test_timing_only_the_position_held_into_the_jump_earns_it():
    # Nothing moves except day 6, which jumps +10%. The position that earns it is
    # the one chosen at the close of day 5, the third step of the episode.
    returns = np.zeros(10)
    returns[6] = UP
    for step in range(5):
        actions = [0] * 5
        actions[step] = 1
        earned = total(run_policy(make(0.0, returns, k=4, days=[4, 5, 6, 7, 8]), actions))
        assert earned == pytest.approx(UP if step == 2 else 0.0, abs=1e-15)


def test_observation_is_the_past_window_then_the_position():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 30)
    env = make(0.0, returns, k=3, obs_scale=100.0)
    obs, _ = env.reset()
    position = 0
    for day in range(3, 30):  # default window: every day from k to the end
        assert obs[:3] == pytest.approx(100.0 * returns[day - 3 : day], abs=1e-5)
        assert obs[3] == position
        position = int(rng.integers(2))
        obs, reward, terminated, truncated, info = env.step(position)
        assert info["day"] == day
        assert reward == pytest.approx(position * returns[day], abs=1e-15)
    assert truncated and not terminated


def test_observation_never_contains_the_future():
    # Rewrite every return from the day about to be traded onwards: what the
    # agent sees before choosing its position must not change at all.
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 40)
    actions = rng.integers(2, size=35).tolist()
    for s in range(35):
        day = 5 + s
        tampered = returns.copy()
        tampered[day:] = rng.normal(0, 0.05, 40 - day)
        seen = []
        for series in (returns, tampered):
            env = make(COST_BPS, series, k=5)
            obs, _ = env.reset()
            for action in actions[:s]:
                obs, *_ = env.step(action)
            seen.append(obs)
        assert np.array_equal(seen[0], seen[1])


def test_peek_is_off_by_default_and_shows_the_next_return_when_on():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 30)
    assert make(0.0, returns, k=4).observation_space.shape == (5,)
    env = make(0.0, returns, k=4, peek=True, obs_scale=100.0)
    assert env.observation_space.shape == (6,)
    obs, _ = env.reset()
    for day in range(4, 30):
        assert obs[5] == pytest.approx(100.0 * returns[day], abs=1e-5)  # the return about to be earned
        assert obs[:4] == pytest.approx(100.0 * returns[day - 4 : day], abs=1e-5)
        obs, _, _, truncated, _ = env.step(1)
    assert truncated and obs[5] == 0.0  # nothing left to peek at after the last day


def test_days_can_be_a_scattered_subset():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, 20)
    days = [4, 5, 9, 15]
    env = make(COST_BPS, returns, k=2, days=days, obs_scale=1.0)
    obs, _ = env.reset()
    for day, action in zip(days, [1, 1, 0, 1]):
        assert obs[:2] == pytest.approx(returns[day - 2 : day], abs=1e-7)  # the true past of that day
        obs, *_ = env.step(action)
    history = env.history
    assert [row["day"] for row in history] == days
    assert [row["gross_return"] for row in history] == pytest.approx(
        [returns[4], returns[5], 0.0, returns[15]], abs=1e-15
    )
    assert [row["trade"] for row in history] == [1, 0, 1, 1]


def test_wealth_path_matches_the_summed_rewards():
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.01, 200)
    history = run_policy(make(COST_BPS, returns, k=10), rng.integers(2, size=190).tolist())
    running = np.cumsum([row["net_return"] for row in history])
    assert [row["wealth"] for row in history] == pytest.approx(np.exp(running), rel=1e-12)
    for row in history:
        assert row["net_return"] == pytest.approx(row["gross_return"] - row["cost"], abs=1e-15)
        assert row["position"] in (0, 1)


def test_same_actions_give_the_same_rewards_and_reset_starts_clean():
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, 60)
    actions = rng.integers(2, size=50).tolist()
    env = make(COST_BPS, returns, k=10)
    first = [dict(row) for row in run_policy(env, actions)]
    second = run_policy(env, actions)  # same environment object, played again
    assert first == second
    obs, info = env.reset(seed=123)
    assert obs[10] == 0 and env.history == [] and info == {}


def test_reset_starts_flat_again_after_an_episode_that_ended_long():
    env = make(COST_BPS)
    first = [dict(row) for row in run_policy(env, ALWAYS_LONG)]  # ends the episode holding the asset
    obs, _ = env.reset()
    assert obs[K] == 0  # flat again
    second = run_policy(env, ALWAYS_LONG)
    assert second == first
    assert second[0]["cost"] == pytest.approx(COST, abs=1e-15)  # so entering costs again


def test_last_observation_is_the_window_ending_on_the_last_day():
    rng = np.random.default_rng(8)
    returns = rng.normal(0, 0.01, 12)
    env = make(0.0, returns, k=3, obs_scale=1.0)
    obs, _ = env.reset()
    truncated = False
    while not truncated:
        obs, _, _, truncated, _ = env.step(1)
    assert obs[:3] == pytest.approx(returns[-3:], abs=1e-7)
    assert obs[3] == 1


def test_records_are_not_changed_by_editing_the_returned_info():
    env = make(0.0)
    env.reset()
    _, _, _, _, info = env.step(1)
    info["episode"] = {"r": 1.0}  # what SB3's Monitor wrapper does when an episode ends
    assert "episode" not in env.history[0]


def test_dates_are_copied_into_the_records():
    dates = [f"2026-01-{d:02d}" for d in range(1, 8)][: len(RETURNS)]
    history = run_policy(make(0.0, dates=dates), ALWAYS_LONG)
    assert [row["date"] for row in history] == dates[1:]


def test_episode_ends_truncated_and_then_refuses_to_step():
    env = make(0.0)
    env.reset()
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and truncated == (i == 4)
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"returns": [0.01, float("nan"), 0.02]},
        {"returns": [[0.01, 0.02]]},
        {"k": 0},
        {"k": 6},  # six returns leave no day to trade
        {"cost_bps": -1.0},
        {"days": [0, 1]},  # day 0 has no earlier return to look at
        {"days": [3, 2]},
        {"days": [2, 2]},
        {"days": [5, 6]},  # day 6 does not exist
        {"days": []},
        {"dates": ["2026-01-01"]},
    ],
)
def test_bad_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_bad_actions_are_rejected():
    env = make(0.0)
    env.reset()
    for action in (2, -1, 0.5, None):
        with pytest.raises(ValueError):
            env.step(action)


def test_cost_must_be_stated_explicitly():
    with pytest.raises(TypeError):
        TradingEnv(RETURNS, k=1)  # no cost given: refuse, do not assume zero


def test_gymnasium_check_env_passes_without_warnings():
    for peek in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env = make(COST_BPS, np.random.default_rng(6).normal(0, 0.01, 50), k=5, peek=peek)
            check_env(env, skip_render_check=True)  # nothing is drawn, so there is no render mode to test


def test_stable_baselines3_check_env_passes_without_warnings():
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sb3_checker.check_env(make(COST_BPS, np.random.default_rng(7).normal(0, 0.01, 50), k=5), warn=True)


def test_environment_can_be_built_by_name():
    env = gym.make(ENV_ID, log_returns=RETURNS, k=K, cost_bps=COST_BPS)
    assert isinstance(env.unwrapped, TradingEnv)
    assert total(run_policy(env, ORACLE)) == pytest.approx(3 * UP - 3 * COST, abs=1e-12)
