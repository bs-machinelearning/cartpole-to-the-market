"""Tests for the trading environment. Run with: uv run python -m pytest Reference

Part 1 scores scripted policies on a toy price series; every expected number
there can be redone by hand. Part 2 checks what the agent is allowed to see.
Part 3 compares the environment with a second, independent statement of the
rules on random cases. Part 4 covers the records, bad inputs and the checkers.
"""

import itertools
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
RETURNS = np.diff(np.log(PRICES))  # [0, UP, UP, DOWN, 0, UP]; position j holds the move into day j+1

# With k = 1 the first return is only history. Five days are traded; they earn UP, UP, DOWN, 0, UP.
K = 1
COST_BPS = 10.0  # 10 basis points = 0.001 per change of position
COST = 0.001

ALWAYS_FLAT = [0, 0, 0, 0, 0]
ALWAYS_LONG = [1, 1, 1, 1, 1]
SWITCH_DAILY = [1, 0, 1, 0, 1]  # earns UP + DOWN + UP, changes position 5 times
ORACLE = [1, 1, 0, 0, 1]  # long exactly on the up days: earns 3 * UP, changes position 3 times
ANTI_ORACLE = [0, 0, 1, 0, 0]  # long exactly on the down day: earns DOWN


def make(cost_bps=0.0, returns=RETURNS, k=K, days="all", **kwargs):
    return TradingEnv(returns, k=k, cost_bps=cost_bps, days=days, **kwargs)


def total(history):
    return sum(row["net_return"] for row in history)


# --- Part 1: scripted policies on the toy series, scored by hand -----------------------------


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


def test_oracle_beats_the_other_scripted_policies_and_nothing_beats_it():
    assert total(run_policy(make(0.0), ORACLE)) == pytest.approx(3 * UP, abs=1e-12)
    assert total(run_policy(make(COST_BPS), ORACLE)) == pytest.approx(3 * UP - 3 * COST, abs=1e-12)
    for cost_bps in (0.0, COST_BPS):
        oracle = total(run_policy(make(cost_bps), ORACLE))
        for other in (ALWAYS_FLAT, ALWAYS_LONG, SWITCH_DAILY, ANTI_ORACLE):
            assert oracle > total(run_policy(make(cost_bps), other))
        # All 32 ways of playing five days: none does better than knowing tomorrow.
        best = max(total(run_policy(make(cost_bps), list(a))) for a in itertools.product((0, 1), repeat=5))
        assert oracle == pytest.approx(best, abs=1e-12)


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


# --- Part 2: what the agent is allowed to see -------------------------------------------------


def test_observation_is_the_past_window_then_the_position():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 30)
    env = make(0.0, returns, k=3, obs_scale=100.0)
    obs, _ = env.reset()
    position = 0
    for day in range(3, 30):  # "all": every day from k to the end
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


def test_nothing_after_the_window_reaches_any_observation():
    # The train window stops at day 19; the data go on to day 39 (a later test period).
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 40)
    tampered = returns.copy()
    tampered[20:] = rng.normal(0, 0.05, 20)
    actions = rng.integers(2, size=15).tolist()
    seen = []
    for series in (returns, tampered):
        env = make(2.0, series, k=5, days=list(range(5, 20)))
        obs, _ = env.reset()
        trail = [obs]
        for action in actions:
            obs, *_ = env.step(action)
            trail.append(obs)
        seen.append(np.array(trail))
    assert np.array_equal(seen[0], seen[1])  # includes the observation returned with the last step


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


def test_every_episode_starts_with_the_same_observation():
    returns = np.random.default_rng(2).normal(0, 0.01, 30)
    for peek in (False, True):
        env = make(2.0, returns, k=4, peek=peek)
        first, _ = env.reset()
        for _ in range(26):
            env.step(1)
        again, _ = env.reset()
        assert np.array_equal(first, again)
        env.step(1)
        env.step(0)
        mid_episode, _ = env.reset()
        assert np.array_equal(first, mid_episode)


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


def test_a_policy_that_reads_the_peek_collects_every_up_day():
    # The one-bar oracle played as a function of the observation: "long when the
    # peek slot is positive" must earn every positive return and nothing else.
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, 60)
    for days in ("all", [8, 9, 15, 16, 17, 30, 44, 59]):
        for scale in (1.0, 100.0):
            env = make(0.0, returns, k=5, days=days, peek=True, obs_scale=scale)
            history = run_policy(env, lambda obs: int(obs[6] > 0))
            earned = returns[5:] if days == "all" else returns[days]
            assert total(history) == pytest.approx(earned[earned > 0].sum(), abs=1e-12)
            assert [row["position"] for row in history] == (earned > 0).astype(int).tolist()


def test_defaults_that_shape_the_observation_are_pinned():
    returns = np.random.default_rng(4).normal(0, 0.01, 10)
    obs, _ = TradingEnv(returns, k=3, cost_bps=0.0, days="all").reset()
    assert obs[:3] == pytest.approx(100.0 * returns[:3], rel=1e-6)  # returns shown in percent
    assert obs.shape == (4,)  # peek off


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


def test_a_gap_in_days_is_stepped_over_as_if_it_were_not_there():
    # Days 6 and 7 fall 10% each and are not listed. A position that is long on both
    # sides of the gap pays nothing to cross it and does not earn the skipped returns.
    returns = np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.01, -0.10, -0.10, 0.0, 0.01, 0.01])
    env = make(COST_BPS, returns, k=2, days=[4, 5, 9, 10], obs_scale=1.0)
    obs, _ = env.reset()
    position_seen = []
    for _ in range(4):
        position_seen.append(obs[2])
        obs, *_ = env.step(1)
    assert position_seen == [0, 1, 1, 1]
    assert [row["cost"] for row in env.history] == pytest.approx([COST, 0, 0, 0], abs=1e-15)
    assert [row["gross_return"] for row in env.history] == pytest.approx([0.01] * 4, abs=1e-15)


# --- Part 3: a second statement of the rules, compared on random cases -------------------------


def reference_run(log_returns, k, days, cost_bps, actions, peek=False, obs_scale=1.0):
    """The rules again in plain Python, written from the plan and not from the environment.

    The action for a listed day d is taken at the close of day d-1: it may use the
    returns of days d-k .. d-1 and the position held so far, and it earns log_returns[d].
    """
    c = cost_bps * 1e-4
    position, log_wealth, rows = 0, 0.0, []
    for d, a in zip(days, actions):
        window = [obs_scale * log_returns[j] for j in range(d - k, d)]
        peeked = obs_scale * log_returns[d] if peek else None
        cost = c * abs(a - position)
        reward = a * log_returns[d] - cost
        log_wealth += reward
        rows.append(dict(window=window, peek=peeked, position_before=position, reward=reward, cost=cost,
                         wealth=math.exp(log_wealth)))
        position = a
    return rows


def random_case(rng):
    k = int(rng.integers(1, 7))
    n = k + int(rng.integers(1, 30))
    kind = rng.choice(["all", "block", "scattered"])
    if kind == "all":
        days = list(range(k, n))
    elif kind == "block":
        start = int(rng.integers(k, n))
        days = list(range(start, int(rng.integers(start, n)) + 1))
    else:
        days = sorted(rng.choice(np.arange(k, n), size=int(rng.integers(1, n - k + 1)), replace=False).tolist())
    return dict(returns=rng.normal(0.0, 0.01, n), k=k, days=days, cost_bps=float(rng.choice([0.0, 2.0, 10.0])),
                peek=bool(rng.integers(2)), obs_scale=float(rng.choice([1.0, 100.0, 37.5])))


def test_matches_the_independent_reference_on_random_cases():
    rng = np.random.default_rng(2026)
    for _ in range(300):
        case = random_case(rng)
        returns, k, days, peek, scale = case["returns"], case["k"], case["days"], case["peek"], case["obs_scale"]
        env = TradingEnv(returns, k=k, cost_bps=case["cost_bps"], days=days, peek=peek, obs_scale=scale)
        for _episode in range(2):  # the second episode reuses the object: nothing may survive reset()
            actions = rng.integers(2, size=len(days)).tolist()
            expected = reference_run(returns.tolist(), k, days, case["cost_bps"], actions, peek, scale)
            obs, _ = env.reset()
            for i, (action, row) in enumerate(zip(actions, expected)):
                assert obs[:k] == pytest.approx(row["window"], rel=1e-6, abs=1e-9)
                assert obs[k] == row["position_before"]
                if peek:
                    assert obs[k + 1] == pytest.approx(row["peek"], rel=1e-6, abs=1e-9)
                obs, reward, terminated, truncated, info = env.step(action)
                assert reward == pytest.approx(row["reward"], abs=1e-14)
                assert info["cost"] == pytest.approx(row["cost"], abs=1e-14)
                assert info["wealth"] == pytest.approx(row["wealth"], rel=1e-12)
                assert not terminated and truncated == (i == len(days) - 1)
            last = days[-1]  # after the last listed day: the window ending on it, nothing newer
            assert obs[:k] == pytest.approx(scale * returns[last + 1 - k : last + 1], rel=1e-6, abs=1e-9)
            assert obs[k] == actions[-1]
            if peek:
                assert obs[k + 1] == 0.0


# --- Part 4: records, resets, bad inputs, checkers ---------------------------------------------


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


def test_records_of_an_earlier_run_survive_a_second_run_on_the_same_env():
    returns = np.random.default_rng(5).normal(0, 0.01, 20)
    env = make(0.0, returns, k=2)
    long_history = run_policy(env, [1] * 18)
    flat_history = run_policy(env, [0] * 18)
    assert long_history is not flat_history
    assert total(long_history) == pytest.approx(returns[2:].sum(), abs=1e-12)


def test_records_are_not_changed_by_editing_the_returned_info():
    env = make(0.0)
    env.reset()
    _, _, _, _, info = env.step(1)
    info["episode"] = {"r": 1.0}  # what SB3's Monitor wrapper does when an episode ends
    assert "episode" not in env.history[0]


def test_editing_the_callers_array_afterwards_does_not_change_the_environment():
    returns = np.random.default_rng(0).normal(0, 0.01, 12)
    env = make(0.0, returns, k=2)
    before = total(run_policy(env, [1] * 10))
    returns[:] = 0.0
    assert total(run_policy(env, [1] * 10)) == before


def test_dates_are_copied_into_the_records():
    dates = [f"2026-01-{d:02d}" for d in range(1, 8)][: len(RETURNS)]
    history = run_policy(make(0.0, dates=dates), ALWAYS_LONG)
    assert [row["date"] for row in history] == dates[1:]


def test_dates_follow_the_day_position_on_a_sub_window():
    returns = np.random.default_rng(6).normal(0, 0.01, 30)
    dates = [f"day-{j:02d}" for j in range(30)]
    history = run_policy(make(0.0, returns, k=2, days=[20, 21, 25], dates=dates), [1, 1, 1])
    assert [row["date"] for row in history] == ["day-20", "day-21", "day-25"]


def test_dates_given_as_a_pandas_series_are_read_by_position():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"date": [f"2026-01-{d:02d}" for d in range(1, 9)],
                          "close": [100.0, 101.0, 100.0, 102.0, 103.0, 101.0, 104.0, 105.0]})
    frame["log_return"] = np.log(frame["close"]).diff()
    frame = frame.dropna()  # the index now starts at 1, positions still start at 0
    history = run_policy(make(0.0, frame["log_return"], dates=frame["date"]), [1] * 6)
    assert [row["date"] for row in history] == frame["date"].tolist()[1:]


def test_episode_ends_truncated_and_then_refuses_to_step():
    env = make(0.0)
    env.reset()
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and truncated == (i == 4)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_run_policy_wants_exactly_one_action_per_day():
    returns = np.random.default_rng(0).normal(0, 0.01, 30)
    signal_by_day = (returns > 0).astype(int).tolist()  # 30 entries, but only 25 days are traded
    for actions in (signal_by_day, signal_by_day[:10]):
        with pytest.raises(ValueError):
            run_policy(make(0.0, returns, k=5), actions)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"returns": [0.01, float("nan"), 0.02]},
        {"returns": [0.0, float("inf"), 0.01, 0.02]},
        {"returns": [[0.01, 0.02]]},
        {"k": 0},
        {"k": 2.5},
        {"k": 6},  # six returns leave no day to trade
        {"cost_bps": -1.0},
        {"cost_bps": float("nan")},
        {"cost_bps": float("inf")},
        {"obs_scale": float("nan")},
        {"obs_scale": 0.0},
        {"days": [0, 1]},  # day 0 has no earlier return to look at
        {"days": [3, 2]},
        {"days": [2, 2]},
        {"days": [5, 6]},  # position 6 does not exist
        {"days": []},
        {"days": [1.5, 2.5]},
        {"days": None},
        {"days": "train"},
        {"days": np.array([4, 2, 5], dtype=np.uint64)},  # out of order, in a type whose differences wrap around
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


def test_cost_and_window_must_be_stated_explicitly():
    with pytest.raises(TypeError):
        TradingEnv(RETURNS, k=1, days="all")  # no cost given: refuse, do not assume zero
    with pytest.raises(TypeError):
        TradingEnv(RETURNS, k=1, cost_bps=0.0)  # no window given: refuse, do not assume "everything"


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
    env = gym.make(ENV_ID, log_returns=RETURNS, k=K, cost_bps=COST_BPS, days="all")
    assert isinstance(env.unwrapped, TradingEnv)
    assert total(run_policy(env, ORACLE)) == pytest.approx(3 * UP - 3 * COST, abs=1e-12)


def test_environment_built_by_name_runs_a_long_window_to_the_end():
    returns = np.random.default_rng(7).normal(0, 0.01, 3005)  # about twelve years of trading days
    env = gym.make(ENV_ID, log_returns=returns, k=5, cost_bps=0.0, days="all")
    assert len(run_policy(env, [1] * 3000)) == 3000
