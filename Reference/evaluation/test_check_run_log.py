"""Tests for the run log checker. Run with: uv run python -m pytest Reference"""

import csv

from Reference.evaluation.check_run_log import COLUMNS, check_log

GOOD_CONTROL = {
    "run_id": "20261115_ppo_cartpole_s0", "date": "2026-11-15", "owner": "Stefan",
    "git_commit": "a1b2c3d", "config": "configs/ppo_cartpole.yaml",
    "environment": "CartPole-v1", "algorithm": "PPO", "seed": "0", "fold": "none",
    "status": "control", "test_sharpe_per_period": "",
    "other_metrics": '{"mean_return": 500}', "returns_path": "",
}


def write_log(tmp_path, rows):
    path = tmp_path / "run_log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_empty_log_passes(tmp_path):
    assert check_log(write_log(tmp_path, [])) == ([], [])


def test_good_control_row_passes(tmp_path):
    assert check_log(write_log(tmp_path, [GOOD_CONTROL])) == ([], [])


def test_bad_values_are_caught(tmp_path):
    bad = dict(GOOD_CONTROL, status="Done", date="15/11/2026", fold="0", other_metrics="500")
    errors, _ = check_log(write_log(tmp_path, [bad]))
    text = " ".join(errors)
    for word in ("status", "date", "fold", "other_metrics"):
        assert word in text


def test_duplicate_run_id_is_caught(tmp_path):
    errors, _ = check_log(write_log(tmp_path, [GOOD_CONTROL, GOOD_CONTROL]))
    assert any("already used on line 2" in e for e in errors)


def test_completed_market_run_needs_sharpe_and_returns(tmp_path):
    market = dict(GOOD_CONTROL, environment="market", fold="0", status="completed")
    errors, _ = check_log(write_log(tmp_path, [market]))
    assert any("test_sharpe_per_period" in e for e in errors)
    assert any("returns_path" in e for e in errors)


def test_annualised_sharpe_gives_warning(tmp_path):
    market = dict(GOOD_CONTROL, environment="market", fold="0", status="abandoned",
                  test_sharpe_per_period="1.8")
    errors, warnings = check_log(write_log(tmp_path, [market]))
    assert errors == []
    assert any("annualised" in w for w in warnings)


def test_wrong_header_is_caught(tmp_path):
    path = tmp_path / "run_log.csv"
    path.write_text("run_id,date\n")
    errors, _ = check_log(path)
    assert errors and errors[0].startswith("header")


def test_unknown_commit_is_a_warning(tmp_path):
    errors, warnings = check_log(write_log(tmp_path, [dict(GOOD_CONTROL, git_commit="unknown")]))
    assert errors == []
    assert any("unknown" in w for w in warnings)
