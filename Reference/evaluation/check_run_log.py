"""Check results/run_log.csv against the rules in results/README.md. Owner: Eval.

Run from the repo root:

    uv run python -m Reference.evaluation.check_run_log

It prints every problem it finds, with the line number in the CSV, and exits
with code 1 if there are errors (so it can later run automatically).

Errors break a rule. Warnings are things that are allowed but look suspicious,
such as a daily Sharpe ratio so large it was probably annualised.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT / "results" / "run_log.csv"

COLUMNS = [
    "run_id", "date", "owner", "git_commit", "config", "environment",
    "algorithm", "seed", "fold", "status", "test_sharpe_per_period",
    "other_metrics", "returns_path",
]
STATUSES = {"completed", "abandoned", "crashed", "control"}
ALGORITHMS = {"PPO", "DQN", "buy_hold", "flat", "random_turnover"}
CONTROL_ENVS = {"CartPole-v1", "Acrobot-v1"}

# A daily Sharpe ratio above this is almost certainly annualised by mistake
# (a daily Sharpe of 0.2 is already about 3.2 annualised).
MAX_PLAUSIBLE_DAILY_SHARPE = 0.5


def _is_int(text: str) -> bool:
    return re.fullmatch(r"-?\d+", text) is not None


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def check_row(row: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one row of the run log."""
    errors: list[str] = []
    warnings: list[str] = []
    get = lambda col: (row.get(col) or "").strip()

    for col in ("run_id", "owner", "config", "environment"):
        if not get(col):
            errors.append(f"{col} is empty")

    try:
        date.fromisoformat(get("date"))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", get("date")):
            raise ValueError
    except ValueError:
        errors.append(f"date '{get('date')}' is not YYYY-MM-DD")

    if get("git_commit") == "unknown":
        warnings.append("git_commit is 'unknown': the run was made outside the repo and can't be reproduced exactly")
    elif not re.fullmatch(r"[0-9a-f]{7,40}", get("git_commit")):
        errors.append(f"git_commit '{get('git_commit')}' is not a short or full git hash")

    if get("algorithm") not in ALGORITHMS:
        errors.append(f"algorithm '{get('algorithm')}' is not one of {sorted(ALGORITHMS)}")

    if not _is_int(get("seed")):
        errors.append(f"seed '{get('seed')}' is not a whole number")

    fold = get("fold")
    if fold != "none" and not (_is_int(fold) and int(fold) >= 0):
        errors.append(f"fold '{fold}' must be 'none' or a fold number (0, 1, 2, ...)")

    status = get("status")
    if status not in STATUSES:
        errors.append(f"status '{status}' is not one of {sorted(STATUSES)}")

    # Control runs (CartPole, Acrobot) have no folds and no Sharpe ratio.
    is_control = status == "control" or get("environment") in CONTROL_ENVS
    if is_control and fold != "none":
        errors.append("control runs must have fold 'none'")

    sharpe = get("test_sharpe_per_period")
    if sharpe:
        if not _is_float(sharpe):
            errors.append(f"test_sharpe_per_period '{sharpe}' is not a number")
        elif abs(float(sharpe)) > MAX_PLAUSIBLE_DAILY_SHARPE:
            warnings.append(
                f"test_sharpe_per_period {sharpe} is very large for a daily Sharpe; "
                "was it annualised by mistake?"
            )
    elif status == "completed" and not is_control:
        errors.append("completed market runs need a test_sharpe_per_period")

    metrics = get("other_metrics")
    if metrics:
        try:
            if not isinstance(json.loads(metrics), dict):
                raise ValueError
        except ValueError:
            errors.append(f"other_metrics is not a JSON object like {{\"mean_return\": 500}}: {metrics}")

    returns_path = get("returns_path")
    if status == "completed" and not is_control:
        if not returns_path:
            errors.append("completed market runs need a returns_path")
        elif not (REPO_ROOT / returns_path).exists():
            warnings.append(f"returns_path '{returns_path}' does not exist on this computer")

    return errors, warnings


def check_log(path: Path = DEFAULT_LOG) -> tuple[list[str], list[str]]:
    """Check the whole log. Returns (errors, warnings), each prefixed with its CSV line."""
    errors: list[str] = []
    warnings: list[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            errors.append(f"header: columns should be {COLUMNS}, found {reader.fieldnames}")
            return errors, warnings

        seen: dict[str, int] = {}
        for line, row in enumerate(reader, start=2):  # line 1 is the header
            if None in row:
                errors.append(f"line {line}: too many values (an unquoted comma?)")
                continue
            row_errors, row_warnings = check_row(row)
            run_id = (row.get("run_id") or "").strip()
            if run_id in seen:
                row_errors.append(f"run_id '{run_id}' already used on line {seen[run_id]}")
            elif run_id:
                seen[run_id] = line
            errors += [f"line {line}: {e}" for e in row_errors]
            warnings += [f"line {line}: {w}" for w in row_warnings]

    return errors, warnings


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    errors, warnings = check_log(path)
    for e in errors:
        print(f"ERROR    {e}")
    for w in warnings:
        print(f"WARNING  {w}")
    if not errors and not warnings:
        print(f"OK: {path.name} follows every rule.")
    else:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s) in {path.name}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
