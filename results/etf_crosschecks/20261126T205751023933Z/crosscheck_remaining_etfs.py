"""Audit saved Yahoo prices against Stooq. Never change primary prices.

Run from the repository root:
uv run python "3. Honest Evaluation/crosscheck_remaining_etfs.py"
Optional first argument: path to the specific results/data_revision snapshot.
Optional second argument: folder containing TLT_stooq.csv, GLD_stooq.csv,
and DBC_stooq.csv for manual downloads when Stooq blocks automatic access.
"""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd

TICKERS = ("TLT", "GLD", "DBC")
START, END = "2006-02-06", "2025-12-31"
# A review filter, not a statistical acceptance criterion or a correction rule.
REVIEW_THRESHOLD_BP = 10.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_prices(path, required):
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or not set(required).issubset(frame.columns):
        raise ValueError(f"Expected Date and {required} in {path}")
    # Preserve the provider's local trading date, including timezone-bearing CSVs.
    frame["Date"] = pd.to_datetime(frame["Date"].astype(str).str[:10], errors="raise")
    if frame["Date"].isna().any() or frame["Date"].duplicated().any():
        raise ValueError(f"Missing or duplicate dates in {path}")
    frame = frame.set_index("Date").sort_index().loc[START:END, list(required)]
    if frame.empty:
        raise ValueError(f"No prices in the requested window: {path}")
    values = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise ValueError(f"Missing, infinite, or nonpositive prices in {path}")
    return values


def compare(yahoo, stooq, expected, output):
    output.mkdir(parents=True, exist_ok=True)
    calendar = pd.DataFrame(index=expected)
    calendar.index.name = "Date"
    calendar["Yahoo present"] = expected.isin(yahoo.index)
    calendar["Stooq present"] = expected.isin(stooq.index)
    calendar.to_csv(output / "calendar_coverage.csv")
    union = yahoo.index.union(stooq.index)
    unmatched = pd.DataFrame(index=union)
    unmatched["Yahoo present"] = union.isin(yahoo.index)
    unmatched["Stooq present"] = union.isin(stooq.index)
    unmatched.loc[unmatched["Yahoo present"] != unmatched["Stooq present"]].to_csv(
        output / "date_differences.csv", index_label="Date")

    aligned = yahoo.rename(columns={"Close": "Yahoo Close", "Adj Close": "Yahoo Adj Close"}).join(
        stooq.rename(columns={"Close": "Stooq Close"}), how="inner").sort_index()
    if len(aligned) < 2:
        raise ValueError("Fewer than two common dates")
    # Join FIRST: all returns below span the same dates for both providers.
    dates = aligned.index
    aligned["Previous common date"] = pd.Series(dates, index=dates).shift(1)
    aligned["Trading-session steps"] = [np.nan] + [
        int(((expected > a) & (expected <= b)).sum())
        for a, b in zip(dates[:-1], dates[1:])]
    for column in ("Yahoo Close", "Yahoo Adj Close", "Stooq Close"):
        aligned[column + " return"] = aligned[column].pct_change(fill_method=None)
    aligned["Adjusted return difference bp"] = 10000 * (
        aligned["Yahoo Adj Close return"] - aligned["Stooq Close return"])
    aligned["Unadjusted return difference bp"] = 10000 * (
        aligned["Yahoo Close return"] - aligned["Stooq Close return"])
    aligned["Absolute adjusted return difference bp"] = aligned["Adjusted return difference bp"].abs()
    aligned.to_csv(output / "aligned_comparison.csv", index_label="Date")
    review = aligned.loc[(aligned["Absolute adjusted return difference bp"] > REVIEW_THRESHOLD_BP)
                         | (aligned["Trading-session steps"] > 1)].copy()
    review["Disposition"] = "Unresolved; investigate source adjustments, actions and dates; do not edit prices"
    review.to_csv(output / "discrepancy_register.csv", index_label="Date")
    aligned.sort_values("Absolute adjusted return difference bp", ascending=False).head(25).to_csv(
        output / "largest_return_differences.csv", index_label="Date")
    diffs = aligned["Absolute adjusted return difference bp"].dropna()
    return {
        "status": "COMPARED_REVIEW_REQUIRED", "common_dates": len(aligned),
        "matched_return_intervals": len(diffs),
        "Yahoo_missing_sessions": len(expected.difference(yahoo.index)),
        "Stooq_missing_sessions": len(expected.difference(stooq.index)),
        "Yahoo_extra_dates": len(yahoo.index.difference(expected)),
        "Stooq_extra_dates": len(stooq.index.difference(expected)),
        "median_absolute_adjusted_return_difference_bp": float(diffs.median()),
        "p95_absolute_adjusted_return_difference_bp": float(diffs.quantile(.95)),
        "max_absolute_adjusted_return_difference_bp": float(diffs.max()),
        "intervals_above_review_threshold": int((diffs > REVIEW_THRESHOLD_BP).sum()),
        "multi_session_intervals": int((aligned["Trading-session steps"] > 1).sum()),
        "review_threshold_bp": REVIEW_THRESHOLD_BP,
        "note": "Diagnostic comparison only. Stooq Close adjustment conventions must be reviewed; equality is not assumed. No automatic pass, correction or protocol approval.",
    }


def main():
    import pandas_market_calendars as mcal
    root = Path.cwd()
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError("Run from the cartpole-to-the-market repository root")
    if len(sys.argv) > 1:
        revision = Path(sys.argv[1]).resolve()
    else:
        candidates = [p for p in (root / "results" / "data_revision").glob("*")
                      if all((p / "raw" / f"{t}_original.csv").is_file() for t in TICKERS)]
        if len(candidates) != 1:
            raise ValueError("Specify the revision folder as the first argument: expected exactly one complete saved snapshot")
        revision = candidates[0]
    manual = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "3. Honest Evaluation" / "stooq_inputs"
    originals = {t: revision / "raw" / f"{t}_original.csv" for t in TICKERS}
    before = {t: sha(p) for t, p in originals.items()}
    now = datetime.now(timezone.utc)
    output = root / "results" / "etf_crosschecks" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    expected = mcal.get_calendar("NYSE").valid_days(START, END).tz_localize(None)
    records = []
    for ticker in TICKERS:
        target = output / ticker
        target.mkdir()
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d&d1=20060206&d2=20251231"
        record = {"ticker": ticker, "Yahoo_path": str(originals[ticker].relative_to(root)),
                  "Yahoo_sha256": before[ticker], "Stooq_download_url": url}
        print(f"Checking {ticker}...", flush=True)
        try:
            local = manual / f"{ticker}_stooq.csv"
            if local.exists():
                payload = local.read_bytes()
                record["Stooq_acquisition"] = f"Manual file: {local}"
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = response.read()
                record["Stooq_acquisition"] = "Automatic download"
            # Preserve the response, including unexpected responses, for the audit.
            source = target / "Stooq_response.csv"
            source.write_bytes(payload)
            record["Stooq_sha256"] = sha(source)
            pd.read_csv(BytesIO(payload), nrows=1)  # Reject unreadable responses.
            yahoo = read_prices(originals[ticker], ("Close", "Adj Close"))
            stooq = read_prices(source, ("Close",))
            record.update(compare(yahoo, stooq, expected, target))
        except Exception as error:
            record.update(status="PENDING_SOURCE_OR_VALIDATION", error=str(error))
            print(f"  Pending: {error}\n  Manual download: {url}", flush=True)
        records.append(record)
    after = {t: sha(p) for t, p in originals.items()}
    if before != after:
        raise RuntimeError("A Yahoo input changed during execution; results must not be used")
    report = {"created_utc": now.isoformat(), "revision": str(revision),
              "sample_start": START, "sample_end": END,
              "status": "REVIEW_REQUIRED; not protocol approval",
              "Yahoo_inputs_unchanged": True, "checks": records,
              "limits": "No sources overwritten, no missing prices filled, no folds changed, no agents trained. Review discrepancies before confirming ETF selection."}
    (output / "crosscheck_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(output / "crosscheck_summary.csv", index=False)
    print(pd.DataFrame(records)[["ticker", "status"]].to_string(index=False))
    print(f"Results saved to: {output}")
    print("Yahoo originals unchanged. Team review still required.")


if __name__ == "__main__":
    main()
