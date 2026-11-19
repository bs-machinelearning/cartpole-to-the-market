"""Prepare Christina's requested data revision; no agents are trained.

Run from the repository root with uv run python <path-to-this-file>.
All selections remain proposals until the team approves the protocol.
"""
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
import shutil
import urllib.request
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

END_EXCLUSIVE = "2026-01-01"
TRAIN_SESSIONS = 756
EMBARGO_SESSIONS = 20
ORIGINAL = ["SPY", "QQQ", "IWM", "EFA", "TLT"]
RECOMMENDED = ["SPY", "TLT", "GLD", "DBC"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2002-12-01&coed=2025-12-31"


def make_folds(dates, train_sessions=TRAIN_SESSIONS, embargo_sessions=EMBARGO_SESSIONS):
    """Same 756-session training / 20-session gap / Jan-Jul test grid as before."""
    dates = pd.DatetimeIndex(dates).sort_values()
    if dates.has_duplicates:
        raise ValueError("Session index has duplicates")
    rows = []
    for start in pd.date_range("2003-01-01", "2025-07-01", freq="6MS"):
        stop = start + pd.DateOffset(months=6)
        test = dates[(dates >= start) & (dates < stop)]
        if test.empty or test[0] >= pd.Timestamp(END_EXCLUSIVE):
            continue
        position = dates.get_loc(test[0])
        train_stop = position - embargo_sessions
        train_start = train_stop - train_sessions
        if train_start < 0:
            continue
        train = dates[train_start:train_stop]
        gap = dates[train_stop:position]
        rows.append({
            "Fold": len(rows) + 1, "Train start": str(train[0].date()),
            "Train end": str(train[-1].date()), "Train sessions": len(train),
            "Embargo start": str(gap[0].date()), "Embargo end": str(gap[-1].date()),
            "Embargo sessions": len(gap), "Test start": str(test[0].date()),
            "Test end": str(test[-1].date()), "Test sessions": len(test),
        })
    return pd.DataFrame(rows)


def main():
    root = Path.cwd()
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError("Run from the repository root")
    stamp = datetime.now(timezone.utc)
    folder = root / "results" / "data_revision" / stamp.strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    raw_folder = folder / "raw"
    raw_folder.mkdir()
    # Keep an exact copy of the executing script beside each snapshot.
    shutil.copy2(Path(__file__), folder / "prepare_data_revision.py")
    frames = {}
    quality_rows = []
    calendar = mcal.get_calendar("NYSE")
    for ticker in ORIGINAL + ["GLD", "DBC"]:
        print(f"Downloading {ticker}...", flush=True)
        raw = yf.Ticker(ticker).history(
            start="2003-01-01", end=END_EXCLUSIVE,
            interval="1d", auto_adjust=False, actions=True
        )
        if raw.empty:
            raise ValueError(f"No data received for {ticker}; no scenario is finalized")
        raw.to_csv(raw_folder / f"{ticker}_original.csv")
        frame = raw.copy()
        frame.index = frame.index.tz_localize(None).normalize()
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unsorted dates for {ticker}")
        frames[ticker] = frame
        price_values = frame[["Open", "High", "Low", "Close", "Adj Close"]]
        expected = calendar.valid_days(
            start_date=frame.index.min(), end_date="2025-12-31"
        ).tz_localize(None)
        missing = expected.difference(frame.index)
        pd.Series(missing, name="Missing session").to_csv(folder / f"{ticker}_missing_sessions.csv", index=False)
        quality_rows.append({
            "Ticker": ticker, "First date": str(frame.index.min().date()),
            "Last date": str(frame.index.max().date()), "Rows": len(frame),
            "Missing sessions": len(missing),
            "Extra dates": len(frame.index.difference(expected)),
            "Missing price values": int(price_values.isna().sum().sum()),
            "Infinite prices": int(np.isinf(price_values.to_numpy()).sum()),
            "Nonpositive prices": int((price_values <= 0).sum().sum()),
        })
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(folder / "price_quality_checks.csv", index=False)

    scenarios = [
        ("Original five from 2010", ORIGINAL, "2010-01-01", 25),
        ("Original five from 2003", ORIGINAL, "2003-01-01", 39),
        ("Original five plus GLD", ORIGINAL + ["GLD"], "2003-01-01", 36),
        ("Original five plus GLD and DBC", ORIGINAL + ["GLD", "DBC"], "2003-01-01", 33),
        ("Recommended four asset exposures", RECOMMENDED, "2003-01-01", 33),
    ]
    scenario_rows = []
    for number, (name, tickers, requested_start, estimate) in enumerate(scenarios, start=1):
        # Calendar window is constrained by actual first observations.
        # No intersection/dropna is used to hide interior missing prices.
        sample_start = max(pd.Timestamp(requested_start), *(frames[t].index.min() for t in tickers))
        sessions = calendar.valid_days(sample_start, "2025-12-31").tz_localize(None)
        folds = make_folds(sessions)
        folds.to_csv(folder / f"scenario_{number}_folds.csv", index=False)
        panel = pd.concat({t: frames[t]["Adj Close"] for t in tickers}, axis=1).reindex(sessions)
        scenario_rows.append({
            "Scenario": name, "Tickers": ", ".join(tickers),
            "Sample start": str(sessions[0].date()), "Sample end": str(sessions[-1].date()),
            "First test": None if folds.empty else folds.iloc[0]["Test start"],
            "Last test": None if folds.empty else folds.iloc[-1]["Test end"],
            "Fold count": len(folds), "Christina estimate": estimate,
            "Estimate matches": len(folds) == estimate,
            "Missing panel values": int(panel.isna().sum().sum()),
            "Test sessions": int(folds["Test sessions"].sum()) if not folds.empty else 0,
        })
        if tickers == RECOMMENDED:
            panel.index.name = "Date"
            panel.to_csv(folder / "proposed_primary_adjusted_closes.csv")
            folds.to_csv(folder / "proposed_walk_forward_folds.csv", index=False)
    scenario_table = pd.DataFrame(scenario_rows)
    scenario_table.to_csv(folder / "scenario_comparison.csv", index=False)

    print("Downloading ^IRX risk-free quote snapshot...", flush=True)
    irx = yf.Ticker("^IRX").history(
        start="2002-12-01", end=END_EXCLUSIVE, interval="1d", auto_adjust=False
    )
    if irx.empty:
        raise ValueError("No ^IRX observations received")
    irx.to_csv(raw_folder / "IRX_original.csv")
    irx.index = irx.index.tz_localize(None).normalize()
    rates = irx[["Close"]].rename(columns={"Close": "IRX quoted annual rate percent"})
    if rates.index.has_duplicates:
        raise ValueError("Duplicate ^IRX quote dates")
    rates.to_csv(folder / "risk_free_quotes.csv")
    # Never use pct_change on yield quotes as a risk-free return.
    rf_check = {"status": "PENDING_FRED_DOWNLOAD"}
    try:
        print("Downloading FRED DTB3 cross-check...", flush=True)
        request = urllib.request.Request(FRED_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw_bytes = response.read()
        fred = pd.read_csv(StringIO(raw_bytes.decode("utf-8-sig")), na_values=["."])
        if "DTB3" not in fred.columns:
            raise ValueError("Unexpected FRED response; expected a DTB3 column")
        (raw_folder / "FRED_DTB3_original.csv").write_bytes(raw_bytes)
        date_column = fred.columns[0]
        fred[date_column] = pd.to_datetime(fred[date_column])
        fred = fred.set_index(date_column).rename(columns={"DTB3": "FRED DTB3 annual discount rate percent"})
        fred.index.name = "Date"
        aligned = rates.join(fred, how="outer").sort_index()
        aligned["Quote difference bp"] = (
            aligned["IRX quoted annual rate percent"]
            - aligned["FRED DTB3 annual discount rate percent"]
        ) * 100
        aligned.to_csv(folder / "risk_free_source_comparison.csv")
        differences = aligned["Quote difference bp"].dropna()
        if differences.empty:
            raise ValueError("No overlapping nonmissing risk-free quotes")
        rf_check = {
            "status": "COMPARED_REVIEW_REQUIRED", "same_date_pairs": len(differences),
            "median_absolute_difference_bp": float(differences.abs().median()),
            "p95_absolute_difference_bp": float(differences.abs().quantile(.95)),
            "max_absolute_difference_bp": float(differences.abs().max()),
            "IRX_nonmissing_quotes": int(rates.iloc[:, 0].notna().sum()),
            "FRED_nonmissing_quotes": int(fred.iloc[:, 0].notna().sum()),
            "note": "Quotes can differ in instrument definition and observation time; no equality requirement imposed",
        }
    except Exception as error:
        rf_check["error"] = str(error)
        rf_check["manual_download_url"] = FRED_URL
        print(f"FRED cross-check pending: {error}", flush=True)

    manifest = {
        "status": "DRAFT revision; not an approved protocol",
        "saved_at_utc": stamp.isoformat(), "yfinance_version": yf.__version__,
        "recommended_tickers": RECOMMENDED,
        "selection_rule": "One predeclared representative for US broad equities, long US Treasuries, physical gold, and broad commodity futures; no return-based selection",
        "selection_limits": "Representatives are proposed, not proof of independent bets or an exhaustive oldest/liquidity screen; DBC includes some gold exposure",
        "end_exclusive": END_EXCLUSIVE,
        "training_sessions": TRAIN_SESSIONS, "embargo_sessions_proposed": EMBARGO_SESSIONS,
        "embargo_approval": "Pending observation-window agreement with Env",
        "cost_assumption": "Env owns the cost model; constant 2bp is not claimed realistic for all historical years",
        "evaluation_method": "Rolling walk-forward only; no 80/20 alternative",
        "risk_free_crosscheck": rf_check,
        "risk_free_use": "Snapshot of quoted annual rates only. Eval must approve quote convention, conversion to interval returns, lagging and missing-quote policy before scoring excess returns. No cash rewards changed.",
        "risk_free_causality": "Never backfill from future quotes; use only quotes available before the return interval, accounting for publication timing. Frozen revised history is not point-in-time vintage data.",
        "price_crosschecks": "Previous SPY investigation retained; other assets still need cross-checks, including new GLD and DBC",
        "file_sha256": {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(folder.rglob("*.csv"))},
    }
    (folder / "revision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSCENARIO COUNTS", flush=True)
    print(scenario_table.to_string(index=False))
    print("\nRISK-FREE CROSS-CHECK", json.dumps(rf_check, indent=2), flush=True)
    print(f"\nSaved revision to: {folder}", flush=True)
    print("No models trained. Nothing published or frozen automatically.", flush=True)


if __name__ == "__main__":
    main()
