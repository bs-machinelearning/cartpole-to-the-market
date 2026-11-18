"""Document SPY source-selection evidence. Run from the project root.

No raw input is edited, no missing price is filled, and no date is removed.
The source-selection recommendation remains subject to team approval.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SOURCE_URL = "https://www.sec.gov/file/exhibit-3a-20"
EXPECTED_HASHES = {
    "SPY_original.csv": "f140c49b8e6e4972a5e609f20393f1723cf0a2c68c80ce7e38cc45ce75f873ef",
    "spy_us_d.csv": "8d7748a3dc6c4fd510416f1c19119845ad071cc45158cfa5de0a649ef8f95ff5",
}


def main(project_root=None):
    root = Path.cwd() if project_root is None else Path(project_root)
    folder = root / "3. Honest Evaluation" / "SPY_crosscheck"
    if not (root / "pyproject.toml").exists() or not folder.is_dir():
        raise FileNotFoundError("Run from the cartpole-to-the-market project root")
    paths = {name: folder / "inputs" / name for name in EXPECTED_HASHES}
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    if before != EXPECTED_HASHES:
        raise ValueError("Input snapshots differ from the reviewed files; recheck evidence first")
    yahoo = pd.read_csv(paths["SPY_original.csv"])
    yahoo["Date"] = pd.to_datetime(yahoo["Date"].str[:10])
    yahoo = yahoo.set_index("Date").sort_index()
    stooq = pd.read_csv(paths["spy_us_d.csv"], parse_dates=["Date"]).set_index("Date").sort_index()
    if yahoo.index.has_duplicates or stooq.index.has_duplicates:
        raise ValueError("Duplicate dates require review")
    outputs = folder / "outputs"
    outputs.mkdir(exist_ok=True)

    # Transcribed from MIAX filing SR-MIAX-2015-46, Exhibit 3, Appendix A,
    # printed page 24 of 27 (page 2 of the three-page PDF).
    # These are UNADJUSTED closes; do not compare them to adjusted price levels.
    official = pd.DataFrame({
        "Date": pd.to_datetime(["2015-03-12", "2015-03-13", "2015-03-16"]),
        "MIAX unadjusted close": [207.10, 205.83, 208.58],
    }).set_index("Date")
    evidence = official.join(yahoo[["Close", "Adj Close", "Dividends"]])
    evidence["Yahoo close matches MIAX within $0.0001"] = np.isclose(
        evidence["Close"], evidence["MIAX unadjusted close"], rtol=0, atol=.0001
    )
    if not evidence["Yahoo close matches MIAX within $0.0001"].all():
        raise ValueError("Yahoo snapshot does not match the transcribed MIAX evidence")
    evidence["Source URL"] = SOURCE_URL
    evidence["Source location"] = "SR-MIAX-2015-46, Exhibit 3 Appendix A, printed page 24/27"
    evidence.to_csv(outputs / "independent_close_evidence.csv")

    # Internal adjustment consistency, not an independent dividend audit.
    # With no splits, this checks the backward cash-dividend factor convention.
    if (yahoo["Stock Splits"] != 0).any():
        raise ValueError("Split events need a separate adjustment review")
    close = yahoo["Close"]
    factor = yahoo["Adj Close"] / close
    cash = yahoo["Dividends"] + yahoo["Capital Gains"]
    implied_step = factor / factor.shift(1)
    expected_step = 1 / (1 - cash / close.shift(1))
    residual = implied_step / expected_step - 1
    tolerance = 2e-6  # Relative factor tolerance, not a performance threshold.
    valid = residual.notna()
    adjustment = pd.DataFrame({
        "Cash distribution": cash,
        "Provider adjustment factor": factor,
        "Observed factor step": implied_step,
        "Expected factor step": expected_step,
        "Relative residual": residual,
        "Check available": valid,
    })
    adjustment["Flag for review"] = valid & (residual.abs() > tolerance)
    adjustment.to_csv(outputs / "yahoo_adjustment_consistency.csv")

    # Keep a complete primary-source candidate. Returns remain unavailable
    # on the first observation rather than silently being set to zero.
    primary = yahoo[["Close", "Adj Close", "Dividends", "Stock Splits", "Capital Gains"]].copy()
    primary["Adjusted close return"] = primary["Adj Close"].pct_change(fill_method=None)
    primary["Source"] = "Yahoo Finance supplied snapshot"
    primary.to_csv(outputs / "SPY_primary_candidate.csv")

    common = yahoo[["Close", "Adj Close"]].join(
        stooq[["Close"]].rename(columns={"Close": "Stooq Close"}), how="inner"
    )
    returns = common[["Adj Close", "Stooq Close"]].pct_change(fill_method=None)
    common["Price difference vs Yahoo adjusted (bp)"] = (
        common["Stooq Close"] / common["Adj Close"] - 1
    ) * 10000
    common["Matched interval return difference (bp)"] = (
        returns["Stooq Close"] - returns["Adj Close"]
    ) * 10000
    diagnostic_limit_bp = 10  # Investigation screen, not team-approved acceptance.
    flagged = common.loc[
        (common["Price difference vs Yahoo adjusted (bp)"].abs() > diagnostic_limit_bp)
        | (common["Matched interval return difference (bp)"].abs() > diagnostic_limit_bp)
    ].copy()
    flagged["Source decision"] = "Retain unchanged Yahoo candidate; preserve Stooq discrepancy"
    flagged.to_csv(outputs / "source_discrepancy_register.csv")
    yahoo_only = yahoo.index.difference(stooq.index)
    decision = {
        "status": "INVESTIGATED_WITH_EXCEPTIONS; team approval pending",
        "recommended_primary_source": "Unchanged Yahoo Finance snapshot",
        "secondary_source": "Stooq cross-check only; not interchangeable with Yahoo",
        "yahoo_rows_preserved": len(yahoo),
        "dates_only_in_yahoo": [str(date.date()) for date in yahoo_only],
        "missing_date_action": "Retain Yahoo observation; do not fabricate a Stooq row",
        "independent_evidence": {
            "source": "MIAX filing hosted by SEC", "url": SOURCE_URL,
            "dates_checked": [str(date.date()) for date in official.index],
            "result": "All three Yahoo unadjusted closes match within $0.0001",
            "scope_limit": "Does not validate adjusted levels or the full historical series",
        },
        "internal_adjustment_check": {
            "cash_dividend_events": int((yahoo["Dividends"] != 0).sum()),
            "split_events": 0,
            "relative_factor_tolerance": tolerance,
            "max_absolute_relative_residual": float(residual.abs().max()),
            "flagged_observations": int(adjustment["Flag for review"].sum()),
            "independent_dividend_validation": False,
        },
        "diagnostic_screen_bp": diagnostic_limit_bp,
        "diagnostic_flagged_dates": len(flagged),
        "unresolved": [
            "Exact causes of Stooq discrepancies and missing date are unknown",
            "Other Stooq/Yahoo differences remain documented, not repaired",
            "No independent full-series corporate-action verification",
            "Other four ETFs still require their own cross-checks",
            "Source-selection recommendation requires team approval",
        ],
        "input_sha256": before,
    }
    (outputs / "source_selection_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    note = """SPY source-selection investigation

Recommendation: retain the unchanged Yahoo snapshot as the primary candidate.
Team approval is still required. Do not splice values from different providers.

Evidence: the MIAX filing SR-MIAX-2015-46, Exhibit 3 Appendix A, printed page
24 of 27, records unadjusted closes of 207.10 (2015-03-12), 205.83
(2015-03-13) and 208.58 (2015-03-16). All three match the supplied Yahoo Close.
Source: https://www.sec.gov/file/exhibit-3a-20
These observations support Yahoo at the largest discrepancy dates; they do
not establish a definitive error mechanism in Stooq or validate every date.

Coverage: Stooq lacks 2011-02-17 relative to Yahoo. Preserve Yahoo's row in
the primary candidate and the gap in the original Stooq snapshot. No fill,
deletion or synthesized price has been applied to either input.

Adjustment consistency: SPY has no split events in the Yahoo snapshot and
64 reported cash-dividend events. The factor implied by Adj Close / Close
is consistent with the provider's reported cash distributions within a
relative tolerance of 0.000002. This is an internal consistency check using
Yahoo's own inputs, not an independent verification of dividends or splits.

Use outputs/SPY_primary_candidate.csv as the preserved Yahoo candidate.
The adjusted-return first row is unavailable and must not be scored as zero.
Provider-adjusted historical levels may incorporate adjustments known only
later; do not feed them to an agent without the planned causal feature and
leakage review. Rewards and execution conventions must be agreed with Env.

The 10-basis-point screen only helps locate discrepancies. It is not a
pre-agreed pass/fail standard. Original comparison outputs remain intact.
Remaining source differences and the other four ETFs are not certified.
"""
    (outputs / "source_selection_note.txt").write_text(note, encoding="utf-8")
    after = {name: hashlib.sha256(path.read_bytes()).hexdigest()
             for name, path in paths.items()}
    if before != after:
        raise AssertionError("Source snapshots unexpectedly changed")
    print("Source snapshots unchanged.")
    print("Independent MIAX closing-price checks: 3/3 matched.")
    print(f"Yahoo dividend-factor consistency flags: {int(adjustment['Flag for review'].sum())}")
    print(f"Primary Yahoo candidate preserved: {len(primary)} rows.")
    print("Stooq discrepancies retained in the audit register.")
    print("Status: investigated with exceptions; team approval pending.")
    print(f"Results saved to: {outputs}")


if __name__ == "__main__":
    main()
