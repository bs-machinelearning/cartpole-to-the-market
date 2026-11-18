"""Compare the supplied SPY snapshots without changing either source."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    base = Path(__file__).resolve().parent
    inputs, outputs = base / "inputs", base / "outputs"
    outputs.mkdir(exist_ok=True)
    yahoo_path = inputs / "SPY_original.csv"
    stooq_path = inputs / "spy_us_d.csv"
    yahoo = pd.read_csv(yahoo_path)
    stooq = pd.read_csv(stooq_path)
    # Preserve each provider's local session date, without converting to UTC.
    yahoo["Date"] = pd.to_datetime(yahoo["Date"].str.slice(0, 10))
    stooq["Date"] = pd.to_datetime(stooq["Date"])
    yahoo = yahoo.set_index("Date").sort_index()
    stooq = stooq.set_index("Date").sort_index()
    for name, frame in [("Yahoo", yahoo), ("Stooq", stooq)]:
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate dates in {name}; investigate before joining")
        values = frame[["Open", "High", "Low", "Close"]].to_numpy()
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid OHLC prices in {name}")
    if not np.isfinite(yahoo["Adj Close"]).all() or (yahoo["Adj Close"] <= 0).any():
        raise ValueError("Invalid Yahoo adjusted closes")

    yahoo_only = yahoo.index.difference(stooq.index)
    stooq_only = stooq.index.difference(yahoo.index)
    date_differences = pd.DataFrame(
        [{"Date": date.date().isoformat(), "Present only in": "Yahoo"}
         for date in yahoo_only]
        + [{"Date": date.date().isoformat(), "Present only in": "Stooq"}
           for date in stooq_only], columns=["Date", "Present only in"]
    )
    date_differences.to_csv(outputs / "date_differences.csv", index=False)

    common = stooq[["Close"]].rename(columns={"Close": "Stooq Close"}).join(
        yahoo[["Close", "Adj Close"]].rename(columns={
            "Close": "Yahoo Close", "Adj Close": "Yahoo Adj Close"
        }), how="inner"
    ).sort_index()
    common["Price difference vs Yahoo adjusted (%)"] = (
        common["Stooq Close"] / common["Yahoo Adj Close"] - 1
    ) * 100
    # Join BEFORE pct_change: both returns must span the same two dates.
    # Around a missing source row, that interval can exceed one session.
    returns = common[["Stooq Close", "Yahoo Close", "Yahoo Adj Close"]].pct_change(
        fill_method=None
    )
    common["Previous common date"] = common.index.to_series().shift(1)
    common["Calendar days in interval"] = common.index.to_series().diff().dt.days
    common["Yahoo sessions in interval"] = (
        pd.Series(yahoo.index.get_indexer(common.index), index=common.index).diff()
    )
    common["Stooq interval return"] = returns["Stooq Close"]
    common["Yahoo adjusted interval return"] = returns["Yahoo Adj Close"]
    common["Return difference (bp)"] = (
        returns["Stooq Close"] - returns["Yahoo Adj Close"]
    ) * 10000
    common.to_csv(outputs / "aligned_comparison.csv")
    largest = common.reindex(
        common["Return difference (bp)"].abs().dropna().sort_values(
            ascending=False
        ).index
    ).head(20)
    largest.to_csv(outputs / "largest_return_discrepancies.csv")

    differences = common["Return difference (bp)"].dropna().abs()
    summary = {
        "scope": "SPY only; supplied 2010-2025 snapshots",
        "status": "REVIEW: date coverage and return discrepancies unresolved",
        "yahoo_rows": len(yahoo), "stooq_rows": len(stooq),
        "common_dates": len(common), "matched_return_intervals": len(differences),
        "dates_only_in_yahoo": [str(date.date()) for date in yahoo_only],
        "dates_only_in_stooq": [str(date.date()) for date in stooq_only],
        "stooq_vs_yahoo_adjusted_return_correlation": float(
            returns[["Stooq Close", "Yahoo Adj Close"]].corr().iloc[0, 1]
        ),
        "stooq_vs_yahoo_unadjusted_return_correlation": float(
            returns[["Stooq Close", "Yahoo Close"]].corr().iloc[0, 1]
        ),
        "median_absolute_return_difference_bp": float(differences.median()),
        "p95_absolute_return_difference_bp": float(differences.quantile(.95)),
        "maximum_absolute_return_difference_bp": float(differences.max()),
        "maximum_discrepancy_interval_end": str(differences.idxmax().date()),
        "interpretation": [
            "Stooq Close is empirically much closer to Yahoo Adj Close than Yahoo Close.",
            "This does not independently establish either provider's adjustment policy.",
            "Correlation alone is not a quality acceptance criterion.",
            "Do not forward-fill, overwrite, or delete discrepancies automatically.",
            "Matched intervals adjacent to a missing session may span multiple sessions.",
            "Investigate March 12, 13 and 16, 2015 using independent price/action evidence.",
            "Other ETFs have not been cross-checked against Stooq."
        ],
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [yahoo_path, stooq_path]
        }
    }
    (outputs / "crosscheck_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
