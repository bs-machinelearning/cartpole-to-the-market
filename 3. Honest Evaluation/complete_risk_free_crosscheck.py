"""Complete the existing revision with a manually downloaded FRED DTB3 CSV.

Run from the repository root. An optional argument supplies the DTB3 CSV path;
otherwise use Downloads/DTB3.csv. No rate is converted to a daily return.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

IRX_SHA256 = "587535c83bf85d922089918d619a01e211e45e49b35cee066aba732a6a7519c8"
IRX_COLUMN = "IRX quoted annual rate percent"
FRED_COLUMN = "FRED DTB3 annual discount rate percent"


def compare_files(irx_path, fred_path, output_folder):
    rates = pd.read_csv(irx_path)
    fred = pd.read_csv(fred_path, na_values=["."])
    rates["Date"] = pd.to_datetime(rates["Date"])
    fred["observation_date"] = pd.to_datetime(fred["observation_date"])
    rates = rates.set_index("Date").sort_index()
    fred = fred.set_index("observation_date")[["DTB3"]].rename(
        columns={"DTB3": FRED_COLUMN}
    ).sort_index()
    fred.index.name = "Date"
    for name, frame in [("IRX", rates), ("DTB3", fred)]:
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate dates in {name}")
        numeric = frame.iloc[:, 0].dropna()
        if not np.isfinite(numeric).all():
            raise ValueError(f"Nonfinite rates in {name}")
    aligned = rates[[IRX_COLUMN]].join(fred, how="outer")
    aligned["Quote difference bp"] = (aligned[IRX_COLUMN] - aligned[FRED_COLUMN]) * 100
    aligned["Both observed"] = aligned[[IRX_COLUMN, FRED_COLUMN]].notna().all(axis=1)
    diff = aligned["Quote difference bp"].dropna()
    if diff.empty:
        raise ValueError("No paired nonmissing quotes")
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_folder / "risk_free_source_comparison.csv")
    aligned.loc[diff.abs().nlargest(25).index].to_csv(
        output_folder / "risk_free_largest_quote_differences.csv"
    )
    summary = {
        "status": "COMPARED_WITH_DIFFERENCES; Eval review pending",
        "IRX_rows": len(rates), "FRED_rows": len(fred),
        "IRX_missing_values": int(rates[IRX_COLUMN].isna().sum()),
        "FRED_missing_values": int(fred[FRED_COLUMN].isna().sum()),
        "same_date_pairs": len(diff),
        "median_absolute_difference_bp": float(diff.abs().median()),
        "p95_absolute_difference_bp": float(diff.abs().quantile(.95)),
        "max_absolute_difference_bp": float(diff.abs().max()),
        "max_difference_date": str(diff.abs().idxmax().date()),
        "mean_signed_difference_bp": float(diff.mean()),
        "paired_quote_correlation": float(aligned[[IRX_COLUMN, FRED_COLUMN]].corr().iloc[0, 1]),
        "units": "Both source columns are quoted annual rates in percent; a 0.01 percentage-point quote gap equals 1 bp",
        "method": "Same-date quote comparison with no filling or interpolation",
        "interpretation": "Related 13-week / 3-month series; exact quote equality is not assumed. Differences are preserved, with no provider automatically declared incorrect.",
        "pending": [
            "Review instrument/quote conventions and largest differences with Eval",
            "Approve conversion from annual quotes to interval risk-free returns",
            "Approve conservative availability lag and missing-quote policy; never backfill from future observations",
            "Approve final rate snapshot; revised historical data are not guaranteed point-in-time vintages",
        ],
        "sources": {
            "IRX": "https://finance.yahoo.com/quote/%5EIRX/history/",
            "FRED_DTB3": "https://fred.stlouisfed.org/series/DTB3",
        },
        "input_sha256": {
            "risk_free_quotes.csv": hashlib.sha256(Path(irx_path).read_bytes()).hexdigest(),
            "DTB3.csv": hashlib.sha256(Path(fred_path).read_bytes()).hexdigest(),
        },
    }
    (output_folder / "risk_free_crosscheck_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    root = Path.cwd()
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError("Run from the repository root")
    fred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "DTB3.csv"
    if not fred_path.is_file():
        raise FileNotFoundError(f"FRED file not found: {fred_path}. Pass its actual path as an argument.")
    matches = [
        p for p in (root / "results" / "data_revision").glob("*/risk_free_quotes.csv")
        if hashlib.sha256(p.read_bytes()).hexdigest() == IRX_SHA256
    ]
    if not matches:
        raise ValueError("Cannot locate the exact IRX snapshot reviewed in the conversation")
    revision = sorted(matches)[-1].parent
    raw = revision / "raw"
    raw.mkdir(exist_ok=True)
    saved_fred = raw / "FRED_DTB3_original.csv"
    if saved_fred.exists() and saved_fred.read_bytes() != fred_path.read_bytes():
        raise ValueError("A different frozen FRED file already exists; review instead of overwriting")
    shutil.copy2(fred_path, saved_fred)
    summary = compare_files(revision / "risk_free_quotes.csv", saved_fred, revision)
    manifest_path = revision / "revision_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["risk_free_crosscheck"] = summary
    manifest["risk_free_review_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("file_sha256", {}).update({
        str(p.relative_to(revision)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in revision.rglob("*.csv")
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Updated existing revision: {revision}")
    print("IRX quotes unchanged. No models trained. Final protocol still requires approval.")


if __name__ == "__main__":
    main()
