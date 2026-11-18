SPY source cross-check: Yahoo Finance versus Stooq

Scope: the two supplied SPY snapshots, January 2010 through December 2025.
This folder is an analysis contribution for the project repository.

Findings
- Yahoo contains 4,024 rows and Stooq contains 4,023 rows.
- February 17, 2011 is present only in Yahoo. No dates occur only in Stooq.
- Stooq Close is numerically much closer to Yahoo Adj Close than Yahoo Close.
- The median absolute difference between matched interval returns is
  0.28358 basis points (0.0028358 percentage points).
- The 95th percentile absolute difference is 2.45035 basis points.
- The maximum absolute difference is 200.96467 basis points, or 2.00965
  percentage points, on the interval ending March 13, 2015.
- Notable discrepancies also occur on March 12 and March 16, 2015.

Conclusion
REVIEW. The sources broadly track each other, but the independent comparison
is not a clean pass. Provider adjustment conventions and the largest
discrepancies require investigation. The data alone does not establish which
provider is correct. Preserve both source files. Do not fill or remove
observations automatically. Other candidate ETFs remain to be cross-checked.

Method
Keep the Yahoo local session date by stripping the time and timezone text.
Check unique dates and finite positive prices. Join on common dates before
calculating percentage changes, so each compared return spans the same
interval. The interval adjacent to the missing Stooq date spans two Yahoo
sessions; it is explicitly marked in aligned_comparison.csv. Price differences
and return differences are distinct. One basis point is 0.01 percentage points.
All diagnostics describe the supplied snapshots; no new downloads were made.
Source files have not been modified. SHA-256 hashes are in the summary JSON.

Files
inputs/: the two unchanged provider snapshots.
outputs/date_differences.csv: unmatched dates.
outputs/aligned_comparison.csv: common prices and matched interval returns.
outputs/largest_return_discrepancies.csv: the 20 largest absolute differences.
outputs/crosscheck_summary.json: metrics, limitations and source hashes.
compare_sources.py: executable comparison using pandas and numpy.

Add to the repository
Extract the ZIP. Copy the entire SPY_crosscheck folder into
3. Honest Evaluation, beside Matteo.ipynb.
From the project root, reproduce the analysis with:

uv run python "3. Honest Evaluation/SPY_crosscheck/compare_sources.py"

The script reads only local input files and regenerates its output files.
Review and commit the folder using the team's Git workflow.
