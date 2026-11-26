# TLT, GLD and DBC source cross-check review

Owner: Data

Status: comparison completed; discrepancies documented; final source/protocol acceptance remains with the team. This report does not claim that every discrepancy is resolved.

## Inputs and scope

Reviewed archive: `20261126T205751023933Z.zip`. SHA-256: `029358bf1d8f56847d7ca4984cc4d7223f299762ece10919a596a457ce8742a2`.

Yahoo revision: `results/data_revision/20261119T203414986964Z`. Comparison output: `results/etf_crosschecks/20261126T205751023933Z`.

Window: 6 February 2006 through 31 December 2025. Each ETF has 5,007 common price dates and 5,006 matched return intervals. Returns were independently recalculated from the aligned price columns, reproducing the reported maximum differences and counts above the diagnostic threshold. All three archived Stooq responses match the SHA-256 values in the summary. Original Yahoo files are not included in the archive; their hashes cannot be independently rechecked here. The execution summary reports that they were unchanged.

## Results

| ETF | Common dates | Median absolute difference (bp) | 95th percentile (bp) | Maximum (bp) | Intervals above 10 bp |
|---|---:|---:|---:|---:|---:|
| TLT | 5007 | 0.373847 | 3.793909 | 101.554850 | 32 |
| GLD | 5007 | 0.000236 | 0.000702 | 16.238516 | 6 |
| DBC | 5007 | 0.134511 | 1.797796 | 58.653511 | 18 |

One basis point is 0.01 percentage points of return. The 10 bp threshold is an investigation filter chosen in the audit script, not an approved acceptance criterion. Small median differences do not settle the largest discrepancies.

## Missing date

Stooq omits 17 February 2011 for all three ETFs; Yahoo includes it. The archived calendar reports 5,008 expected sessions and no missing Yahoo sessions. The return ending on 18 February consequently spans two trading sessions in the common-date comparison. It must not be described as a one-session return or filled with an invented price. The cause of the shared Stooq omission is not established.

## Independent GLD closing-price evidence

The SPDR Gold Trust prospectus filed on 22 August 2008 explicitly reports a NYSE Arca closing price of USD 82.30 on 21 August 2008.

Source: https://www.sec.gov/Archives/edgar/data/1222333/000095012308010029/y63337sv3asr.htm (prospectus opening paragraph; accessed 26 November 2026).

On that date the archived Yahoo Close is 82.30000305; Stooq Close is 82.17000000. The issuer filing supports Yahoo to within USD 0.0001 on this specific date. It does not independently validate the entire GLD series. The six GLD return intervals above 10 bp occur in three adjacent-date pairs: 26/29 January 2007, 21/22 February 2007, and 21/22 August 2008. A disputed endpoint affects two consecutive returns; these are not necessarily six separate price errors. The other dates remain unresolved.

## TLT and DBC exceptions

TLT has 32 intervals above 10 bp, including the largest differences on 13 and 16 March 2015. DBC has 18, with its maximum on 9 October 2006. The largest absolute differences are approximately 1.016 and 0.587 percentage points, respectively. Differences may involve price observations or adjustment conventions; neither cause nor the correct provider is established for these cases by the two-source comparison alone. Corporate-action records are needed before attributing adjustment discrepancies to dividends.

## Disposition and remaining work

Preserve all Yahoo and Stooq snapshots and the discrepancy registers. Do not overwrite prices, delete affected dates, interpolate missing prices, or change folds to obtain agreement. Yahoo remains the proposed primary source; retain a documented exception register for team review. Further issuer/exchange evidence is needed to close unresolved cases or the team must explicitly accept the remaining exceptions before the protocol is treated as final.

The ETF selection rule and document formatting are separate tasks. Evaluation has reported that the 20-session gap and T-bill conversion/timing were settled in Metrics; the exact Metrics wording must be read before updating those references. No models were trained as part of this review.
