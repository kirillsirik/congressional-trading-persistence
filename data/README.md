# Data

`data/v4/` holds the stored results of the analysis: `make exhibits` generates the paper's tables, figures and
printed macros from these files. The licensed and restricted inputs from which the results
were computed are not distributed; they are described below with instructions for obtaining them.

## Stored results (`data/v4/`)

These are the stored results that the paper's exhibits and text use. The analysis stages write more files;
those the paper does not use are not stored here, and `make analysis` writes them again. The files contain
aggregate and household-level statistics. Households appear only under anonymous indices
(`household_index`); no member of Congress is named, and no security-level return from CRSP is included.

| Path | Contents | Written by |
|---|---|---|
| `results.csv`, `results_time_to_detect.csv` | Household persistence estimates and detection horizons | `scripts/v4/run_results.py` |
| `existing/` | The average-return analysis on the de-duplicated sample: Table 1 counts, horizons, the funded account, the appendix tables | `trading_on_disclosure_v4.existing` |
| `ext/` | Record ledgers, decompositions, overlap exclusions, power by household activity, forward prediction, the inference ladder | `trading_on_disclosure_v4.ext` |
| `final/` | Sample flow, decomposition and spreads on the extended sample | `trading_on_disclosure_v4.final` |
| `calibration/` | Size, power and detection frontiers on placebo datasets, the upper-bound validation | `trading_on_disclosure_v4.calibration` |
| `round2/closeout/closeout_2026-11-10/` | Household means, the paired household bootstrap, influence, limits, small-cluster inference | `trading_on_disclosure_v4.round2` (`round2/paired_engine.py` for `paired/`) |
| `writing_sample/` | The continuation analysis: stored components, `results.csv`, the summary tables in `tables/` | `trading_on_disclosure_v4.writing_sample`, `scripts/writing_sample/` |
| `coherence/` | Persistence on the households common to both entry-date conventions | `scripts/coherence/aggregate_checks.py` |
| `diagnostics/`, `pooled/`, `delay/`, `pyun/`, `nullside/`, `figures/`, `round2/macro_sources.json` | Outcome-free counts, pooled outcome cutoffs, the filing-delay regression, the bridge to Pyun (2026), and figure inputs, read by `paper/generate_assets.py` | `trading_on_disclosure_v4` |
| `environment_freeze.txt` | Python packages of the environment that produced the results | |

The extension and closeout stages record the SHA-256 digests of the restricted inputs they read
(`ext/input_manifest.json`, `final/input_manifest.json`), with the paths in use when each stage ran. `final/baseline_freeze/SHA256SUMS_restricted_inputs.txt`
lists the digests of the restricted intermediates as of 6 November 2026.
`round2/closeout/closeout_2026-11-10/baseline_reconciliation.csv`
records the reproduction of the principal estimates from the restricted inputs with the original estimators
on 10 November 2026 (77 of 77 checks pass; largest relative difference 6.9e-15).

## Restricted inputs (not included)

| Input | Source and access | Used for | Expected location |
|---|---|---|---|
| Periodic transaction reports, 2012 onward | U.S. House Clerk (disclosures-clerk.house.gov) and Senate eFD (efdsearch.senate.gov). Public records; Senate reports carry the use restrictions of 5 U.S.C. app. §105(c) | Trades, owners, amounts and dates of each filing | Official-filing store built by `make collect-official` (`OFFICIAL_STORE_DIR`) |
| Capitol Trades transaction table, snapshot of 21 July 2026 | capitoltrades.com (2iQ Research). Not redistributed, and the collection code is not included; obtain the records from 2iQ Research under its terms | Vendor records and publication dates | `$DROPBOX_PROJECT_ROOT/capitol_trades/20260721_221640/` |
| CRSP Daily Stock File, version 2 (CIZ format) | Wharton Research Data Services; requires a subscription | Returns, delisting returns, security names and identifiers | `$DROPBOX_PROJECT_ROOT/data/raw/crsp/` |
| Fama–French five factors and momentum, daily | Kenneth R. French Data Library (free) | Factor benchmarks | `$DROPBOX_PROJECT_ROOT/data/raw/factors/factors_daily.csv` |

**CRSP extraction.** From WRDS, CRSP Quarterly (or Monthly) Update, Daily Stock File version 2: all
PERMNOs, no screens in the query, the columns `PERMNO, DlyCalDt, Ticker, TradingSymbol, HdrCUSIP, PERMCO,
SICCD, PrimaryExch, SecurityType, ShareType, DlyCap, DlyRet` plus the index returns `vwretd` and `sprtrn`.
The analysis used three pulls: `crsp_v2_daily_1980_present.csv.gz` (1980-01-02 to 2026-03-31),
`crsp_v2_daily_202604_202606.csv.gz` (April–June 2026) and `crsp_v2_daily_202607_202607.csv.gz` (July 2026).
`congressional_trading.data.crsp_loader.trim_crsp_v2_file` applies the common-stock screens
(`PrimaryExch` in N, A, Q; `SecurityType` EQTY; `ShareType` NS); the market-panel builder reads the
trimmed extract from 2012 onward as `crsp_trimmed_2012plus.csv` and re-attaches each security's
delisting-return row, which CIZ records as its own daily observation.

**Factors.** "Fama/French 5 Factors (2x3) [Daily]" and "Momentum Factor (Mom) [Daily]", merged and
converted to decimals by `congressional_trading.data.factor_loader.build_factors_from_french`; columns
`date, mkt_rf, smb, hml, rmw, cma, mom, rf`. The file used ends on 2026-05-29.

**Verifying inputs.** `data/v4/final/baseline_freeze/SHA256SUMS_restricted_inputs.txt` and the input
manifests list the digest of every restricted file the analysis read. A replicator with access can compare
their own files against them before running the restricted stages.
