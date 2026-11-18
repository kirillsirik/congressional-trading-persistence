# On Persistence in Congressional Stock Selection

Kirill Sirik, Princeton University. Working paper, November 2026.

- [Full paper](paper/Congressional_Trading_Complete.pdf) (main text, references and Appendices A–G)
- [Paper without appendix](paper/Congressional_Trading_Reading_Copy.pdf)
- [Appendix](paper/Congressional_Trading_Appendix.pdf)

## Abstract

Do the same congressional households repeatedly pick stocks that outperform? For 7,964 purchases by 38 households in 2015–2020 and 2021–2025, persistence is negative under a factor-only benchmark and positive under an event-study benchmark, which carries forward recent alpha; neither estimate is statistically significant. The positive estimate can reflect repeated underperformance: recent alpha barely predicts later returns (forecast coefficient 0.057, where the benchmark assumes one), yet the same households keep buying stocks with high recent alpha. A clustered pooled test can miss an arbitrarily large gain to one household; for a household specified in advance making 50 purchases a year, detecting a 5 percent annual advantage would take about 74 years. Benchmarks carrying recent alpha forward should first be checked against later returns.

## Reproducing the exhibits

`make exhibits` rebuilds every table, figure and macro that the paper prints from the stored results in `data/v4/`,
which contain no licensed or proprietary data:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
make install          # the pinned packages of requirements.txt
make check-exhibits   # rebuild the exhibits and compare them with the committed files (about a minute)
```

It runs four steps: the paired household bootstrap of the closeout from the stored household means
(`make paired-bootstrap`), the macros (`make macros`), the tables (`make tables`) and the figures (`make figures`).
No restricted file is read. The table files and the macros that the paper prints come out byte for byte as the
paper loads them, and the generators check the counts quoted in captions and notes against the stored results.
`make check-exhibits` fails if any committed output changes. `make continuation-results` rebuilds
`data/v4/writing_sample/results.csv` from its stored components and runs that stage's checks.

The figures use the URW base35 P052 fonts (https://github.com/ArtifexSoftware/urw-base35-fonts, or the Debian
package `fonts-urw-base35`); if they are not installed, set `P052_FONT_DIR` to a folder holding `P052-Roman.otf`,
`P052-Italic.otf`, `P052-Bold.otf` and `P052-BoldItalic.otf`. Tested with Python 3.11 on an Apple M2 Max (macOS).

## Data availability and provenance

The author has legitimate access to, and permission to use, the data used in the paper. Some of these data
cannot be redistributed:

- **CRSP** daily stock returns are licensed through Wharton Research Data Services (WRDS) and are not
  included.
- **Vendor transaction records** were taken from the Capitol Trades website (2iQ Research). They are not
  redistributed, and the code used to collect them is not included; researchers who want the records
  should obtain them from 2iQ Research under its terms.
- **Periodic transaction reports** filed under the STOCK Act are public records, available from the U.S.
  House Clerk and the Senate eFD system. They are not redistributed here (Senate reports carry statutory
  use restrictions); the collector in this repository downloads them.
- **Fama–French factors** are freely available from the Kenneth R. French Data Library.

The stored results in `data/v4/` contain aggregate and household-level statistics only. Households appear
under anonymous indices; no member of Congress is named and no security-level CRSP return is included.
`data/README.md` describes each input, how to obtain it, where the code expects it, and the SHA-256
digests against which a replicator can verify it.

| Data | Source | Coverage used | Included |
|---|---|---|---|
| Periodic transaction reports | House Clerk, Senate eFD | Filings from July 2012, when the STOCK Act reporting requirement took effect | No (collector provided) |
| Vendor transaction records | capitoltrades.com (2iQ Research) | Snapshot of 21 July 2026 | No (available from 2iQ Research) |
| CRSP Daily Stock File v2 (CIZ) | WRDS | 1980-01-02 to July 2026 | No |
| Fama–French 5 factors and momentum, daily | Kenneth R. French Data Library | Through 2026-05-29 | No (public download) |
| Values quoted from Pyun (2026) | The cited paper | — | `paper/data/pyun_quoted.csv` |
| Stored results that the paper uses | This project | — | `data/v4/` |

## Rebuilding the stored results from the restricted inputs

1. Obtain the inputs listed in `data/README.md`, copy `.env.example` to `.env`, set the paths, and export
   them (`set -a; . ./.env; set +a`).
2. Build the sample: place the vendor snapshot under `$DROPBOX_PROJECT_ROOT/capitol_trades/`, then run
   `make collect-official`, `make market-panel`, `make identity-crosswalk` and `make sample`. The last
   step writes the sample frames to `outputs/frames/` and runs only from a clean Git checkout.
3. Run the analysis: `make analysis`, or its five stages in order: `analysis-initial` (sample panels, placebo
   calibration of size and power, the household estimates and the average-return analysis),
   `analysis-extensions` (coverage ledgers, the benchmark decomposition, overlap exclusions, power by household
   activity and data repairs), `analysis-closeout` (data corrections and the extended sample), `analysis-round2`
   (the paired household bootstrap, influence, limits of clustered statistics and small-cluster inference) and
   `analysis-continuation` (forecasts of returns from recent alpha, persistence under partial continuation, and
   the related robustness, placebo and inference calculations). Each stage writes its stored
   results under `data/v4/` and its restricted intermediates under `data/v4/restricted/`. The stages also write
   results that the paper does not use; those are not stored in this repository.
4. Run `make exhibits`.

One stored component, `data/v4/writing_sample/components/factual_checks.csv` (counts for the two
households whose early filings are scanned), is not rebuilt by this repository: the check that produced it
compares member identities with the cited papers and is not distributed.

## Contents

| Path | Contents |
|---|---|
| `src/congressional_trading/` | Collection and parsing of the official filings (House Clerk, Senate eFD); reading of the vendor snapshot; CRSP and factor loaders |
| `src/trading_on_disclosure_v2/` | Sample construction: official and vendor records, filer identities, amendments, security matching, disclosure and trade dates, signals, returns, and the average-return analysis |
| `src/trading_on_disclosure_v4/` | The persistence analysis: event tables, return cache, placebo panels, estimators, inference, calibration, and the later analysis stages (`ext/`, `final/`, `round2/`, `writing_sample/`) |
| `scripts/` | Entry points of the sample-construction stages, the household estimates, the continuation analysis, and `check_exhibits.py` |
| `paper/` | The three PDFs, the generators of the exhibits (`generate_assets.py`, `scripts/`) and the exhibits they write |
| `data/` | Stored results (`data/README.md`) |
| `configs/` | Settings read by the sample construction and the filing collector, and the schema of the filing validator |

## Tables and figures

Exhibit numbers follow the posted paper. Generated files are under `paper/` and stored results under `data/v4/`.
Tables A1, D1 (Panel A), F3, F8 and G1–G3 are set directly in the paper's LaTeX; every number in them appears
in the stored results named in their rows.

| Exhibit | Written by (file loaded by the paper) | Stored results |
|---|---|---|
| Table 1 | `paper/scripts/writing_sample_exhibits.py` (`writing_sample_generated/table01_sample.tex`) | `existing/table1_dedup.csv`, `existing/appendix_dedup/attrition.csv`, `ext/ledger_tier_d_classification.csv`, `ext/ledger_households.csv`, `final/sample_flow_ledger.csv`, `writing_sample/tables/S1_continuation_purchases.csv`, `round2/closeout/closeout_2026-11-10/inputs/principal_household_means.csv` |
| Table 2 | `paper/scripts/writing_sample_core_tables.py` (`table03_persistence.tex`) | `writing_sample/tables/table3_persistence_both_benchmarks.csv`, `round2/closeout/closeout_2026-11-10/paired/`, `round2/closeout/closeout_2026-11-10/influence/specification_statistics.csv` |
| Figure 1 | `writing_sample_exhibits.py` (`figure02_households.tex`, `figures/writing_sample_households.pdf`) | `round2/closeout/closeout_2026-11-10/inputs/benchmark_figure.csv`, `principal_household_means.csv`; `results.csv` |
| Table 3 | `writing_sample_core_tables.py` (`table04_levels.tex`) | `writing_sample/tables/table4_panelA_levels.csv`, `S1_continuation_purchases.csv`, `S1_continuation_formation_grid.csv`; `ext/ledger_households.csv` |
| Table 4 | `writing_sample_core_tables.py` (`table05_decomposition.tex`) | `round2/closeout/closeout_2026-11-10/paired/paired_summary.csv` |
| Figure 2 | `scripts/writing_sample/figure3.py` (`figures/writing_sample_alpha_curve.pdf`) | `writing_sample/results.csv` |
| Table 5 | `writing_sample_core_tables.py` (`table06_robustness.tex`) | `ext/overlap_exclusion.csv`, `final/decomposition_corrected.csv`, `round2/closeout/closeout_2026-11-10/paired/point_statistics.csv` |
| Table 6 | `writing_sample_exhibits.py` (`table02_average.tex`) | `existing/appendix_dedup/horizons.csv`, `existing/table1_dedup.csv` |
| Figure 3 | `writing_sample_exhibits.py` (`figure04_power.tex`, `figures/writing_sample_power.pdf`) | `ext/activity_power_surface.csv`; the critical value and maxima quoted in Section 6.3 are in `ext/activity_power_observed.csv` (`trading_on_disclosure_v4.ext.activity_power_observed`) |
| Table A1 | set in the paper; `writing_sample_exhibits.py` checks its counts | `ext/ledger_tier_d_classification.csv` |
| Table B1 | `paper/scripts/abcd_tables.py` (row macros in `revised_generated/abcd_overrides.tex`) | `existing/appendix_dedup/full_results.csv` |
| Table C1 | `writing_sample_appendix_tables.py` (`appendix_units_dates.tex`, with rows from `abcd_overrides.tex`) | `existing/appendix_dedup/full_results.csv`, `writing_sample/tables/tableC1_six_factor_spread.csv` |
| Figure C1 | `writing_sample_exhibits.py` (`figure01_timeline.tex`, `figures/writing_sample_timeline.pdf`) | none (a diagram) |
| Table C2 | `annals_closeout_tables.py`, `writing_sample_appendix_tables.py` (`appendix_clustering.tex`) | `ext/inference_ladder.csv`, `final/spread_corrected.csv` |
| Table C3 | `writing_sample_appendix_tables.py` (`appendix_small_cluster.tex`) | `round2/closeout/closeout_2026-11-10/small_cluster/small_cluster_inference.csv` |
| Table D1 | set in the paper, with rows from `abcd_overrides.tex` and macros from `generated/existing_rerun_macros.tex` | `existing/portfolio_dedup_summary.csv` |
| Figure F1 | `scripts/writing_sample/figure3.py` (`figures/writing_sample_alpha_affine.pdf`) | `writing_sample/results.csv` |
| Table F1 | `coherence_appendix_tables.py`, `writing_sample_appendix_tables.py` (`appendix_alternative_decompositions.tex`) | `ext/decomposition.csv` |
| Table F2 | `writing_sample_appendix_tables.py` (`appendix_paired_inference.tex`) | `writing_sample/tables/S5_conclusion_matrix.csv` |
| Table F3 | set in the paper | `ext/overlap_exclusion.csv` |
| Table F4 | `writing_sample_appendix_tables.py` (`appendix_common_households.tex`) | `coherence/common_household_statistics.csv` |
| Table F5 | `writing_sample_core_tables.py` (`appendix_shifted_dates.tex`) | `writing_sample/tables/table3_persistence_both_benchmarks.csv`, `ext/decomposition.csv` |
| Tables F6, F7 | `writing_sample_appendix_tables.py` (`appendix_sensitivity.tex`, `appendix_sensitivity_disclosure.tex`) | `writing_sample/tables/tableF5_both_benchmarks.csv`, `writing_sample/results.csv` |
| Table F8 | set in the paper | `ext/forward_prediction_historical.csv` |
| Table F9 | `writing_sample_appendix_tables.py` (`appendix_continuation.tex`) | `writing_sample/tables/S1_continuation_purchases.csv`, `S1_continuation_formation_grid.csv` |
| Table F10 | `writing_sample_appendix_tables.py` (`appendix_placebo_levels.tex`) | `writing_sample/tables/S1_placebo_levels.csv` |
| Table G1 | set in the paper; the same numbers are in `generated/results_macros.tex` (`trading_on_disclosure_v4.ledger.macros`) | `results.csv` |
| Table G2 | set in the paper | `ext/activity_power_frontiers.csv` |
| Table G3 | set in the paper | `round2/closeout/closeout_2026-11-10/limits/dataset_limits.csv`, `writing_sample/results.csv`, `existing/table1_dedup.csv` |
| Figure G1 | `paper/scripts/annals_figures.py` (`figures/annals_figure3_detection_trade.pdf`) | `calibration/figure2_detection_region_v3_*.csv`, `calibration/frontiers_80.csv` |

The figures in the posted PDFs were drawn by the same scripts from the same data; their font rendering can differ
slightly from a rebuild on another machine.

## License

The code is released under the MIT License (`LICENSE`). The license does not cover the third-party data or
the cited papers.

## References

Center for Research in Security Prices (2026). CRSP US Stock Database, Daily Stock File, version 2 (CIZ).
Wharton Research Data Services.

French, K. R. (2026). Data Library: Fama/French 5 Factors (2x3) and Momentum Factor, daily.
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Pyun, C. (2026). The filing is not the event: Congressional trading and the market's response to disclosure.
SSRN Working Paper 5295880, version of 13 September 2026.

U.S. House of Representatives, Office of the Clerk. Financial disclosure reports. https://disclosures-clerk.house.gov

U.S. Senate, Office of Public Records. Electronic Financial Disclosures (eFD). https://efdsearch.senate.gov
