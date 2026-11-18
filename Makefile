# Replication package for "On Persistence in Congressional Stock Selection".
#
#   make install         install the pinned Python environment (requirements.txt)
#   make exhibits        rebuild every table, macro and figure from the stored results in data/
#   make check-exhibits  rebuild the exhibits and confirm they match the committed versions
#
# The remaining targets rebuild the sample and the stored results. They need the licensed and
# restricted inputs described in README.md and cannot run from this repository alone.

PY ?= python3
export PYTHONPATH := $(CURDIR)/src
export MPLBACKEND := Agg
# Fixed timestamp so rebuilt PDFs are byte-stable.
export SOURCE_DATE_EPOCH ?= 1794355200

.PHONY: help install exhibits paired-bootstrap macros tables figures check-exhibits \
        collect-official market-panel identity-crosswalk sample \
        analysis analysis-initial analysis-extensions analysis-closeout analysis-round2 \
        analysis-continuation continuation-results

help:
	@sed -n '3,10p' Makefile | sed 's/^# //; s/^#//'

install:
	$(PY) -m pip install -r requirements.txt

# ── Exhibits: stored results -> LaTeX macros, table bodies and figures (no restricted input) ──

exhibits: paired-bootstrap macros tables figures

# The paired household bootstrap of the closeout (4,999 resamples), from the stored household means.
paired-bootstrap:
	$(PY) src/trading_on_disclosure_v4/round2/paired_engine.py \
	    --input data/v4/round2/closeout/closeout_2026-11-10/inputs/principal_household_means.csv \
	    --output data/v4/round2/closeout/closeout_2026-11-10/paired

# The macro files whose macros the paper prints.
macros:
	$(PY) paper/generate_assets.py
	$(PY) -m trading_on_disclosure_v4.ledger.macros
	$(PY) -m trading_on_disclosure_v4.existing.macros

tables:
	$(PY) -m trading_on_disclosure_v4.writing_sample.tables
	$(PY) paper/scripts/annals_closeout_tables.py
	$(PY) paper/scripts/abcd_tables.py
	$(PY) paper/scripts/coherence_appendix_tables.py
	$(PY) paper/scripts/writing_sample_core_tables.py
	$(PY) paper/scripts/writing_sample_appendix_tables.py

figures:
	$(PY) paper/scripts/annals_figures.py
	$(PY) paper/scripts/writing_sample_exhibits.py
	$(PY) scripts/writing_sample/figure3.py

check-exhibits:
	$(PY) scripts/check_exhibits.py snapshot
	$(MAKE) exhibits
	$(PY) scripts/check_exhibits.py compare

# ── Sample construction (restricted inputs; see README, "Data availability") ──

collect-official:
	$(PY) -m congressional_trading.official discover --chambers house,senate --start-year 2012 --end-year auto
	$(PY) -m congressional_trading.official backfill --chambers house,senate --start-year 2012 --end-year auto
	$(PY) -m congressional_trading.official build

market-panel:
	$(PY) scripts/v2/22_build_market_panel_2014plus.py

identity-crosswalk:
	$(PY) scripts/v2/07_build_identity_crosswalk.py

sample:
	$(PY) scripts/v2/21_canonical_top_journal_release.py --confirm-canonical-run

# ── Analysis (restricted inputs) ──

analysis: analysis-initial analysis-extensions analysis-closeout analysis-round2 analysis-continuation

# Sample panels, placebo calibration of size and power, the household estimates, and the average-return
# analysis. The upper-bound calibration reads the first estimates, and the second run of
# scripts/v4/run_results.py records the calibrated bound.
analysis-initial:
	$(PY) -m trading_on_disclosure_v4.data.build
	$(PY) -m trading_on_disclosure_v4.data.vendor_only_status
	$(PY) -m trading_on_disclosure_v4.outcomes.forward_cache
	$(PY) -m trading_on_disclosure_v4.outcomes.validate
	$(PY) -m trading_on_disclosure_v4.outcomes.pooled
	for c in T P; do $(PY) -m trading_on_disclosure_v4.null.run_null_panels --clock $$c; done
	$(PY) -m trading_on_disclosure_v4.calibration.run_size_v2
	$(PY) -m trading_on_disclosure_v4.calibration.run_power_v2
	$(PY) -m trading_on_disclosure_v4.calibration.run_power_k
	$(PY) -m trading_on_disclosure_v4.calibration.run_power_x
	$(PY) -m trading_on_disclosure_v4.calibration.power_common
	$(PY) -m trading_on_disclosure_v4.calibration.run_report_tables_v2
	$(PY) scripts/v4/run_results.py
	$(PY) -m trading_on_disclosure_v4.calibration.run_bound_calibration
	$(PY) -m trading_on_disclosure_v4.calibration.inverted_bound
	$(PY) -m trading_on_disclosure_v4.calibration.bound_validation
	$(PY) -m trading_on_disclosure_v4.calibration.sparse_trade_grid
	$(PY) -m trading_on_disclosure_v4.calibration.bcc
	$(PY) scripts/v4/run_results.py
	$(PY) -m trading_on_disclosure_v4.nullside.run_false_winners --sets all
	$(PY) -m trading_on_disclosure_v4.nullside.run_time_to_detect
	$(PY) -m trading_on_disclosure_v4.existing.rerun
	$(PY) -m trading_on_disclosure_v4.existing.table1
	$(PY) -m trading_on_disclosure_v4.existing.units
	$(PY) -m trading_on_disclosure_v4.existing.placebo_spread
	$(PY) -m trading_on_disclosure_v4.existing.appendix
	$(PY) -m trading_on_disclosure_v4.existing.portfolio
	$(PY) -m trading_on_disclosure_v4.pyun.replicate
	$(PY) -m trading_on_disclosure_v4.delay.stage1

# Coverage ledgers, the benchmark decomposition, overlap exclusions, power by household activity, the
# inference ladder, forward prediction, and repairs of identities, delistings and scanned histories.
analysis-extensions:
	$(PY) -m trading_on_disclosure_v4.ext.reproduce
	$(PY) -m trading_on_disclosure_v4.ext.ledgers
	$(PY) -m trading_on_disclosure_v4.ext.two_households
	$(PY) -m trading_on_disclosure_v4.ext.windows --record-level-counts data/v4/ext/ledger_tier_d_classification.csv
	$(PY) -m trading_on_disclosure_v4.ext.endpoint
	$(PY) -m trading_on_disclosure_v4.ext.decompose
	$(PY) -m trading_on_disclosure_v4.ext.overlap
	$(PY) -m trading_on_disclosure_v4.ext.activity_power --workers 8
	$(PY) -m trading_on_disclosure_v4.ext.inference
	$(PY) -m trading_on_disclosure_v4.ext.forward
	$(PY) -m trading_on_disclosure_v4.ext.delisting
	$(PY) -m trading_on_disclosure_v4.ext.repair
	$(PY) -m trading_on_disclosure_v4.ext.pilot_validation
	$(PY) -m trading_on_disclosure_v4.ext.history_extraction --workers 8
	$(PY) -m trading_on_disclosure_v4.ext.history_recovery

# Data corrections and the extended sample, with reference distributions for the persistence estimates.
analysis-closeout:
	$(PY) -m trading_on_disclosure_v4.final.reproduce
	$(PY) -m trading_on_disclosure_v4.final.duplicate_audit
	$(PY) -m trading_on_disclosure_v4.final.cache_extension
	$(PY) -m trading_on_disclosure_v4.final.cumulative
	$(PY) -m trading_on_disclosure_v4.final.reconcile
	$(PY) -m trading_on_disclosure_v4.final.reference --clocks T,P
	$(PY) -m trading_on_disclosure_v4.final.size_level_rank
	$(PY) -m trading_on_disclosure_v4.final.factor_delisting
	$(PY) -m trading_on_disclosure_v4.final.history_reread --workers 8
	$(PY) -m trading_on_disclosure_v4.final.history_recovery_final
	$(PY) -m trading_on_disclosure_v4.final.review_set
	$(PY) -m trading_on_disclosure_v4.final.power_verification

# Diagnostics added after the principal estimates: the paired household bootstrap, influence, limits of
# clustered statistics, small-cluster inference and variance checks.
analysis-round2:
	$(PY) -m trading_on_disclosure_v4.round2.exports
	$(PY) -m trading_on_disclosure_v4.round2.paired
	$(PY) -m trading_on_disclosure_v4.round2.influence
	$(PY) -m trading_on_disclosure_v4.round2.limits
	$(PY) -m trading_on_disclosure_v4.round2.small_cluster_run
	$(PY) -m trading_on_disclosure_v4.round2.variance_audit
	$(PY) -m trading_on_disclosure_v4.round2.factor_fits
	$(PY) -m trading_on_disclosure_v4.round2.lag_sensitivity
	$(PY) -m trading_on_disclosure_v4.round2.baseline

# Continuation analysis (src/trading_on_disclosure_v4/writing_sample/): forecasts of returns
# from recent alpha, persistence under partial continuation, and the supporting robustness, placebo and inference
# calculations. `continuation-results` rebuilds its results file and figure from the stored components only.
analysis-continuation:
	$(PY) scripts/writing_sample/build_computation.py --restricted
	$(PY) scripts/coherence/aggregate_checks.py

continuation-results:
	$(PY) scripts/writing_sample/build_computation.py
